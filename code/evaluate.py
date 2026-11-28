"""Evaluación: líneas base, benchmarks, métricas y bitácora de operaciones.

Responsabilidades (secciones 3.5, 3.7, 3.8 y 3.9 de los lineamientos):
- Líneas base triviales con costos: aleatorio, siempre largo y siempre plano.
- Benchmarks con el mismo costo: comprar y mantener BTCUSDT e indexar al S&P 500.
- Métricas por fold y agregadas (media y desviación estándar): retorno total y anualizado,
  volatilidad anualizada, Sharpe, Sortino, máximo drawdown, Calmar, hit rate, duración
  promedio de las operaciones, rotación y alfa frente a cada benchmark.
- Bitácora de operaciones (artifacts/trade_log_fold_k.csv) y resumen de métricas
  (artifacts/metrics_summary.csv).
- Experimentos de sensibilidad: resolución intrabarra y retraso de una barra.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from data import SEMILLA
from env import ACCION_LARGO, ACCION_PLANO, COLUMNAS_BITACORA
from reward import COSTO_POR_LADO  # una sola fuente para el 0.125 % por lado

RAIZ = Path(__file__).resolve().parents[1]
RUTA_SP500 = RAIZ / "data" / "raw" / "sp500_diario.csv"
RUTA_ARTEFACTOS = RAIZ / "artifacts"
# Hora UTC desde la que vale cada cierre del S&P 500. Las 16:00 de Nueva York son las 21:00 UTC en
# invierno y las 20:00 UTC en verano; usar 21:00 todo el año garantiza que nunca se use un cierre
# antes de que exista (en verano se usa con una hora de retraso).
HORA_CIERRE_SP500_UTC = 21
DIAS_POR_ANIO = 365.25  # BTCUSDT opera 24/7: se anualiza con días calendario


def cargar_sp500(ruta: Path = RUTA_SP500) -> pd.Series:
    """Cierres diarios del S&P 500 (índice ^GSPC) para el benchmark de indexar al S&P 500.

    La primera vez se descargan con yfinance y se guardan en data/raw/sp500_diario.csv; desde
    entonces se leen del CSV, así el pipeline no depende de internet y siempre usa los mismos
    datos. Solo se usan para el benchmark (sección 3.7): nunca entran al estado del agente.

    Supuesto: ^GSPC es un índice de precio y no incluye dividendos (del orden de 1.5 % anual), así
    que el benchmark subestima ligeramente lo que habría ganado un fondo indexado.
    """
    if not ruta.exists():
        import yfinance as yf  # solo se importa si hay que descargar

        crudo = yf.download("^GSPC", start="2017-08-01", end="2026-04-28", auto_adjust=True, progress=False)
        cierres = crudo["Close"].squeeze().rename("close")
        cierres.index = pd.to_datetime(cierres.index).tz_localize(None)
        cierres.index.name = "fecha"
        ruta.parent.mkdir(parents=True, exist_ok=True)
        cierres.to_csv(ruta)
    return pd.read_csv(ruta, index_col="fecha", parse_dates=True)["close"]


def sp500_horario(cierres: pd.Series, indice: pd.DatetimeIndex) -> pd.Series:
    """Lleva los cierres diarios del S&P 500 a la malla horaria de BTCUSDT.

    Cada cierre vale a partir de las HORA_CIERRE_SP500_UTC (21:00 UTC) de su día y se mantiene hasta
    el siguiente cierre: en las noches y los fines de semana el índice no cambia.
    """
    con_hora = cierres.copy()
    con_hora.index = con_hora.index + pd.Timedelta(hours=HORA_CIERRE_SP500_UTC)
    malla = con_hora.reindex(con_hora.index.union(indice)).ffill()
    return malla.reindex(indice).rename("sp500")


def comprar_y_mantener(precios: pd.Series, costo: float = COSTO_POR_LADO, capital_inicial: float = 1.0) -> pd.Series:
    """Curva de capital de comprar en la primera barra y vender en la última (sección 3.7).

    Parte de la fórmula de buy-and-hold del S18, capital_t = capital_inicial * (1 - costo) * p_t / p_0,
    y además cobra la venta en la última barra: el agente cierra sus posiciones pagando la salida,
    así que el benchmark también. El primer valor es el capital antes de comprar, para que el
    costo de entrada cuente en las métricas.
    """
    capital = capital_inicial * (1 - costo) * precios / precios.iloc[0]
    capital.iloc[0] = capital_inicial
    capital.iloc[-1] *= 1 - costo
    return capital.rename("capital")


def bitacora_comprar_y_mantener(precios: pd.Series, costo: float = COSTO_POR_LADO, capital_inicial: float = 1.0) -> pd.DataFrame:
    """La única operación de comprar y mantener, con los mismos números que su curva de capital.

    Compra con todo el capital después del costo de entrada y vende en la última barra.
    """
    p0, pT = precios.iloc[0], precios.iloc[-1]
    tamano = capital_inicial * (1 - costo) / p0
    pnl_bruto = tamano * (pT - p0)
    costos = costo * capital_inicial + costo * tamano * pT
    return pd.DataFrame([{
        "entrada": precios.index[0], "salida": precios.index[-1], "direccion": 1, "razon_salida": "cierre",
        "precio_entrada": p0, "precio_salida": pT, "tamano": tamano,
        "pnl_bruto": pnl_bruto, "costos": costos, "pnl_neto": pnl_bruto - costos,
    }], columns=COLUMNAS_BITACORA)


def precios_benchmark(barras: pd.DataFrame, cierres_sp500: pd.Series) -> dict[str, pd.Series]:
    """Precios de BTCUSDT y del S&P 500 en la malla horaria de las barras de un periodo."""
    return {"btc": barras["close"], "sp500": sp500_horario(cierres_sp500, barras.index)}


def benchmarks(barras: pd.DataFrame, cierres_sp500: pd.Series) -> dict[str, pd.Series]:
    """Curvas de capital de comprar y mantener BTCUSDT y el S&P 500 en las mismas barras."""
    return {nombre: comprar_y_mantener(precios) for nombre, precios in precios_benchmark(barras, cierres_sp500).items()}


def anios(indice: pd.DatetimeIndex) -> float:
    """Años calendario entre la primera y la última barra."""
    return (indice[-1] - indice[0]) / pd.Timedelta(days=DIAS_POR_ANIO)


def periodos_por_anio(capital: pd.Series) -> float:
    """Rendimientos por año observados en la serie.

    Se calcula con los datos y no con 8,760 fijo porque faltan horas en el histórico (1,114 en
    total): así la anualización es consistente con los rendimientos que de verdad hay.
    """
    return (len(capital) - 1) / anios(capital.index)


def rendimientos(capital: pd.Series) -> pd.Series:
    """Rendimientos simples barra a barra de una curva de capital."""
    return capital.pct_change().dropna()


def retorno_total(capital: pd.Series) -> float:
    """Ganancia o pérdida acumulada del periodo como fracción del capital inicial."""
    return capital.iloc[-1] / capital.iloc[0] - 1


def retorno_anualizado(capital: pd.Series) -> float:
    """Retorno compuesto por año: (1 + retorno total)^(1 / años) - 1."""
    return (1 + retorno_total(capital)) ** (1 / anios(capital.index)) - 1


def volatilidad_anualizada(capital: pd.Series) -> float:
    """Desviación estándar de los rendimientos barra a barra llevada a escala anual."""
    return rendimientos(capital).std() * np.sqrt(periodos_por_anio(capital))


def drawdown(capital: pd.Series) -> pd.Series:
    """Caída de la curva de capital respecto a su máximo previo (0 o negativa)."""
    return capital / capital.cummax() - 1


def maximo_drawdown(capital: pd.Series) -> float:
    """La peor caída desde un máximo previo durante el periodo (valor negativo)."""
    return drawdown(capital).min()


def sharpe(capital: pd.Series) -> float:
    """Sharpe anualizado con tasa libre de riesgo de 0. NaN si la curva no se mueve (siempre plano)."""
    r = rendimientos(capital)
    desviacion = r.std()
    if desviacion == 0:
        return np.nan
    return r.mean() / desviacion * np.sqrt(periodos_por_anio(capital))


def sortino(capital: pd.Series) -> float:
    """Como el Sharpe, pero solo penaliza la volatilidad a la baja (objetivo de 0).

    La desviación a la baja es la raíz del promedio de min(r, 0)^2 sobre todas las barras.
    """
    r = rendimientos(capital)
    desviacion_baja = np.sqrt((np.minimum(r, 0) ** 2).mean())
    if desviacion_baja == 0:
        return np.nan
    return r.mean() / desviacion_baja * np.sqrt(periodos_por_anio(capital))


def calmar(capital: pd.Series) -> float:
    """Retorno anualizado entre el máximo drawdown en valor absoluto. NaN si nunca hubo caída."""
    caida = maximo_drawdown(capital)
    if caida == 0:
        return np.nan
    return retorno_anualizado(capital) / abs(caida)


def metricas_capital(capital: pd.Series) -> dict[str, float]:
    """Métricas de la sección 3.7 que salen solo de la curva de capital."""
    metricas = {
        "retorno_total": retorno_total(capital),
        "retorno_anualizado": retorno_anualizado(capital),
        "volatilidad_anualizada": volatilidad_anualizada(capital),
        "sharpe": sharpe(capital),
        "sortino": sortino(capital),
        "maximo_drawdown": maximo_drawdown(capital),
        "calmar": calmar(capital),
    }
    return {nombre: float(valor) for nombre, valor in metricas.items()}


def correr_politica(entorno, politica, semilla: int = SEMILLA) -> tuple[pd.Series, pd.DataFrame]:
    """Recorre un episodio completo del entorno con politica(obs) -> acción, como correr() del S18.

    Para evaluar, el entorno se crea con inicio_aleatorio=False, así el episodio es todo el periodo.
    Devuelve la curva de capital barra por barra y la bitácora de operaciones del episodio.
    """
    obs, _ = entorno.reset(seed=semilla)
    entorno.action_space.seed(semilla)  # para que la política aleatoria también sea reproducible
    terminado = truncado = False
    while not (terminado or truncado):
        obs, _, terminado, truncado, _ = entorno.step(politica(obs))
    return entorno.curva_capital(), entorno.bitacora_df()


def guardar_bitacora(bitacora: pd.DataFrame, fold: int, carpeta: Path = RUTA_ARTEFACTOS) -> Path:
    """Escribe artifacts/trade_log_fold_k.csv con las columnas en el orden de COLUMNAS_BITACORA."""
    faltan = set(COLUMNAS_BITACORA) - set(bitacora.columns)
    if faltan:
        raise ValueError(f"A la bitácora le faltan columnas: {sorted(faltan)}")
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"trade_log_fold_{fold}.csv"
    bitacora[COLUMNAS_BITACORA].to_csv(ruta, index=False)
    return ruta


def metricas_operaciones(bitacora: pd.DataFrame, capital: pd.Series) -> dict[str, float]:
    """Métricas de la sección 3.7 que salen de la bitácora.

    - hit_rate: fracción de operaciones con P&L neto positivo.
    - duracion_promedio_h: horas promedio entre la entrada y la salida.
    - rotacion_anual: monto operado por año (compra más venta de cada operación) entre el capital
      promedio. Una rotación de 10 significa que cada año se compra y vende 10 veces el capital.
    Sin operaciones (siempre plano), hit rate y duración quedan en NaN y la rotación en 0.
    """
    if bitacora.empty:
        return {"operaciones": 0.0, "hit_rate": np.nan, "duracion_promedio_h": np.nan, "rotacion_anual": 0.0}
    duracion = (bitacora["salida"] - bitacora["entrada"]) / pd.Timedelta(hours=1)
    monto = (bitacora["tamano"] * (bitacora["precio_entrada"] + bitacora["precio_salida"])).sum()
    return {
        "operaciones": float(len(bitacora)),
        "hit_rate": float((bitacora["pnl_neto"] > 0).mean()),
        "duracion_promedio_h": float(duracion.mean()),
        "rotacion_anual": float(monto / capital.mean() / anios(capital.index)),
    }


def alfa(capital: pd.Series, capital_benchmark: pd.Series) -> float:
    """Retorno en exceso: retorno anualizado de la estrategia menos el del benchmark.

    Se anualiza para que los folds se puedan promediar aunque el 4 dure solo cuatro meses.
    """
    return retorno_anualizado(capital) - retorno_anualizado(capital_benchmark)


def metricas(capital: pd.Series, bitacora: pd.DataFrame, curvas_benchmark: dict[str, pd.Series]) -> dict[str, float]:
    """Todas las métricas de la sección 3.7 para una estrategia en un fold."""
    resultado = metricas_capital(capital) | metricas_operaciones(bitacora, capital)
    for nombre, curva in curvas_benchmark.items():
        resultado[f"alfa_{nombre}"] = float(alfa(capital, curva))
    return resultado


def metricas_benchmarks(barras: pd.DataFrame, cierres_sp500: pd.Series) -> dict[str, dict[str, float]]:
    """Métricas de la sección 3.7 de los dos benchmarks en las barras de un periodo.

    Cada benchmark es una sola operación (comprar al inicio y vender al final), así que su bitácora
    sale de bitacora_comprar_y_mantener() y sus métricas se calculan igual que las del agente.
    """
    precios = precios_benchmark(barras, cierres_sp500)
    curvas = {nombre: comprar_y_mantener(serie) for nombre, serie in precios.items()}
    return {
        nombre: metricas(curvas[nombre], bitacora_comprar_y_mantener(precios[nombre]), curvas)
        for nombre in precios
    }


def politicas_base(entorno) -> dict:
    """Las tres líneas base triviales del S19: quien no piensa. El agente tiene que superarlas."""
    return {
        "aleatorio": lambda obs: entorno.action_space.sample(),
        "siempre_largo": lambda obs: ACCION_LARGO,
        "siempre_plano": lambda obs: ACCION_PLANO,
    }


def metricas_lineas_base(crear_entorno, curvas_benchmark: dict[str, pd.Series]) -> dict[str, dict[str, float]]:
    """Métricas de la sección 3.7 de cada línea base, con los mismos costos que el agente.

    crear_entorno() devuelve un entorno nuevo del periodo (inicio_aleatorio=False), uno por línea base.
    """
    resultados = {}
    for nombre in ("aleatorio", "siempre_largo", "siempre_plano"):
        entorno = crear_entorno()
        curva, bitacora = correr_politica(entorno, politicas_base(entorno)[nombre])
        resultados[nombre] = metricas(curva, bitacora, curvas_benchmark)
    return resultados


def metricas_entre_semillas(por_semilla: dict[int, dict[str, float]]) -> dict[str, float]:
    """Métricas de un agente en un fold: el promedio entre semillas de cada métrica, más la desviación
    estándar del Sharpe entre semillas para reportar su dispersión (sección 3.5)."""
    tabla = pd.DataFrame.from_dict(por_semilla, orient="index")
    return {**tabla.mean().to_dict(), "sharpe_desviacion_semillas": float(tabla["sharpe"].std())}


def tabla_metricas(resultados: dict[str, dict[int, dict[str, float]]]) -> pd.DataFrame:
    """Métricas por fold y agregadas, con la forma de artifacts/metrics_summary.csv.

    resultados[estrategia][fold] es la salida de metricas(). Por cada estrategia hay una fila
    por fold y dos agregadas: la media y la desviación estándar muestral entre folds.
    """
    filas = []
    for estrategia, por_fold in resultados.items():
        folds = pd.DataFrame.from_dict(por_fold, orient="index")
        folds.index = folds.index.astype(str)
        folds.loc["media"] = folds.mean()
        folds.loc["desviacion"] = folds.iloc[:-1].std()
        filas.append(folds.rename_axis("fold").reset_index().assign(estrategia=estrategia))
    tabla = pd.concat(filas, ignore_index=True)
    return tabla[["estrategia", "fold"] + [c for c in tabla.columns if c not in ("estrategia", "fold")]]


def guardar_tabla_metricas(tabla: pd.DataFrame, carpeta: Path = RUTA_ARTEFACTOS) -> Path:
    """Escribe artifacts/metrics_summary.csv."""
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / "metrics_summary.csv"
    tabla.to_csv(ruta, index=False)
    return ruta
