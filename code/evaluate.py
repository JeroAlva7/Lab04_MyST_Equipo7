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

RUTA_SP500 = Path(__file__).resolve().parents[1] / "data" / "raw" / "sp500_diario.csv"
HORA_CIERRE_SP500_UTC = 21  # 16:00 en Nueva York: cada cierre se usa a partir de esta hora
DIAS_POR_ANIO = 365.25  # BTCUSDT opera 24/7: se anualiza con días calendario


def cargar_sp500(ruta: Path = RUTA_SP500) -> pd.Series:
    """Cierres diarios del S&P 500 (índice ^GSPC) para el benchmark de indexar al S&P 500.

    La primera vez se descargan con yfinance y se guardan en data/raw/sp500_diario.csv; desde
    entonces se leen del CSV, así el pipeline no depende de internet y siempre usa los mismos
    datos. Solo se usan para el benchmark (sección 3.7): nunca entran al estado del agente.
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

    Cada cierre vale a partir de las 21:00 UTC de su día (16:00 en Nueva York) y se mantiene hasta
    el siguiente cierre: en las noches y los fines de semana el índice no cambia.
    """
    con_hora = cierres.copy()
    con_hora.index = con_hora.index + pd.Timedelta(hours=HORA_CIERRE_SP500_UTC)
    malla = con_hora.reindex(con_hora.index.union(indice)).ffill()
    return malla.reindex(indice).rename("sp500")


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
    return capital.iloc[-1] / capital.iloc[0] - 1


def retorno_anualizado(capital: pd.Series) -> float:
    """Retorno compuesto por año: (1 + retorno total)^(1 / años) - 1."""
    return (1 + retorno_total(capital)) ** (1 / anios(capital.index)) - 1


def volatilidad_anualizada(capital: pd.Series) -> float:
    return rendimientos(capital).std() * np.sqrt(periodos_por_anio(capital))


def drawdown(capital: pd.Series) -> pd.Series:
    """Caída de la curva de capital respecto a su máximo previo (0 o negativa)."""
    return capital / capital.cummax() - 1


def maximo_drawdown(capital: pd.Series) -> float:
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
