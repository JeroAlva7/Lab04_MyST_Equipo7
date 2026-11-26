"""Datos: carga, limpieza, variables causales, folds walk-forward y normalización.

Responsabilidades (secciones 3.1, 3.2 y 3.6 de los lineamientos):
- Cargar data/raw/Binance_BTCUSDT_1h.csv, ordenarlo en el tiempo y limpiarlo
  (horas duplicadas, formatos de fecha mixtos, marcas fuera de la hora exacta y huecos).
- Calcular variables derivadas de forma causal: el valor en t usa solo información
  disponible en el momento de la decisión o antes.
- Definir los cuatro folds de ventana expansiva (entrenamiento, validación y prueba).
- Ajustar la normalización solo con el entrenamiento de cada fold y aplicarla sin
  cambios a validación y prueba.
- Calcular los estadísticos del análisis exploratorio y del análisis de variables por fold.
"""
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

RUTA_CSV = Path(__file__).resolve().parents[1] / "data" / "raw" / "Binance_BTCUSDT_1h.csv"

HORAS_POR_ANIO = 24 * 365.25  # BTCUSDT opera todas las horas del año

# Horizonte objetivo de tenencia, en horas. En el entrenamiento de los cuatro folds, el costo de ida
# y vuelta (0.25 %) equivale a entre 80 % y 93 % del movimiento típico de una hora y a entre 13 % y
# 16 % del de 24 horas (ver tabla_horizontes), así que plazos más cortos quedan dominados por los
# costos. El análisis de variables mide su relación con el rendimiento de este mismo plazo.
HORIZONTE_HORAS = 24

# Un régimen cambia cuando el precio se aleja más de este porcentaje de su último máximo o mínimo.
# Con umbrales de 25 % a 50 % los rebotes de unos días dentro de un mercado bajista cuentan como
# regímenes (48 y 13 cambios); con 70 % quedan los siete ciclos mayores de 2017 a 2024.
UMBRAL_REGIMEN = 0.70

# Nombres de columnas del CSV de CryptoDataDownload y su nombre dentro del proyecto.
COLUMNAS = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume BTC": "volume",
    "Volume USDT": "volume_usdt",
    "tradecount": "trades",
}

# Fold: (último año de entrenamiento, año de validación, año de prueba). Ventana expansiva:
# el entrenamiento siempre empieza en 2017-08 y crece un año por fold (sección 3.6).
FOLDS = {
    1: (2021, 2022, 2023),
    2: (2022, 2023, 2024),
    3: (2023, 2024, 2025),
    4: (2024, 2025, 2026),
}


# Variables del estado de información y de creencia, agrupadas por familia para la ablación.
# Las ventanas se miden en horas de calendario.
FAMILIAS = {
    "rendimientos": ["ret_1h", "ret_4h", "ret_24h", "ret_168h"],
    "volatilidad": ["vol_24h", "vol_168h", "parkinson_24h", "ratio_vol"],
    "tendencia": ["dist_media_168h", "dist_media_720h", "caida_maximo_720h"],
    "volumen": ["volumen_z_168h"],
    "creencia": ["sigma_ewma", "tstat_tendencia_168h"],
}
VARIABLES = [variable for familia in FAMILIAS.values() for variable in familia]


def leer_csv(ruta: Path = RUTA_CSV) -> pd.DataFrame:
    """Lee el CSV tal como viene de Canvas, sin modificarlo.

    La primera línea del archivo es la URL de la fuente, por eso se omite.
    """
    return pd.read_csv(ruta, skiprows=1)


def _marca_de_tiempo(unix: pd.Series) -> pd.DatetimeIndex:
    """Convierte la columna Unix a fechas.

    El archivo mezcla milisegundos (13 dígitos) y microsegundos (16 dígitos). Cualquier valor
    mayor o igual a 10^14 solo puede estar en microsegundos, así que se divide entre 1000.
    """
    valores = unix.astype("int64").to_numpy()
    milisegundos = np.where(valores >= 10**14, valores // 1000, valores)
    return pd.DatetimeIndex(pd.to_datetime(milisegundos, unit="ms"))


def limpiar(crudo: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Limpia los datos crudos y devuelve las barras horarias junto con un reporte.

    Pasos, en orden:
    1. La hora se toma de la columna Unix; la columna Date mezcla dos formatos de texto.
    2. Las barras desfasadas (43 en febrero de 2018, con minuto 28) se alinean al inicio de
       su hora. Ninguna choca con una hora que ya exista.
    3. Se ordena de la barra más antigua a la más reciente (el archivo viene al revés).
    4. Se eliminan las horas duplicadas: son la misma barra repetida con el Unix en otra
       unidad, con valores idénticos.
    5. Se verifica la consistencia OHLC: precios positivos y high y low envolviendo a
       open y close.

    Las horas faltantes no se rellenan, porque inventar precios crearía rendimientos y
    volatilidades falsas. La columna horas_desde_anterior registra el tamaño de cada hueco.

    Returns:
        (datos, reporte): barras indexadas por su hora de apertura y un diccionario con el
        conteo de cada paso de limpieza.
    """
    tiempo = _marca_de_tiempo(crudo["Unix"])
    alineado = tiempo.floor("h")

    datos = crudo[list(COLUMNAS)].rename(columns=COLUMNAS)
    datos.index = alineado
    datos.index.name = "hora"
    datos = datos.sort_index()
    duplicadas = datos.index.duplicated(keep="first")
    datos = datos[~duplicadas].astype(float)

    inconsistentes = (
        (datos[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | (datos["high"] < datos[["open", "close"]].max(axis=1))
        | (datos["low"] > datos[["open", "close"]].min(axis=1))
    )
    if inconsistentes.any():
        raise ValueError(f"{int(inconsistentes.sum())} barras con OHLC inconsistente.")

    horas = datos.index.to_series().diff().dt.total_seconds().div(3600)
    datos["horas_desde_anterior"] = horas.fillna(1.0).to_numpy()

    esperadas = pd.date_range(datos.index[0], datos.index[-1], freq="h")
    reporte = {
        "filas_leidas": int(len(crudo)),
        "barras_desfasadas_alineadas": int((tiempo != alineado).sum()),
        "horas_duplicadas_eliminadas": int(duplicadas.sum()),
        "barras_finales": int(len(datos)),
        "primera_barra": str(datos.index[0]),
        "ultima_barra": str(datos.index[-1]),
        "horas_faltantes": int(len(esperadas) - len(datos)),
        "huecos_mayores_a_1h": int((datos["horas_desde_anterior"] > 1).sum()),
        "hueco_maximo_horas": float(datos["horas_desde_anterior"].max()),
        "barras_con_volumen_cero": int((datos["volume"] == 0).sum()),
    }
    return datos, reporte


def cargar_datos(ruta: Path = RUTA_CSV) -> tuple[pd.DataFrame, dict]:
    """Lee y limpia el CSV de BTCUSDT horario. Ver limpiar() para el detalle."""
    return limpiar(leer_csv(ruta))


def particionar(datos: pd.DataFrame, fold: int) -> dict[str, pd.DataFrame]:
    """Divide los datos en entrenamiento, validación y prueba para un fold.

    Usa años calendario completos según la tabla de la sección 3.6. Las tres partes no se
    traslapan y respetan el orden del tiempo: entrenamiento < validación < prueba.
    """
    fin_entrenamiento, anio_validacion, anio_prueba = FOLDS[fold]
    anio = datos.index.year
    return {
        "entrenamiento": datos[anio <= fin_entrenamiento],
        "validacion": datos[anio == anio_validacion],
        "prueba": datos[anio == anio_prueba],
    }


def rendimiento_horario(datos: pd.DataFrame) -> pd.Series:
    """Rendimiento logarítmico entre barras consecutivas, en escala de una hora.

    Cuando la barra anterior está a más de una hora por un hueco en los datos, el rendimiento se
    divide entre la raíz de las horas transcurridas para que su escala sea comparable.
    """
    return np.log(datos["close"]).diff() / np.sqrt(datos["horas_desde_anterior"])


def _precio_hace(close: pd.Series, horas: int) -> pd.Series:
    """Último precio de cierre disponible hace al menos `horas` horas. Solo mira al pasado."""
    pasado = close.reindex(close.index - pd.Timedelta(hours=horas), method="ffill")
    return pd.Series(pasado.to_numpy(), index=close.index)


def calcular_variables(datos: pd.DataFrame) -> pd.DataFrame:
    """Calcula las variables del estado de información y de creencia.

    Todas son causales: el valor en la hora t usa solo barras ya cerradas hasta t. Las ventanas
    se miden en horas de calendario ('24h', '168h', '720h'), así que un hueco en los datos deja
    menos barras dentro de la ventana en lugar de estirarla. Las primeras semanas quedan en NaN
    mientras las ventanas se llenan.

    Familias:
    - rendimientos: rendimiento logarítmico de las últimas 1, 4, 24 y 168 horas.
    - volatilidad: desviación de los rendimientos horarios en 24 y 168 horas, estimador de
      Parkinson con el rango alto-bajo de 24 horas y el cociente entre volatilidad corta y larga.
    - tendencia: distancia del precio a su media de 168 y 720 horas y caída desde el máximo de
      720 horas.
    - volumen: puntaje z del volumen en USDT (en logaritmo) frente a sus últimas 168 horas.
    - creencia: volatilidad estimada con EWMA (lambda = 0.94) y estadístico t de la tendencia
      de 168 horas (rendimiento medio dividido entre su error estándar).
    """
    close = datos["close"]
    log_close = np.log(close)
    r = rendimiento_horario(datos)
    v = pd.DataFrame(index=datos.index)

    for horas in (1, 4, 24, 168):
        v[f"ret_{horas}h"] = log_close - np.log(_precio_hace(close, horas))

    v["vol_24h"] = r.rolling("24h", min_periods=12).std()
    v["vol_168h"] = r.rolling("168h", min_periods=84).std()
    rango = np.log(datos["high"] / datos["low"]) ** 2 / (4 * np.log(2))
    v["parkinson_24h"] = np.sqrt(rango.rolling("24h", min_periods=12).mean())
    v["ratio_vol"] = v["vol_24h"] / v["vol_168h"]

    v["dist_media_168h"] = close / close.rolling("168h", min_periods=84).mean() - 1
    v["dist_media_720h"] = close / close.rolling("720h", min_periods=360).mean() - 1
    v["caida_maximo_720h"] = close / close.rolling("720h", min_periods=360).max() - 1

    log_volumen = np.log1p(datos["volume_usdt"])
    media = log_volumen.rolling("168h", min_periods=84).mean()
    desviacion = log_volumen.rolling("168h", min_periods=84).std()
    v["volumen_z_168h"] = (log_volumen - media) / desviacion

    v["sigma_ewma"] = np.sqrt((r**2).ewm(alpha=0.06, adjust=False).mean())
    n = r.rolling("168h", min_periods=84).count()
    v["tstat_tendencia_168h"] = r.rolling("168h", min_periods=84).mean() / (v["vol_168h"] / np.sqrt(n))
    return v[VARIABLES]


class Normalizador:
    """Normalización z con los estadísticos del entrenamiento de un fold.

    Se ajusta una sola vez con el entrenamiento y se aplica sin cambios a validación y prueba
    (sección 3.1), así que cada fold tiene su propio normalizador. Los valores se recortan a
    +-LIMITE desviaciones: con colas tan pesadas (curtosis cercana a 30) y horas sin volumen,
    un solo dato extremo dominaría la entrada de la red.
    """

    LIMITE = 5.0

    def __init__(self) -> None:
        self.media: pd.Series | None = None
        self.desviacion: pd.Series | None = None

    def ajustar(self, entrenamiento: pd.DataFrame) -> "Normalizador":
        """Calcula media y desviación estándar de cada variable con el entrenamiento."""
        self.media = entrenamiento.mean()
        self.desviacion = entrenamiento.std(ddof=0).replace(0.0, 1.0)
        return self

    def transformar(self, variables: pd.DataFrame) -> pd.DataFrame:
        """Aplica la normalización ya ajustada, sin recalcular nada."""
        if self.media is None or self.desviacion is None:
            raise RuntimeError("El normalizador debe ajustarse con el entrenamiento antes de usarse.")
        z = (variables[self.media.index] - self.media) / self.desviacion
        return z.clip(-self.LIMITE, self.LIMITE)

    def a_dict(self) -> dict:
        """Estadísticos en un formato que se puede guardar como JSON."""
        return {"media": self.media.to_dict(), "desviacion": self.desviacion.to_dict(), "limite": self.LIMITE}


@dataclass
class ParteFold:
    """Una parte de un fold: las barras para simular la ejecución y sus observaciones normalizadas."""

    barras: pd.DataFrame
    observaciones: pd.DataFrame


def preparar_fold(datos: pd.DataFrame, fold: int) -> tuple[dict[str, ParteFold], Normalizador]:
    """Arma entrenamiento, validación y prueba de un fold con su normalizador.

    Las variables se calculan sobre la serie completa, lo cual es válido porque cada una es
    causal, y se descartan las horas iniciales en las que alguna ventana todavía no se llena.
    El normalizador se ajusta solo con las observaciones de entrenamiento.
    """
    variables = calcular_variables(datos)
    validas = variables.notna().all(axis=1)
    barras = particionar(datos[validas], fold)
    observaciones = particionar(variables[validas], fold)
    normalizador = Normalizador().ajustar(observaciones["entrenamiento"])
    partes = {
        nombre: ParteFold(barras=barras[nombre], observaciones=normalizador.transformar(observaciones[nombre]))
        for nombre in barras
    }
    return partes, normalizador


# ---------------------------------------------------------------------------------------------
# Análisis exploratorio y análisis de variables (secciones 3.2 y 3.8, preguntas 1 y 2).
# Todo se calcula únicamente con datos de entrenamiento.
# ---------------------------------------------------------------------------------------------


def autocorrelacion(serie: np.ndarray, rezagos: int) -> np.ndarray:
    """Autocorrelación muestral de los rezagos 1 a `rezagos`."""
    x = np.asarray(serie, dtype=float) - np.mean(serie)
    denominador = np.dot(x, x)
    return np.array([np.dot(x[k:], x[:-k]) / denominador for k in range(1, rezagos + 1)])


def ljung_box(serie: np.ndarray, rezagos: int) -> tuple[float, float]:
    """Prueba de Ljung-Box: H0 = no hay autocorrelación en los primeros `rezagos` rezagos."""
    n = len(serie)
    rho = autocorrelacion(serie, rezagos)
    q = n * (n + 2) * np.sum(rho**2 / (n - np.arange(1, rezagos + 1)))
    return float(q), float(stats.chi2.sf(q, rezagos))


def prueba_estacionariedad(serie: pd.Series) -> dict[str, float]:
    """Pruebas ADF (H0: raíz unitaria) y KPSS (H0: estacionaria) sobre una serie.

    ADF usa un rezago fijo de 24 horas, el ciclo diario de los datos horarios. KPSS solo reporta
    valores p entre 0.01 y 0.10 (los límites de su tabla), así que un 0.10 significa "0.10 o más".
    result_object=False pide a statsmodels 0.15 la tupla de siempre y evita su aviso de cambio
    de formato.
    Conclusión, con nivel de 5 %:
    - estacionaria: ADF rechaza la raíz unitaria y KPSS no rechaza la estacionariedad.
    - persistente: ADF rechaza la raíz unitaria, pero KPSS sí rechaza; la serie regresa a su
      media, aunque muy lento o con cambios de nivel entre regímenes.
    - raiz_unitaria: ADF no rechaza y KPSS rechaza, como el precio.
    - indeterminada: ninguna de las dos rechaza.
    """
    valores = serie.dropna().to_numpy()
    adf_estadistico, adf_p = adfuller(valores, maxlag=24, autolag=None, result_object=False)[:2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        kpss_estadistico, kpss_p = kpss(valores, regression="c", nlags="auto", result_object=False)[:2]
    if adf_p < 0.05:
        conclusion = "estacionaria" if kpss_p > 0.05 else "persistente"
    else:
        conclusion = "indeterminada" if kpss_p > 0.05 else "raiz_unitaria"
    return {
        "adf_estadistico": float(adf_estadistico),
        "adf_p": float(adf_p),
        "kpss_estadistico": float(kpss_estadistico),
        "kpss_p": float(kpss_p),
        "conclusion": conclusion,
    }


def puntos_de_giro(close: pd.Series, umbral: float = UMBRAL_REGIMEN) -> list[pd.Timestamp]:
    """Fechas de los máximos y mínimos que separan los regímenes.

    Un régimen alcista termina en su máximo cuando después el precio cae más de `umbral` desde ahí;
    uno bajista termina en su mínimo cuando el precio sube más de `umbral`. La regla necesita ver
    la trayectoria completa, así que sirve para describir el pasado y nunca entra al estado.
    """
    precios = close.to_numpy()
    giros, tendencia, extremo = [0], 0, 0
    for i in range(1, len(precios)):
        cambio = precios[i] / precios[extremo] - 1
        if tendencia == 0:
            if abs(cambio) >= umbral:
                tendencia, extremo = (1 if cambio > 0 else -1), i
        elif tendencia * (precios[i] - precios[extremo]) > 0:
            extremo = i
        elif -tendencia * cambio >= umbral:
            giros.append(extremo)
            tendencia, extremo = -tendencia, i
    giros.append(len(precios) - 1)
    return [close.index[k] for k in giros]


def regimenes(datos: pd.DataFrame, umbral: float = UMBRAL_REGIMEN) -> pd.DataFrame:
    """Regímenes del periodo por rango de fechas, con su tendencia, volatilidad y drawdown (pregunta 1)."""
    close = datos["close"]
    r = rendimiento_horario(datos)
    giros = puntos_de_giro(close, umbral)
    filas = []
    for inicio, fin in zip(giros[:-1], giros[1:]):
        precio = close.loc[inicio:fin]
        retorno = precio.iloc[-1] / precio.iloc[0] - 1
        filas.append({
            "inicio": inicio,
            "fin": fin,
            "dias": (fin - inicio) / pd.Timedelta(days=1),
            "tendencia": "alcista" if retorno > 0 else "bajista",
            "retorno_total": retorno,
            "volatilidad_anual": r.loc[inicio:fin].iloc[1:].std() * np.sqrt(HORAS_POR_ANIO),
            "maximo_drawdown": (precio / precio.cummax() - 1).min(),
        })
    return pd.DataFrame(filas)


def estadisticos_rendimientos(r: pd.Series) -> dict[str, float]:
    """Forma de la distribución de los rendimientos horarios."""
    x = r.dropna().to_numpy()
    z = (x - x.mean()) / x.std()
    return {
        "observaciones": float(len(x)),
        "media": float(x.mean()),
        "desviacion": float(x.std()),
        "volatilidad_anual": float(x.std() * np.sqrt(HORAS_POR_ANIO)),
        "sesgo": float(stats.skew(x)),
        "curtosis_exceso": float(stats.kurtosis(x)),
        "jarque_bera_p": float(stats.jarque_bera(x).pvalue),
        "cuantil_1": float(np.quantile(x, 0.01)),
        "cuantil_99": float(np.quantile(x, 0.99)),
        "fraccion_mas_de_3_desviaciones": float(np.mean(np.abs(z) > 3)),
    }


def rendimiento_futuro(close: pd.Series, horas: int) -> pd.Series:
    """Rendimiento logarítmico de las siguientes `horas` horas (objetivo de análisis, mira al futuro).

    Usa el primer precio disponible a partir de t + horas dentro de la misma serie: si la serie es
    solo el entrenamiento, las últimas horas quedan en NaN en lugar de tomar precios de validación.
    """
    futuro = close.reindex(close.index + pd.Timedelta(hours=horas), method="bfill")
    return pd.Series(np.log(futuro.to_numpy() / close.to_numpy()), index=close.index)


def tabla_horizontes(close: pd.Series, horizontes: tuple[int, ...] = (1, 4, 12, 24, 72, 168)) -> pd.Series:
    """Mediana del movimiento absoluto del precio en cada plazo, para compararla con los costos."""
    return pd.Series(
        {h: float(rendimiento_futuro(close, h).abs().median()) for h in horizontes},
        name="mediana_movimiento_absoluto",
    ).rename_axis("horizonte_horas")


def tabla_folds(datos: pd.DataFrame) -> pd.DataFrame:
    """Primera y última barra y número de barras de cada parte de cada fold."""
    filas = []
    for fold in FOLDS:
        for parte, barras in particionar(datos, fold).items():
            filas.append({"fold": fold, "parte": parte, "inicio": barras.index[0], "fin": barras.index[-1],
                          "barras": len(barras)})
    return pd.DataFrame(filas)


def resumen_eda(datos: pd.DataFrame, rezagos: int = 168) -> dict[str, pd.DataFrame]:
    """Tablas del análisis exploratorio, calculadas solo con el entrenamiento de cada fold.

    Devuelve: folds (fechas y barras de cada parte), regimenes y correlacion_variables (Spearman
    entre las variables del estado), ambas sobre el entrenamiento más largo, el del fold 4, que
    contiene a los demás; y por fold: distribucion, autocorrelacion (rezagos 1 a 168 de r y |r|),
    ljung_box, estacionariedad (log precio, r y |r|) y horizontes.
    """
    distribucion, autocorrelaciones, ljung, estacionariedad, horizontes = {}, {}, [], [], {}
    for fold in FOLDS:
        entrenamiento = particionar(datos, fold)["entrenamiento"]
        r = rendimiento_horario(entrenamiento).dropna()
        series = {"log_precio": np.log(entrenamiento["close"]), "rendimiento": r, "rendimiento_absoluto": r.abs()}
        distribucion[fold] = estadisticos_rendimientos(r)
        autocorrelaciones[f"rendimiento_fold_{fold}"] = autocorrelacion(r.to_numpy(), rezagos)
        autocorrelaciones[f"rendimiento_absoluto_fold_{fold}"] = autocorrelacion(r.abs().to_numpy(), rezagos)
        for nombre in ("rendimiento", "rendimiento_absoluto"):
            for k in (24, 168):
                q, p = ljung_box(series[nombre].to_numpy(), k)
                ljung.append({"fold": fold, "serie": nombre, "rezagos": k, "estadistico": q, "p": p})
        for nombre, serie in series.items():
            estacionariedad.append({"fold": fold, "serie": nombre, **prueba_estacionariedad(serie)})
        horizontes[fold] = tabla_horizontes(entrenamiento["close"])
    ultimo = particionar(datos, max(FOLDS))["entrenamiento"]
    return {
        "folds": tabla_folds(datos),
        "regimenes": regimenes(ultimo),
        "correlacion_variables": calcular_variables(ultimo).dropna().corr(method="spearman"),
        "distribucion": pd.DataFrame(distribucion).T.rename_axis("fold"),
        "autocorrelacion": pd.DataFrame(autocorrelaciones, index=pd.RangeIndex(1, rezagos + 1, name="rezago")),
        "ljung_box": pd.DataFrame(ljung),
        "estacionariedad": pd.DataFrame(estacionariedad),
        "horizontes": pd.DataFrame(horizontes).rename_axis("horizonte_horas"),
    }
