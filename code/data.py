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
from pathlib import Path

import numpy as np
import pandas as pd

RUTA_CSV = Path(__file__).resolve().parents[1] / "data" / "raw" / "Binance_BTCUSDT_1h.csv"

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
