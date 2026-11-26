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

import pandas as pd

RUTA_SP500 = Path(__file__).resolve().parents[1] / "data" / "raw" / "sp500_diario.csv"
HORA_CIERRE_SP500_UTC = 21  # 16:00 en Nueva York: cada cierre se usa a partir de esta hora


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
