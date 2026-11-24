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
