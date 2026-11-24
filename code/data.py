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
