"""Punto de entrada único del pipeline del Lab 04.

Uso:
    python main.py

Etapas (se agregan conforme se implementan los módulos de code/):
    1. Datos: carga y limpieza, variables causales, folds y normalización (code/data.py), y
       análisis exploratorio: artifacts/btcusdt_eda.pdf y artifacts/feature_analysis_fold_k.csv.
    2. Entorno: validación con check_env y líneas base (code/env.py, code/reward.py).
    3. Entrenamiento: algoritmos de SB3 por fold y semilla, búsqueda en validación (code/train.py).
    4. Evaluación: prueba una sola vez, benchmarks, métricas y bitácoras (code/evaluate.py).
    5. Figuras y artefactos en artifacts/ (code/plots.py).
"""
import sys
import time
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent
# code/ no es un paquete de Python porque "code" ya es un módulo de la librería estándar.
# Por eso se agrega la carpeta al path y los módulos se importan por su nombre (import data, import env, ...).
sys.path.insert(0, str(RAIZ / "code"))

import data  # noqa: E402  (los módulos de code/ se importan después de agregar la carpeta al path)
import evaluate  # noqa: E402
import plots  # noqa: E402


def etapa_eda(datos: pd.DataFrame, reporte: dict) -> None:
    """Análisis exploratorio y análisis de variables, solo con el entrenamiento de cada fold (sección 3.8)."""
    resumen = data.resumen_eda(datos)
    analisis = {fold: data.analisis_variables(datos, fold) for fold in data.FOLDS}
    for fold, tabla in analisis.items():
        ruta = data.guardar_analisis_variables(tabla, fold, evaluate.RUTA_ARTEFACTOS)
        print(f"   {ruta.relative_to(RAIZ)}")
    ruta = plots.pdf_eda(datos, reporte, resumen, analisis, evaluate.COSTO_POR_LADO,
                         evaluate.RUTA_ARTEFACTOS / "btcusdt_eda.pdf")
    print(f"   {ruta.relative_to(RAIZ)}")


def main() -> None:
    """Ejecuta el pipeline completo."""
    inicio = time.perf_counter()
    print("1. Datos y análisis exploratorio")
    datos, reporte = data.cargar_datos()
    etapa_eda(datos, reporte)
    print("Las etapas de entorno, entrenamiento y evaluación todavía no están conectadas.")
    print(f"Tiempo total: {time.perf_counter() - inicio:.0f} s")


if __name__ == "__main__":
    main()
