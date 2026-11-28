"""Punto de entrada único del pipeline del Lab 04.

Uso:
    python main.py

Etapas (se agregan conforme se implementan los módulos de code/):
    1. Datos: carga y limpieza, variables causales, folds y normalización (code/data.py), y
       análisis exploratorio: artifacts/btcusdt_eda.pdf y artifacts/feature_analysis_fold_k.csv.
    2. Entrenamiento y búsqueda de hiperparámetros en validación, por fold y semilla, con DQN y
       PPO sobre el entorno validado con check_env (code/train.py, code/env.py, code/reward.py):
       artifacts/validation_results_fold_k.csv.
    3. Evaluación: prueba una sola vez, líneas base, benchmarks, métricas y bitácoras (code/evaluate.py).
    4. Figuras y artefactos en artifacts/ (code/plots.py).

Los modelos quedan en models/ (fuera del repositorio) y se reutilizan si ya existen con la misma
configuración, así volver a correr main.py no repite el entrenamiento. Desde un clon limpio se
entrena todo.
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent
# code/ no es un paquete de Python porque "code" ya es un módulo de la librería estándar.
# Por eso se agrega la carpeta al path y los módulos se importan por su nombre (import data, import env, ...).
sys.path.insert(0, str(RAIZ / "code"))

import data  # noqa: E402  (los módulos de code/ se importan después de agregar la carpeta al path)
import evaluate  # noqa: E402
import plots  # noqa: E402
import train  # noqa: E402


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


def etapa_validacion(datos: pd.DataFrame) -> dict[int, dict[str, int]]:
    """Búsqueda de hiperparámetros en validación de cada fold (sección 3.5).

    Entrena cada configuración de train.REJILLA con cada semilla de train.SEMILLAS, la evalúa en
    validación contra los dos benchmarks y guarda artifacts/validation_results_fold_k.csv.
    Cada fold corre en su propio proceso: los folds no comparten nada y cada corrida tiene su
    semilla, así que el resultado es el mismo que en serie. Devuelve la configuración elegida de
    cada algoritmo por fold (mayor Sharpe medio).
    """
    cierres_sp500 = evaluate.cargar_sp500()
    procesos = min(len(data.FOLDS), max(1, (os.cpu_count() or 2) // 2))
    with ProcessPoolExecutor(max_workers=procesos) as grupo:
        por_fold = dict(zip(data.FOLDS, grupo.map(train.busqueda_en_fold, repeat(datos), data.FOLDS,
                                                  repeat(cierres_sp500))))
    elegidas = {}
    for fold, resultados in por_fold.items():
        ruta = train.guardar_resultados_validacion(resultados, fold)
        elegidas[fold] = train.mejor_configuracion(resultados)
        print(f"   {ruta.relative_to(RAIZ)}: configuraciones elegidas {elegidas[fold]}", flush=True)
    return elegidas


def main() -> None:
    """Ejecuta el pipeline completo."""
    inicio = time.perf_counter()
    print("1. Datos y análisis exploratorio", flush=True)
    datos, reporte = data.cargar_datos()
    etapa_eda(datos, reporte)
    print("2. Entrenamiento y búsqueda de hiperparámetros en validación", flush=True)
    etapa_validacion(datos)
    print("Las etapas de evaluación en prueba y figuras todavía no están conectadas.")
    print(f"Tiempo total: {time.perf_counter() - inicio:.0f} s")


if __name__ == "__main__":
    main()
