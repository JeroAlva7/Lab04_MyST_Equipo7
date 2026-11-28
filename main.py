"""Punto de entrada único del pipeline del Lab 04.

Uso:
    python main.py

Etapas (se agregan conforme se implementan los módulos de code/):
    1. Datos: carga y limpieza, variables causales, folds y normalización (code/data.py), y
       análisis exploratorio: artifacts/btcusdt_eda.pdf y artifacts/feature_analysis_fold_k.csv.
    2. Entrenamiento y búsqueda de hiperparámetros en validación, por fold y semilla, con DQN y
       PPO sobre el entorno validado con check_env (code/train.py, code/env.py, code/reward.py):
       artifacts/validation_results_fold_k.csv.
    3. Evaluación en prueba, una sola vez por fold, de la configuración elegida de cada algoritmo con
       sus tres semillas, contra las líneas base y los dos benchmarks (code/evaluate.py, code/plots.py):
       artifacts/trade_log_fold_k.csv, training_curves_fold_k.png, equity_curves_test_fold_k.png y
       metrics_summary.csv.

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
from env import CAPITAL_INICIAL, EntornoTrading  # noqa: E402


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


def etapa_prueba(datos: pd.DataFrame, elegidas: dict[int, dict[str, int]]) -> pd.DataFrame:
    """Evaluación en prueba de la configuración elegida en validación, una sola vez por fold (3.6 a 3.8).

    Cada algoritmo se evalúa con sus tres semillas y en la tabla entra el promedio entre semillas, más
    la desviación del Sharpe entre semillas. La bitácora de cada fold es la del algoritmo principal con
    la semilla del proyecto. Las líneas base y los benchmarks pagan los mismos costos.
    """
    cierres_sp500 = evaluate.cargar_sp500()
    resultados: dict[str, dict[int, dict[str, float]]] = {}
    for fold in data.FOLDS:
        partes, _ = data.preparar_fold(datos, fold)
        prueba = partes["prueba"]
        curvas_benchmark = evaluate.benchmarks(prueba.barras, cierres_sp500)
        curvas_agentes, monitores = {}, {}
        for algoritmo, por_semilla in train.modelos_elegidos(fold, partes, elegidas[fold]).items():
            metricas_semillas, curvas_agentes[algoritmo], monitores[algoritmo] = {}, {}, {}
            for semilla, (modelo, carpeta) in por_semilla.items():
                entorno = EntornoTrading(prueba, inicio_aleatorio=False)
                curva, bitacora = evaluate.correr_politica(entorno, train.politica_del_modelo(modelo))
                metricas_semillas[semilla] = evaluate.metricas(curva, bitacora, curvas_benchmark)
                curvas_agentes[algoritmo][semilla], monitores[algoritmo][semilla] = curva, train.leer_monitor(carpeta)
                if algoritmo == train.ALGORITMO_PRINCIPAL and semilla == data.SEMILLA:
                    print(f"   {evaluate.guardar_bitacora(bitacora, fold).relative_to(RAIZ)}", flush=True)
            resultados.setdefault(algoritmo, {})[fold] = evaluate.metricas_entre_semillas(metricas_semillas)
        lineas_base = evaluate.metricas_lineas_base(lambda: EntornoTrading(prueba, inicio_aleatorio=False), curvas_benchmark)
        for nombre, metricas in lineas_base.items():
            resultados.setdefault(nombre, {})[fold] = metricas
        for nombre, metricas in evaluate.metricas_benchmarks(prueba.barras, cierres_sp500).items():
            resultados.setdefault(f"benchmark_{nombre}", {})[fold] = metricas

        descripciones = {a: f"lr = {train.REJILLA[a][c]['learning_rate']:g}, gamma = {train.REJILLA[a][c]['gamma']} por hora"
                         for a, c in elegidas[fold].items()}
        entrenamiento = partes["entrenamiento"].barras.index
        periodo = f"entrenamiento del {entrenamiento[0]:%Y-%m-%d} al {entrenamiento[-1]:%Y-%m-%d}"
        fin_exploracion = train.HIPERPARAMETROS_BASE["DQN"]["exploration_fraction"] * train.PASOS
        for ruta in (plots.curvas_entrenamiento(fold, monitores, descripciones, fin_exploracion, periodo,
                                                evaluate.RUTA_ARTEFACTOS / f"training_curves_fold_{fold}.png"),
                     plots.curvas_capital_prueba(fold, curvas_agentes, curvas_benchmark, CAPITAL_INICIAL,
                                                 evaluate.RUTA_ARTEFACTOS / f"equity_curves_test_fold_{fold}.png")):
            print(f"   {ruta.relative_to(RAIZ)}", flush=True)
    tabla = evaluate.tabla_metricas(resultados)
    print(f"   {evaluate.guardar_tabla_metricas(tabla).relative_to(RAIZ)}", flush=True)
    return tabla


def main() -> None:
    """Ejecuta el pipeline completo."""
    inicio = time.perf_counter()
    print("1. Datos y análisis exploratorio", flush=True)
    datos, reporte = data.cargar_datos()
    etapa_eda(datos, reporte)
    print("2. Entrenamiento y búsqueda de hiperparámetros en validación", flush=True)
    elegidas = etapa_validacion(datos)
    print("3. Evaluación en prueba", flush=True)
    etapa_prueba(datos, elegidas)
    print(f"Tiempo total: {time.perf_counter() - inicio:.0f} s")


if __name__ == "__main__":
    main()
