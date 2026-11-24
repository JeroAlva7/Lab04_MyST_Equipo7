"""Punto de entrada único del pipeline del Lab 04.

Uso:
    python main.py

Etapas (se agregan conforme se implementan los módulos de code/):
    1. Datos: carga y limpieza, variables causales, folds y normalización (code/data.py).
    2. Entorno: validación con check_env y líneas base (code/env.py, code/reward.py).
    3. Entrenamiento: algoritmos de SB3 por fold y semilla, búsqueda en validación (code/train.py).
    4. Evaluación: prueba una sola vez, benchmarks, métricas y bitácoras (code/evaluate.py).
    5. Figuras y artefactos en artifacts/ (code/plots.py).
"""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
# code/ no es un paquete de Python porque "code" ya es un módulo de la librería estándar.
# Por eso se agrega la carpeta al path y los módulos se importan por su nombre (import data, import env, ...).
sys.path.insert(0, str(RAIZ / "code"))


def main() -> None:
    """Ejecuta el pipeline completo."""
    raise SystemExit("Pipeline en construcción: todavía no hay etapas implementadas.")


if __name__ == "__main__":
    main()
