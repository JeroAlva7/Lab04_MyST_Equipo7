# Lab04_MyST_Equipo7 — Agente de Reinforcement Learning para operar BTCUSDT

Microestructuras y Sistemas de Trading · ITESO · Módulo 4 (Reinforcement Learning)

## Integrantes

| Integrante | Usuario de GitHub | Responsabilidades principales |
|---|---|---|
| Jerónimo Rojas Alvarado | [@JeroAlva7](https://github.com/JeroAlva7) | Datos y variables causales, entorno Gymnasium, función de contribución, pruebas |
| Raúl Zanatta Casas | [@rzanatta0](https://github.com/rzanatta0) | Entrenamiento con Stable-Baselines3, evaluación, benchmarks y gráficas |

Ambos integrantes participan en el reporte, la presentación y la revisión cruzada del código.

## Nivel de alcance

Alcance completo: se implementan todos los componentes que exigen los lineamientos.

- Formulación completa del problema en el marco de Powell (sección 3.2) y cierre de posiciones semi-Markov
  con resolución intrabarra explícita (sección 3.3).
- Función de contribución con costos de 0.125 % por lado (sección 3.4).
- Entrenamiento con Stable-Baselines3: DQN como algoritmo principal y PPO como segundo algoritmo, con varias
  semillas por fold (sección 3.5).
- Protocolo walk-forward de cuatro folds con un modelo por fold (sección 3.6), benchmarks y métricas
  (sección 3.7), artefactos obligatorios (sección 3.8) y las cuatro pruebas de pytest (sección 3.11).
- Ablación por familias de variables, sensibilidad a la resolución intrabarra y al retraso de una barra, y
  respuesta a las preguntas de análisis (apartado 4).

## Descripción

Agente de reinforcement learning que opera BTCUSDT con barras horarias. El trabajo se centra en el diseño
del entorno: la formulación del problema de decisión secuencial en el marco de Powell, el estado, la función
de contribución y un simulador con costos de 0.125 % por lado. El aprendizaje usa Stable-Baselines3 y la
evaluación sigue un protocolo walk-forward de cuatro folds con ventana expansiva.

## Diseño del entorno

El detalle y las alternativas descartadas están en `docs/reporte.pdf` y en el docstring de `code/env.py`.

- **Decisiones (semi-Markov):** plano, largo o corto, con todo el capital y sin apalancamiento. En plano el
  agente decide cada hora; con una posición abierta decide cada 24 horas, y repetir la dirección la mantiene
  sin pagar costos.
- **Cierre:** stop-loss y take-profit a ±2 desviaciones esperadas de 24 horas (sigma EWMA × √24). Si una
  barra toca ambos, se supone que el stop fue primero; si abre más allá de una barrera, se llena a la apertura.
- **Ejecución:** decide al cierre de la barra t y llena en la apertura de t+1, con 0.125 % por lado.
- **Estado:** físico (posición, rendimiento abierto, tiempo en posición, drawdown), información (12 variables
  de mercado) y creencia (sigma EWMA y estadístico t de la tendencia).
- **Contribución:** log del capital al final de cada intervalo de decisión entre el del inicio, con costos.
- **Episodio:** ventanas de 30 días con inicio aleatorio al entrenar y el periodo completo al validar y
  probar; termina antes si el capital cae a la mitad.

## Estructura del repositorio

```
Lab04_MyST_Equipo7/
├── README.md
├── requirements.txt     # dependencias con versiones fijadas
├── .gitignore
├── main.py              # punto de entrada único del pipeline
├── code/
│   ├── data.py          # carga, variables causales, normalización
│   ├── env.py           # entorno Gymnasium
│   ├── reward.py        # funciones de contribución
│   ├── train.py         # entrenamiento SB3 por fold y semilla
│   ├── evaluate.py      # métricas, benchmarks, bitácora
│   └── plots.py         # figuras
├── data/raw/            # datos de entrada (CSV de BTCUSDT horario)
├── artifacts/           # artefactos generados por main.py
├── tests/               # pruebas con pytest
└── docs/                # reporte, walkthrough y presentación en PDF
```

La carpeta `data/` no forma parte de la estructura mínima de los lineamientos; se incluye para que el
pipeline corra con un solo comando, sin descargas manuales.

## Instalación

Requiere Python 3.12.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1      # Windows (PowerShell)
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

## Reproducción

```powershell
python main.py
pytest
```

`main.py` corre por etapas y escribe los artefactos en `artifacts/`:

1. Datos y análisis exploratorio (menos de un minuto): `btcusdt_eda.pdf` y `feature_analysis_fold_k.csv`.
2. Búsqueda de hiperparámetros en validación: entrena 96 modelos de 100 mil pasos (4 folds, DQN y PPO,
   4 configuraciones, 3 semillas), un fold por proceso, y escribe `validation_results_fold_k.csv`. Tardó 43
   minutos en una laptop con Windows y 8 hilos.
3. Evaluación en prueba de la configuración elegida de cada algoritmo, con sus 3 semillas, contra las líneas
   base y los dos benchmarks: `trade_log_fold_k.csv`, `training_curves_fold_k.png`,
   `equity_curves_test_fold_k.png` y `metrics_summary.csv`.

Los modelos se guardan en `models/`, que no se sube al repositorio. Si `main.py` se vuelve a correr, los
modelos con la misma configuración se cargan en vez de entrenarse otra vez y todo el pipeline tarda unos
minutos; para reentrenar todo se borra `models/`. Desde un clon limpio se entrena todo.

## Datos

- `data/raw/Binance_BTCUSDT_1h.csv`: archivo publicado en Canvas (fuente: CryptoDataDownload, Binance) con
  barras horarias OHLCV de BTCUSDT del 2017-08-17 04:00 al 2026-04-26 23:00.
- `data/raw/sp500_diario.csv`: cierres diarios del índice S&P 500 (^GSPC, Yahoo Finance mediante yfinance)
  del 2017-08-01 al 2026-04-27. Se usan solo para el benchmark de indexar al S&P 500 y nunca entran al
  estado del agente. ^GSPC es un índice de precio, así que el benchmark no incluye dividendos.

## Uso de herramientas de IA

Se usó Claude (Anthropic) como asistente, conforme a la política de la sección 6 de los lineamientos:

- Lectura de los lineamientos y la rúbrica, y planeación del trabajo entre los integrantes.
- Propuesta del código de `data.py` (limpieza, variables causales, normalización y análisis exploratorio),
  `env.py`, `reward.py`, `plots.py` y `main.py`, y revisión de `evaluate.py` y `train.py`, incluida la
  comparación de los buffers semi-Markov con el código fuente de Stable-Baselines3 2.9.2.
- Propuesta de las pruebas de `tests/`, con las operaciones de referencia calculadas a mano.
- Cálculos de apoyo para las decisiones de diseño: frecuencia de salidas por stop y take-profit con barreras
  fijas y escaladas por volatilidad, y el costo de cerrar y reabrir cada 24 horas.

Las decisiones de diseño del entorno (acciones, revisión a las 24 horas, barreras y episodio) las tomó el
equipo con las cifras del análisis exploratorio. El código se ejecutó en las computadoras de los integrantes
y se validó con las pruebas de `tests/` y con los datos; las cifras del reporte salen de los artefactos que
genera `main.py`. Los dos integrantes revisaron el código y pueden explicar cualquier parte de él.
