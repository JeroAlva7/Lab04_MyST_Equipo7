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

## Datos

`data/raw/Binance_BTCUSDT_1h.csv`: archivo publicado en Canvas (fuente: CryptoDataDownload, Binance) con
barras horarias OHLCV de BTCUSDT del 2017-08-17 04:00 al 2026-04-26 23:00.

## Uso de herramientas de IA

Se usó Claude (Anthropic) como asistente. Esta sección se completa al cierre del proyecto con el detalle de
las partes en las que se usó.
