"""Entorno Gymnasium de trading de BTCUSDT con cierre de posiciones semi-Markov.

Formulación de Powell (sección 3.2 de los lineamientos). Cada época de decisión k ocurre al cierre
de una barra horaria:

- Estado S_k = (R_k, I_k, B_k):
  - Físico R_k: posición (+1 largo, 0 plano, -1 corto), precio de entrada, horas en posición y
    capital (efectivo más la posición marcada a mercado). El agente observa la posición, el
    rendimiento abierto contra el precio de entrada, el tiempo en posición en horizontes y el
    drawdown del capital.
  - Información I_k: las variables de mercado de data.FAMILIAS, normalizadas con el entrenamiento.
  - Creencia B_k: sigma_ewma (volatilidad estimada) y tstat_tendencia_168h (tendencia estimada
    entre su error estándar).
- Decisión x_k: la posición objetivo, 0 plano, 1 largo o 2 corto, con todo el capital y sin
  apalancamiento. En plano se decide cada hora; con posición, solo al cumplir el horizonte, y
  repetir la dirección la mantiene sin pagar costos. Como solo se decide en esos momentos, no
  existen acciones inválidas (por ejemplo, abrir sobre una posición abierta); una acción fuera del
  espacio levanta un error en vez de corregirse en silencio.
- Información exógena W_{k+1}: las barras OHLC entre la ejecución y la siguiente época. Deciden si
  se toca el stop-loss o el take-profit y cuánto dura el intervalo.
- Transición S^M: la orden se llena en la apertura de la barra siguiente con 0.125 % por lado. Una
  posición se cierra en el primero de: stop-loss, take-profit (barreras a ±MULTIPLO_BARRERAS
  desviaciones esperadas del horizonte, medidas con sigma_ewma) o cumplir el horizonte sin que el
  agente la mantenga. Si el stop y el take-profit caben en la misma barra, decide regla_intrabarra
  (por defecto el stop primero). La duración del intervalo, en horas de calendario, va en
  info["duracion"] para los buffers semi-Markov de train.py.
- Contribución C(S_k, x_k, W_{k+1}): la de reward.py, por paso por defecto.
- Objetivo: max_pi E[sum_k gamma^{t_k} C(S_k, X^pi(S_k), W_{k+1})], con t_k en horas.

Episodio: en entrenamiento, ventanas de HORAS_EPISODIO con inicio aleatorio; en validación y prueba,
el periodo completo. terminated solo por ruina (el capital cae a FRACCION_RUINA del inicial);
truncated al terminar la ventana o los datos. Al terminar los datos o en la ruina se liquida la
posición abierta, así la bitácora cuadra con el capital.
"""
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

import data
import reward

# Columnas de la bitácora (sección 3.8). direccion: +1 largo, -1 corto. tamano: unidades de BTC.
# razon_salida: "horizonte", "stop", "take_profit" o "cierre".
COLUMNAS_BITACORA = [
    "entrada", "salida", "direccion", "razon_salida", "precio_entrada", "precio_salida",
    "tamano", "pnl_bruto", "costos", "pnl_neto",
]
CAPITAL_INICIAL = 10_000.0  # USDT al inicio de cada episodio
MULTIPLO_BARRERAS = 2.0  # stop y take-profit a ±2 desviaciones esperadas del horizonte
HORAS_EPISODIO = 30 * 24  # ventanas de 30 días en entrenamiento
FRACCION_RUINA = 0.5  # el gestor detiene la estrategia si el capital cae a la mitad
LIMITE_OBSERVACION = data.Normalizador.LIMITE
ACCION_PLANO, ACCION_LARGO, ACCION_CORTO = 0, 1, 2
POSICION_DE_ACCION = (0, 1, -1)  # posición objetivo de cada acción: plano, largo, corto
VARIABLES_FISICAS = ["posicion", "rendimiento_abierto", "tiempo_en_posicion", "drawdown"]
REGLAS_INTRABARRA = ("stop_primero", "take_profit_primero")
CONTRIBUCIONES = ("por_paso", "terminal")
UNA_HORA = pd.Timedelta(hours=1)


class EntornoTrading(gym.Env):
    """Entorno semi-Markov de BTCUSDT horario sobre una parte de un fold (ver el docstring del módulo).

    Args:
        parte: entrenamiento, validación o prueba de data.preparar_fold().
        inicio_aleatorio: True en entrenamiento (ventanas de horas_episodio con inicio aleatorio);
            False en validación y prueba (todo el periodo en un episodio).
        horas_episodio: duración de las ventanas de entrenamiento, en horas.
        horizonte_horas: horas hasta la revisión de una posición abierta.
        multiplo_barreras: distancia del stop y del take-profit en desviaciones esperadas del horizonte.
        regla_intrabarra: "stop_primero" (supuesto conservador) o "take_profit_primero".
        retraso_barras: barras de retraso de los datos que ve el agente (0 o 1, pregunta 6).
        variables: columnas de mercado que se observan; por defecto todas (ablación por familia).
        contribucion: "por_paso" (la elegida) o "terminal" (la alternativa, para comparar).
    """

    metadata = {"render_modes": []}

    def __init__(self, parte: data.ParteFold, *, inicio_aleatorio: bool = True, horas_episodio: float = HORAS_EPISODIO,
                 horizonte_horas: float = data.HORIZONTE_HORAS, multiplo_barreras: float = MULTIPLO_BARRERAS,
                 regla_intrabarra: str = "stop_primero", retraso_barras: int = 0, variables: list[str] | None = None,
                 contribucion: str = "por_paso"):
        super().__init__()
        if regla_intrabarra not in REGLAS_INTRABARRA:
            raise ValueError(f"regla_intrabarra debe ser una de {REGLAS_INTRABARRA}")
        if contribucion not in CONTRIBUCIONES:
            raise ValueError(f"contribucion debe ser una de {CONTRIBUCIONES}")
        if len(parte.barras) < retraso_barras + 2:
            raise ValueError("La parte necesita al menos dos barras después del retraso.")
        self.inicio_aleatorio = inicio_aleatorio
        self.horas_episodio = horas_episodio
        self.horizonte = horizonte_horas
        self.multiplo_barreras = multiplo_barreras
        self.regla_intrabarra = regla_intrabarra
        self.retraso = retraso_barras
        self.contribucion = contribucion
        self.variables = list(variables) if variables is not None else list(parte.observaciones.columns)

        barras = parte.barras
        self.tiempo = barras.index
        self.horas = ((barras.index - barras.index[0]) / UNA_HORA).to_numpy(dtype=float)
        self.apertura = barras["open"].to_numpy(dtype=float)
        self.maximo = barras["high"].to_numpy(dtype=float)
        self.minimo = barras["low"].to_numpy(dtype=float)
        self.cierre = barras["close"].to_numpy(dtype=float)
        self.sigma = parte.volatilidad.to_numpy(dtype=float)
        self.mercado = parte.observaciones[self.variables].to_numpy(dtype=np.float32)

        self.action_space = spaces.Discrete(len(POSICION_DE_ACCION))
        self.observation_space = spaces.Box(-LIMITE_OBSERVACION, LIMITE_OBSERVACION,
                                            shape=(len(self.variables) + len(VARIABLES_FISICAS),), dtype=np.float32)

    # ----- Gymnasium ------------------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        ultimo = len(self.horas) - 1
        if self.inicio_aleatorio:
            # La ventana completa debe caber antes del final de los datos.
            limite = int(np.searchsorted(self.horas, self.horas[-1] - self.horas_episodio, side="right")) - 1
            self.i = int(self.np_random.integers(self.retraso, max(limite, self.retraso) + 1))
            self.fin = min(int(np.searchsorted(self.horas, self.horas[self.i] + self.horas_episodio)), ultimo)
        else:
            self.i, self.fin = self.retraso, ultimo
        self._quedar_en_plano()
        self.capital = self.pico = CAPITAL_INICIAL
        self.bitacora: list[dict] = []
        self._curva = {self.i: self.capital}
        self.operaciones_ambiguas = 0
        return self._observacion(), {"capital": self.capital}

    def step(self, accion):
        if not self.action_space.contains(accion):
            raise ValueError(f"Acción fuera del espacio de decisión: {accion!r}")
        if self.i >= len(self.horas) - 1:
            raise RuntimeError("El episodio ya terminó: llame a reset().")
        objetivo = POSICION_DE_ACCION[int(accion)]
        inicio, capital_inicio, cerradas_antes = self.i, self.capital, len(self.bitacora)

        # 1. Ejecutar la decisión en la apertura de la barra siguiente.
        j = self.i + 1
        if self.posicion and objetivo != self.posicion:
            self._cerrar(self.apertura[j], "horizonte", self.tiempo[j])
        if objetivo and objetivo != self.posicion:
            self._abrir(j, objetivo)
        elif objetivo:  # repite la dirección: mantiene la posición y vuelve a colocar las barreras
            self._colocar_barreras(self.apertura[j])
            self.inicio_tramo = self.horas[j]

        # 2. Simular hasta la siguiente época de decisión.
        if self.posicion:
            evento = self._avanzar_con_posicion(j)
        else:
            self.i, evento = j, "plano"
            self._curva[j] = self.capital

        # 3. Ruina, fin de la ventana o fin de los datos.
        terminado = self.capital <= FRACCION_RUINA * CAPITAL_INICIAL
        fin_de_datos = self.i >= len(self.horas) - 1
        if self.posicion and (terminado or fin_de_datos):
            self._cerrar(self.cierre[self.i], "cierre", self.tiempo[self.i] + UNA_HORA)
            self._curva[self.i] = self.capital
            evento = "cierre"
        truncado = not terminado and (fin_de_datos or self.i >= self.fin)
        self.pico = max(self.pico, self.capital)

        if self.contribucion == "por_paso":
            recompensa = reward.contribucion_por_paso(capital_inicio, self.capital)
        else:
            recompensa = reward.contribucion_terminal(self.bitacora[cerradas_antes:])
        info = {"duracion": float(self.horas[self.i] - self.horas[inicio]), "capital": self.capital,
                "posicion": self.posicion, "evento": evento}
        return self._observacion(), recompensa, bool(terminado), bool(truncado), info

    # ----- Resultados del episodio ----------------------------------------------------------------

    def bitacora_df(self) -> pd.DataFrame:
        """Operaciones cerradas con las columnas de COLUMNAS_BITACORA.

        Agrega la hora de la decisión de entrada y el capital al abrir, que sirven para el análisis
        de operaciones (sección 3.9).
        """
        return pd.DataFrame(self.bitacora, columns=COLUMNAS_BITACORA + ["hora_decision", "capital_apertura"])

    def curva_capital(self) -> pd.Series:
        """Capital marcado a mercado al cierre de cada barra del episodio.

        Si el episodio terminó antes por ruina, el capital se queda constante hasta el final del
        periodo, como en un fondo que deja de operar; así la curva cubre las mismas barras que los
        benchmarks.
        """
        indices = sorted(self._curva)
        curva = pd.Series([self._curva[k] for k in indices], index=self.tiempo[indices], name="capital")
        return curva.reindex(self.tiempo[indices[0]:max(indices[-1], self.fin) + 1]).ffill()

    # ----- Estado y transición --------------------------------------------------------------------

    def _observacion(self) -> np.ndarray:
        """Variables de mercado y de creencia (con el retraso de datos) más las físicas."""
        visible = self.i - self.retraso
        if self.posicion:
            abierto = self.posicion * np.log(self.cierre[visible] / self.precio_entrada)
            en_posicion = (self.horas[self.i] + 1 - self.hora_entrada) / self.horizonte
        else:
            abierto = en_posicion = 0.0
        fisicas = np.array([self.posicion, abierto, en_posicion, self.capital / self.pico - 1], dtype=np.float32)
        observacion = np.concatenate([self.mercado[visible], fisicas])
        return np.clip(observacion, -LIMITE_OBSERVACION, LIMITE_OBSERVACION).astype(np.float32)

    def _colocar_barreras(self, precio: float) -> None:
        """Stop-loss y take-profit a ±multiplo_barreras * sigma * sqrt(horizonte) en log del precio."""
        distancia = self.multiplo_barreras * self.sigma[self.i - self.retraso] * np.sqrt(self.horizonte)
        abajo, arriba = precio * np.exp(-distancia), precio * np.exp(distancia)
        self.stop, self.take_profit = (abajo, arriba) if self.posicion == 1 else (arriba, abajo)

    def _abrir(self, j: int, direccion: int) -> None:
        """Abre una posición con todo el capital en la apertura de la barra j; el costo sale del capital."""
        precio = self.apertura[j]
        self.posicion, self.precio_entrada, self.hora_entrada = direccion, precio, self.horas[j]
        self.capital_apertura = self.capital
        self.cantidad = (self.capital - reward.costo(self.capital)) / precio
        self.entrada, self.hora_decision = self.tiempo[j], self.tiempo[self.i]
        self.inicio_tramo = self.horas[j]
        self._colocar_barreras(precio)

    def _cerrar(self, precio: float, razon: str, salida: pd.Timestamp) -> None:
        """Cierra la posición a `precio`, registra la operación y deja el capital realizado."""
        bruto = self.posicion * self.cantidad * (precio - self.precio_entrada)
        costos = reward.costo(self.capital_apertura) + reward.costo(self.cantidad * precio)
        self.bitacora.append({
            "entrada": self.entrada, "salida": salida, "direccion": self.posicion, "razon_salida": razon,
            "precio_entrada": self.precio_entrada, "precio_salida": precio, "tamano": self.cantidad,
            "pnl_bruto": bruto, "costos": costos, "pnl_neto": bruto - costos,
            "hora_decision": self.hora_decision, "capital_apertura": self.capital_apertura,
        })
        self.capital = self.capital_apertura + bruto - costos
        self._quedar_en_plano()

    def _quedar_en_plano(self) -> None:
        """Deja el estado físico en plano: sin posición, precio de entrada ni barreras."""
        self.posicion, self.cantidad = 0, 0.0
        self.precio_entrada = self.hora_entrada = self.capital_apertura = np.nan
        self.stop = self.take_profit = self.inicio_tramo = np.nan
        self.entrada = self.hora_decision = None

    def _marcar_a_mercado(self, precio: float) -> float:
        """Capital si la posición se valuara a `precio`, sin el costo de salida."""
        return self.capital_apertura - reward.costo(self.capital_apertura) + self.posicion * self.cantidad * (
            precio - self.precio_entrada)

    def _revisar_barreras(self, k: int) -> tuple[str, float] | None:
        """Razón y precio de salida si la barra k toca el stop o el take-profit; None si no toca ninguno.

        Si la barra abre más allá de una barrera (hueco), se llena a la apertura. Si toca ambas, el
        orden dentro de la barra no se observa y decide regla_intrabarra.
        """
        apertura, maximo, minimo = self.apertura[k], self.maximo[k], self.minimo[k]
        if self.posicion == 1:
            abre_stop, abre_tp = apertura <= self.stop, apertura >= self.take_profit
            toca_stop, toca_tp = minimo <= self.stop, maximo >= self.take_profit
        else:
            abre_stop, abre_tp = apertura >= self.stop, apertura <= self.take_profit
            toca_stop, toca_tp = maximo >= self.stop, minimo <= self.take_profit
        if abre_stop:
            return "stop", apertura
        if abre_tp:
            return "take_profit", apertura
        if toca_stop and toca_tp:
            self.operaciones_ambiguas += 1
            toca_stop = self.regla_intrabarra == "stop_primero"
            toca_tp = not toca_stop
        if toca_stop:
            return "stop", self.stop
        if toca_tp:
            return "take_profit", self.take_profit
        return None

    def _avanzar_con_posicion(self, j: int) -> str:
        """Recorre las barras desde j hasta el stop, el take-profit, el horizonte o el fin de los datos.

        Deja self.i en la barra de la siguiente época de decisión y devuelve qué la provocó.
        """
        ultimo = len(self.horas) - 1
        k = j
        while True:
            salida = self._revisar_barreras(k)
            if salida:
                self._cerrar(salida[1], salida[0], self.tiempo[k] + UNA_HORA)
                self.i, self._curva[k] = k, self.capital
                return salida[0]
            self.capital = self._marcar_a_mercado(self.cierre[k])
            self._curva[k] = self.capital
            if self.horas[k] + 1 - self.inicio_tramo >= self.horizonte or k == ultimo:
                self.i = k  # al final de los datos, step() liquida la posición
                return "horizonte"
            k += 1
