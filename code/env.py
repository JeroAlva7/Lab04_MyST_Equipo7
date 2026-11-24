"""Entorno Gymnasium de trading de BTCUSDT con cierre de posiciones semi-Markov.

Responsabilidades (secciones 3.2 y 3.3 de los lineamientos):
- Estado con sus tres componentes: física (posición, precio de entrada, tiempo en
  posición, efectivo), de información (variables de mercado) y de creencia.
- Espacio de decisiones y tratamiento de acciones inválidas.
- Transición: una posición se cierra en el primero de estos eventos: horizonte objetivo,
  stop-loss, take-profit o cierre explícito. Cada paso del entorno va de una decisión a
  la siguiente y reporta la duración real (formulación semi-Markov).
- Resolución explícita cuando el stop-loss y el take-profit caben en la misma barra
  (supuesto conservador por defecto: el stop se ejecuta primero).
- terminated solo por ruina; truncated por fin de datos.
"""
