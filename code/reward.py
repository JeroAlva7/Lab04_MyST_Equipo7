"""Funciones de contribución (recompensa) del agente.

Responsabilidades (sección 3.4 de los lineamientos):
- Calcular la contribución de cada operación incluyendo los costos de 0.125 % por lado.
- Hacer explícito si la contribución es por paso o terminal (pagada al cierre).
- Mantener la contribución separada del entorno para poder comparar alternativas.

Contribución elegida, por paso: en la formulación semi-Markov un paso es el intervalo entre dos
decisiones, y su contribución es el log del capital marcado a mercado al final del intervalo entre
el capital al inicio, ya con los costos pagados en él. Mientras una posición sigue abierta se paga en
cada revisión, así que cada decisión (abrir, mantener, cerrar o invertir) recibe el resultado del
intervalo que controló. La suma de un episodio es log(capital final / capital inicial): maximizar
la suma esperada es maximizar el crecimiento del capital.

Alternativa, terminal: todo el resultado de la posición se paga al cerrarla y las revisiones
intermedias reciben 0. Con gamma < 1 una pérdida pospuesta pesa menos, así que premia aguantar las
pérdidas y cerrar pronto las ganancias (efecto disposición). El entorno la ofrece como opción para
comparar ambas.
"""
import numpy as np

COSTO_POR_LADO = 0.00125  # 0.125 % sobre el monto de cada compra o venta (secciones 3.1 y 3.4)


def costo(monto: float) -> float:
    """Costo de transacción de una compra o una venta por `monto` USDT."""
    return COSTO_POR_LADO * abs(monto)


def contribucion_por_paso(capital_inicio: float, capital_fin: float) -> float:
    """Contribución elegida: log del capital al final del intervalo entre el capital al inicio."""
    return float(np.log(capital_fin / capital_inicio))


def contribucion_terminal(operaciones_cerradas: list[dict]) -> float:
    """Alternativa: log del resultado de cada posición cerrada en el intervalo; 0 si no cerró ninguna.

    Cada operación trae capital_apertura (el capital antes del costo de entrada) y pnl_neto.
    """
    return float(sum(np.log1p(op["pnl_neto"] / op["capital_apertura"]) for op in operaciones_cerradas))
