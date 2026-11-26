"""Prueba obligatoria 3: operaciones cortas calculadas a mano que el entorno reproduce exactamente.

Se usan barras sintéticas con sigma constante, elegida para que las barreras queden a ±5 % de la
entrada: 2 * sigma * sqrt(24) = ln(1.05). Con entrada a 100, el take-profit de un largo es 105 y su
stop 100 / 1.05. El costo es 0.125 % por lado y el capital inicial 10,000 USDT, así que una
entrada a 100 compra 10,000 * (1 - 0.00125) / 100 = 99.875 BTC.
"""
import numpy as np
import pandas as pd
import pytest

import data
from env import CAPITAL_INICIAL, EntornoTrading

COSTO = 0.00125
SIGMA = np.log(1.05) / (2 * np.sqrt(24))
PLANO, LARGO, CORTO = 0, 1, 2


def entorno(ohlc: list[tuple[float, float, float, float]], horas: list[int] | None = None, **opciones) -> EntornoTrading:
    """Entorno sobre barras sintéticas: observaciones en cero y sigma constante."""
    horas = list(range(len(ohlc))) if horas is None else horas
    indice = pd.Timestamp("2024-01-01") + pd.to_timedelta(horas, unit="h")
    barras = pd.DataFrame(ohlc, columns=["open", "high", "low", "close"], index=indice)
    parte = data.ParteFold(barras=barras, observaciones=pd.DataFrame(0.0, index=indice, columns=data.VARIABLES),
                           volatilidad=pd.Series(SIGMA, index=indice))
    resultado = EntornoTrading(parte, inicio_aleatorio=False, **opciones)
    resultado.reset(seed=0)
    return resultado


def test_largo_que_sale_por_take_profit():
    # Barra 0: decisión. Barra 1: entrada a 100. La barra 3 llega a 106 y toca el take-profit en 105.
    e = entorno([(100, 100, 100, 100), (100, 101, 99.5, 100.5), (100.5, 103, 100, 102),
                 (102, 106, 101.5, 105.5), (105.5, 105.5, 105.5, 105.5), (105.5, 105.5, 105.5, 105.5)])
    _, recompensa, terminado, truncado, info = e.step(LARGO)
    operacion = e.bitacora_df().iloc[0]
    assert operacion["razon_salida"] == "take_profit"
    assert operacion["entrada"] == pd.Timestamp("2024-01-01 01:00") and operacion["salida"] == pd.Timestamp("2024-01-01 04:00")
    assert info["duracion"] == 3  # del cierre de la barra 0 al cierre de la barra 3
    # A mano: 99.875 BTC; bruto 99.875 * (105 - 100) = 499.375; costos 12.5 + 0.00125 * 99.875 * 105 = 25.60859375
    assert np.isclose(operacion["precio_salida"], 105)
    assert np.isclose(operacion["tamano"], 99.875)
    assert np.isclose(operacion["pnl_bruto"], 499.375)
    assert np.isclose(operacion["costos"], 25.60859375)
    assert np.isclose(operacion["pnl_neto"], 473.76640625)
    assert np.isclose(info["capital"], 10_473.76640625)
    assert np.isclose(recompensa, np.log(10_473.76640625 / CAPITAL_INICIAL))
    assert info["posicion"] == 0 and not terminado and not truncado


def test_mantener_en_el_horizonte_no_paga_costos():
    # Precio que oscila entre 99.5 y 100.5, lejos de las barreras: solo sale por decisión del agente.
    cierres = [100.0] + [100 + 0.5 * ((k % 3) - 1) for k in range(1, 60)]
    ohlc = [(cierres[max(k - 1, 0)], max(cierres[max(k - 1, 0)], c) + 0.2, min(cierres[max(k - 1, 0)], c) - 0.2, c)
            for k, c in enumerate(cierres)]
    e = entorno(ohlc)
    _, _, _, _, info = e.step(LARGO)  # entra a 100 en la barra 1; revisión al cierre de la barra 24
    assert info["duracion"] == 24 and info["posicion"] == 1 and info["evento"] == "horizonte"
    _, _, _, _, info = e.step(LARGO)  # mantiene: sin operación nueva; revisión al cierre de la barra 48
    assert info["duracion"] == 24 and info["posicion"] == 1 and e.bitacora == []
    _, _, _, _, info = e.step(PLANO)  # cierra en la apertura de la barra 49 por horizonte
    operacion = e.bitacora_df().iloc[0]
    precio_salida = cierres[48]
    assert operacion["razon_salida"] == "horizonte" and len(e.bitacora) == 1
    assert operacion["salida"] - operacion["entrada"] == pd.Timedelta(hours=48)
    # Un solo costo de entrada y uno de salida, aunque la posición pasó por dos revisiones.
    assert np.isclose(operacion["costos"], COSTO * CAPITAL_INICIAL + COSTO * 99.875 * precio_salida)
    assert np.isclose(info["capital"], CAPITAL_INICIAL * (1 - COSTO) ** 2 * precio_salida / 100)
    assert info["duracion"] == 1  # en plano decide de nuevo una hora después


@pytest.mark.parametrize("regla, razon, precio", [("stop_primero", "stop", 100 / 1.05),
                                                  ("take_profit_primero", "take_profit", 105.0)])
def test_barra_que_toca_ambas_barreras_la_decide_la_regla(regla, razon, precio):
    # La barra 2 baja a 95 y sube a 106: toca el stop (95.24) y el take-profit (105) sin saber cuál fue primero.
    e = entorno([(100, 100, 100, 100), (100, 100.5, 99.5, 100), (100, 106, 95, 100), (100, 100, 100, 100),
                 (100, 100, 100, 100)], regla_intrabarra=regla)
    _, _, _, _, info = e.step(LARGO)
    operacion = e.bitacora_df().iloc[0]
    assert operacion["razon_salida"] == razon and np.isclose(operacion["precio_salida"], precio)
    assert e.operaciones_ambiguas == 1
    assert np.isclose(info["capital"], CAPITAL_INICIAL * (1 - COSTO) ** 2 * precio / 100)


def test_corto_que_abre_mas_alla_del_stop_se_llena_a_la_apertura():
    # Corto a 100 con stop en 105. Tras un hueco de 4 horas en los datos, la barra abre en 108:
    # el stop no se pudo ejecutar a 105 y la salida es a 108.
    e = entorno([(100, 100, 100, 100), (100, 100.5, 99.5, 100.2), (108, 109, 107, 108.5), (108.5, 108.5, 108.5, 108.5),
                 (108.5, 108.5, 108.5, 108.5)], horas=[0, 1, 6, 7, 8])
    _, recompensa, _, _, info = e.step(CORTO)
    operacion = e.bitacora_df().iloc[0]
    assert operacion["razon_salida"] == "stop" and operacion["direccion"] == -1
    assert np.isclose(operacion["precio_salida"], 108)
    assert info["duracion"] == 6  # del cierre de la barra 0 (hora 1) al cierre de la barra 2 (hora 7)
    # A mano: bruto -99.875 * (108 - 100) = -799; costos 12.5 + 0.00125 * 99.875 * 108 = 25.983125
    assert np.isclose(operacion["pnl_bruto"], -799.0)
    assert np.isclose(operacion["pnl_neto"], -824.983125)
    assert np.isclose(info["capital"], 9_175.016875)
    assert np.isclose(recompensa, np.log(9_175.016875 / CAPITAL_INICIAL))
