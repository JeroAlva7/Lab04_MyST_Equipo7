"""Prueba obligatoria 4: el capital final es el inicial más la suma del P&L neto de la bitácora."""
import numpy as np
import pytest

import data
from env import CAPITAL_INICIAL, EntornoTrading


@pytest.fixture(scope="module")
def partes():
    return data.preparar_fold(data.cargar_datos()[0], 1)[0]


@pytest.mark.parametrize("parte", ["validacion", "prueba"])
def test_capital_final_cuadra_con_la_bitacora(partes, parte):
    """Una política aleatoria recorre todo el periodo; al final no queda ninguna posición abierta."""
    entorno = EntornoTrading(partes[parte], inicio_aleatorio=False)
    entorno.reset(seed=data.SEMILLA)
    entorno.action_space.seed(data.SEMILLA)
    recompensas, terminado, truncado = [], False, False
    while not (terminado or truncado):
        _, recompensa, terminado, truncado, info = entorno.step(entorno.action_space.sample())
        recompensas.append(recompensa)
    bitacora = entorno.bitacora_df()
    assert len(bitacora) > 20 and entorno.posicion == 0
    assert np.isclose(info["capital"], CAPITAL_INICIAL + bitacora["pnl_neto"].sum(), rtol=1e-10)
    assert np.allclose(bitacora["pnl_neto"], bitacora["pnl_bruto"] - bitacora["costos"])
    curva = entorno.curva_capital()
    assert np.isclose(curva.iloc[-1], info["capital"]) and curva.index[-1] == partes[parte].barras.index[-1]
    # La suma de las contribuciones por paso es el log del crecimiento del capital.
    assert np.isclose(sum(recompensas), np.log(info["capital"] / CAPITAL_INICIAL))
