"""Prueba obligatoria 1: el entorno pasa check_env de Stable-Baselines3 sin advertencias."""
import warnings

import pytest
from stable_baselines3.common.env_checker import check_env

import data
from env import EntornoTrading


@pytest.fixture(scope="module")
def partes():
    return data.preparar_fold(data.cargar_datos()[0], 1)[0]


@pytest.mark.parametrize("parte, inicio_aleatorio", [("entrenamiento", True), ("prueba", False)])
def test_check_env_sin_advertencias(partes, parte, inicio_aleatorio):
    """Se revisa el entorno como se usa al entrenar (ventanas aleatorias) y al evaluar (periodo completo)."""
    entorno = EntornoTrading(partes[parte], inicio_aleatorio=inicio_aleatorio)
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        check_env(entorno, warn=True)
    assert not avisos, [str(aviso.message) for aviso in avisos]
