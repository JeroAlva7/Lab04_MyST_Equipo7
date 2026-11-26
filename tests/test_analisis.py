"""El análisis exploratorio usa solo el entrenamiento y cubre lo que pide la pregunta 1."""
import numpy as np
import pandas as pd
import pytest

import data


@pytest.fixture(scope="module")
def datos():
    return data.cargar_datos()[0]


def test_regimenes_cubren_el_entrenamiento(datos):
    """Al menos cuatro regímenes contiguos, alternando tendencia, de la primera a la última barra."""
    entrenamiento = data.particionar(datos, max(data.FOLDS))["entrenamiento"]
    tabla = data.regimenes(entrenamiento)
    assert len(tabla) >= 4
    assert tabla["inicio"].iloc[0] == entrenamiento.index[0]
    assert tabla["fin"].iloc[-1] == entrenamiento.index[-1]
    assert (tabla["fin"].iloc[:-1].to_numpy() == tabla["inicio"].iloc[1:].to_numpy()).all()
    assert (tabla["tendencia"].iloc[:-1].to_numpy() != tabla["tendencia"].iloc[1:].to_numpy()).all()


def test_objetivo_no_usa_precios_fuera_del_entrenamiento(datos):
    """El rendimiento futuro de las últimas horas del entrenamiento queda vacío en vez de tomar validación."""
    close = data.particionar(datos, 1)["entrenamiento"]["close"]
    objetivo = data.rendimiento_futuro(close, data.HORIZONTE_HORAS)
    limite = close.index[-1] - pd.Timedelta(hours=data.HORIZONTE_HORAS)
    assert objetivo[close.index > limite].isna().all()
    t = close.index[1000]
    esperado = np.log(close.loc[close.index >= t + pd.Timedelta(hours=data.HORIZONTE_HORAS)].iloc[0] / close.loc[t])
    assert np.isclose(objetivo.loc[t], esperado)
