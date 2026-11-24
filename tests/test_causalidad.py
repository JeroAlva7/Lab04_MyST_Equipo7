"""Prueba 2 de la sección 3.11: causalidad de las variables."""
import numpy as np
import pytest

import data


@pytest.fixture(scope="module")
def datos():
    return data.cargar_datos()[0]


def test_variables_causales(datos):
    """La observación en el índice i calculada solo con datos hasta i es igual a la de la serie completa."""
    completas = data.calcular_variables(datos)
    validas = np.flatnonzero(completas.notna().all(axis=1).to_numpy())
    rng = np.random.default_rng(42)
    indices = sorted({int(validas[0]), int(validas[-1]), *rng.choice(validas, size=4, replace=False).tolist()})
    for i in indices:
        truncadas = data.calcular_variables(datos.iloc[: i + 1])
        np.testing.assert_allclose(truncadas.iloc[-1].to_numpy(), completas.iloc[i].to_numpy(), rtol=1e-12, atol=0)
