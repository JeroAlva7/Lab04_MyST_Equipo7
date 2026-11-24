"""Los estadísticos de normalización salen solo del entrenamiento de cada fold (sección 3.1)."""
import numpy as np
import pytest

import data


@pytest.fixture(scope="module")
def datos():
    return data.cargar_datos()[0]


@pytest.mark.parametrize("fold", sorted(data.FOLDS))
def test_normalizador_usa_solo_entrenamiento(datos, fold):
    """El normalizador del fold coincide con uno ajustado a mano solo con el entrenamiento."""
    partes, normalizador = data.preparar_fold(datos, fold)
    variables = data.calcular_variables(datos).loc[partes["entrenamiento"].barras.index]
    np.testing.assert_allclose(normalizador.media, variables.mean(), rtol=1e-12)
    np.testing.assert_allclose(normalizador.desviacion, variables.std(ddof=0), rtol=1e-12)
    for nombre in ("validacion", "prueba"):
        assert partes[nombre].barras.index.min() > partes["entrenamiento"].barras.index.max()
