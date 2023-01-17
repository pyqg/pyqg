import numpy as np
import pyqg
import pytest
import unittest

def QG():
    return pyqg.QGModel

def Layered():
    return pyqg.LayeredModel

def SQG():
    return pyqg.SQGModel

def BT():
    return pyqg.BTModel

@pytest.fixture(params=[QG, Layered, SQG, BT])
def model(request):
    klass = request.param()
    model = klass()
    model._step_forward()
    return model

def test_parameterizations(model):
    for parameterization_type in pyqg.parameterization_types:
        if parameterization_type == pyqg.HybridSymbolicRLPGZ2022 and model != QG:
            continue
        parameterization = parameterization_type()
        predicted_forcing = parameterization(model)

def test_addition_and_scaling(model, rtol=1e-11):
    back = pyqg.BackscatterBiharmonic()
    smag = pyqg.Smagorinsky()
    zb20 = pyqg.ZannaBolton2020()
    comb = 0.5*smag + 0.75*zb20
    du, dv = comb(model)
    np.testing.assert_allclose(du, 0.5*smag(model)[0] + 0.75*zb20(model)[0],
            rtol=rtol)
    np.testing.assert_allclose(dv, 0.5*smag(model)[1] + 0.75*zb20(model)[1],
            rtol=rtol)

    # can't add uv and q parameterizations
    with pytest.raises(AssertionError):
        back + smag

if __name__ == "__main__":
    unittest.main()
