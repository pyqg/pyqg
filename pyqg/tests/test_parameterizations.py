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

def test_smagorinsky(model):
    smag = pyqg.Smagorinsky()
    du, dv = smag(model)

def test_backscatter_biharmonic(model):
    back = pyqg.BackscatterBiharmonic()
    dq = back(model)

def test_zb2020(model):
    zb20 = pyqg.ZannaBolton2020()
    du, dv = zb20(model)

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

def test_ring(model):
    ring = pyqg.RingForcing()
    dq = ring(model)

if __name__ == "__main__":
    unittest.main()
