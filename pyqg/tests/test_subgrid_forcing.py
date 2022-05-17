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

@pytest.fixture(params=[Layered, QG, SQG, BT])
def model(request):
    klass = request.param()
    model = klass()
    model._step_forward()
    return model

def test_forcing_generality(model):
    # make sure it doesn't error for any model class
    model.subgrid_forcing()

def test_qg_subgrid_forcing():
    # forcing with random initial conditions should be nonzero
    m = pyqg.QGModel()
    m._step_forward()
    Sq = m.subgrid_forcing(32, 'q')
    assert np.abs(Sq).sum() > 0

    # forcing with no filter should be larger
    Sq2 = m.subgrid_forcing(32, 'q',
            spectral_filter=lambda m2: np.ones_like(m2.filtr))
    assert np.abs(Sq2).sum() > np.abs(Sq).sum()

    # forcing with soft filter should be smaller
    Sq3 = m.subgrid_forcing(32, 'q',
            spectral_filter=lambda m2: np.exp(-m2.wv**2 * (2*m2.dx)**2 / 24))
    assert np.abs(Sq3).sum() < np.abs(Sq).sum()

    # forcing with all-0s filter should be 0
    Sq4 = m.subgrid_forcing(32, 'q',
            spectral_filter=lambda m2: np.zeros_like(m2.filtr))
    assert np.abs(Sq4).sum() == 0

    # can calculate forcing for many resolutions; average magnitude
    # should change accordingly
    res_vals = [62, 38, 6]
    forcings = [m.subgrid_forcing(res) for res in res_vals]
    assert np.abs(forcings[0]).mean() < np.abs(forcings[1]).mean()
    assert np.abs(forcings[1]).mean() < np.abs(forcings[2]).mean()

    # can pass multiple variables
    Su, Sv, Sq = m.subgrid_forcing(16, ['u','v','q'])
    assert Su.shape == (2,16,16)
    assert Sv.shape == (2,16,16)
    assert Sq.shape == (2,16,16)

    # test for invalid arguments
    with pytest.raises(AssertionError): m.subgrid_forcing(64)
    with pytest.raises(AssertionError): m.subgrid_forcing(63)
    with pytest.raises(AssertionError): m.subgrid_forcing(8, 'qh')
    with pytest.raises(AssertionError): m.subgrid_forcing(8, ':)')
