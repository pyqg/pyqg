from future.utils import iteritems
import pytest
import unittest
import numpy as np
import pyqg
from pyqg import diagnostic_tools as diag

def test_describe_diagnostics():
    """ Test whether describe_diagnostics runs without error """

    m = pyqg.QGModel(1)
    m.describe_diagnostics()

def test_paramspec_diagnostics(rtol=1e-12):
    # Initialize four models with different (deterministic) parameterizations
    m1 = pyqg.QGModel()

    dq = np.random.normal(size=m1.q.shape)
    du = np.random.normal(size=m1.u.shape)
    dv = np.random.normal(size=m1.v.shape)

    m2 = pyqg.QGModel(q_parameterization=lambda m: dq)
    m3 = pyqg.QGModel(uv_parameterization=lambda m: (du,dv))
    m4 = pyqg.QGModel(q_parameterization=lambda m: dq,
                      uv_parameterization=lambda m: (du,dv))

    # Give them the same initial conditions
    for m in [m1,m2,m3,m4]:
        m.q = m1.q

    # Step them forward and manually increment diagnostics
    for m in [m1,m2,m3,m4]:
        m._step_forward()
        m._increment_diagnostics()

    # Unparameterized model should have 0 for its parameterization spectrum
    np.testing.assert_allclose(m1.get_diagnostic('paramspec'), 0., rtol=rtol)

    # Parameterized models should have nonzero values
    for m in [m2,m3,m4]:
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(m.get_diagnostic('paramspec'), 0., rtol=rtol)

    # Model with both parameterizations should have the sum
    np.testing.assert_allclose(
            (m2.get_diagnostic('paramspec') + m3.get_diagnostic('paramspec')),
            m4.get_diagnostic('paramspec'),
            rtol=rtol)



