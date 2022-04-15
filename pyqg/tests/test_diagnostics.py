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

def test_paramspec_diagnostics(rtol=1e-10):
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

def test_Dissspec_diagnostics(atol=1e-20):

    # Run model for some timesteps
    dt = 3600
    tmax = dt * 1000
    m = pyqg.QGModel(tavestart=tmax, taveint=1, tmax=tmax, dt=dt)
    m.run()
    
    # Need to run _calc_diagnostics() once more to use the most recent state variables
    m._calc_diagnostics() 

    # Calculate spectral contribution of dissipation offline
    diss_spectrum, rhs_unfiltered = np.zeros_like(m.qh), np.zeros_like(m.qh)
    ones = np.ones_like(m.filtr)

    # Get AB coefficients
    if m.ablevel==0:
        # forward euler
        dt1 = m.dt
        dt2 = 0.0
        dt3 = 0.0
    elif m.ablevel==1:
        # AB2 at step 2
        dt1 = 1.5*m.dt
        dt2 = -0.5*m.dt
        dt3 = 0.0
    else:
        # AB3 from step 3 on
        dt1 = 23./12.*m.dt
        dt2 = -16./12.*m.dt
        dt3 = 5./12.*m.dt

    for k in range(m.nz):
        rhs_unfiltered[k] = m.qh[k] + dt1*m.dqhdt[k] + dt2*m.dqhdt_p[k] + dt3*m.dqhdt_pp[k]
        diss_spectrum[k] = (m.filtr - ones) * rhs_unfiltered[k]

    diss_contribution = -np.real(np.tensordot(m.Hi, np.conj(m.ph)*diss_spectrum, axes=(0, 0)))/m.H/m.dt
    diss_contribution_model = m.get_diagnostic('Dissspec')

    # Ensure that the above calculation is consistent with the model's internal calculation
    np.testing.assert_allclose(diss_contribution, diss_contribution_model, atol=atol)

    # Obtain filtered contribution, which is used in the model
    qh_new = m.qh.copy()
    for k in range(m.nz):
        qh_new[k] = m.filtr * rhs_unfiltered[k]
    rhs_contribution_filtered = -np.real(np.tensordot(m.Hi, np.conj(m.ph)*qh_new, axes=(0, 0)))/m.H/m.dt
    rhs_contribution_unfiltered = -np.real(np.tensordot(m.Hi, np.conj(m.ph)*rhs_unfiltered, axes=(0, 0)))/m.H/m.dt

    # Ensure that the difference between the filtered contribution and the unfiltered contribution is 
    # completely the effect of dissipation
    np.testing.assert_allclose(diss_contribution_model, 
                               rhs_contribution_filtered - rhs_contribution_unfiltered, 
                               atol=atol)

