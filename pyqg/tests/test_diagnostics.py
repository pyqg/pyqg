from future.utils import iteritems
import pytest
import unittest
import numpy as np
import pyqg
import pickle
import os
from pyqg import diagnostic_tools as diag

def test_describe_diagnostics():
    """ Test whether describe_diagnostics runs without error """

    m = pyqg.QGModel(1)
    m.describe_diagnostics()

def old_qgmodel_calc_paramspec(self, dqh1, dqh2):
    del1 = self.del1
    del2 = self.del2
    F1 = self.F1
    F2 = self.F2
    wv2 = self.wv2
    ph = self.ph
    return np.real(
        (del1 / (wv2 + F1 + F2) * (-(wv2 + F2) * dqh1 - F1 * dqh2) * np.conj(ph[0])) +
        (del2 / (wv2 + F1 + F2) * (-F2 * dqh1 - (wv2 + F1) * dqh2) * np.conj(ph[1])) +
        (del1 * F1 / (wv2 + F1 + F2) * (dqh2 - dqh1) * np.conj(ph[0] - ph[1]))
    )

def test_paramspec_decomposition(rtol=1e-10):
    # Initialize a model with a parameterization, step it forward and compute paramspec
    dq = np.random.normal(size=(2,64,64))
    m = pyqg.QGModel(q_parameterization = lambda m: dq)
    m._step_forward()
    m._increment_diagnostics()

    # Compute the parameterization spectrum at least two ways
    height_ratios = (m.Hi / m.H)[:,np.newaxis,np.newaxis]
    dqh = m.fft(dq)
    ps1 = -np.real((height_ratios * np.conj(m.ph) * dqh).sum(axis=0)) / m.M**2
    ps2 = old_qgmodel_calc_paramspec(m, dqh[0], dqh[1]) / m.M**2
    ps3 = m.get_diagnostic('paramspec') 

    # Ensure they're identical
    np.testing.assert_allclose(ps1, ps2, rtol=rtol)
    np.testing.assert_allclose(ps1, ps3, rtol=rtol)

    # Now test it can be decomposed into separate KE and APE components
    apeflux_term = np.einsum("ij, jk..., k... -> i...", m.S, m.a, dqh)
    keflux_term  = np.einsum("ij..., j... -> i...", m.a, dqh)
    height_ratios = (m.Hi/m.H)[:,np.newaxis,np.newaxis]
    paramspec_apeflux = -np.real(height_ratios*m.ph.conj()*apeflux_term).sum(axis=0) / m.M**2
    paramspec_keflux  = m.wv2*np.real(height_ratios*m.ph.conj()* keflux_term).sum(axis=0) / m.M**2
    
    ps4 = paramspec_apeflux + paramspec_keflux
    np.testing.assert_allclose(ps1, ps4, rtol=rtol)

    # Test these terms match the subterms from QGModel
    np.testing.assert_allclose(paramspec_apeflux,
            m.get_diagnostic('paramspec_APEflux'), rtol=rtol)
    np.testing.assert_allclose(paramspec_keflux,
            m.get_diagnostic('paramspec_KEflux'), rtol=rtol)

def test_paramspec_additivity(rtol=1e-10):
    # Test over multiple model classes
    for model_class in [pyqg.QGModel, pyqg.LayeredModel]:
        # Initialize four models with different (deterministic) parameterizations
        m1 = model_class()

        dq = np.random.normal(size=m1.q.shape)
        du = np.random.normal(size=m1.u.shape)
        dv = np.random.normal(size=m1.v.shape)

        m2 = model_class(q_parameterization=lambda m: dq)
        m3 = model_class(uv_parameterization=lambda m: (du,dv))
        m4 = model_class(q_parameterization=lambda m: dq,
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

    diss_contribution = -np.real(np.tensordot(m.Hi, np.conj(m.ph)*diss_spectrum, axes=(0, 0)))/m.H/m.dt/m.M**2
    diss_contribution_model = m.get_diagnostic('Dissspec')

    # Ensure that the above calculation is consistent with the model's internal calculation
    np.testing.assert_allclose(diss_contribution, diss_contribution_model, atol=atol)

    # Obtain filtered contribution, which is used in the model
    qh_new = m.qh.copy()
    for k in range(m.nz):
        qh_new[k] = m.filtr * rhs_unfiltered[k]
    rhs_contribution_filtered = -np.real(np.tensordot(m.Hi, np.conj(m.ph)*qh_new, axes=(0, 0)))/m.H/m.dt/m.M**2
    rhs_contribution_unfiltered = -np.real(np.tensordot(m.Hi, np.conj(m.ph)*rhs_unfiltered, axes=(0, 0)))/m.H/m.dt/m.M**2

    # Ensure that the difference between the filtered contribution and the unfiltered contribution is 
    # completely the effect of dissipation
    np.testing.assert_allclose(diss_contribution_model, 
                               rhs_contribution_filtered - rhs_contribution_unfiltered, 
                               atol=atol)

def test_diagnostic_magnitude():
    # Load a set of pre-run fixture models from
    # examples/diagnostic_normalization.ipynb (running from scratch would take
    # a bit too long for a test)
    fixtures_path = f"{os.path.dirname(os.path.realpath(__file__))}/fixtures"

    with open(f"{fixtures_path}/LayeredModel_params.pkl", 'rb') as f:
        # Common set of parameters for each model
        params = pickle.load(f)

    m1 = pyqg.LayeredModel(nx=96, **params)
    m2 = pyqg.LayeredModel(nx=64, **params)
    m1.q = np.load(f"{fixtures_path}/LayeredModel_nx96_q.npy")
    m2.q = np.load(f"{fixtures_path}/LayeredModel_nx64_q.npy")
    for m in [m1, m2]:
        m._invert()
        m._calc_derived_fields()

    # Loop through all diagnostics
    for diagnostic in m1.diagnostics.keys():
        if diagnostic == 'Dissspec':
            continue

        # Get the maximum-magnitude instantaneous value of each diagnostic,
        # re-evaluating the function rather than relying on any saved
        # diagnostics (gives a rough idea of order of magnitude)
        max_hi = np.abs(m1.diagnostics[diagnostic]['function'](m1)).max()
        max_lo = np.abs(m2.diagnostics[diagnostic]['function'](m2)).max()
        if max_lo == 0:
            assert max_hi == 0
        else:
            # Ensure they're the same order of magnitude -- no more than a
            # factor of 3 different. If these assertions fail for a new
            # diagnostic, you're probably missing a division by M**2.
            assert max_hi/max_lo < 3, f"{diagnostic} should be normalized"
            assert max_hi/max_lo > 0.33, f"{diagnostic} should be normalized"
