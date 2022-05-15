import numpy as np
import pyqg
import pytest
import os
import pickle
from pyqg.diagnostic_tools import calc_ispec

def test_calc_ispec_peak():
    # Create a radial sine wave spiraling out from the center of the model's
    # spatial field (with a given frequency)
    m = pyqg.QGModel()
    radius = np.sqrt((m.x-m.x.mean())**2 + (m.y-m.y.mean())**2)
    frequency = m.k[0][20]
    radial_sine = np.sin(radius * frequency)

    # Take its FFT
    radial_sine_fft = m.fft(np.array([radial_sine, radial_sine]))[0]

    # Compute an isotropic spectrum
    iso_wavenumbers, iso_spectrum = calc_ispec(m, radial_sine_fft)

    # Its peak should be at the closest frequency to the true frequency
    spectrum_peak_idx = np.argmax(iso_spectrum)
    sinewave_freq_idx = np.argmin(np.abs(iso_wavenumbers - frequency))
    assert spectrum_peak_idx == sinewave_freq_idx

def test_calc_ispec_units(rtol=1e-5):
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
        for a in ['q','p']:
            for z in [0,1]:
                signal2d = getattr(m, a)[z]
                spectral = getattr(m, a+'h')[z]
                power = np.abs(spectral)**2/m.M**2
                k, ispec = calc_ispec(m, power, averaging=False)
                dk = k[1]-k[0]
                np.testing.assert_allclose(
                    signal2d.var(),
                    ispec.sum()*dk,
                    rtol,
                    err_msg=f"ispec should have correct integral for {a}{z+1}"
                )

def test_calc_ispec_sum(): 

    for nx in [16, 64, 256]:
        ny = nx
        m = pyqg.QGModel(nx = nx, ny = ny) 
        p = np.random.rand(ny, nx)

        # Get energy field and its spectrum
        E = np.abs(p)**2 # Energy in real space
        Eh_numpy = np.abs(np.fft.fft2(p))**2 # Energy spectrum in full plane
        Eh_model = np.abs(m.fft(p.reshape(1, nx, ny))[0])**2 # Energy spectrum in model (half) plane

        # Do the same in calc_ispec to avoid double counting 0th wavenumber and the largest wavenumber
        Eh_model_2 = Eh_model.copy()
        Eh_model_2[...,0] /= 2
        Eh_model_2[...,-1] /= 2

        # Get variance (or average energy)
        E_total = E.mean()

        # Check that the half and full planes both satisfy Parseval's theorem
        np.testing.assert_allclose(E_total, Eh_numpy.sum()/m.M**2)
        np.testing.assert_allclose(E_total, Eh_model_2.sum()/m.M**2*2) # Note the factor of 2 here

        ## Test calc_ispec()
        # Check that Parseval's theorem is roughly satisfied with averaging and truncation
        kr, Ehr = calc_ispec(m, Eh_model, truncate=True, averaging=True)
        E_total_radial = np.cumsum(Ehr * (kr[1]-kr[0]))[-1]
        assert E_total_radial/Eh_numpy.sum() > 0.5 and E_total_radial/Eh_numpy.sum() < 2,\
            f"Parseval's theorem is not roughly satisfied by calc_ispec"

        # Check that Parseval's theorem is exactly satisfied without averaging or truncation
        kr, Ehr = calc_ispec(m, Eh_model, truncate=False, averaging=False)
        E_total_radial = Ehr.sum() * (kr[1]-kr[0])
        np.testing.assert_allclose(E_total_radial, Eh_numpy.sum())

if __name__ == "__main__":
    test_calc_ispec_peak()
    test_calc_ispec_units()
    test_calc_ispec_sum()
