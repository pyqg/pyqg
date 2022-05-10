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

def test_calc_ispec_units(rtol=1e-2):
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
                ah = a+'h'
                signal2d = getattr(m, a)[z]
                power = np.abs(getattr(m, ah)[z])**2/m.M**2
                k, ispec = calc_ispec(m, power, averaging=False)
                np.testing.assert_allclose(
                    signal2d.var()/2,
                    ispec.sum()*(k[1]-k[0])/2,
                    rtol,
                    err_msg=f"ispec should have correct units for {a} at z={z}"
                )

if __name__ == "__main__":
    test_calc_ispec_peak()
    test_calc_ispec_units()
