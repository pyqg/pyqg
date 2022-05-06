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

def test_calc_ispec_units():
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

    for diagnostic in m1.diagnostics.keys():
        if 'Dissspec' in diagnostic:
            continue

        if m1.diagnostics[diagnostic]['dims'] == ('l','k'):
            spec1 = m1.diagnostics[diagnostic]['function'](m1)
            spec2 = m2.diagnostics[diagnostic]['function'](m2)
            spec1x = spec1.sum(axis=0)
            spec1y = spec1.sum(axis=1)
            spec2x = spec2.sum(axis=0)
            spec2y = spec2.sum(axis=1)
            _, spec1r = calc_ispec(m1, spec1)
            _, spec2r = calc_ispec(m2, spec2)

            scales = [
                np.abs(spec).max()
                for spec in [spec1x, spec1y, spec1r,
                             spec2x, spec2y, spec2r]
            ]

            if max(scales) == 0:
                continue

            scale_ratio = max(scales) / min(scales)

            assert scale_ratio < 5, \
                f"calc_ispec should preserve units of {diagnostic}"
            assert scale_ratio > 0.2, \
                f"calc_ispec should preserve units of {diagnostic}"

