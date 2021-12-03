import numpy as np
import pyqg
import pytest
from pyqg.diagnostic_tools import calc_ispec

def test_calc_ispec():
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
