import numpy as np
from numpy import pi
import time
import pytest

# hack to skip tests if we don't have pyfft
try:
    import pyfftw
    skip=False
except ImportError:
    skip=True

@pytest.mark.skipif(skip, reason="Can't test fftw without pyfftw")
def test_fftw_rfft2(Nx = 64, Ny = None, n = 7200):
    """ A verification of rfft2/irfft2 pyfftw accuracy lost... """

    # settings
    if Ny is None: Ny = Nx

    # set random stream function
    A = np.random.randn(Nx,Ny)
    Ai = A.copy()

    tstart = time.time()
    for i in range(n):
        Ah = pyfftw.interfaces.numpy_fft.rfft2(Ai, threads=1)
        Ai = pyfftw.interfaces.numpy_fft.irfft2(Ah, threads=1)
    tend = time.time()

    # error after nmax fft cycles
    abs_err = np.abs(A-Ai).max()
    rel_err = np.abs((A-Ai)/A).max()

    print("RFFT2 test with Nx = %i, Ny = %i" %(Nx,Ny))
    print("Performed %i Forward/Inverse FFTs in %f5 seconds, CPU time" %(n,tend-tstart))
    print("Absolute error after %i Forward/Inverse FFTs = %e" %(n,abs_err))
    print("Relative error after %i Forward/Inverse FFTs = %e " %(n,rel_err))
    print(" ")

@pytest.mark.skipif(skip, reason="Can't test fftw without pyfftw")
def test_fftw_rfft(Nx = 64, n = 100000):
    """ A verification of rfft/irfft accuracy lost """

    # set random stream function
    A = np.random.randn(Nx)
    Ai = A.copy()

    tstart = time.time()
    for i in range(n):
        Ah = pyfftw.interfaces.numpy_fft.rfft(Ai, threads=1)
        Ai = pyfftw.interfaces.numpy_fft.irfft(Ah, threads=1)
    tend = time.time()

    # error after nmax fft cycles
    abs_err = np.abs(A-Ai).max()
    rel_err = np.abs((A-Ai)/A).max()

    print("RFFT test with Nx = %i" %Nx)
    print("Performed %i Forward/Inverse FFTs in %f5 seconds, CPU time" %(n,tend-tstart))
    print("Absolute error after %i Forward/Inverse FFTs = %e" %(n,abs_err))
    print("Relative error after %i Forward/Inverse FFTs = %e " %(n,rel_err))
    print(" ")

if __name__ == "__main__":

#    N = 2**np.arange(6,11)
#    for n in N:
#        test_fftw_rfft(Nx = n)

    N = 2**np.arange(5,8)
    for n in N:
        test_fftw_rfft2(Nx = n)
