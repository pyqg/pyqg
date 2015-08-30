import numpy as np
import pyqg

def test_parseval(rtol=1.e-15):
    """ Make sure 2D fft from QGModel satisfy Parseval's relation
            within machine double precision """

    m = pyqg.BTModel(L=2.*np.pi)

    # set random stream function
    p1 = np.random.randn(1,m.nx,m.ny)
    p1 -= p1.mean()

    ph1 = m.fft(p1)

    # var(P) from Fourier coefficients
    P_var_spec = m.spec_var(ph1) 
    print "Variance from spectrum: %5.16f" %P_var_spec

    # var(P) in physical space
    P_var_phys = p1.var()
    print "Variance in physical space: %5.16f" %P_var_phys

    # relative error
    error = np.abs(P_var_phys - P_var_spec)/P_var_phys
    print "error = %5.16f" %error

    assert error<rtol, " *** BTModel FFT2 does not satisfy Parseval's relation "

if __name__ == "__main__":
    test_parseval()
