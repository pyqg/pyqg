import numpy as np
from pyqg import qg_model

def test_parseval(rtol=1.e-15):
    """ Make sure 2D fft from QGModel satisfy Parseval's relation
            within machine double precision """

    m = qg_model.QGModel(L=2.*np.pi)

    # set random stream function
    p1 = np.random.randn(m.nx,m.ny)
    m.p1 = p1 - p1.mean()

    m.ph1 = m.fft2(m.p1)

    # var(P) from Fourier coefficients
    P_var_spec = qg_model.spec_var(m,m.ph1) 
    print "Variance from spectrum: %5.16f" %P_var_spec

    # var(P) in physical space
    P_var_phys = m.p1.var()
    print "Variance in physical space: %5.16f" %P_var_phys

    # relative error
    error = np.abs(P_var_phys - P_var_spec)/P_var_phys
    print "error = %5.16f" %error

    assert error<rtol, " *** QGModel FFT2 does not satisfy Parseval's relation "

if __name__ == "__main__":
    test_parseval()
