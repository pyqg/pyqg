import numpy as np
import unittest
import pyqg
from pyqg.diagnostic_tools import spec_var

class ModelFFTMixin:

    def test_parseval(self, rtol=1.e-15):
        """ Make sure 2D fft satisfies Parseval's relation
                within machine double precision """

        # set random stream function
        p1 = np.random.randn(self.m.nz, self.m.ny, self.m.nx)
        p1 -= p1.mean(axis=(-2,-1))[:,np.newaxis,np.newaxis]

        ph1 = self.m.fft(p1)

        # var(P) from Fourier coefficients
        P_var_spec = spec_var(self.m, ph1)
        #print("Variance from spectrum: %5.16f" % P_var_spec)

        # var(P) in physical space
        P_var_phys = p1.var(axis=(-2,-1))
        #print("Variance in physical space: %5.16f" % P_var_phys)

        # relative error
        error = np.abs((P_var_phys - P_var_spec)/P_var_phys).sum()
        print("error = %5.16f" %error)

        self.assertTrue(error<rtol,
                "fft does not satisfy Parseval's relation")

    def test_forward_backward(self, rtol=1e-5):
        """Test that forward and backward transform give the right result"""

        r = np.random.randn(self.m.nz, self.m.ny, self.m.nx)
        r -= r.mean(axis=(-2,-1))[:,np.newaxis,np.newaxis]
        rh = self.m.fft(r)
        rf = self.m.ifft(rh)
        self.assertTrue(np.allclose(r, rf, rtol=rtol))

class BTModelFFTTester(ModelFFTMixin, unittest.TestCase):

    def setUp(self):
        self.m = pyqg.BTModel(L=2.*np.pi, beta=0., U=0., rd=0., filterfac=0)

class QGModelFFTTester(ModelFFTMixin, unittest.TestCase):

    def setUp(self):
        self.m = pyqg.QGModel(L=2.*np.pi, beta=0., U1=0., U2=0., filterfac=0.)


if __name__ == "__main__":
    test_parseval()
