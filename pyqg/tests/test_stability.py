import unittest
import numpy as np
from numpy import sqrt
import pyqg

class LinearStabilityTester(unittest.TestCase):

    def setUp(self):

        self.atol=1.e-16

    def test_two_layer_stability(self):
        """ Make growth rates calculates numerically
                agree with the exact dispersion relationship
                for nz = 2 """

        m = pyqg.LayeredModel(L=1.e6,rd = 15.e3,nx=256,U=np.array([.1,0.]),V=np.array([0.,0.]),
                H=np.array([2000,2000.]),delta=1.,nz=2,f=1.)

        # numerical results
        evals, evecs = m.stability_analysis()

        # analytical results
        kb = sqrt(m.beta/(m.Us/2.))
        wv4 = m.wv2**2
        kd2 = m.rd**-2
        kd4 = kd2**2
        kb4 = kb**4
        omg_ana = np.zeros_like(m.wv2) + 0.j

        D = 1. +  (4.*wv4*(wv4 - kd4))/(kb4[:,np.newaxis]*kd4)

        fneg = D<0.
        omg_ana[fneg] = 1j*m.k[fneg]*(m.beta/(m.wv2[fneg] + kd2))*( (kd2/(2.*m.wv2[fneg]))*sqrt(-D[fneg]) )

        res = np.abs(omg_ana.imag-evals.imag).max()

        np.testing.assert_allclose(res,0.,atol=self.atol,
                err_msg=' Residual  of growth rate is larger than %1.1e' %res)

if __name__ == "__main__":
    unittest.main()
