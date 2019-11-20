from __future__ import print_function
import unittest
import numpy as np
import pyqg

class VerticalModesTester(unittest.TestCase):

    def setUp(self):

        self.m = pyqg.LayeredModel(
                    nz = 2,
                    U  = [.1,.05],
                    V  = [.1,.0],
                    H  = [1.,1.],
                    delta = 1.,
                    rd = 5.,
                    f  = 1.,
                    beta = 0.,)

        self.atol=1.e-16

        self.m.set_q(np.random.randn(self.m.nz,self.m.ny,self.m.nx))

        self.m.vertical_modes()

    def test_radii(self):
        """ Check deformation radii are computed correctly """


        radii = np.array([np.sqrt(self.m.g*self.m.H)/np.abs(self.m.f), self.m.rd])
        np.testing.assert_allclose(self.m.radii,radii,atol=self.atol,
                err_msg=' Wrong deformation radii')

    def test_modes(self):
        """ Check if vertical modes are computes and normalized correctly """

        p = (self.m.Hi[:,np.newaxis]*self.m.pmodes)/self.m.H

        print(p)

        # barotropic mode must be depth invariant
        np.testing.assert_allclose(p[0,0],p[1,0],atol=self.atol,
                err_msg=' Barotropic mode is not depth invariant')

        # baroclinic modes should have zero integral
        np.testing.assert_allclose(p[0,1]+p[1,1],0.,atol=self.atol,
                err_msg=' Baroclinic mode does not integrate to zero')

        # check normalization
        np.testing.assert_allclose(p[0,0]+p[1,0],1.,atol=self.atol,
                err_msg=' Wrong normalization')


    def test_projection(self):
        """ Check modal projection """

        qn = self.m.modal_projection(self.m.qh,forward=True)
        qh = self.m.modal_projection(qn, forward=False)

        np.testing.assert_allclose(self.m.qh,qh,atol=self.atol,
                err_msg=' Modal projection does not satisfy inversion')


if __name__ == "__main__":
    unittest.main()
