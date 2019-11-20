import unittest
import numpy as np
import pyqg

class LayeredModelTester(unittest.TestCase):

    def setUp(self):

        self.m = pyqg.LayeredModel(
                    nz = 3,
                    U  = [.1,.05,.0],
                    V  = [.1,.05,.0],
                    rho= [.1,.3,.5],
                    H  = [.1,.1,.3],
                    f  = 1.,
                    beta = 0.)

        self.atol=1.e-16

        # creates stretching matrix from scratch
        self.S = np.zeros((self.m.nz,self.m.nz))

        F11 = self.m.f2/self.m.gpi[0]/self.m.Hi[0]
        F12 = self.m.f2/self.m.gpi[0]/self.m.Hi[1]
        F22 = self.m.f2/self.m.gpi[1]/self.m.Hi[1]
        F23 = self.m.f2/self.m.gpi[1]/self.m.Hi[2]

        self.S[0,0], self.S[0,1] = -F11, F11
        self.S[1,0], self.S[1,1], self.S[1,2] = F12, -(F12+F22), F22
        self.S[2,1], self.S[2,2] = F23, -F23

    def test_stretching(self):
        """ Check if stretching matrix is consistent
                and satisfies basic properties """

        # the columns of the S must add to zero (i.e, S is singular)
        err_msg = ' Zero is not an eigenvalue of S'
        assert np.all(self.m.S.sum(axis=1)==0.) , err_msg

        # the matrix Hi * S must by a symmetric matrix
        HS = np.dot(np.diag(self.m.Hi),self.m.S)
        np.testing.assert_allclose(HS,HS.T,atol=self.atol,
                err_msg=' Hi*S is not symmetric')

        np.testing.assert_allclose(self.m.S,self.S,atol=self.atol,
                err_msg= ' Unmatched stretching matrix')

    def test_init_background(self):
        """ Check the initialization of the mean PV gradiends  """
        Qy = -np.einsum('ij...,j...->i...',self.S,self.m.Ubg)
        Qx = np.einsum('ij...,j...->i...',self.S,self.m.Vbg)

        np.testing.assert_allclose(Qy,self.m.Qy,atol=self.atol,
                err_msg=' Unmatched Qy')
        np.testing.assert_allclose(Qx,self.m.Qx,atol=self.atol,
                err_msg=' Unmatched Qx ')


    def test_inversion_matrix(self):
        """ Check the inversion matrix """

        # it suffices to test for a single wavenumber
        M = self.S - np.eye(self.m.nz)*self.m.wv2[5,5]
        Minv = np.zeros_like(M)

        detM = np.linalg.det(M)

        Minv[0,0] = M[1,1]*M[2,2] - M[1,2]*M[2,1]
        Minv[0,1] = M[0,2]*M[2,1] - M[0,1]*M[2,2]
        Minv[0,2] = M[0,1]*M[1,2] - M[0,2]*M[1,1]

        Minv[1,0] = M[1,2]*M[2,0] - M[1,0]*M[2,2]
        Minv[1,1] = M[0,0]*M[2,2] - M[0,2]*M[2,0]
        Minv[1,2] = M[0,2]*M[1,0] - M[0,0]*M[1,2]

        Minv[2,0] = M[1,0]*M[2,1] - M[1,1]*M[2,0]
        Minv[2,1] = M[0,1]*M[2,0] - M[0,0]*M[2,1]
        Minv[2,2] = M[0,0]*M[1,1] - M[0,1]*M[1,0]

        Minv = Minv/detM

        np.testing.assert_allclose(self.m.a[:,:,5,5], Minv,atol=self.atol,
                err_msg= ' Unmatched inversion matrix ')

if __name__ == "__main__":
    unittest.main()
