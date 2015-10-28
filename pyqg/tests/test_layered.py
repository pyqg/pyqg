import numpy as np
import pyqg

def test_the_model(atol=1.e-16):

    """ Test methods specific to LayeredModel subclass """ 

    m = pyqg.LayeredModel(
            nz = 3,
            U  = [.1,.05,.0],                    
            V  = [.1,.05,.0],                    
            rho= [.1,.3,.5],
            H  = [.1,.1,.3],
            f  = 1.,
            beta = 0.)
    
    ## now construct and test stretching matrix

    # creates stretching matrix from scratch
    S = np.zeros((m.nz,m.nz))
    
    F11 = m.f2/m.gpi[0]/m.Hi[0]
    F12 = m.f2/m.gpi[0]/m.Hi[1]
    F22 = m.f2/m.gpi[1]/m.Hi[1]
    F23 = m.f2/m.gpi[1]/m.Hi[2]

    S[0,0], S[0,1] = -F11, F11 
    S[1,0], S[1,1], S[1,2] = F12, -(F12+F22), F22
    S[2,1], S[2,2] = F23, -F23

    # the columns of the S must add to zero (i.e, S is singular)
    assert np.all(m.S.sum(axis=1)==0.) , " Zero is not an eigenvalue of S "

    # the matrix Hi * S must by a symmetric matrix
    HS = np.dot(np.diag(m.Hi),m.S)
    np.testing.assert_allclose(HS,HS.T,atol=atol) , " Hi*S is not symmetric "

    np.testing.assert_allclose(m.S,S,atol=atol) , " Unmatched stretching matrix "
   
    ## test init background    
    Qy = -np.einsum('ij,j->i',S,m.Ubg)
    Qx = np.einsum('ij,j->i',S,m.Vbg)
    
    np.testing.assert_allclose(Qy,m.Qy,atol=atol) , " Unmatched Qy "
    np.testing.assert_allclose(Qx,m.Qx,atol=atol) , " Unmatched Qx "

    ## test inversion matrix

    # it suffices to test for a single wavenumber
    M = S - np.eye(m.nz)*m.wv2[5,5]
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

    np.testing.assert_allclose(m.a[:,:,5,5], Minv,atol=atol), " Unmatched inversion matrix "

if __name__ == "__main__":
    test_the_model()   
