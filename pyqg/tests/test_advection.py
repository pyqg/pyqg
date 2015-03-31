import numpy as np
from pyqg import qg_model

def test_advect(rtol=1.e-13):
    """ Make sure advective term vanishes for plane wave """

    m = qg_model.QGModel(L=2.*np.pi,nx=256)

    
    kx = np.array([1.,5.,10.,0.,24.,2.])
    ky = np.array([2.,0.,10.,21.,12.,49.])

    for i in range(kx.size):

        # set plane wave PV
        m.set_q1q2(
                np.cos( kx[i] * m.x + ky[i] * m.y ),
                np.zeros_like(m.x) )

        # compute psi
        m.ph1,m.ph2 = m.invph(m.qh1,m.qh2)

        # diagnose vel.
        m.u1,m.v1 = m.caluv(m.ph1)

        # compute advection
        jacobh = m.advect(m.q1,m.u1,m.v1)
        jacob = qg_model.ifft2(m,jacobh)

        # residual
        res = np.abs(jacob).std()
        print "residual = %1.5e" %res

        assert res<rtol, " *** Jacobian residual is larger than %1.1e" %rtol

if __name__ == "__main__":
    test_advect()
