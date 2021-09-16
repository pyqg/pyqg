import numpy as np
import pyqg

def test_advect(rtol=1.e-13):
    """ Make sure advective term vanishes for plane wave
        It is an unpleasant fact that we cannot to get
        a double precision accuracy when the plane wave is
        slanted (kx != 0 and ky !=0 ) """

    #m = bt_model.BTModel(L=2.*np.pi,nx=256)
    m = pyqg.QGModel(L=2.*np.pi,nx=256,U1=0)

    # there are some magic combinations that
    #   fails the test...try kx=12,ky=24
    #   should investigate what's going on...

    kx = np.array([1.,5.,10.,0.,24.,2.])
    ky = np.array([2.,0.,10.,21.,12.,49.])


    for i in range(kx.size):

        # set plane wave PV
        #m.set_q(
        #        np.cos( kx[i] * m.x + ky[i] * m.y ))

        m.set_q1q2(
                np.cos( kx[i] * m.x + ky[i] * m.y ),
                np.zeros_like(m.x) )

        # compute psi
        m._invert()
        #m.ph1,m.ph2 = m.invph(m.qh1,m.qh2)

        # diagnose vel.
        #m.u1,m.v1 = m.caluv(m.ph1)

        # compute advection
        #jacobh = m.advect(m.q[0],m.u[0],m.v[0])
        jacobh = m._advect(m.q,m.u,m.v)
        #jacobh = m.advect(m.q,m.u,m.v)
        jacob = m.ifft(jacobh)

        # residual -- the L2-norm of the jacobian
        res = np.abs(jacob).sum()*m.dx*m.dy/(m.L**2)

        print("residual = %1.5e" %res)

        assert res<rtol, " *** Jacobian residual is larger than %1.1e" %rtol

if __name__ == "__main__":
    test_advect()
