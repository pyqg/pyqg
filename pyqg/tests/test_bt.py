import numpy as np
from numpy import pi
import pyqg

def test_the_model(rtol=1e-5):
    """ Test the BT model. Numbers come from a simulations in
            Cesar's 2014 MacPro with Anaconda 64-bit """

    # the model object
    year = 1.
    m = pyqg.BTModel(L=2.*pi,nx=128, tmax = 10*year,
            beta = 0., H = 1., rek = 0., rd = None, dt = 0.001,
                         taveint=year,twrite=1000, ntd=1)

    # a vortex merger IC with unit energy
    p = np.exp(-(2.*(m.x-1.75*pi/2))**2.-(2.*(m.y-pi))**2) +\
            np.exp(-(2.*(m.x-2.25*pi/2))**2.-(2.*(m.y-pi))**2)

    ph = m.fft(p[np.newaxis,...])
    KEaux = m.spec_var(m.filtr*m.wv*ph )/2.
    pih = ( ph/np.sqrt(KEaux) )
    qih = -m.wv2*pih
    qi = m.ifft(qih)
    m.set_q(qi)

    np.testing.assert_allclose(m.q, qi, rtol)

    m.run()

    qnorm = (m.q**2).sum()
    mp = m.ifft(m.ph)
    pnorm = (mp**2).sum()
    ke = m._calc_ke()

    print 'time:       %g' % m.t
    assert m.t == 10.000999999999896

    np.testing.assert_allclose(qnorm, 356981.55844167515, rtol)
    np.testing.assert_allclose(pnorm, 5890.857144590821, rtol)
    np.testing.assert_allclose(ke, 0.99872441481956509, rtol)

if __name__ == "__main__":
    test_the_model()
