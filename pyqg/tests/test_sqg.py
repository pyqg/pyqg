import numpy as np
from numpy import pi
from pyqg import sqg_model

def test_the_model(rtol=1e-14):
    """ Test the sqg model similatly to BT model. Numbers come from 
            a simulations in Cesar's 2014 MacPro with Anaconda 64-bit """

    # the model object
    year = 1.
    m = sqg_model.SQGModel(L=2.*pi,nx=128, tmax = 10*year,
            beta = 0., H = 1., rek = 0., rd = None, dt = 1.e-3,
                         taveint=year, twrite=1000,use_fftw=True, ntd=1)

    # a vortex merger IC with unit energy
    p = np.exp(-(2.*(m.x-1.75*pi/2))**2.-(2.*(m.y-pi))**2) +\
            np.exp(-(2.*(m.x-2.25*pi/2))**2.-(2.*(m.y-pi))**2)

    ph = m.fft(p[np.newaxis,:,:])
    KEaux = m.spec_var( m.filtr*m.wv*ph )/2.
    ph = ( ph/np.sqrt(KEaux) )
    qih = m.wv*ph
    qi = m.ifft(qih)
    m.set_q(qi)

    m.run()

    qnorm = (m.q**2).sum()
    mp = m.ifft(m.ph)
    pnorm = (mp**2).sum()
    ke = m._calc_ke()

    print 'time:       %g' % m.t
    assert m.t == 10.000999999999896
    
    np.testing.assert_allclose(qnorm, 31517.690603406099, rtol)
    np.testing.assert_allclose(pnorm, 5398.52096250875, rtol)
    np.testing.assert_allclose(ke, 0.96184358530902392, rtol)

if __name__ == "__main__":
    test_the_model()
