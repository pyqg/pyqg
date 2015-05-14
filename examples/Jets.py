import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from pyqg import bt_model

# the model object
year = 1.
m = bt_model.BTModel(L=2.*pi,nx=256, tmax = 50*year,
        beta = 20., H = 1., rek = 0., rd = None, dt = 0.001,
                     taveint=year, use_fftw=True, ntd=4)

fk = m.wv != 0
ckappa = np.zeros_like(m.wv2)
ckappa[fk] = np.sqrt( m.wv2[fk]*(1. + (m.wv2[fk]/36.)**2) )**-1

nhx,nhy = m.wv2.shape

Pi_hat = np.random.randn(nhx,nhy)*ckappa +1j*np.random.randn(nhx,nhy)*ckappa

Pi = m.ifft2( Pi_hat )
Pi = Pi - Pi.mean()

Pi = Pi - Pi.mean()
Pi_hat = m.fft2( Pi )
KEaux = bt_model.spec_var( m, m.wv*Pi_hat )

pih = ( Pi_hat/np.sqrt(KEaux) )
qih = -m.wv2*pih
qi = m.ifft2(qih)
m.set_q(qi,check=False)

# run the model
for snapshot in m.run_with_snapshots(tsnapstart=0, tsnapint=200*m.dt):
    plt.clf()
    plt.imshow(m.q + m.beta * m.y)
    #plt.clim([-55., 55.])
    plt.show()
    plt.pause(0.01)

