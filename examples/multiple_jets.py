import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
import pyqg

# the model object
year = 1.
m = pyqg.BTModel(L=2.*pi,nx=256, tmax = 50*year,
        beta = 21., H = 1., rek = 0., rd = None, dt = 0.001,
                     taveint=year, ntd=4)

fk = m.wv != 0
ckappa = np.zeros_like(m.wv2)
ckappa[fk] = np.sqrt( m.wv2[fk]*(1. + (m.wv2[fk]/36.)**2) )**-1

nhx,nhy = m.wv2.shape

Pi_hat = np.random.randn(nhx,nhy)*ckappa +1j*np.random.randn(nhx,nhy)*ckappa

Pi = m.ifft( Pi_hat[np.newaxis,:,:] )
Pi = Pi - Pi.mean()

Pi = Pi - Pi.mean()
Pi_hat = m.fft( Pi )
KEaux = m.spec_var( m.wv*Pi_hat )

pih = ( Pi_hat/np.sqrt(KEaux) )
qih = -m.wv2*pih
qi = m.ifft(qih)
m.set_q(qi)

# run the model
plt.rcParams['image.cmap'] = 'YlOrRd_r'

for snapshot in m.run_with_snapshots(tsnapstart=0, tsnapint=200*m.dt):
    plt.clf()
    plt.imshow(m.q.squeeze() + m.beta * m.y)
    plt.clim([-20., 130.])
    #plt.title(str(m.t))
    plt.pause(0.01)

    plt.draw()

