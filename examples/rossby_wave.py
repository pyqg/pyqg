import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
import pyqg

# the model object
year = 1.
m = pyqg.BTModel(L=2.*pi,nx=128, tmax = 1*year,
        beta = 20., H = 1., rek = 0., rd = None, dt = 0.001,
                     taveint=year, ntd=4)

# Gaussian IC
fk = m.wv != 0
ckappa = np.zeros_like(m.wv2)
ckappa[fk] = np.sqrt( m.wv2[fk]*(1. + (m.wv2[fk]/36.)**2) )**-1

nhx,nhy = m.wv2.shape

R = pi/6.
Pi = -np.exp(-((m.x-3*pi/2)**2 + (m.y-pi)**2)/R**2)

Pi = Pi - Pi.mean()
Pi_hat = m.fft( Pi[np.newaxis,:,:] )
KEaux = m.spec_var(m.wv*Pi_hat )

pih = ( Pi_hat/np.sqrt(KEaux) )
qih = -m.wv2*pih
qi = m.ifft(qih)
m.set_q(qi)

# run the model
plt.rcParams['image.cmap'] = 'RdBu'

plt.ion()

for snapshot in m.run_with_snapshots(tsnapstart=0, tsnapint=10*m.dt):
    plt.clf()
    plt.imshow(m.q.squeeze())
    plt.clim([-20., 20.])
    plt.xticks([])
    plt.yticks([])

    plt.pause(0.01)
    plt.draw()
    plt.ioff()
    

