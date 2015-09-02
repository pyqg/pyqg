import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
import pyqg
from pyqg.diagnostic_tools import spec_var

# the model object
year = 1.
m = pyqg.BTModel(L=2.*pi,nx=128, tmax = 200*year,
        beta = 0., H = 1., rek = 0., rd = None, dt = 0.005,
                     taveint=year, ntd=2)

# McWilliams 84 IC condition
fk = m.wv != 0
ckappa = np.zeros_like(m.wv2)
ckappa[fk] = np.sqrt( m.wv2[fk]*(1. + (m.wv2[fk]/36.)**2) )**-1

nhx,nhy = m.wv2.shape

Pi_hat = np.random.randn(nhx,nhy)*ckappa +1j*np.random.randn(nhx,nhy)*ckappa

Pi = m.ifft( Pi_hat[np.newaxis] )
Pi = Pi - Pi.mean()
Pi_hat = m.fft( Pi )
KEaux = spec_var( m, m.filtr*m.wv*Pi_hat )

pih = ( Pi_hat/np.sqrt(KEaux) )
qih = -m.wv2*pih
qi = m.ifft(qih)
m.set_q(qi)

# run the model and plot some figs
plt.rcParams['image.cmap'] = 'RdBu_r'

plt.ion()

for snapshot in m.run_with_snapshots(tsnapstart=0, tsnapint=15*m.dt):

    plt.clf()
    p1 = plt.imshow(m.q[0] + m.beta * m.y)
    plt.clim([-30., 30.])
    plt.title('t='+str(m.t))
    
    plt.xticks([])
    plt.yticks([])

    plt.pause(0.01)

    plt.draw()

plt.show()
plt.ion()
