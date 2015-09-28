import matplotlib.pyplot as plt
import numpy as np
from numpy import pi

import pyqg
from pyqg.diagnostic_tools import spec_var

# the model object
year = 1.

m = pyqg.BTModel(L=2.*pi,nx=128, tmax = 200*year,
        beta = 15., H = 1., rek = 0., rd = None, dt = 0.005,
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

ke = []
t = []

for snapshot in m.run_with_snapshots(tsnapstart=0, tsnapint=15*m.dt):

    ke.append(m._calc_ke())
    t.append(m.t)

    plt.clf()
    plt.subplot(121)
    p1 = plt.imshow(np.fliplr(m.q[0,:,:]))
    plt.ylim(0,m.nx)
    plt.clim([-30., 30.])
    plt.title('t='+str(m.t))
    
    #plt.xticks([])
    #plt.yticks([])

    plt.subplot(122)

    p2 = plt.plot(t,ke)
    p2 = plt.plot(t[-1],ke[-1],'ro')
    plt.xlim([0., 200.])
    plt.ylim([.45,.5])

    plt.pause(0.01)

    plt.draw()

plt.show()
plt.ion()
