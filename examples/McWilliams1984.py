import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from pyqg import bt_model

# the model object
year = 1.
m = bt_model.BTModel(L=2.*pi,nx=128, tmax = 200*year,
        beta = 0., H = 1., rek = 0., rd = None, dt = 0.005,
                     taveint=year, use_fftw=True, ntd=1)

# McWilliams 84 IC condition
fk = m.wv != 0
ckappa = np.zeros_like(m.wv2)
ckappa[fk] = np.sqrt( m.wv2[fk]*(1. + (m.wv2[fk]/36.)**2) )**-1

nhx,nhy = m.wv2.shape

Pi_hat = np.random.randn(nhx,nhy)*ckappa +1j*np.random.randn(nhx,nhy)*ckappa

Pi = m.ifft2( Pi_hat )
Pi = Pi - Pi.mean()
Pi_hat = m.fft2( Pi )
KEaux = bt_model.spec_var( m, m.filtr*m.wv*Pi_hat )

pih = ( Pi_hat/np.sqrt(KEaux) )
qih = -m.wv2*pih
qi = m.ifft2(qih)
m.set_q(qi,check=True)

# run the model and plot some figs
plt.rcParams['image.cmap'] = 'RdBu'

plt.ion()

for snapshot in m.run_with_snapshots(tsnapstart=0, tsnapint=15*m.dt):

    plt.clf()
    p1 = plt.imshow(m.q + m.beta * m.y)
    plt.clim([-30., 30.])
    plt.title('t='+str(m.t))
    
    plt.xticks([])
    plt.yticks([])

    plt.pause(0.01)

    plt.draw()

plt.show()
plt.ion()
