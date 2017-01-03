import numpy as np
from matplotlib import pyplot as plt
import pyqg
from pyqg.diagnostic_tools import spec_var

a = 0.25
c=20.
nx = 256.
b=nx/2.
U1 = 0.0 + a * np.exp(-((np.arange(nx,dtype=float)-b)**2)/(2*c**2))
U2 = U1 / 2.0
m = pyqg.QGModel(tavestart=0,  dt=500, U1=U1,U2=U2, nx=nx)

fk = m.wv != 0
ckappa = np.zeros_like(m.wv2)
ckappa[fk] = np.sqrt( m.wv2[fk]*(1. + (m.wv2[fk]/36.)**2) )**-1

nhx,nhy = m.wv2.shape

Pi_hat = np.random.randn(nhx,nhy)*ckappa +1j*np.random.randn(nhx,nhy)*ckappa

Pi = m.ifft( Pi_hat[np.newaxis] )
Pi = Pi - Pi.mean()
Pi_hat = m.fft( Pi )
KEaux = spec_var( m, m.filtr*m.wv*Pi_hat )

pih = ( Pi_hat/np.sqrt(KEaux[:,np.newaxis,np.newaxis]) )
qih = -m.wv2*pih
qi = m.ifft(qih)
m.set_q(qi)
i=0
plt.figure(figsize=[12,10])
clevels = np.arange(-0.0005, 0.00075, 0.000025)
levels = np.arange(-0.0005,0.0008,0.0001)
for snapshot in m.run_with_snapshots(
        tsnapstart=0, tsnapint=250*m.dt):
    plt.clf()
    # plt.imshow(m.ufull[0,:])
    Q1 = np.expand_dims(m.Qy1 - np.gradient(np.gradient(m.U1, m.dy), m.dy), axis=1) * m.y + np.gradient(m.U1, m.dy)
    # plt.imshow(m.q[0] + Q1)
    f = plt.contourf(Q1 + m.q[0,:],levels=clevels, extend='both')
    plt.contour(Q1 + m.q[0,:],levels=levels, extend='both', colors='#444444')
    # f = plt.contourf(m.q[0,:],levels=clevels, extend='both')
    # plt.contour(m.q[0,:],levels=levels, extend='both', colors='#444444')
    # plt.imshow(m.q[0,:])
    # plt.imshow(m.ufull[0,:])
    # plt.imshow(m.q[0,:])
    # plt.clim([-0.00175,0.00175])
    plt.clim([-0.0005,0.00075])
    plt.colorbar(f)
    plt.pause(0.01)
    plt.draw()
    plt.savefig('Q_full_with_jet_' + ('%04d' % i) + '.png', bbox_inches='tight')
    plt.clf()
    f = plt.contourf(m.q[0,:],levels=clevels, extend='both')
    plt.contour(m.q[0,:],levels=levels, extend='both', colors='#444444')
    plt.clim([-0.0005,0.00075])
    plt.colorbar(f)
    plt.pause(0.01)
    plt.draw()
    plt.savefig('Q_with_jet_' + ('%04d' % i) + '.png', bbox_inches='tight')
    i += 1

# now the model is done
