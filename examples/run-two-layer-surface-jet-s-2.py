import numpy as np
from matplotlib import pyplot as plt
import pyqg
from pyqg.diagnostic_tools import spec_var
import subprocess

a = 0.25
c=10.
nx = 256.
b=nx/2.
jet = a * np.exp(-((np.arange(nx,dtype=float)-b)**2)/(2*c**2))
U1 = 0.005 + jet
U2 = 0.000 + jet*0.99
m = pyqg.QGModel(tavestart=0, dt=500, U1=U1,U2=U2, nx=nx, tmax=500*500*1000, ntd=4, beta=-1.45e-11)
Q1 = np.expand_dims(m.beta + m.F1*(m.U1 - m.U2), axis=1) * m.y + np.expand_dims(np.gradient(m.U1, m.dy),axis=1) * np.ones((nx))

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
plt.figure(figsize=[14,8])
for snapshot in m.run_with_snapshots(
        tsnapstart=0, tsnapint=250*m.dt):
    clevels = np.arange(0.0000, 0.000005, 0.0000005)
    levels = np.arange(0.000,0.000005,0.000001)
    plt.clf()
    plt.subplot(1,2,1)
    f = plt.contourf(Q1,levels=clevels, extend='both')
    plt.contour(Q1,levels=levels, extend='both', colors='#444444')
    plt.clim([-0.0005,0.00125])
    # plt.colorbar(f)
    # plt.pause(0.01)
    # plt.draw()
    # plt.savefig('output/Q_full_with_surf_jet_s2_' + ('%04d' % i) + '.png', bbox_inches='tight')

    plt.subplot(1,2,2)
    clevels = np.arange(-0.0005, 0.00125, 0.000025)
    levels = np.arange(-0.0005,0.0013,0.0001)
    f = plt.contourf(m.q[0,:],levels=clevels, extend='both')
    plt.contour(m.q[0,:],levels=levels, extend='both', colors='#444444')
    plt.clim([-0.0005,0.00125])
    # plt.colorbar(f)
    plt.pause(0.01)
    plt.draw()
    plt.savefig('output/Q_with_surf_jet_s2_' + ('%04d' % i) + '.png', bbox_inches='tight')
    i += 1
subprocess.call("ffmpeg -y -f image2 -i Q_full_with_surf_jet_s_%04d.png -f mp4 -vcodec libx264 -r 10 -pix_fmt yuv420p -vf scale=978:844 output/q_full_with_surf_jet_s2_25.mp4",shell=True)
subprocess.call("ffmpeg -y -f image2 -i output/Q_with_surf_jet_s_%04d.png -f mp4 -vcodec libx264 -r 10 -pix_fmt yuv420p -vf scale=978:844 output/q_with_surf_jet_s2_25.mp4",shell=True)