import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

from pyqg import qg2d as qg_model

m = qg_model.BTModel(
        tmax=20, tavestart=100, twrite=10,
        nx = 256, L = 2*np.pi, beta = 0.,
        rek = .00,
        dt = 0.0025*.25,
        fftw = False,
        ntd = 3)

kx,ky = m.kk,m.ll
kappa2 = m.wv2
kappa = m.wv
fk = kappa2 != 0
ckappa = np.zeros_like(kappa2)
ckappa[fk] = 1./np.sqrt( ( kappa2[fk]*(1. + (kappa2[fk]/36.)**2) ))
Pi_hat = np.random.randn(kappa2.size).reshape(kappa2.shape)*ckappa +\
        1.j*np.random.randn(kappa2.size).reshape(kappa2.shape)*ckappa

Pi = qg_model.ifft2(m, Pi_hat )
Pi = Pi - Pi.mean()
Pi_hat = qg_model.fft2(m , Pi )
KEaux = qg_model.spec_var( m, kappa*Pi_hat )

pih = ( Pi_hat/np.sqrt(KEaux) )
qih = -m.wv2*pih
qi = qg_model.ifft2(m,qih)
m.set_q(qi)
ens = .5* qg_model.spec_var(m, m.wv2*pih)

# initial eddy turnover time
Te_i = 2.*np.pi/np.sqrt(ens)
tmax = Te_i

n1=dt.datetime.now()
m.run()
n2=dt.datetime.now()

#
#print "  "
#print " Elapsed time %i sec" %((n2-n1).seconds)


