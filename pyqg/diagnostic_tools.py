import numpy as np
from numpy import pi

def spec_var(self,ph):
    """ compute variance of p from Fourier coefficients ph """
    var_dens = 2. * np.abs(ph)**2 / self.M**2
    # only half of coefs [0] and [nx/2+1] due to symmetry in real fft2
    var_dens[:,0],var_dens[:,-1] = var_dens[:,0]/2.,var_dens[:,-1]/2.
    return var_dens.sum()

def calc_ispec(self, Spec):
    """ calculates isotropic spectrum from 2D spectrum """
    if self.kk.max()>self.ll.max():
        kmax = self.ll.max()
    else:
        kmax = self.kk.max()

    # create radial wavenumber
    dkr = np.sqrt(self.dk**2 + self.dl**2)
    kr =  np.arange(dkr/2.,kmax+dkr,dkr)
    ispec = np.zeros(kr.size)

    for i in range(kr.size):
        fkr =  (self.wv>=kr[i]-dkr/2) & (self.wv<=kr[i]+dkr/2)
        dth = pi / (fkr.sum()-1)
        ispec[i] = Spec[fkr].sum() * kr[i] * dth
        
    return kr, ispec
