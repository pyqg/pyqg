"""Utility functions for pyqg model data."""

import numpy as np
from numpy import pi

def spec_var(model, ph):
    """Compute variance of ``p`` from Fourier coefficients ``ph``.
    
    Parameters
    ----------
    model : pyqg.Model instance
        The model object from which `ph` originates
    ph : complex array
        The field on which to compute the variance
        
    Returns
    -------
    var_dens : float
        The variance of `ph`
    """

    var_dens = 2. * np.abs(ph)**2 / model.M**2
    # only half of coefs [0] and [nx/2+1] due to symmetry in real fft2
    var_dens[...,0] /= 2
    var_dens[...,-1] /= 2
    return var_dens.sum(axis=(-1,-2))
    
    
def calc_ispec(model, ph):
    """Compute isotropic spectrum `phr` of `ph` from 2D spectrum.
    
    Parameters
    ----------
    model : pyqg.Model instance
        The model object from which `ph` originates
    ph : complex array
        The field on which to compute the variance
        
    Returns
    -------
    kr : array
        isotropic wavenumber
    phr : array
        isotropic spectrum
    """
    
    if model.kk.max()>model.ll.max():
        kmax = model.ll.max()
    else:
        kmax = model.kk.max()

    # create radial wavenumber
    dkr = np.sqrt(model.dk**2 + model.dl**2)
    kr =  np.arange(dkr/2.,kmax+dkr,dkr)
    phr = np.zeros(kr.size)

    for i in range(kr.size):
        fkr =  (model.wv>=kr[i]-dkr/2) & (model.wv<=kr[i]+dkr/2)
        dth = pi / (fkr.sum()-1)
        phr[i] = ph[fkr].sum() * kr[i] * dth
        
    return kr, phr
