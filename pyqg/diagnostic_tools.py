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


def spec_sum(ph2):
    """Compute total spectral sum of the real spectral quantity``ph^2``.

    Parameters
    ----------
    model : pyqg.Model instance
        The model object from which `ph` originates
    ph2 : real array
        The field on which to compute the sum

    Returns
    -------
    var_dens : float
        The sum of `ph2`
    """

    ph2 = 2.*ph2
    ph2[...,0] = ph2[...,0]/2.
    ph2[...,-1] = ph2[...,-1]/2.

    return ph2.sum(axis=(-1,-2))

def calc_ispec(model, _var_dens, averaging = True, truncate=True, nd_wavenumber=False, nfactor = 1):
    """Compute isotropic spectrum `phr` from 2D spectrum of variable `signal2d`
    such that `signal2d.var() = phr.sum() * (kr[1] - kr[0])`.

    Parameters
    ----------
    model : pyqg.Model instance
        The model object from which `var_dens` originates
    
    var_dens : squared modulus of fourier coefficients like this:
        `np.abs(signal2d_fft)**2/m.M**2`

    averaging: If True, spectral density is estimated with averaging over circles,
        otherwise summation is used and Parseval identity holds

    truncate: If True, maximum wavenumber corresponds to inner circle in Fourier space,
        otherwise - outer circle
    
    nd_wavenumber: If True, wavenumber is nondimensional: 
        minimum wavenumber is 1 and corresponds to domain length/width,
        otherwise - wavenumber is dimensional [m^-1]

    nfactor: width of the bin in sqrt(dk^2+dl^2) units

    Returns
    -------
    kr : array
        isotropic wavenumber
    phr : array
        isotropic spectrum
    """

    # account for complex conjugate
    var_dens = np.copy(_var_dens)
    var_dens[...,0] /= 2
    var_dens[...,-1] /= 2

    ll_max = np.abs(model.ll).max()
    kk_max = np.abs(model.kk).max()

    if truncate:
        kmax = np.minimum(ll_max, kk_max)
    else:
        kmax = np.sqrt(ll_max**2 + kk_max**2)
    
    kmin = 0

    dkr = np.sqrt(model.dk**2 + model.dl**2) * nfactor

    # left border of bins
    kr = np.arange(kmin, kmax, dkr)
    
    phr = np.zeros(kr.size)

    for i in range(kr.size):
        if i == kr.size-1:
            fkr = (model.wv>=kr[i]) & (model.wv<=kr[i]+dkr)
        else:
            fkr = (model.wv>=kr[i]) & (model.wv<kr[i+1])
        if averaging:
            phr[i] = var_dens[fkr].mean() * (kr[i]+dkr/2) * pi / (model.dk * model.dl)
        else:
            phr[i] = var_dens[fkr].sum() / dkr

        phr[i] *= 2 # include full circle
    
    # convert left border of the bin to center
    kr = kr + dkr/2

    # convert to non-dimensional wavenumber 
    # preserving integral over spectrum
    if nd_wavenumber:
        kr = kr / kmin
        phr = phr * kmin

    return kr, phr
