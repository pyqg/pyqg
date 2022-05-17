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

def diagnostic_differences(m1, m2, reduction='rmse', instantaneous=False):
    """Compute a dictionary of differences in the diagnostics of two models at
    possibly different resolutions (e.g. for quantifying the effects of
    parameterizations). Applies normalization/isotropization to certain
    diagnostics before comparing them and skips others. Also computes
    differences for each vertical layer separately.

    Parameters
    ----------
    m1 : pyqg.Model instance
        The first model to compare
    m2 : pyqg.Model instance
        The second model to compare
    reduction : string or function
        A function that takes two arrays of diagnostics and computes a distance
        metric. Defaults to the root mean squared difference ('rmse').
    instantaneous : boolean
        If true, compute difference metrics for the instantaneous values of a
        diagnostic, rather than its time average. Defaults to false.

    Returns
    -------
    diffs : dict
        A dictionary of diagnostic name => distance. If the diagnostic is
        defined over multiple layers, separate keys are included with an
        appended z index.
    """

    diffs = {}

    # Compute the minimum common wavenumber in case we're comparing two
    # models with different resolutions

    kr1, _ = calc_ispec(m1, m1.diagnostics['KEspec']['function'](m1)[0])
    kr2, _ = calc_ispec(m2, m2.diagnostics['KEspec']['function'](m2)[0])
    min_kr_length = min(len(kr1), len(kr2))

    # Helper to get a normalized version of diagnostics
    def get_normalized_diagnostic(model, diag_name, layer=None):
        # Get the raw diagnostic
        attrs = model.diagnostics[diag_name]
        if instantaneous:
            diag = attrs['function'](model)
        else:
            diag = model.get_diagnostic(diag_name)

        # Check if we need to add other terms to this diagnostic (e.g.
        # KEflux + paramspec_KEflux)
        for diag_name2 in attrs.get('sums_with', []):
            if instantaneous:
                diag += model.diagnostics[diag_name2]['function'](model)
            else:
                diag += model.get_diagnostic(diag_name2)

        # Potentially limit to a layer
        if layer is not None:
            diag = diag[layer]

        # Potentially convert to isotropic spectrum, keeping only the
        # wavenumbers common to both models
        if attrs['dims'][-2:] == ('l','k'):
            kr, diag = calc_ispec(model, diag)
            diag = diag[:min_kr_length]

        # Return the normalized diagnostic
        return diag

    # Loop through all diagnostics
    for diag_name, attrs in m1.diagnostics.items():
        # Skip diagnostics flagged as not for comparison (TODO: diagnostics
        # should be objects and this should be a method, rather than a
        # dictionary key)
        if attrs.get('skip_comparison', False):
            continue

        # Skip diagnostics not present in the second model (usually not
        # necessary)
        if diag_name not in m2.diagnostics:
            continue

        # If we have multiple layers in this diagnostic, we want to consider
        # them separately with different keys
        if attrs['dims'][0] == 'lev':
            layers = range(m1.nz)
        elif attrs['dims'][0] == 'lev_mid':
            layers = range(m1.nz - 1)
        else:
            layers = [None]

        for layer in layers:
            diag1 = get_normalized_diagnostic(m1, diag_name, layer)
            diag2 = get_normalized_diagnostic(m2, diag_name, layer)
            label = f"{diag_name}{'' if layer is None else layer+1}"
            # Compute the error
            if reduction == 'rmse':
                diff = np.sqrt(np.mean((diag1-diag2)**2))
            else:
                diff = reduction(diag1, diag2)
            diffs[label] = diff

    return diffs

def diagnostic_similarities(model, target, baseline, **kw):
    """Like `diagnostic_differences`, but returning a dictionary of similarity
    scores between negative infinity and 1 which quantify how much closer the
    diagnostics of a given `model` are to a `target` with respect to a
    `baseline`. Scores approach 1 when the distance between the model and the
    target is small compared to the baseline and are negative when that
    distance is greater.

    Parameters
    ----------
    model : pyqg.Model instance
        The model for which we want to compute similiarity scores (e.g. a
        parameterized low resolution model)
    target : pyqg.Model instance
        The target model (e.g. a high resolution model)
    baseline : pyqg.Model instance
        The baseline against which we check for improvement or degradation
        (e.g. an unparameterized low resolution model)

    Returns
    -------
    sims : dict
        A dictionary of diagnostic name => similarity. If the diagnostic is
        defined over multiple layers, separate keys are included with an
        appended z index.
    """
    d1 = diagnostic_differences(model, target, **kw)
    d2 = diagnostic_differences(baseline, target, **kw)
    sims = dict((k, 1-d1[k]/d2[k]) for k in d1.keys())
    return sims
