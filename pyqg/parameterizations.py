import numpy as np

class Parameterization:
    pass

class ZannaBolton2020(Parameterization):
    def __init__(self, constant=-46761284):
        self.constant = constant

    def __call__(self, m):
        # Compute ZB2020 basis functions
        vx = m.ifft(m.vh * m.ik)
        vy = m.ifft(m.vh * m.il)
        ux = m.ifft(m.uh * m.ik)
        uy = m.ifft(m.uh * m.il)
        rel_vort = vx - uy
        shearing = vx + uy
        stretching = ux - vy
        # Combine them in real space and take their FFT
        rv_stretch = m.fft(rel_vort * stretching)
        rv_shear = m.fft(rel_vort * shearing)
        sum_sqs = m.fft(rel_vort**2 + shearing**2 + stretching**2) / 2.0
        # Take spectral-space derivatives and multiply by the scaling factor
        kappa = self.constant
        du = kappa * m.ifft(m.ik*(sum_sqs - rv_shear) + m.il*rv_stretch)
        dv = kappa * m.ifft(m.il*(sum_sqs + rv_shear) + m.ik*rv_stretch)
        return du, dv

class Smagorinsky(Parameterization):
    def __init__(self, constant=0.1):
        self.constant = constant

    def __call__(self, m, just_viscosity=False):
        Sxx = m.ifft(m.uh * m.ik)
        Syy = m.ifft(m.vh * m.il)
        Sxy = 0.5 * m.ifft(m.uh * m.il + m.vh * m.ik)
        nu = (self.constant * m.dx)**2 * np.sqrt(2 * (Sxx**2 + Syy**2 + 2 * Sxy**2))
        if just_viscosity:
            return nu
        du = 2 * (m.ifft(nu * Sxx * m.ik) + m.ifft(nu * Sxy * m.il))
        dv = 2 * (m.ifft(nu * Sxy * m.ik) + m.ifft(nu * Syy * m.il))
        return du, dv

class BackscatterBiharmonic(Parameterization):
    def __init__(self, smag_constant=0.1, back_constant=0.9, eps=1e-32):
        self.smagorinsky = Smagorinsky(smag_constant)
        self.back_constant = back_constant
        self.eps = eps

    def __call__(self, m):
        lap = m.ik**2 + m.il**2
        psi = m.ifft(m.ph)
        lap_lap_psi = m.ifft(lap**2 * m.ph)
        dissipation = -m.ifft(lap * m.fft(lap_lap_psi * self.smagorinsky(m, just_viscosity=True)))
        backscatter = -self.back_constant * lap_lap_psi * (
            (np.sum(m.Hi * np.mean(psi * dissipation, axis=(-1,-2)))) /
            (np.sum(m.Hi * np.mean(psi * lap_lap_psi, axis=(-1,-2))) + self.eps)) 
        dq = dissipation + backscatter
        return dq
