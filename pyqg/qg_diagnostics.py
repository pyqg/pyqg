import numpy as np
from numpy import pi
from . import model

class QGDiagnostics(model.Model):
    def _initialize_model_diagnostics(self):
        self.add_diagnostic('entspec',
                description='barotropic enstrophy spectrum',
                function= (lambda self:
                    np.abs((self.Hi[:,np.newaxis,np.newaxis]*self.qh).sum(axis=0)/self.H)**2/self.M**2),
                units='m s^-2',
                dims=('l','k')
        )

        self.add_diagnostic('paramspec_APEflux',
            description='total additional APE flux due to subgrid parameterization',
            function=(lambda self: 
                -self._calc_paramspec_contribution(np.einsum("ij, jk..., k... -> i...", 
                        self.S, self.a, self._calc_parameterization_contribution()))
                ),
            units='m^2 s^-3',
            dims=('l','k')
        )

        self.add_diagnostic('paramspec_KEflux',
            description='total additional KE flux due to subgrid parameterization',
            function=(lambda self: 
                self.wv2*self._calc_paramspec_contribution(np.einsum("ij..., j... -> i...", 
                        self.a, self._calc_parameterization_contribution()))
                ),
            units='m^2 s^-3',
            dims=('l','k')
        )

        self.add_diagnostic('ENSflux',
                 description='barotropic enstrophy flux',
                 function = (lambda self: (-self.Hi[:,np.newaxis,np.newaxis]*
                              (self.qh.conj()*self.Jq).real).sum(axis=0)/self.H/self.M**2),
                units='s^-3',
                dims=('l','k')
        )

        self.add_diagnostic('ENSgenspec',
                    description='the spectrum of the rate of generation of barotropic enstrophy',
                    function = (lambda self:
                                (self.Hi[:,np.newaxis,np.newaxis]*((self.ilQx-self.ikQy)*
                                self.Sph.conj()*self.ph).real).sum(axis=0)/self.H/self.M**2),
                units='s^-3',
                dims=('l','k')
        )

        self.add_diagnostic('ENSfrictionspec',
                    description='the spectrum of the rate of dissipation of barotropic enstrophy due to bottom friction',
                    function = (lambda self: self.rek*self.Hi[-1]/self.H*self.wv2*
                        (self.qh[-1].conj()*self.ph[-1]).real/self.M**2), 
                units='s^-3',
                dims=('l','k')
        )

        self.add_diagnostic('APEgenspec',
                    description='the spectrum of the rate of generation of available potential energy',
                    function =(lambda self: (self.Hi[:,np.newaxis,np.newaxis]*
                                (self.Ubg[:,np.newaxis,np.newaxis]*self.k +
                                 self.Vbg[:,np.newaxis,np.newaxis]*self.l)*
                                (1j*self.ph.conj()*self.Sph).real).sum(axis=0)/self.H/self.M**2),
                units='m^2 s^-3',
                dims=('l','k')
        )

    def _calc_paramspec_contribution(self, term):
        height_ratios = (self.Hi/self.H)[:,np.newaxis,np.newaxis]
        return np.real(height_ratios*self.ph.conj()*term).sum(axis=0) / self.M**2

    def _calc_cfl(self):
        return np.abs(
            np.hstack([self.u + self.Ubg[:,np.newaxis,np.newaxis], self.v])
        ).max()*self.dt/self.dx

    # calculate KE: this has units of m^2 s^{-2}
    #   (should also multiply by H1 and H2...)
    def _calc_ke(self):
        ke = 0.
        for j in range(self.nz):
            ke += .5*self.Hi[j]*self.spec_var(self.wv*self.ph[j])
        return ke.sum() / self.H

    # calculate eddy turn over time
    # (perhaps should change to fraction of year...)
    def _calc_eddy_time(self):
        """ estimate the eddy turn-over time in days """
        ens = 0.
        for j in range(self.nz):
            ens = .5*self.Hi[j] * self.spec_var(self.wv2*self.ph[j])

        return 2.*pi*np.sqrt( self.H / ens.sum() ) / 86400
