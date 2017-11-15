from __future__ import print_function
import numpy as np
from numpy import pi
from . import model


class BTModel(model.Model):
    r"""Single-layer (barotropic) quasigeostrophic model.
    This class can represent both pure two-dimensional flow
    and also single reduced-gravity layers with deformation
    radius ``rd``.

    The equivalent-barotropic quasigeostrophic evolution equations is

    .. math::

       \partial_t q + J(\psi, q ) + \beta \psi_x = \text{ssd}

    The potential vorticity anomaly is

    .. math::

       q = \nabla^2 \psi - \kappa_d^2 \psi

    """

    def __init__(self, beta=0.,  rd=0., H=1., U=0., **kwargs):
        """
        Parameters
        ----------

        beta : number, optional
            Gradient of coriolis parameter. Units: meters :sup:`-1`
            seconds :sup:`-1`
        rd : number, optional
            Deformation radius. Units: meters.
        U : number *or* array-like, optional
            Upper layer flow. Units: meters.
        """

        self.beta = beta
        self.rd = rd
        self.H = H
        self.Hi = np.array(H)[np.newaxis,...]
        self.U = U

        self.nz = 1

        # deformation wavenumber
        if rd:
            self.kd2 = rd**-2
        else:
            self.kd2 = 0.

        super(BTModel, self).__init__(**kwargs)

        # initial conditions: (PV anomalies)
        self.set_q(1e-3*np.random.rand(1,self.ny,self.nx))

    def _initialize_background(self):
        """Set up background state (zonal flow and PV gradients)."""

        if len(np.shape(self.U)) == 0:
          self.U = self.U * np.ones((self.ny)) 
        # the meridional PV gradients in each layer
        self.Qy = (self.beta + np.gradient(np.gradient(self.U, self.dy), self.dy))[np.newaxis,...]

        # background vel.
        self.set_U(self.U)
        # complex versions, multiplied by k, speeds up computations to pre-compute
        self.ikQy = np.expand_dims(self.Qy, axis=2) *  1j * self.k

        self.ilQx = 0.

    def _initialize_inversion_matrix(self):
        """ the inversion """
        # The bt model is diagonal. The inversion is simply qh = -kappa**2 ph
        self.a = -(self.wv2i+self.kd2)[np.newaxis, np.newaxis, :, :]

    def _initialize_forcing(self):
        pass

    # def _initialize_forcing(self):
    #     """Set up frictional filter."""
    #     # this defines the spectral filter (following Arbic and Flierl, 2003)
    #     cphi=0.65*pi
    #     wvx=np.sqrt((self.k*self.dx)**2.+(self.l*self.dy)**2.)
    #     self.filtr = np.exp(-self.filterfac*(wvx-cphi)**4.)
    #     self.filtr[wvx<=cphi] = 1.
    #
    # def _filter(self, q):
    #     return self.filtr * q

    def set_U(self, U):
        """Set background zonal flow.

        Parameters
        ----------

        U : number *or* array-like
            Upper layer flow. Units meters.
        """
        self.Ubg = np.asarray(U)[np.newaxis, ...]

    def _calc_diagnostics(self):
        # here is where we calculate diagnostics
        if (self.t>=self.dt) and (self.tc%self.taveints==0):
            self._increment_diagnostics()

    ### All the diagnostic stuff follows. ###
    def _calc_cfl(self):
        return np.abs(
            np.hstack([self.u + self.Ubg, self.v])
        ).max()*self.dt/self.dx

    # calculate KE: this has units of m^2 s^{-2}
    def _calc_ke(self):
        ke = .5*self.spec_var(self.wv*self.ph)
        return ke.sum()

    # calculate eddy turn over time
    # (perhaps should change to fraction of year...)
    def _calc_eddy_time(self):
        """ estimate the eddy turn-over time in days """
        ens = .5*self.H * self.spec_var(self.wv2*self.ph)
        return 2.*pi*np.sqrt( self.H / ens ) / year

    # def _calc_derived_fields(self):
    #     self.p = self.ifft2( self.ph)
    #     self.xi =self.ifft2( -self.wv2*self.ph)
    #     self.Jptpc = -self.advect(self.p,self.u,self.v)
    #     self.Jpxi = self.advect(self.xi, self.u, self.v)
