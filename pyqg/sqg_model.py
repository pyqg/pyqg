import numpy as np
from . import model


class SQGModel(model.Model):
    r"""Surface  quasigeostrophic model.

    The surface quasigeostrophic evolution equations is

    .. math::

       \partial_t b + J(\psi, b) = \text{ssd}

    The buoyancy anomaly is in physical and Fourier space

    .. math::

       b = \partial_z \psi

       bh = -K*Nb/f_0*ph

    """

    def __init__(self, Nb=1., f_0=1., U=0., **kwargs):
        """
        Parameters
        ----------

        Nb : number, optional
            Buoyancy frequency. Units: seconds :sup:`-1`.
        f_0 : number, optional
            Coriolis frequency. Units: seconds :sup:`-1`.
        U : number, optional
            Upper layer flow. Units: meters seconds :sup:`-1`.
        """

        self.Nb = Nb
        self.f_0 = f_0
        self.U = U

        self.nz = 1

        self.SQG = 1

        super().__init__(**kwargs)

        # initial conditions: (buoyancy anomalies)
        # FJP: what amplitude do we pick here?
        # FJP: do we specify a background U?  We should
        self.set_b(1e-3*np.random.rand(1,self.ny,self.nx))

    ### PRIVATE METHODS - not meant to be called by user ###

    def _initialize_background(self):
        """Set up background state (zonal flow and buoyancy gradient)."""

        # the meridional buoyancy gradient
        self.By = np.asarray(self.Nb)[np.newaxis, ...]

        # background vel.
        self.set_U(self.U)

        # complex versions, multiplied by k, speeds up computations to pre-compute
        self.ikBy = self.By * 1j * self.k

        self.ilBx = 0.

    def _initialize_inversion_matrix(self):
        """ the inversion """
        # The sqg inversion is ph = f / (N * kappa) qh (see documentation) 
        # FJP: need to change qh to bh
        self.a = np.asarray(self.f_0/self.Nb*np.sqrt(self.wv2i))[np.newaxis, np.newaxis, :, :]

    def _initialize_forcing(self):
        pass

    def set_U(self, U):
        """Set background zonal flow.

        Parameters
        ----------

        U : number
            Flow. Units: meters seconds :sup:`-1`.
        """
        """Set background zonal flow"""
        self.Ubg = np.asarray(U)[np.newaxis,...]

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
    #def _calc_eddy_time(self):
    #    """ estimate the eddy turn-over time in days """
    #    ens = .5*self.H * spec_var(self, self.wv2*self.ph)
    #    return 2.*np.pi*np.sqrt( self.H / ens ) / year
    # FJP: what is the analogue of this for SQG?  Should not need to know H.