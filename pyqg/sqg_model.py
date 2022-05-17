import numpy as np
from numpy import pi
from . import model


class SQGModel(model.Model):
    """Surface quasigeostrophic model."""

    def __init__(
        self,
        beta=0.,                    # gradient of coriolis parameter
        Nb = 1.,                    # Buoyancy frequency
        f_0 = 1.,                   # coriolis parameter
        H = 1.,                     # depth of layer
        U=0.,                       # max vel. of base-state
        **kwargs
        ):
        """
        Parameters
        ----------

        beta : number
            Gradient of coriolis parameter. Units: meters :sup:`-1`
            seconds :sup:`-1`
        Nb : number
            Buoyancy frequency. Units: seconds :sup:`-1`.
        U : number
            Background zonal flow. Units: meters seconds :sup:`-1`.
        """

        # physical
        self.beta = beta
        self.Nb = Nb
        self.f_0 = f_0
        self.H = H
        self.Hi = np.array(H)[np.newaxis,...]
        self.U = U
        #self.filterfac = filterfac

        super().__init__(nz=1, **kwargs)

        # initial conditions: (PV anomalies)
        self.set_q(1e-3*np.random.rand(1,self.ny,self.nx))

    ### PRIVATE METHODS - not meant to be called by user ###

    def _initialize_background(self):
        """Set up background state (zonal flow and PV gradients)."""

        # the meridional PV gradients in each layer
        self.Qy = np.asarray(self.beta)[np.newaxis, ...]

        # background vel.
        self.set_U(self.U)

        # complex versions, multiplied by k, speeds up computations to pre-compute
        self.ikQy = self.Qy * 1j * self.k

        self.ilQx = 0.

    def _initialize_inversion_matrix(self):
        """ the inversion """
        # The sqg inversion is ph = f / (kappa * N) qh (see documentation) 
        self.a = np.asarray(self.f_0/self.Nb*np.sqrt(self.wv2i))[np.newaxis, np.newaxis, :, :]

    def _initialize_forcing(self):
        pass

    def set_U(self, U):
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
    def _calc_eddy_time(self):
        """ estimate the eddy turn-over time in days """
        ens = .5*self.H * spec_var(self, self.wv2*self.ph)
        return 2.*pi*np.sqrt( self.H / ens ) / year
