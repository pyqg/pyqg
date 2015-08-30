import numpy as np
import model
from numpy import pi


class SQGModel(model.Model):
    """Surface quasigeostrophic model."""
    
    def __init__(
        self,
        beta=0.,                    # gradient of coriolis parameter
        Nb = 1.,                    # Buoyancy frequency
        rd=0.,                      # deformation radius
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
            Background zonal flow. Units: meters.
        """

        # physical
        self.beta = beta
        self.Nb = Nb
        #self.rek = rek
        self.rd = rd
        self.H = H
        self.U = U
        #self.filterfac = filterfac
        
        self.nz = 1
       
        super(SQGModel, self).__init__(**kwargs)
     
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
        # The sqg model is diagonal. The inversion is simply qh = -kappa**2 ph
        self.a = np.asarray(self.Nb*np.sqrt(self.wv2i))[np.newaxis, np.newaxis, :, :]

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
