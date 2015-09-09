import numpy as np
import model
from numpy import pi
try:   
    import mkl
    np.use_fastnumpy = True
except ImportError:
    pass

try:
    import pyfftw
    pyfftw.interfaces.cache.enable() 
except ImportError:
    pass

class QGModel(model.Model):
    r"""N-layer quasigeostrophic model.
    
    This model is meant to representflows driven by baroclinic instabilty of a
    base-state shear :math:`U_1-U_2`. The upper and lower
    layer potential vorticity anomalies :math:`q_1` and :math:`q_2` are
    
    .. math::
    
        q_1 &= \nabla^2\psi_1 + F_1(\psi_2 - \psi_1) \\
        q_2 &= \nabla^2\psi_2 + F_2(\psi_1 - \psi_2)

    with
    
    .. math::
        F_1 &\equiv \frac{k_d^2}{1 + \delta^2} \\
        F_2 &\equiv \delta F_1 \ .

    The layer depth ratio is given by :math:`\delta = H_1 / H_2`.
    The total depth is :math:`H = H_1 + H_2`.

    The background potential vorticity gradients are

    .. math::

        \beta_1 &= \beta + F_1(U_1 - U_2) \\
        \beta_2 &= \beta - F_2( U_1 - U_2) \ .
    
    The evolution equations for :math:`q_1` and :math:`q_2` are

    .. math::

        \partial_t\,{q_1} + J(\psi_1\,, q_1) + \beta_1\,
        {\psi_1}_x &= \text{ssd} \\
        \partial_t\,{q_2} + J(\psi_2\,, q_2)+ \beta_2\, {\psi_2}_x
        &= -r_{ek}\nabla^2 \psi_2 + \text{ssd}\,.

    where `ssd` represents small-scale dissipation and :math:`r_{ek}` is the
    Ekman friction parameter.
    
    """
    
    def __init__(
        self,
        beta=1.5e-11,               # gradient of coriolis parameter
        #rek=5.787e-7,               # linear drag in lower layer
        rd=15000.0,                 # deformation radius
        delta=0.25,                 # layer thickness ratio (H1/H2)
        H1 = 500,                   # depth of layer 1 (H1)
        U1=0.025,                   # upper layer flow
        U2=0.0,                     # lower layer flow
        nz = 4,
        **kwargs
        ):
        """
        Parameters
        ----------

        beta : number
            Gradient of coriolis parameter. Units: meters :sup:`-1`
            seconds :sup:`-1`
        rek : number
            Linear drag in lower layer. Units: seconds :sup:`-1`
        rd : number
            Deformation radius. Units: meters.
        delta : number
            Layer thickness ratio (H1/H2)
        U1 : number
            Upper layer flow. Units: m/s
        U2 : number
            Lower layer flow. Units: m/s
        """

        # physical
        self.g = 9.81    # acceleration due to gravity
        self.f = 1.2e-4  # Coriolis frequency
        self.f2 = self.f**2
        self.beta = beta
        #self.rek = rek
        self.rd = rd
        self.delta = delta
        self.H1 = H1
        self.H2 = H1/delta
        self.U1 = U1
        self.U2 = U2
        #self.filterfac = filterfac
       
        # H is an array
        #self.Hi = np.array([100.,700.,2000.])
        #self.rhoi = np.array([1024.,1025.,1026.])
        #self.Ubg = np.array([0.05,0.025,0])
        #self.Vbg = np.array([0.05,0.025,0])
        self.Hi = np.array([500,2000.])
        self.rhoi = np.array([1025.,1025.83])
        self.Ubg = np.array([0.025,.0])
        self.Vbg = np.array([0.,0])
         
        self.gpi = self.g*(self.rhoi[1:]-self.rhoi[:-1])/self.rhoi[:-1]
        self.nz = nz

        assert self.Hi.size == self.nz, "size of Hi does not match number\
                of vertical levels nz" 

        assert self.rhoi.size == self.nz, "size of rhoi does not match number\
                of vertical levels nz" 

        # stretching matrix
        self._initialize_stretching_matrix()

        super(QGModel, self).__init__(**kwargs)
        
        # initial conditions: (PV anomalies)
        #self.set_q1q2(
        #    1e-7*np.random.rand(self.ny,self.nx) + 1e-6*(
        #        np.ones((self.ny,1)) * np.random.rand(1,self.nx) ),
        #        np.zeros_like(self.x) )   
                        
    ### PRIVATE METHODS - not meant to be called by user ###
       
    def _initialize_stretching_matrix(self):
        """ Set up the stretching matrix """
   
        self.S = np.zeros((self.nz, self.nz))
      
        for i in range(self.nz):
            
            if i == 0:
                self.S[i,i]   = -self.f2/self.Hi[i]/self.gpi[i] #- self.f2/self.Hi[i]/self.g
                self.S[i,i+1] =  self.f2/self.Hi[i]/self.gpi[i]

                self.S[i,i]   = -3.5555555555555554e-09 #- self.f2/self.Hi[i]/self.g
                self.S[i,i+1] =  3.5555555555555554e-09

            elif i == self.nz-1:
                self.S[i,i]   = -self.f2/self.Hi[i]/self.gpi[i-1] 
                self.S[i,i-1] =  self.f2/self.Hi[i]/self.gpi[i-1]
                
                self.S[i,i]   = -8.888888888888889e-10 
                self.S[i,i-1] =  8.888888888888889e-10

            else:
                self.S[i,i-1] = self.f2/self.Hi[i]/self.gpi[i-1]
                self.S[i,i]   = self.f2/self.Hi[i]/self.gpi[i] - self.f2/self.Hi[i]/self.gpi[i-1]
                self.S[i,i+1] = self.f2/self.Hi[i]/self.gpi[i]

    def _initialize_background(self):
        """Set up background state (zonal flow and PV gradients)."""
        
        # Background zonal flow (m/s):
        self.H = self.Hi.sum()

        # the meridional PV gradients in each layer
        self.Qy = self.beta - np.dot(self.S,self.Ubg)
        self.Qx = np.dot(self.S,self.Vbg)

        # complex versions, multiplied by k, speeds up computations to precompute 
        for i in range(self.nz): 
            if i == 0:
                self.ikQyi = self.Qy[i] * 1j * self.k
                self.ikQy = self.ikQyi[np.newaxis,...]
                self.ilQxi = self.Qx[i] * 1j * self.l
                self.ilQx = self.ilQxi[np.newaxis,...]
            else:
                self.ikQyi = self.Qy[i] * 1j * self.k
                self.ikQy = np.vstack([self.ikQy, self.ikQyi[np.newaxis,...]]) 
                self.ilQxi = self.Qx[i] * 1j * self.l
                self.ilQx = np.vstack([self.ilQx, self.ilQxi[np.newaxis,...]]) 
         
    def _initialize_inversion_matrix(self):
        
        a = np.ma.zeros((self.nz, self.nz, self.nl, self.nk), np.dtype('float64'))        

        for i in range(self.nl):
            for j in range(self.nk):
                if self.wv2[i,j]== 0:
                    a[:,:,i,j] = 0.
                else:
                    a[:,:,i,j] = np.linalg.inv(self.S - np.eye(self.nz)*self.wv2[i,j])

        self.a = np.ma.masked_invalid(a).filled(0.)
        
    def _initialize_forcing(self):
        pass
        #"""Set up frictional filter."""
        # this defines the spectral filter (following Arbic and Flierl, 2003)
        # cphi=0.65*pi
        # wvx=np.sqrt((self.k*self.dx)**2.+(self.l*self.dy)**2.)
        # self.filtr = np.exp(-self.filterfac*(wvx-cphi)**4.)
        # self.filtr[wvx<=cphi] = 1.
        
    def set_qi(self, qi, check=False):
        """Set PV anomalies.
        
        Parameters
        ----------
        
        qi : array-like
              layer PV anomaly in spatial coordinates.
        """
        self.set_q(q)
        
#    def set_U1U2(self, U1, U2):
#        """Set background zonal flow.
#        
#        Parameters
#        ----------
#        
#        U1 : number
#            Upper layer flow. Units: m/s
#        U2 : number
#            Lower layer flow. Units: m/s
#        """
#        self.U1 = U1
#        self.U2 = U2
#        #self.Ubg = np.array([U1,U2])[:,np.newaxis,np.newaxis]
#        #self.Ubg = np.array([U1,U2])
#        self.Ubg = self.U[:,np.newaxis,np.newaxis]

    def _calc_diagnostics(self):
        # here is where we calculate diagnostics
        if (self.t>=self.dt) and (self.tc%self.taveints==0):
            self._increment_diagnostics()

    ### All the diagnostic stuff follows. ###
    def _calc_cfl(self):
        return np.abs(
            np.hstack([self.u + self.Ubg[:,np.newaxis,np.newaxis], self.v])
        ).max()*self.dt/self.dx

    # calculate KE: this has units of m^2 s^{-2}
    #   (should also multiply by H1 and H2...)
    def _calc_ke(self):
        ke1 = .5*self.H1*self.spec_var(self.wv*self.ph[0])
        ke2 = .5*self.H2*self.spec_var(self.wv*self.ph[1]) 
        return ( ke1.sum() + ke2.sum() ) / self.H

    # calculate eddy turn over time 
    # (perhaps should change to fraction of year...)
    def _calc_eddy_time(self):
        """ estimate the eddy turn-over time in days """

        ens = .5*self.H1 * self.spec_var(self.wv2*self.ph1) + \
            .5*self.H2 * self.spec_var(self.wv2*self.ph2)

        return 2.*pi*np.sqrt( self.H / ens ) / 86400

#    def _calc_derived_fields(self):
#        self.p = self.ifft(self.ph)
#        self.xi =self.ifft(-self.wv2*self.ph)
#        self.Jptpc = -self._advect(
#                    (self.p[0] - self.p[1]),
#                    (self.del1*self.u[0] + self.del2*self.u[1]),
#                    (self.del1*self.v[0] + self.del2*self.v[1]))
#        # fix for delta.neq.1
#        self.Jpxi = self._advect(self.xi, self.u, self.v)
#
#    def _initialize_model_diagnostics(self):
#        """Extra diagnostics for two-layer model"""
#
#        self.add_diagnostic('entspec',
#            description='barotropic enstrophy spectrum',
#            function= (lambda self:
#                      np.abs(self.del1*self.qh[0] + self.del2*self.qh[1])**2.)
#        )
#            
#        self.add_diagnostic('APEflux',
#            description='spectral flux of available potential energy',
#            function= (lambda self:
#              self.rd**-2 * self.del1*self.del2 *
#              np.real((self.ph[0]-self.ph[1])*np.conj(self.Jptpc)) )
#        )
#        
#        self.add_diagnostic('KEflux',
#            description='spectral flux of kinetic energy',
#            function= (lambda self:
#              np.real(self.del1*self.ph[0]*np.conj(self.Jpxi[0])) + 
#              np.real(self.del2*self.ph[1]*np.conj(self.Jpxi[1])) )
#        )
#        
#        self.add_diagnostic('APEgenspec',
#            description='spectrum of APE generation',
#            function= (lambda self: self.U * self.rd**-2 * self.del1 * self.del2 *
#                       np.real(1j*self.k*(self.del1*self.ph[0] + self.del2*self.ph[0]) *
#                                  np.conj(self.ph[0] - self.ph[1])) )
#        )
#        
#        self.add_diagnostic('APEgen',
#            description='total APE generation',
#            function= (lambda self: self.U * self.rd**-2 * self.del1 * self.del2 *
#                       np.real(1j*self.k*
#                           (self.del1*self.ph[0] + self.del2*self.ph[1]) *
#                            np.conj(self.ph[0] - self.ph[1])).sum() / 
#                            (self.nx*self.ny) )
#        )
 
