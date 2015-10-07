import numpy as np
import model
from numpy import pi
import scipy as sp
import scipy.linalg
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
    
    This model is meant to represent flows driven by baroclinic instabilty of a
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
        nz = 4,                     # number of layers
        rd=15000.0,                 # deformation radius
        H = None,                   # layer thickness 
        U=None,                     # zonal base state flow
        V=None,                     # meridional base state flow
        rho = None,
        delta = None,
        g = 9.81,
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
        self.g = g    # acceleration due to gravity
        self.beta = beta
        #self.rek = rek
        self.rd = rd
        self.delta = delta
        #self.filterfac = filterfac
       
        # H is an array
        #self.Hi = np.array([100.,700.,2000.])
        #self.rhoi = np.array([1024.,1025.,1026.])
        #self.Ubg = np.array([0.05,0.025,0])
        #self.Vbg = np.array([0.05,0.025,0])
        self.nz = nz
        self.U = U
        self.V = V
        self.H = H
        self.rho = rho

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
      
        if (self.nz==2)&(self.rd is not None):

            self.del1 = self.delta/(self.delta+1.)
            self.del2 = (self.delta+1.)**-1
            self.Us = self.U[0]-self.U[1]

            self.F1 = self.rd**-2 / (1.+self.delta)
            self.F2 = self.delta*self.F1
            self.S[0,0], self.S[0,1] = -self.F1,  self.F1
            self.S[1,0], self.S[1,1] =  self.F2, -self.F2

        else:
            
            for i in range(self.nz):

                if i == 0:
                    self.S[i,i]   = -self.f2/self.Hi[i]/self.gpi[i] #- self.f2/self.Hi[i]/self.g
                    self.S[i,i+1] =  self.f2/self.Hi[i]/self.gpi[i]

                elif i == self.nz-1:
                    self.S[i,i]   = -self.f2/self.Hi[i]/self.gpi[i-1] 
                    self.S[i,i-1] =  self.f2/self.Hi[i]/self.gpi[i-1]
                    
                else:
                    self.S[i,i-1] = self.f2/self.Hi[i]/self.gpi[i-1]
                    self.S[i,i]   = -(self.f2/self.Hi[i]/self.gpi[i] +
                                        self.f2/self.Hi[i]/self.gpi[i-1])
                    self.S[i,i+1] = self.f2/self.Hi[i]/self.gpi[i]

    def _initialize_background(self):
        """Set up background state (zonal flow and PV gradients)."""
        
        # Need to figure out a warning to user when
        # these entries are not provided
        self.Hi = self.H
        self.Ubg = self.U
        self.Vbg = self.V
        self.rhoi = self.rho

#        assert self.Hi.size == self.nz, "size of Hi does not match number\
#                of vertical levels nz" 
#
#        assert self.rhoi.size == self.nz, "size of rhoi does not match number\
#                of vertical levels nz" 
#
#        assert self.Ubg.size == self.nz, "size of Ubg does not match number\
#                of vertical levels nz" 
#
#        assert self.Vbg.size == self.nz, "size of Vbg does not match number\
#                of vertical levels nz" 
#
        #self.Hi = np.array([500,2000.])
        #self.rhoi = np.array([1025.,1025.83])
        #self.Ubg = np.array([0.05,.0])
        #self.Vbg = np.array([0.,0])
         
        if not self.nz==2:
            self.gpi = self.g*(self.rhoi[1:]-self.rhoi[:-1])/self.rhoi[:-1]

        self.H = self.Hi.sum()

        self._initialize_stretching_matrix()

        # the meridional PV gradients in each layer
        self.Qy = self.beta - np.dot(self.S,self.Ubg)
        self.Qx = np.dot(self.S,self.Vbg)

        self.hb = self.hb * self.f/self.Hi[-1]
   

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

        if (self.nz==2):
            det_inv =  np.ma.masked_equal(
                    ( (self.S[0,0]-self.wv2)*(self.S[1,1]-self.wv2) -\
                            self.S[0,1]*self.S[1,0] ), 0.)**-1
            a[0,0] = (self.S[1,1]-self.wv2)*det_inv
            a[0,1] = -self.S[0,1]*det_inv
            a[1,0] = -self.S[1,0]*det_inv
            a[1,1] = (self.S[0,0]-self.wv2)*det_inv
        else:
            I = np.eye(self.nz)[:,:,np.newaxis,np.newaxis]
            a = np.linalg.inv((self.S[:,:,np.newaxis,np.newaxis]-I*self.wv2).T).T
            a[:,:,0,0] = 0.

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
        self.set_q(qi)
      
    
    def set_q1q2(self, q1, q2, check=False):
        """Set upper and lower layer PV anomalies.
        
        Parameters
        ----------
        
        q1 : array-like
            Upper layer PV anomaly in spatial coordinates.
        q1 : array-like
            Lower layer PV anomaly in spatial coordinates.
        """
        self.set_q(np.vstack([q1[np.newaxis,:,:], q2[np.newaxis,:,:]]))
        #self.q[0] = q1
        #self.q[1] = q2

        # initialize spectral PV
        #self.qh = self.fft2(self.q)
        
        # check that it works
        if check:
            np.testing.assert_allclose(self.q1, q1)
            np.testing.assert_allclose(self.q1, self.ifft2(self.qh1))


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

    def _calc_derived_fields(self):

        if self.nz == 2:
            self.p = self.ifft(self.ph)
            self.xi =self.ifft(-self.wv2*self.ph)
            self.Jptpc = -self._advect(
                        (self.p[0] - self.p[1]),
                        (self.del1*self.u[0] + self.del2*self.u[1]),
                        (self.del1*self.v[0] + self.del2*self.v[1]))
            # fix for delta.neq.1
            self.Jpxi = self._advect(self.xi, self.u, self.v)

        else:
            pass
            #raise NotImplementedError('Not implemented yet')
    

    def _initialize_model_diagnostics(self):
        """Extra diagnostics for two-layer model"""

        if self.nz == 2:

            self.add_diagnostic('entspec',
                description='barotropic enstrophy spectrum',
                function= (lambda self:
                          np.abs(self.del1*self.qh[0] + self.del2*self.qh[1])**2.)
            )
                
            self.add_diagnostic('APEflux',
                description='spectral flux of available potential energy',
                function= (lambda self:
                  self.rd**-2 * self.del1*self.del2 *
                  np.real((self.ph[0]-self.ph[1])*np.conj(self.Jptpc)) )
            )
            
            self.add_diagnostic('KEflux',
                description='spectral flux of kinetic energy',
                function= (lambda self:
                  np.real(self.del1*self.ph[0]*np.conj(self.Jpxi[0])) + 
                  np.real(self.del2*self.ph[1]*np.conj(self.Jpxi[1])) )
            )
            
            self.add_diagnostic('APEgenspec',
                description='spectrum of APE generation',
                function= (lambda self: self.Us * self.rd**-2 * self.del1 * self.del2 *
                           np.real(1j*self.k*(self.del1*self.ph[0] + self.del2*self.ph[0]) *
                                      np.conj(self.ph[0] - self.ph[1])) )
            )
            
            self.add_diagnostic('APEgen',
                description='total APE generation',
                function= (lambda self: self.Us * self.rd**-2 * self.del1 * self.del2 *
                           np.real(1j*self.k*
                               (self.del1*self.ph[0] + self.del2*self.ph[1]) *
                                np.conj(self.ph[0] - self.ph[1])).sum() / 
                                (self.nx*self.ny) )
            )

        else:
            pass
            #raise NotImplementedError('Not implemented yet')
    

