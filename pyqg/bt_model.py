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

class BTModel(model.Model):
    """A class that represents the single-layer QG model."""
    
    def __init__(
        self,
        beta=1.5e-11,               # gradient of coriolis parameter
        rek=5.787e-7,               # linear drag in lower layer
        rd=15000.0,                 # deformation radius
        H = 4000,                   # depth of layer 1 (H)
        U=0.,                       # max vel. of base-state
        filterfac=23.6,             # the factor for use in the exponential filter
        **kwargs
        ):
        """Initialize the single-layer QG model.

        beta -- gradient of coriolis parameter, units m^-1 s^-1
        rek -- linear drag in lower layer, units seconds^-1
        rd -- deformation radius, units meters
        (NOTE: currently some diagnostics assume delta==1)
        U -- upper layer flow, units m/s
        filterfac -- amplitdue of the spectral spherical filter
                     (originally 18.4, later changed to 23.6)
        """

        # physical
        self.beta = beta
        self.rek = rek
        self.rd = rd
        self.H = H
        self.U = U
        self.filterfac = filterfac
        
        self.nz = 1
        
        super(BTModel, self).__init__(**kwargs)
        
        # initial conditions: (PV anomalies)
        self.set_q(1e-3*np.random.rand(self.ny,self.nx))
     
    ### PRIVATE METHODS - not meant to be called by user ###
        
    def _initialize_background(self):
        """Set up background state (zonal flow and PV gradients)."""
        
        # the meridional PV gradients in each layer
        self.Qy = self.beta

        # background vel.
        self.set_U(self.U)        

        # complex versions, multiplied by k, speeds up computations to precompute
        self.ikQy = self.Qy * 1j * self.k
        
        self.ilQx = 0.
    

    def _initialize_inversion_matrix(self):
        """ the inversion """ 
        # The bt model is diagonal. The inversions is simply qh = -kappa**2 ph
        self.a = np.ma.masked_invalid(-self.wv2i).filled(0.)

    def _initialize_forcing(self):
        """Set up frictional filter."""
        # this defines the spectral filter (following Arbic and Flierl, 2003)
        cphi=0.65*pi
        wvx=np.sqrt((self.k*self.dx)**2.+(self.l*self.dy)**2.)
        self.filtr = np.exp(-self.filterfac*(wvx-cphi)**4.)  
        self.filtr[wvx<=cphi] = 1.
        
    def _filter(self, q):
        return self.filtr * q

    def _initialize_state_variables(self):
        
        # shape and datatype of real data
        dtype_real = np.dtype('float64')
        shape_real = (self.ny, self.nx)
        # shape and datatype of complex (fourier space) data
        dtype_cplx = np.dtype('complex128')
        shape_cplx = (self.nl, self.nk)
        
        # qgpv
        self.q  = np.zeros(shape_real, dtype_real)
        self.qh = np.zeros(shape_cplx, dtype_cplx)
        # streamfunction
        self.p  = np.zeros(shape_real, dtype_real)
        self.ph = np.zeros(shape_cplx, dtype_cplx)
        # velocity (only need real version)
        self.u = np.zeros(shape_real, dtype_real)
        self.v = np.zeros(shape_real, dtype_real)
        # tendencies (only need complex version)
        self.dqhdt_adv = np.zeros(shape_cplx, dtype_cplx)
        self.dqhdt_forc = np.zeros(shape_cplx, dtype_cplx)
        self.dqhdt = np.zeros(shape_cplx, dtype_cplx)
        # also need to save previous tendencies for Adams Bashforth
        self.dqhdt_p = np.zeros(shape_cplx, dtype_cplx)
        self.dqhdt_pp = np.zeros(shape_cplx, dtype_cplx)
                
     
    def set_q(self, q, check=False):
        """ Set PV anomaly """
        self.q = q

        # initialize spectral PV
        self.qh = self.fft2(self.q)
        
        # check that it works
        if check:
            np.testing.assert_allclose(self.q, q)
            np.testing.assert_allclose(self.q, self.ifft2(self.qh))
    
    def set_U(self, U):
        """Set background zonal flow"""
        self.Ubg = U

    def _invert(self):
        """ invert qgpv to find streamfunction. """
        # this matrix multiplication is an obvious target for optimization
        self.ph = self.a*self.qh
        self.u = self.ifft2(-self.lj* self.ph) + self.Ubg
        self.v = self.ifft2(self.kj * self.ph)

    def _forcing_tendency(self):
        """Calculate tendency due to forcing."""
        #self.dqh1dt_forc = # just leave blank
        # apply only in bottom layer
        self.dqhdt_forc = self.rek * self.wv2 * self.ph

#    def _calc_diagnostics(self):
#        # here is where we calculate diagnostics
#        if (self.t>=self.dt) and (self.tc%self.taveints==0):
#            self._increment_diagnostics()
#
    ### All the diagnostic stuff follows. ###
    def _calc_cfl(self):
        return np.abs(
            np.hstack([self.u + self.Ubg, self.v])
        ).max()*self.dt/self.dx

    # calculate KE: this has units of m^2 s^{-2}
    #   (should also multiply by H1 and H2...)
    def _calc_ke(self):
        ke = .5*spec_var(self, self.wv*self.ph)
        return ke.sum()

    # calculate eddy turn over time 
    # (perhaps should change to fraction of year...)
    def _calc_eddy_time(self):
        """ estimate the eddy turn-over time in days """

        ens = .5*self.H * spec_var(self, self.wv2*self.ph1)

        return 2.*pi*np.sqrt( self.H / ens ) / 86400

#    def _calc_derived_fields(self):
#        self.p = self.ifft2( self.ph)
#        self.xi =self.ifft2( -self.wv2*self.ph)
#        self.Jptpc = -self.advect(
#                    (self.p[0] - self.p[1]),
#                    (self.del1*self.u[0] + self.del2*self.u[1]),
#                    (self.del1*self.v[0] + self.del2*self.v[1]))
#        # fix for delta.neq.1
#        self.Jpxi = self.advect(self.xi, self.u, self.v)
#
#    def _initialize_diagnostics(self):
#        # Initialization for diagnotics
#        self.diagnostics = dict()
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
#
#        )
#        
#        self.add_diagnostic('KEflux',
#            description='spectral flux of kinetic energy',
#            function= (lambda self:
#              np.real(self.del1*self.ph[0]*np.conj(self.Jpxi[1])) + 
#              np.real(self.del2*self.ph[1]*np.conj(self.Jpxi[0])) )
#        )
#
#        self.add_diagnostic('KE1spec',
#            description='upper layer kinetic energy spectrum',
#            function=(lambda self: 0.5*self.wv2*np.abs(self.ph[0])**2)
#        )
#        
#        self.add_diagnostic('KE2spec',
#            description='lower layer kinetic energy spectrum',
#            function=(lambda self: 0.5*self.wv2*np.abs(self.ph[1])**2)
#        )
#        
#        self.add_diagnostic('q1',
#            description='upper layer QGPV',
#            function= (lambda self: self.q[0])
#        )
#
#        self.add_diagnostic('q2',
#            description='lower layer QGPV',
#            function= (lambda self: self.q[1])
#        )
#
#        self.add_diagnostic('EKE1',
#            description='mean upper layer eddy kinetic energy',
#            function= (lambda self: 0.5*(self.v[0]**2 + self.u[0]**2).mean())
#        )
#
#        self.add_diagnostic('EKE2',
#            description='mean lower layer eddy kinetic energy',
#            function= (lambda self: 0.5*(self.v[1]**2 + self.u[1]**2).mean())
#        )
#        
#        self.add_diagnostic('EKEdiss',
#            description='total energy dissipation by bottom drag',
#            function= (lambda self:
#                       (self.del2*self.rek*self.wv2*
#                        np.abs(self.ph[1])**2./(self.nx*self.ny)).sum())
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
#

# some off-class diagnostics
def spec_var(self,ph):
    """ compute variance of p from Fourier coefficients ph """
    var_dens = 2. * np.abs(ph)**2 / self.M**2
    # only half of coefs [0] and [nx/2+1] due to symmetry in real fft2
    var_dens[:,0],var_dens[:,-1] = var_dens[:,0]/2.,var_dens[:,-1]/2.
    return var_dens.sum()

