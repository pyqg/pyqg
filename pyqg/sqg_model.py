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

class SQGModel(model.Model):
    """A subclass that represents the Surface QG model."""
    
    def __init__(
        self,
        beta=0.,                    # gradient of coriolis parameter
        Nb = 1.,                    # Buoyancy frequency
        rek=0.,                     # linear drag in lower layer
        rd=0.,                      # deformation radius
        H = 1.,                     # depth of layer
        U=0.,                       # max vel. of base-state
        filterfac = 23.6,           # the factor for use in the exponential filter
        **kwargs
        ):
        """Initialize the surface QG model.

        beta -- gradient of coriolis parameter, units m^-1 s^-1
        (NOTE: currently some diagnostics assume delta==1)
        U -- upper layer flow, units m/s
        filterfac -- amplitdue of the spectral spherical filter
                     (originally 18.4, later changed to 23.6)
        """

        # physical
        self.beta = beta
        self.Nb = Nb
        self.rek = rek
        self.rd = rd
        self.H = H
        self.U = U
        self.filterfac = filterfac
        
        self.nz = 1
       
        super(SQGModel, self).__init__(**kwargs)
     
        # initial conditions: (PV anomalies)
        self.set_q(1e-3*np.random.rand(self.ny,self.nx))
 
    ### PRIVATE METHODS - not meant to be called by user ###
        
    def _initialize_background(self):
        """Set up background state (zonal flow and PV gradients)."""
        
        # the meridional PV gradients in each layer
        self.Qy = self.beta

        # background vel.
        self.set_U(self.U)        

        # complex versions, multiplied by k, speeds up computations to pre-compute
        self.ikQy = self.Qy * 1j * self.k
        
        self.ilQx = 0.

    def _initialize_inversion_matrix(self):
        """ the inversion """ 
        # The sqg model is diagonal. The inversion is simply qh = -kappa**2 ph
        self.a = self.wv/self.Nb

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
        self.ph = self.a*self.qh
        self.u = self.ifft2(-self.lj* self.ph) + self.Ubg
        self.v = self.ifft2(self.kj * self.ph)

    def _forcing_tendency(self):
        """Calculate tendency due to forcing."""
        # apply only in bottom layer
        self.dqhdt_forc = self.rek * self.wv2 * self.ph

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
        ke = .5*spec_var(self, self.wv*self.ph)
        return ke.sum()

    # calculate eddy turn over time 
    # (perhaps should change to fraction of year...)
    def _calc_eddy_time(self):
        """ estimate the eddy turn-over time in days """

        ens = .5*self.H * spec_var(self, self.wv2*self.ph)

        return 2.*pi*np.sqrt( self.H / ens ) / year

    def _calc_derived_fields(self):
        self.p = self.ifft2( self.ph)
        self.xi =self.ifft2( -self.wv2*self.ph)
        self.Jptpc = -self.advect(self.p,self.u,self.v)
        self.Jpxi = self.advect(self.xi, self.u, self.v)


    def _initialize_diagnostics(self):
        # Initialization for diagnotics
        self.diagnostics = dict()

        self.add_diagnostic('Ensspec',
            description='enstrophy spectrum',
            function= (lambda self: np.abs(self.qh)**2/self.M**2)
            )
            
        self.add_diagnostic('KEspec',
            description=' kinetic energy spectrum',
            function=(lambda self: self.wv2*np.abs(self.ph)**2/self.M**2)
            )  # factor of 2 to account for the fact that we have only half of 
               #    the Fourier coefficients.

        self.add_diagnostic('q',
            description='QGPV',
            function= (lambda self: self.q)
        )

        self.add_diagnostic('EKEdiss',
            description='total energy dissipation by bottom drag',
            function= (lambda self:
                       (self.rek*self.wv2*
                        np.abs(self.ph)**2./(self.M**2)).sum())
        )
        
# some off-class diagnostics
def spec_var(self,ph):
    """ compute variance of p from Fourier coefficients ph """
    var_dens = 2. * np.abs(ph)**2 / self.M**2
    # only half of coefs [0] and [nx/2+1] due to symmetry in real fft2
    var_dens[:,0],var_dens[:,-1] = var_dens[:,0]/2.,var_dens[:,-1]/2.
    return var_dens.sum()

