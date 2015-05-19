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
    """A class that represents the two-layer QG model."""
    
    def __init__(
        self,
        beta=1.5e-11,               # gradient of coriolis parameter
        #rek=5.787e-7,               # linear drag in lower layer
        rd=15000.0,                 # deformation radius
        delta=0.25,                 # layer thickness ratio (H1/H2)
        H1 = 500,                   # depth of layer 1 (H1)
        U1=0.025,                   # upper layer flow
        U2=0.0,                     # lower layer flow
        filterfac=23.6,             # the factor for use in the exponential filter
        **kwargs
        ):
        """Initialize the two-layer QG model.

        beta -- gradient of coriolis parameter, units m^-1 s^-1
        rek -- linear drag in lower layer, units seconds^-1
        rd -- deformation radius, units meters
        delta -- layer thickness ratio (H1/H2)
        (NOTE: currently some diagnostics assume delta==1)
        U1 -- upper layer flow, units m/s
        U2 -- lower layer flow, units m/s
        filterfac -- amplitdue of the spectral spherical filter
                     (originally 18.4, later changed to 23.6)
        """

        # physical
        self.beta = beta
        #self.rek = rek
        self.rd = rd
        self.delta = delta
        self.H1 = H1
        self.H2 = H1/delta
        self.U1 = U1
        self.U2 = U2
        self.filterfac = filterfac
        
        self.nz = 2
        
        super(QGModel, self).__init__(**kwargs)
        
        # initial conditions: (PV anomalies)
        self.set_q1q2(
            1e-7*np.random.rand(self.ny,self.nx) + 1e-6*(
                np.ones((self.ny,1)) * np.random.rand(1,self.nx) ),
                np.zeros_like(self.x) )   
        
    
                
    ### PRIVATE METHODS - not meant to be called by user ###
        
    def _initialize_background(self):
        """Set up background state (zonal flow and PV gradients)."""
        
        # Background zonal flow (m/s):
        self.H = self.H1 + self.H2
        self.set_U1U2(self.U1, self.U2)
        self.U = self.U1 - self.U2        

        # the F parameters
        self.F1 = self.rd**-2 / (1.+self.delta)
        self.F2 = self.delta*self.F1

        # the meridional PV gradients in each layer
        self.Qy1 = self.beta + self.F1*(self.U1 - self.U2)
        self.Qy2 = self.beta - self.F2*(self.U1 - self.U2)
        self.Qy = np.array([self.Qy1, self.Qy2])
        # complex versions, multiplied by k, speeds up computations to precompute
        self.ikQy1 = self.Qy1 * 1j * self.k
        self.ikQy2 = self.Qy2 * 1j * self.k

        # vector version
        self.ikQy = np.vstack([self.ikQy1[np.newaxis,...], 
                               self.ikQy2[np.newaxis,...]]) 
        self.ilQx = 0.

        # layer spacing
        self.del1 = self.delta/(self.delta+1.)
        self.del2 = (self.delta+1.)**-1
        
    def _initialize_inversion_matrix(self):
        
        # The matrix multiplication will look like this
        # ph[0] = a[0,0] * self.qh[0] + a[0,1] * self.qh[1]
        # ph[1] = a[1,0] * self.qh[0] + a[1,1] * self.qh[1]

        a = np.ma.zeros((self.nz, self.nz, self.nl, self.nk), np.dtype('float64'))        
        # inverse determinant
        det_inv =  np.ma.masked_equal(
                self.wv2 * (self.wv2 + self.F1 + self.F2), 0.)**-1
        a[0,0] = -(self.wv2 + self.F2)*det_inv
        a[0,1] = -self.F1*det_inv
        a[1,0] = -self.F2*det_inv
        a[1,1] = -(self.wv2 + self.F1)*det_inv
        
        self.a = np.ma.masked_invalid(a).filled(0.)
        
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
        shape_real = (self.nz, self.ny, self.nx)
        # shape and datatype of complex (fourier space) data
        dtype_cplx = np.dtype('complex128')
        shape_cplx = (self.nz, self.nl, self.nk)
        
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
                
     
    def set_q1q2(self, q1, q2, check=False):
        """Set upper and lower layer PV anomalies."""
        self.set_q(np.vstack([q1[np.newaxis,:,:], q2[np.newaxis,:,:]]))
        #self.q[0] = q1
        #self.q[1] = q2

        # initialize spectral PV
        #self.qh = self.fft2(self.q)
        
        # check that it works
        if check:
            np.testing.assert_allclose(self.q1, q1)
            np.testing.assert_allclose(self.q1, self.ifft2(self.qh1))
    
    def set_U1U2(self, U1, U2):
        """Set background zonal flow"""
        self.U1 = U1
        self.U2 = U2
        #self.Ubg = np.array([U1,U2])[:,np.newaxis,np.newaxis]
        self.Ubg = np.array([U1,U2])

    def _invert_old(self):
        """invert qgpv to find streamfunction."""
        # this matrix multiplication is an obvious target for optimization
        self.ph = np.einsum('ijkl,jkl->ikl', self.a, self.qh)
        self.u = self.ifft2(-self.lj* self.ph) + self.Ubg
        self.v = self.ifft2(self.kj * self.ph)

    def _invert_test(self):
        """invert qgpv to find streamfunction."""
        # this matrix multiplication is an obvious target for optimization
        ph = np.einsum('ijkl,jkl->ikl', self.a, self.qh)
        u = self.ifft2(-self.lj* ph)
        v = self.ifft2(self.kj * ph)
        return ph, u, v
        
    def _do_friction(self):
        """Calculate tendency due to forcing."""
        #self.dqh1dt_forc = # just leave blank
        # apply only in bottom layer
        #self.dqhdt_forc[-1] = self.rek * self.wv2 * self.ph[-1]
        self.dqhdt[-1] += self.rek * self.wv2 * self.ph[-1]

    def _forcing_tendency(self):
        """Calculate tendency due to forcing."""
        #self.dqh1dt_forc = # just leave blank
        # apply only in bottom layer
        #self.dqhdt_forc[-1] = self.rek * self.wv2 * self.ph[-1]
        self.dqhdt[-1] += self.rek * self.wv2 * self.ph[-1]

                    
        #     print 't=%16d, tc=%10d: cfl=%5.6f, ke=%9.9f, T_e=%9.9f' % (
        #            self.t, self.tc, self.calc_cfl(), \
        #                    self.ke[-1], self.eddy_time[-1] )
        #
        #     # append ke and time
        #     if self.tc > 0.:
        #         self.ke = np.append(self.ke,self.calc_ke())
        #         self.eddy_time = np.append(self.eddy_time,self.calc_eddy_time())
        #         self.time = np.append(self.time,self.t)


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
        ke1 = .5*self.H1*spec_var(self, self.wv*self.ph[0])
        ke2 = .5*self.H2*spec_var(self, self.wv*self.ph[1]) 
        return ( ke1.sum() + ke2.sum() ) / self.H

    # calculate eddy turn over time 
    # (perhaps should change to fraction of year...)
    def _calc_eddy_time(self):
        """ estimate the eddy turn-over time in days """

        ens = .5*self.H1 * spec_var(self, self.wv2*self.ph1) + \
            .5*self.H2 * spec_var(self, self.wv2*self.ph2)

        return 2.*pi*np.sqrt( self.H / ens ) / 86400

    def _calc_derived_fields(self):
        self.p = self.ifft2( self.ph)
        self.xi =self.ifft2( -self.wv2*self.ph)
        self.Jptpc = -self.advect(
                    (self.p[0] - self.p[1]),
                    (self.del1*self.u[0] + self.del2*self.u[1]),
                    (self.del1*self.v[0] + self.del2*self.v[1]))
        # fix for delta.neq.1
        self.Jpxi = self.advect(self.xi, self.u, self.v)

    def _initialize_diagnostics(self, diagnostics_list):
        # Initialization for diagnotics
        self.diagnostics = dict()

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
              np.real(self.del1*self.ph[0]*np.conj(self.Jpxi[1])) + 
              np.real(self.del2*self.ph[1]*np.conj(self.Jpxi[0])) )
        )

        self.add_diagnostic('KE1spec',
            description='upper layer kinetic energy spectrum',
            function=(lambda self: 0.5*self.wv2*np.abs(self.ph[0])**2)
        )
        
        self.add_diagnostic('KE2spec',
            description='lower layer kinetic energy spectrum',
            function=(lambda self: 0.5*self.wv2*np.abs(self.ph[1])**2)
        )
        
        self.add_diagnostic('q1',
            description='upper layer QGPV',
            function= (lambda self: self.q[0])
        )

        self.add_diagnostic('q2',
            description='lower layer QGPV',
            function= (lambda self: self.q[1])
        )

        self.add_diagnostic('EKE1',
            description='mean upper layer eddy kinetic energy',
            function= (lambda self: 0.5*(self.v[0]**2 + self.u[0]**2).mean())
        )

        self.add_diagnostic('EKE2',
            description='mean lower layer eddy kinetic energy',
            function= (lambda self: 0.5*(self.v[1]**2 + self.u[1]**2).mean())
        )
        
        self.add_diagnostic('EKEdiss',
            description='total energy dissipation by bottom drag',
            function= (lambda self:
                       (self.del2*self.rek*self.wv2*
                        np.abs(self.ph[1])**2./(self.nx*self.ny)).sum())
        )
        
        self.add_diagnostic('APEgenspec',
            description='spectrum of APE generation',
            function= (lambda self: self.U * self.rd**-2 * self.del1 * self.del2 *
                       np.real(1j*self.k*(self.del1*self.ph[0] + self.del2*self.ph[0]) *
                                  np.conj(self.ph[0] - self.ph[1])) )
        )
        
        self.add_diagnostic('APEgen',
            description='total APE generation',
            function= (lambda self: self.U * self.rd**-2 * self.del1 * self.del2 *
                       np.real(1j*self.k*
                           (self.del1*self.ph[0] + self.del2*self.ph[1]) *
                            np.conj(self.ph[0] - self.ph[1])).sum() / 
                            (self.nx*self.ny) )
        )


# some off-class diagnostics
def spec_var(self,ph):
    """ compute variance of p from Fourier coefficients ph """
    var_dens = 2. * np.abs(ph)**2 / self.M**2
    # only half of coefs [0] and [nx/2+1] due to symmetry in real fft2
    var_dens[:,0],var_dens[:,-1] = var_dens[:,0]/2.,var_dens[:,-1]/2.
    return var_dens.sum()

