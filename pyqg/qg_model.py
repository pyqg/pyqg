import numpy as np
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

class QGModel(object):
    """A class that represents the two-layer QG model."""
    
    def __init__(
        self,
        # grid size parameters
        nx=64,                     # grid resolution
        ny=None,
        L=1e6,                     # domain size is L [m]
        W=None,
        # physical parameters
        beta=1.5e-11,               # gradient of coriolis parameter
        rek=5.787e-7,               # linear drag in lower layer
        rd=15000.0,                 # deformation radius
        delta=0.25,                 # layer thickness ratio (H1/H2)
        H1 = 500,                   # depth of layer 1 (H1)
        U1=0.025,                   # upper layer flow
        U2=0.0,                     # lower layer flow
        filterfac=23.6,             # the factor for use in the exponential filter
        # timestepping parameters
        dt=7200.,                   # numerical timestep
        tplot=10000.,               # interval for plots (in timesteps)
        twrite=1000.,               # interval for cfl and ke writeout (in timesteps)
        tmax=1576800000.,           # total time of integration
        tavestart=315360000.,       # start time for averaging
        taveint=86400.,             # time interval used for summation in longterm average in seconds
        tpickup=31536000.,          # time interval to write out pickup fields ("experimental")
        useAB2=False,               # use second order Adams Bashforth timestepping instead of 3rd
        # diagnostics parameters
        diagnostics_list='all',     # which diagnostics to output
        # fft parameters
        use_fftw = False,               # fftw flag 
        ntd = 1,                    # number of threads to use in fftw computations
        ):
        """Initialize the two-layer QG model.
        
        The model parameters are passed as keyword arguments.
        They are grouped into the following categories
        
        Grid Parameter Keyword Arguments:
        nx -- number of grid points in the x direction
        ny -- number of grid points in the y direction (default: nx)
        L -- domain length in x direction, units meters 
        W -- domain width in y direction, units meters (default: L)
        (WARNING: some parts of the model or diagnostics might
        actuallye assume nx=ny -- check before making different choice!)
        
        Physical Paremeter Keyword Arguments:
        beta -- gradient of coriolis parameter, units m^-1 s^-1
        rek -- linear drag in lower layer, units seconds^-1
        rd -- deformation radius, units meters
        delta -- layer thickness ratio (H1/H2)
        (NOTE: currently some diagnostics assume delta==1)
        U1 -- upper layer flow, units m/s
        U2 -- lower layer flow, units m/s
        filterfac -- amplitdue of the spectral spherical filter
                     (originally 18.4, later changed to 23.6)
        
        Timestep-related Keyword Arguments:
        dt -- numerical timstep, units seconds
        tplot -- interval for plotting, units number of timesteps
        tcfl -- interval for cfl writeout, units number of timesteps
        tmax -- total time of integration, units seconds
        tavestart -- start time for averaging, units seconds
        tsnapstart -- start time for snapshot writeout, units seconds
        taveint -- time interval for summation in diagnostic averages,
                   units seconds
           (for performance purposes, averaging does not have to
            occur every timestep)
        tsnapint -- time interval for snapshots, units seconds 
        tpickup -- time interval for writing pickup files, units seconds
        (NOTE: all time intervals will be rounded to nearest dt interval)
        useAB2 -- use second order Adams Bashforth timestepping instead of third
        """

        if ny is None: ny = nx
        if W is None: W = L
       
        # put all the parameters into the object
        # grid
        self.nx = nx
        self.ny = ny
        self.L = L
        self.W = W
        # physical
        self.beta = beta
        self.rek = rek
        self.rd = rd
        self.delta = delta
        self.H1 = H1
        self.H2 = H1/delta
        self.H = self.H1 + self.H2
        self.U1 = U1
        self.U2 = U2
        self.filterfac = filterfac
        # timestepping
        self.dt = dt
        self.tplot = tplot
        self.twrite = twrite
        self.tmax = tmax
        self.tavestart = tavestart
        self.taveint = taveint
        self.tpickup = tpickup
        self.useAB2 = useAB2
        # fft 
        self.use_fftw = use_fftw
        self.ntd = ntd

        # compute timestep stuff
        self.taveints = np.ceil(taveint/dt)      

        self.x,self.y = np.meshgrid(
            np.arange(0.5,self.nx,1.)/self.nx*self.L,
            np.arange(0.5,self.ny,1.)/self.ny*self.W )
        
        # initialize fft
        self.initialize_fft()
        
        # initial conditions: (PV anomalies)
        self.set_q1q2(
            1e-7*np.random.rand(self.ny,self.nx) + 1e-6*(
                np.ones((self.ny,1)) * np.random.rand(1,self.nx) ),
                np.zeros_like(self.x) )   

        # Background zonal flow (m/s):
        self.U = self.U1 - self.U2

        # Notice: at xi=1 U=beta*rd^2 = c for xi>1 => U>c

        # wavenumber one (equals to dkx/dky)
        self.dk = 2.*pi/self.L
        self.dl = 2.*pi/self.W

        # wavenumber grids
        self.ll = self.dl*np.append( np.arange(0.,self.nx/2), \
            np.arange(-self.nx/2,0.) )
        self.kk = self.dk*np.arange(0.,self.nx/2+1)

        self.k, self.l = np.meshgrid(self.kk, self.ll)
        # complex versions, speeds up computations to precompute
        self.kj = 1j*self.k
        self.lj = 1j*self.l
        # physical grid spacing
        self.dx = self.L / self.nx
        self.dy = self.W / self.ny

        # constant for spectral normalizations
        self.M = nx*ny

        # the F parameters
        self.F1 = self.rd**-2 / (1.+self.delta)
        self.F2 = self.delta*self.F1

        # the meridional PV gradients in each layer
        self.beta1 = self.beta + self.F1*(self.U1 - self.U2)
        self.beta2 = self.beta - self.F2*(self.U1 - self.U2)
        # complex versions, multiplied by k, speeds up computations to precompute
        self.beta1jk = self.beta1 * 1j * self.k
        self.beta2jk = self.beta2 * 1j * self.k

        # layer spacing
        self.del1 = self.delta/(self.delta+1.)
        self.del2 = (self.delta+1.)**-1

        # isotropic wavenumber^2 grid
        # the inversion is not defined at kappa = 0 
        # it is better to be explicit and not compute
        self.wv2 = self.k**2 + self.l**2
        self.wv = np.sqrt( self.wv2 )

        iwv2 = self.wv2 != 0.
        self.wv2i = np.zeros(self.wv2.shape)
        self.wv2i[iwv2] = self.wv2[iwv2]**-2
        
        # determine inversion matrix: psi = A q (i.e. A=M_2**(-1) where q=M_2*psi)
        self.a11 = np.zeros(self.wv2.shape)
        self.a12,self.a21 = self.a11.copy(), self.a11.copy()
        self.a22 = self.a11.copy()

        det = self.wv2 * (self.wv2 + self.F1 + self.F2)
        self.a11[iwv2] = -((self.wv2[iwv2] + self.F2)/det[iwv2])
        self.a12[iwv2] = -((self.F1)/det[iwv2])
        self.a21[iwv2] = -((self.F2)/det[iwv2])
        self.a22[iwv2] = -((self.wv2[iwv2] + self.F1)/det[iwv2])

        # this defines the spectral filter (following Arbic and Flierl, 2003)
        cphi=0.65*pi
        wvx=np.sqrt((self.k*self.dx)**2.+(self.l*self.dy)**2.)
        self.filtr = np.exp(-self.filterfac*(wvx-cphi)**4.)  
        self.filtr[wvx<=cphi] = 1.  

        # initialize timestep
        self.t=0        # actual time
        self.tc=0       # timestep number
        
        # initialize tendencies to zero
        self.dqh1dt_adv = np.zeros_like(self.wv2)
        self.dqh2dt_adv = np.zeros_like(self.wv2)
        self.dqh1dt_forc = np.zeros_like(self.wv2)
        self.dqh2dt_forc = np.zeros_like(self.wv2)
        self.dqh1dt = np.zeros_like(self.wv2)
        self.dqh2dt = np.zeros_like(self.wv2)

        # Set time-stepping parameters for very first timestep (forward Euler stepping).
        # Second-order Adams Bashford (AB2) is used at the second setep
        #   and  third-order AB (AB3) is used thereafter
        self.dqh1dt_p, self.dqh1dt_p = np.zeros(self.wv2.shape), np.zeros(self.wv2.shape) 
        self.dqh2dt_p, self.dqh2dt_p = self.dqh1dt_p.copy(), self.dqh1dt_p.copy()
        self.dqh1dt_pp, self.dqh1dt_pp = self.dqh1dt_p.copy(), self.dqh1dt_p.copy()
        self.dqh2dt_pp, self.dqh2dt_pp =  self.dqh1dt_p.copy(), self.dqh1dt_p.copy()

        self._initialize_diagnostics()
        if diagnostics_list == 'all':
            pass # by default, all diagnostics are active
        elif diagnostics_list == 'none':
            self.set_active_diagnostics([])
        else:
            self.set_active_diagnostics(diagnostics_list)
            
    def initialize_fft(self):
        # set up fft functions for use later
        if self.use_fftw:
            
            ## drop in replacement for numpy fft
            ## more than twice as fast as the previous code
            self.fft2 = (lambda x :
                    pyfftw.interfaces.numpy_fft.rfft2(x, threads=self.ntd))
            self.ifft2 = (lambda x :
                    pyfftw.interfaces.numpy_fft.irfft2(x, threads=self.ntd))
            
            ## This does some weird stuff that I don't understand
            ## and in the end does not work.
            ## Have to better understand the whole byte align thing...
            #aw = pyfftw.n_byte_align_empty(self.x.shape, 8, 'float64')
            #fft2 = pyfftw.builders.rfft2(aw,threads=self.ntd)
            #self.fft2 = (lambda g : fft2(g).copy())
            ## get an example fourier representation
            #ah = self.fft2(self.x)
            
            #awh = pyfftw.n_byte_align_empty(ah.shape, 16, 'complex128')
            #ifft2 = pyfftw.builders.irfft2(awh,threads=self.ntd)
            #self.ifft2 = (lambda g : ifft2(g).copy())
            
            ## this was very slow because it re-initializes the fft plan with every call
            #def fftw_fft(a):
            #    aw = pyfftw.n_byte_align_empty(a.shape, 8, 'float64')                 
            #    aw[:] = a.copy()
            #    return pyfftw.builders.rfft2(aw,threads=self.ntd)()
            #def fftw_ifft(ah):
            #    awh = pyfftw.n_byte_align_empty(ah.shape, 16, 'complex128')
            #    awh[:]= ah.copy()
            #    return pyfftw.builders.irfft2(awh,threads=self.ntd)()
            #self.fft2 = fftw_fft
            #self.ifft2 = fftw_ifft
            
        else:
            self.fft2 = np.fft.rfft2
            self.ifft2 = np.fft.irfft2
     
    ### list of all the variables that get transformed
    ### FFT:
    ###      self.q1, self.q2, u*q, v*q
    ### IFFT:
    ###     -1j*self.l*ph, 1j*self.k*ph, self.qh1, self.qh2
    ###     (for diagnostics):
    ###      self.ph1, self.ph2, -self.wv2*self.ph1, -self.wv2*self.ph2
     
    def set_q1q2(self, q1, q2, check=False):
        self.q1 = q1
        self.q2 = q2

        # initialize spectral PV
        self.qh1 = self.fft2(self.q1)
        self.qh2 = self.fft2(self.q2)
        
        # check that it works
        if check:
            np.testing.assert_allclose(self.q1, q1)
            np.testing.assert_allclose(self.q1, self.ifft2(self.qh1))
        

    # compute advection in grid space (returns qdot in fourier space)
    def advect(self, q, u, v):
        return 1j*self.k*self.fft2(u*q) + 1j*self.l*self.fft2(v*q)
        
    # compute grid space u and v from fourier streafunctions
    def caluv(self, ph):
        u = self.ifft2(-1j*self.l*ph)
        v = self.ifft2(1j*self.k*ph)
        return u, v
  
    # Invert PV for streamfunction
    def invph(self, zh1, zh2):
        """ From q_i compute psi_i, i= 1,2"""
        ph1 = self.a11*zh1 + self.a12*zh2
        ph2 = self.a21*zh1 + self.a22*zh2
        return ph1, ph2
    
    def run_with_snapshots(self, tsnapstart=0., tsnapint=432000.):
        """ Run the model forward until the next snapshot, then yield."""
        
        tsnapints = np.ceil(tsnapint/self.dt)
        nt = np.ceil(np.floor((self.tmax-tsnapstart)/self.dt+1)/tsnapints)
        
        while(self.t < self.tmax):
            self._step_forward()
            if self.t>=tsnapstart and (self.tc%tsnapints)==0:
                yield self.t
        return
                
    def run(self):
        """ Run the model forward without stopping until the end."""
        while(self.t < self.tmax): 
            self._step_forward()

            if np.isnan(self.qh1.sum()):
                print " *** Blow up  "
                break

    def _step_forward(self):

        # the basic steps are
        
        self.invert()
        # find streamfunction from pv
        
        self.advection_tendency()
        # use streamfunction to calculate advection tendency
        
        self.forcing_tendency()
        # apply friction and external forcing
        
        self.calc_diagnostics()
        # do what has to be done with diagnostics
        
        self.forward_timestep()
        # apply tendencies to step the model forward
        # (filter gets called here)


    def invert(self):
        """invert qgpv to find streamfunction."""
        # this matrix multiplication is an obvious target for optimization
        self.ph1, self.ph2 = self.invph(self.qh1, self.qh2)
        np.add(
            np.multiply(self.a11, self.qh1),
            np.multiply(self.a12, self.qh2),
            self.ph1 # output to this variable
        )
        np.add(
            np.multiply(self.a21, self.qh1),
            np.multiply(self.a22, self.qh2),
            self.ph2 # output to this variable
        )
                        
    def advection_tendency(self):
        """Calculate tendency due to advection."""
        # compute real space qgpv and velocity
        self.q1 = self.ifft2(self.qh1)
        self.q2 = self.ifft2(self.qh2)
        self.u1 = self.ifft2(-1j*self.l * self.ph1)
        self.v1 = self.ifft2(1j*self.k * self.ph1)
        self.u2 = self.ifft2(-1j*self.l * self.ph2)
        self.v2 = self.ifft2(1j*self.k * self.ph2)
        
        # multiply velocity and qgpv to get fluxes
        uq1 = np.multiply(self.q1, self.u1 + self.U1)
        vq1 = np.multiply(self.q1, self.v1)        
        uq2 = np.multiply(self.q2, self.u2 + self.U2)
        vq2 = np.multiply(self.q2, self.v2)
        
        # derivatives in spectral space (including background advection)
        ddx_uq1 = 1j * self.k * self.fft2(uq1)
        ddy_vq1 = 1j * self.l * self.fft2(vq1) + (self.beta1jk * self.ph1)
        ddx_uq2 = 1j * self.k * self.fft2(uq2)
        ddy_vq2 = 1j * self.l * self.fft2(vq2) + (self.beta2jk * self.ph2)
        
        # divergence of advective flux
        self.dqh1dt_adv = -(ddx_uq1 + ddy_vq1)
        self.dqh2dt_adv = -(ddx_uq2 + ddy_vq2)        

    def forcing_tendency(self):
        """Calculate tendency due to forcing."""
        #self.dqh1dt_forc = # just leave blank
        self.dqh2dt_forc = self.rek * self.wv2 * self.ph2


        # this is stuff the Cesar added
        
        # if self.tc==0:
        #     assert self.calc_cfl()<1., " *** time-step too large "
        #     # initialize ke and time arrays
        #     self.ke = np.array([self.calc_ke()])
        #     self.eddy_time = np.array([self.calc_eddy_time()])
        #     self.time = np.array([0.])
        
        
        ## write out
        # if (self.tc % self.twrite)==0:
        #     print 't=%16d, tc=%10d: cfl=%5.6f, ke=%9.9f, T_e=%9.9f' % (
        #            self.t, self.tc, self.calc_cfl(), \
        #                    self.ke[-1], self.eddy_time[-1] )
        #
        #     # append ke and time
        #     if self.tc > 0.:
        #         self.ke = np.append(self.ke,self.calc_ke())
        #         self.eddy_time = np.append(self.eddy_time,self.calc_eddy_time())
        #         self.time = np.append(self.time,self.t)


    def calc_diagnostics(self):
        # here is where we calculate diagnostics
        if (self.t>=self.dt) and (self.tc%self.taveints==0):
            self._increment_diagnostics()


    def forward_timestep(self):
        """Step forward based on tendencies"""
        self.dqh1dt = self.dqh1dt_adv + self.dqh1dt_forc
        self.dqh2dt = self.dqh2dt_adv + self.dqh2dt_forc
        
        #self.dqh1dt = (-self.advect(self.q1, self.u1 + self.U1, self.v1)
        #          -self.beta1*1j*self.k*self.ph1)
        #self.dqh2dt = (-self.advect(self.q2, self.u2 + self.U2, self.v2)
        #          -self.beta2*1j*self.k*self.ph2 + self.rek*self.wv2*self.ph2)
              
        # Note that Adams-Bashforth is not self-starting
        if self.tc==0:
            # forward Euler at the first step
            q1tend = tendency_forward_euler(self.dt, self.dqh1dt)
            q2tend = tendency_forward_euler(self.dt, self.dqh2dt)
        elif (self.tc==1) or (self.useAB2):
            # AB2 at step 2
            q1tend = tendency_ab2(self.dt, self.dqh1dt, self.dqh1dt_p)
            q2tend = tendency_ab2(self.dt, self.dqh2dt, self.dqh2dt_p)
        else:
            # AB3 from step 3 on
            q1tend = tendency_ab3(self.dt, self.dqh1dt,
                        self.dqh1dt_p, self.dqh1dt_pp)
            q2tend = tendency_ab3(self.dt, self.dqh2dt,
                        self.dqh2dt_p, self.dqh2dt_pp)
            
        # add tendency and filter
        self.qh1 = self.filtr*(self.qh1 + q1tend)
        self.qh2 = self.filtr*(self.qh2 + q2tend)  
        
        # remember previous tendencies
        self.dqh1dt_pp = self.dqh1dt_p.copy()
        self.dqh2dt_pp = self.dqh2dt_p.copy() 
        self.dqh1dt_p = self.dqh1dt.copy()
        self.dqh2dt_p = self.dqh2dt.copy()
                
        # augment timestep and current time
        self.tc += 1
        self.t += self.dt

    ### All the diagnostic stuff follows. ###
    def calc_cfl(self):
        return np.abs(np.hstack([self.u1 + self.U1, self.v1,
                          self.u2 + self.U2, self.v2])).max()*self.dt/self.dx

    # calculate KE: this has units of m^2 s^{-2}
    #   (should also multiply by H1 and H2...)
    def calc_ke(self):
        ke1 = .5*self.H1*spec_var(self, self.wv*self.ph1)
        ke2 = .5*self.H2*spec_var(self, self.wv*self.ph2) 
        return ( ke1.sum() + ke2.sum() ) / self.H

    # calculate eddy turn over time 
    # (perhaps should change to fraction of year...)
    def calc_eddy_time(self):
        """ estimate the eddy turn-over time in days """

        ens = .5*self.H1 * spec_var(self, self.wv2*self.ph1) + \
            .5*self.H2 * spec_var(self, self.wv2*self.ph2)

        return 2.*pi*np.sqrt( self.H / ens ) / 86400


    def set_active_diagnostics(self, diagnostics_list):
        for d in self.diagnostics:
            self.diagnostics[d]['active'] == (d in diagnostics_list)

    def _initialize_diagnostics(self):
        # Initialization for diagnotics
        self.diagnostics = dict()

        self.add_diagnostic('entspec',
            description='barotropic enstrophy spectrum',
            function= (lambda self:
                      np.abs(self.del1*self.qh1 + self.del2*self.qh2)**2.)
        )
            
        self.add_diagnostic('APEflux',
            description='spectral flux of available potential energy',
            function= (lambda self:
              self.rd**-2 * self.del1*self.del2 *
              np.real((self.ph1-self.ph2)*np.conj(self.Jptpc)) )

        )
        
        self.add_diagnostic('KEflux',
            description='spectral flux of kinetic energy',
            function= (lambda self:
              np.real(self.del1*self.ph1*np.conj(self.Jp1xi1)) + 
              np.real(self.del2*self.ph2*np.conj(self.Jp2xi2)) )
        )

        self.add_diagnostic('KE1spec',
            description='upper layer kinetic energy spectrum',
            function=(lambda self: 0.5*self.wv2*np.abs(self.ph1)**2)
        )
        
        self.add_diagnostic('KE2spec',
            description='lower layer kinetic energy spectrum',
            function=(lambda self: 0.5*self.wv2*np.abs(self.ph2)**2)
        )
        
        self.add_diagnostic('q1',
            description='upper layer QGPV',
            function= (lambda self: self.q1)
        )

        self.add_diagnostic('q2',
            description='lower layer QGPV',
            function= (lambda self: self.q2)
        )

        self.add_diagnostic('EKE1',
            description='mean upper layer eddy kinetic energy',
            function= (lambda self: 0.5*(self.v1**2 + self.u1**2).mean())
        )

        self.add_diagnostic('EKE2',
            description='mean lower layer eddy kinetic energy',
            function= (lambda self: 0.5*(self.v2**2 + self.u2**2).mean())
        )
        
        self.add_diagnostic('EKEdiss',
            description='total energy dissipation by bottom drag',
            function= (lambda self:
                       (self.del2*self.rek*self.wv2*
                        np.abs(self.ph2)**2./(self.nx*self.ny)).sum())
        )
        
        self.add_diagnostic('APEgenspec',
            description='spectrum of APE generation',
            function= (lambda self: self.U * self.rd**-2 * self.del1 * self.del2 *
                       np.real(1j*self.k*(self.del1*self.ph1 + self.del2*self.ph2) *
                                  np.conj(self.ph1 - self.ph2)) )
        )
        
        self.add_diagnostic('APEgen',
            description='total APE generation',
            function= (lambda self: self.U * self.rd**-2 * self.del1 * self.del2 *
                       np.real(1j*self.k*
                           (self.del1*self.ph1 + self.del2*self.ph2) *
                            np.conj(self.ph1 - self.ph2)).sum() / 
                            (self.nx*self.ny) )
        )

    def add_diagnostic(self, diag_name, description=None, units=None, function=None):
        # create a new diagnostic dict and add it to the object array
        
        # make sure the function is callable
        assert hasattr(function, '__call__')
        
        # make sure the name is valid
        assert isinstance(diag_name, str)
        
        # by default, diagnostic is active
        self.diagnostics[diag_name] = {
           'description': description,
           'units': units,
           'active': True,
           'count': 0,
           'function': function, }
           
    def _increment_diagnostics(self):
        # compute intermediate quantities needed for some diagnostics
        self.p1 = self.ifft2( self.ph1)
        self.p2 = self.ifft2( self.ph2)
        self.xi1 =self.ifft2( -self.wv2*self.ph1)
        self.xi2 =self.ifft2( -self.wv2*self.ph2)
        self.Jptpc = -self.advect(
                    (self.p1 - self.p2),
                    (self.del1*self.u1 + self.del2*self.u2),
                    (self.del1*self.v1 + self.del2*self.v2))
        # fix for delta.neq.1
        self.Jp1xi1 = self.advect(self.xi1, self.u1, self.v1)
        self.Jp2xi2 = self.advect(self.xi2, self.u2, self.v2)
        
        for dname in self.diagnostics:
            if self.diagnostics[dname]['active']:
                res = self.diagnostics[dname]['function'](self)
                if self.diagnostics[dname]['count']==0:
                    self.diagnostics[dname]['value'] = res
                else:
                    self.diagnostics[dname]['value'] += res
                self.diagnostics[dname]['count'] += 1
                
    def get_diagnostic(self, dname):
        return (self.diagnostics[dname]['value'] / 
                self.diagnostics[dname]['count'])


# DFT functions
# def fft2(self, a):
#     if self.fftw:
#         aw = pyfftw.n_byte_align_empty(a.shape, 8, 'float64')
#         aw[:]= a.copy()
#         return pyfftw.builders.rfft2(aw,threads=self.ntd)()
#     else:
#         return np.fft.rfft2(a)
#
# def ifft2(self, ah):
#     if self.fftw:
#         awh = pyfftw.n_byte_align_empty(ah.shape, 16, 'complex128')
#         awh[:]= ah.copy()
#         return pyfftw.builders.irfft2(awh,threads=self.ntd)()
#     else:
#         return np.fft.irfft2(ah)

# some off-class diagnostics 
def spec_var(self,ph):
    """ compute variance of p from Fourier coefficients ph """
    var_dens = 2. * np.abs(ph)**2 / self.M**2 
    # only half of coefs [0] and [nx/2+1] due to symmetry in real fft2
    var_dens[:,0],var_dens[:,-1] = var_dens[:,0]/2.,var_dens[:,-1]/2.
    return var_dens.sum()


# general purpose timestepping routines
def tendency_forward_euler(dt, dqdt):
    """Compute tendency using forward euler timestepping."""
    return dt * dqdt

def tendency_ab2(dt, dqdt, dqdt_p):
    """Compute tendency using Adams Bashforth 2nd order timestepping."""
    return (1.5*dt) * dqdt + (-0.5*dt) * dqdt_p

def tendency_ab3(dt, dqdt, dqdt_p, dqdt_pp):
    """Compute tendency using Adams Bashforth 3nd order timestepping."""
    return (23/12.*dt) * dqdt + (-16/12.*dt) * dqdt_p + (5/12.*dt) * dqdt_p 
