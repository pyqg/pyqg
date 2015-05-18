import numpy as np
import spectral_kernel
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

class Model(spectral_kernel.PseudoSpectralKernel):
    """A class that represents a generic pseudo-spectral inversion model."""
    
    def __init__(
        self,
        # grid size parameters
        nx=64,                     # grid resolution
        ny=None,
        L=1e6,                     # domain size is L [m]
        W=None,
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
        teststyle = False,            # use fftw with "estimate" planner to get reproducibility
        ntd = 1,                    # number of threads to use in fftw computations
        quiet = False,
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
        use_fftw  -- if True fftw is used with "estimate" planner to get reproducibility (hopefully)
        useAB2 -- use second order Adams Bashforth timestepping instead of third
        """

        if ny is None: ny = nx
        if W is None: W = L
       
        # put all the parameters into the object
        # grid
        self.nx = nx
        self.ny = ny
        self.nz = 2     # two layers
        self.L = L
        self.W = W

        # timestepping
        self.dt = dt
        self.tplot = tplot
        self.twrite = twrite
        self.tmax = tmax
        self.tavestart = tavestart
        self.taveint = taveint
        self.tpickup = tpickup
        self.quiet = quiet
        self.useAB2 = useAB2
        # fft 
        self.use_fftw = use_fftw
        self.teststyle= teststyle
        self.ntd = ntd

        self._initialize_grid()
        self._initialize_fft()
        self._initialize_background()
        self._initialize_forcing()
        self._initialize_inversion_matrix()
        #self._initialize_state_variables()
        self._initialize_kernel()
        self._initialize_time()                
       
        self._initialize_diagnostics()
        if diagnostics_list == 'all':
            pass # by default, all diagnostics are active
        elif diagnostics_list == 'none':
            self.set_active_diagnostics([])
        else:
            self.set_active_diagnostics(diagnostics_list)
   
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
                
    ### PRIVATE METHODS - not meant to be called by user ###

    def _step_forward(self):

        # the basic steps are
        self._print_status() 

        self._invert()
        # find streamfunction from pv

        #self._advection_tendency()
        self.dqhdt_adv = self._advection_tendency()
        # use streamfunction to calculate advection tendency
        
        self._forcing_tendency()
        # apply friction and external forcing
        
        #self._calc_diagnostics()
        # do what has to be done with diagnostics
        
        self._forward_timestep()
        # apply tendencies to step the model forward
        # (filter gets called here)
        
    def _initialize_time(self):
        """Set up timestep stuff"""
        self.t=0        # actual time
        self.tc=0       # timestep number
        self.taveints = np.ceil(self.taveint/self.dt)      
        
    ### initialization routines, only called once at the beginning ###
    def _initialize_grid(self):
        """Set up spatial and spectral grids and related constants"""
        self.x,self.y = np.meshgrid(
            np.arange(0.5,self.nx,1.)/self.nx*self.L,
            np.arange(0.5,self.ny,1.)/self.ny*self.W )

        # Notice: at xi=1 U=beta*rd^2 = c for xi>1 => U>c
        # wavenumber one (equals to dkx/dky)
        self.dk = 2.*pi/self.L
        self.dl = 2.*pi/self.W

        # wavenumber grids
        self.nl = self.ny
        self.nk = self.nx/2+1
        self.ll = self.dl*np.append( np.arange(0.,self.nx/2), 
            np.arange(-self.nx/2,0.) )
        self.kk = self.dk*np.arange(0.,self.nk)

        self.k, self.l = np.meshgrid(self.kk, self.ll)
        # complex versions, speeds up computations to precompute
        self.kj = 1j*self.k
        self.lj = 1j*self.l
        # physical grid spacing
        self.dx = self.L / self.nx
        self.dy = self.W / self.ny

        # constant for spectral normalizations
        self.M = self.nx*self.ny
        
        # isotropic wavenumber^2 grid
        # the inversion is not defined at kappa = 0 
        self.wv2 = self.k**2 + self.l**2
        self.wv = np.sqrt( self.wv2 )

        iwv2 = self.wv2 != 0.
        self.wv2i = np.zeros_like(self.wv2)
        self.wv2i[iwv2] = self.wv2[iwv2]**-1
        
    def _initialize_background(self):        
        raise NotImplementedError(
            'needs to be implemented by Model subclass')
        
    def _initialize_inversion_matrix(self):
        raise NotImplementedError(
            'needs to be implemented by Model subclass')
            
    def _initialize_kernel(self):
        #super(spectral_kernel.PseudoSpectralKernel, self).__init__(
        self._kernel_init(
            self.nz, self.ny, self.nx,
            self.a, self.kk, self.ll,
            self.Ubg, self.Qy
        )
        
        # still need to initialize a few state variables here, outside kernel
        # this is sloppy
        self.dqhdt_forc = np.zeros_like(self.qh)
        self.dqhdt_p = np.zeros_like(self.qh)
        self.dqhdt_pp = np.zeros_like(self.qh)
        
        
    def _initialize_forcing(self):
        raise NotImplementedError(
            'needs to be implemented by Model subclass')
            
    def _initialize_state_variables(self):
        """Set up model basic state variables.
        This is universal. All models should store their state this way."""
        # shape and datatype of data
        dtype_real = np.dtype('float64')            
        dtype_cplx = np.dtype('complex128')
        if self.nz > 1:
            shape_real = (self.nz, self.ny, self.nx)
            shape_cplx = (self.nz, self.nl, self.nk)
        else:
            shape_real = (self.ny, self.nx)
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
            
    def _initialize_fft(self):
        # set up fft functions for use later
        if self.use_fftw:
            
            if self.teststyle:
                ## drop in replacement for numpy fft
                ## more than twice as fast as the previous code
                self.fft2 = (lambda x :
                        pyfftw.interfaces.numpy_fft.rfft2(x, threads=self.ntd, planner_effort='FFTW_ESTIMATE'))
                self.ifft2 = (lambda x :
                        pyfftw.interfaces.numpy_fft.irfft2(x, threads=self.ntd, planner_effort='FFTW_ESTIMATE'))
            else:
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
    
    # def set_q(self, q):
    #     """This should work with all models."""
    #     self.q = q
    #     self.qh = self.fft2(self.q)

    # compute advection in grid space (returns qdot in fourier space)
    # *** don't remove! needed for diagnostics (but not forward model) ***
    def advect(self, q, u, v):
        return self.kj*self.fft2(u*q) + self.lj*self.fft2(v*q)

    def _invert_old(self):
        """invert qgpv to find streamfunction."""
        raise NotImplementedError(
            'needs to be implemented by Model subclass')
                        
    def _advection_tendency_old(self):
        """Calculate tendency due to advection."""
        # this is actually the same for all Models
        
        # compute real space qgpv and velocity
        self.q = self.ifft2(self.qh)
        # multiply velocity and qgpv to get fluxes
        uq = np.multiply(self.q, self.u)
        vq = np.multiply(self.q, self.v)
        ddx_uq = self.kj * self.fft2(uq)
        # derivatives in spectral space (including background advection)
        ddx_uq = self.kj * self.fft2(uq) + (self.ilQx * self.ph)
        ddy_vq = self.lj * self.fft2(vq) + (self.ikQy * self.ph)
        # convergence of advective flux
        self.dqhdt_adv = -(ddx_uq + ddy_vq)

    def _forcing_tendency(self):
        """Calculate tendency due to forcing."""
        raise NotImplementedError(
            'needs to be implemented by Model subclass')

    def _filter(self, q):
        """Apply filter to field q."""
        return q
        
    def _calc_derived_fields(self):
        """For use by diagnostics."""
        pass
        
    def _print_status(self):
        """Output some basic stats."""
        if (not self.quiet) and ((self.tc % self.twrite)==0) and self.tc>0.:
            ke = self._calc_ke()
            cfl = self._calc_cfl()
            print 't=%16d, tc=%10d: cfl=%5.6f, ke=%9.9f' % (
                   self.t, self.tc, cfl, ke)
            assert cfl<1., "CFL condition violated"

    def _calc_diagnostics(self):
        # here is where we calculate diagnostics
        if (self.t>=self.dt) and (self.tc%self.taveints==0):
            self._increment_diagnostics()

    def _forward_timestep(self):
        """Step forward based on tendencies"""
       

        self.dqhdt = self.dqhdt_adv + self.dqhdt_forc
              
        # Note that Adams-Bashforth is not self-starting
        if self.tc==0:
            # forward Euler at the first step
            qtend = tendency_forward_euler(self.dt, self.dqhdt)
        elif (self.tc==1) or (self.useAB2):
            # AB2 at step 2
            qtend = tendency_ab2(self.dt, self.dqhdt, self.dqhdt_p)
        else:
            # AB3 from step 3 on
            qtend = tendency_ab3(self.dt, self.dqhdt,
                        self.dqhdt_p, self.dqhdt_pp)
            
        # add tendency and filter
        self.set_qh(self._filter(self.qh + qtend))
        
        # remember previous tendencies
        self.dqhdt_pp = self.dqhdt_p.copy()
        self.dqhdt_p = self.dqhdt.copy()
                
        # augment timestep and current time
        self.tc += 1
        self.t += self.dt

    ### All the diagnostic stuff follows. ###
    def _calc_cfl(self):
        raise NotImplementedError(
            'needs to be implemented by Model subclass')
    
    # this is stuff the Cesar added
    
    # if self.tc==0:
    #     assert self.calc_cfl()<1., " *** time-step too large "
    #     # initialize ke and time arrays
    #     self.ke = np.array([self.calc_ke()])
    #     self.eddy_time = np.array([self.calc_eddy_time()])
    #     self.time = np.array([0.])
           

    def _calc_ke(self):
        raise NotImplementedError(
            'needs to be implemented by Model subclass')        

    def _set_active_diagnostics(self, diagnostics_list):
        for d in self.diagnostics:
            self.diagnostics[d]['active'] == (d in diagnostics_list)

    def _initialize_diagnostics(self):
        # Initialization for diagnotics
        self.diagnostics = dict()

        ## example diagnostics
        #self.add_diagnostic('entspec',
        #    description='barotropic enstrophy spectrum',
        #    function= (lambda self:
        #              np.abs(self.del1*self.qh[0] + self.del2*self.qh[1])**2.)
        #)
            
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
        
        self._calc_derived_fields()
        
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


# general purpose timestepping routines
def tendency_forward_euler(dt, dqdt):
    """Compute tendency using forward euler timestepping."""
    return dt * dqdt

def tendency_ab2(dt, dqdt, dqdt_p):
    """Compute tendency using Adams Bashforth 2nd order timestepping."""
    return (1.5*dt) * dqdt + (-0.5*dt) * dqdt_p

def tendency_ab3(dt, dqdt, dqdt_p, dqdt_pp):
    """Compute tendency using Adams Bashforth 3nd order timestepping."""
    return (23/12.*dt) * dqdt + (-16/12.*dt) * dqdt_p + (5/12.*dt) * dqdt_pp 
