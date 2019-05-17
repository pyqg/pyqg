# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from numpy import pi
import logging
import warnings

from .kernel import PseudoSpectralKernel, tendency_forward_euler, tendency_ab2, tendency_ab3
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

class Model(PseudoSpectralKernel):
    """A generic pseudo-spectral inversion model.

    Attributes
    ----------
    nx, ny : int
        Number of real space grid points in the `x`, `y` directions (cython)
    nk, nl : int
        Number of spectal space grid points in the `k`, `l` directions (cython)
    nz : int
        Number of vertical levels (cython)
    kk, ll : real array
        Zonal and meridional wavenumbers (`nk`) (cython)
    a : real array
        inversion matrix (`nk`, `nk`, `nl`, `nk`) (cython)
    q : real array
        Potential vorticity in real space (`nz`, `ny`, `nx`) (cython)
    qh : complex array
        Potential vorticity in spectral space (`nk`, `nl`, `nk`) (cython)
    ph : complex array
        Streamfunction in spectral space (`nk`, `nl`, `nk`) (cython)
    u, v : real array
        Zonal and meridional velocity anomalies in real space (`nz`, `ny`, `nx`) (cython)
    Ubg : real array
        Background zonal velocity (`nk`, `ny`) (cython)
    Qy : real array
        Background potential vorticity gradient (`nk`, `nl`) (cython)
    ufull, vfull : real arrays
        Zonal and meridional full velocities in real space (`nz`, `ny`, `nx`) (cython)
    uh, vh : complex arrays
        Velocity anomaly components in spectral space (`nk`, `nl`, `nk`) (cython)
    rek : float
        Linear drag in lower layer (cython)
    rbg : float
        Linear drag within each layer (cython)
    t : float
        Model time (cython)
    tc : int
        Model timestep (cython)
    dt : float
        Numerical timestep (cython)
    L, W : float
        Domain length in x and y directions
    filterfac : float
        Amplitdue of the spectral spherical filter
    twrite : int
        Interval for cfl writeout (units: number of timesteps)
    tmax : float
        Total time of integration (units: model time)
    tavestart : float
        Start time for averaging (units: model time)
    tsnapstart : float
        Start time for snapshot writeout (units: model time)
    taveint : float
        Time interval for accumulation of diagnostic averages.
        (units: model time)
    tsnapint : float
        Time interval for snapshots (units: model time)
    ntd : int
        Number of threads to use. Should not exceed the number of cores on
        your machine.
    pmodes : real array
        Vertical pressure modes (unitless)
    radii :  real array
        Deformation radii  (units: model length)
    """

    def __init__(
        self,
        # grid size parameters
        nz=1,
        nx=64,                     # grid resolution
        ny=None,
        L=1e6,                     # domain size is L [m]
        W=None,
        # timestepping parameters
        dt=7200.,                   # numerical timestep
        twrite=1000.,               # interval for cfl and ke writeout (in timesteps)
        tmax=1576800000.,           # total time of integration
        tavestart=315360000.,       # start time for averaging
        taveint=86400.,             # time interval used for summation in longterm average in seconds
        useAB2=False,               # use second order Adams Bashforth timestepping instead of 3rd
        # friction parameters
        rek=5.787e-7,               # linear drag in lower layer
        rbg=5.787e-7,               # linear drag in each layer
        filterfac=23.6,             # the factor for use in the exponential filter
        # constants
        f = None,                   # coriolis parameter (not necessary for two-layer model
                                    #  if deformation radius is provided)
        g= 9.81,                    # acceleration due to gravity
        # diagnostics parameters
        diagnostics_list='all',     # which diagnostics to output
        # fft parameters
        # removed because fftw is now manditory
        #use_fftw = False,               # fftw flag
        #teststyle = False,            # use fftw with "estimate" planner to get reproducibility
        ntd = 1,                       # number of threads to use in fftw computations
        log_level = 1,                 # logger level: from 0 for quiet (no log) to 4 for verbose
                                       #     logger (see  https://docs.python.org/2/library/logging.html)
        logfile = None,                # logfile; None prints to screen
        ):
        """
        .. note:: All of the test cases use ``nx==ny``. Expect bugs if you choose
                  these parameters to be different.
        .. note:: All time intervals will be rounded to nearest `dt` interval.

        Parameters
        ----------
        nx : int
            Number of grid points in the x direction.
        ny : int
            Number of grid points in the y direction (default: nx).
        L : number
            Domain length in x direction. Units: meters.
        W :
            Domain width in y direction. Units: meters (default: L).
        rek : number
            linear drag in lower layer. Units: seconds :sup:`-1`.
        filterfac : number
            amplitdue of the spectral spherical filter (originally 18.4, later
            changed to 23.6).
        dt : number
            Numerical timstep. Units: seconds.
        twrite : int
            Interval for cfl writeout. Units: number of timesteps.
        tmax : number
            Total time of integration. Units: seconds.
        tavestart : number
            Start time for averaging. Units: seconds.
        tsnapstart : number
            Start time for snapshot writeout. Units: seconds.
        taveint : number
            Time interval for accumulation of diagnostic averages.
            Units: seconds. (For performance purposes, averaging does not have to
            occur every timestep)
        tsnapint : number
            Time interval for snapshots. Units: seconds.
        ntd : int
            Number of threads to use. Should not exceed the number of cores on
            your machine.
        """

        if ny is None:
            ny = nx
        if W is None:
            W = L

        # TODO: be more clear about what attributes are cython and what
        # attributes are python
        PseudoSpectralKernel.__init__(self, nz, ny, nx, ntd)

        self.L = L
        self.W = W

        # timestepping
        self.dt = dt
        self.twrite = twrite
        self.tmax = tmax
        self.tavestart = tavestart
        self.taveint = taveint
        self.logfile = logfile
        self.log_level = log_level
        self.useAB2 = useAB2
        self.ntd = ntd

        # friction
        self.rek = rek
        self.rbg = rbg
        self.filterfac = filterfac

        # constants
        self.g = g
        if f:
            self.f = f
            self.f2 = f**2

        # TODO: make this less complicated!
        # Really we just need to initialize the grid here. It's not necessary
        # to have all these silly methods. Maybe we need "hooks" instead.
        self._initialize_logger()
        self._initialize_grid()
        self._initialize_background()
        self._initialize_forcing()
        self._initialize_filter()
        self._initialize_time()
        self._initialize_inversion_matrix()
        self._initialize_diagnostics(diagnostics_list)


    def run_with_snapshots(self, tsnapstart=0., tsnapint=432000.):
        """Run the model forward, yielding to user code at specified intervals.

        Parameters
        ----------

        tsnapstart : int
            The timestep at which to begin yielding.
        tstapint : int
            The interval at which to yield.
        """

        tsnapints = np.ceil(tsnapint/self.dt)

        while(self.t < self.tmax):
            self._step_forward()
            if self.t>=tsnapstart and (self.tc%tsnapints)==0:
                yield self.t
        return

    def run(self):
        """Run the model forward without stopping until the end."""
        while(self.t < self.tmax):
            self._step_forward()


    def vertical_modes(self):
        """ Calculate standard vertical modes. Simply
            the eigenvectors of the stretching matrix S """

        evals,evecs = np.linalg.eig(-self.S)

        asort = evals.argsort()

        # deformation wavenumbers and radii
        self.kdi2 = evals[asort]
        self.radii = np.zeros_like(self.kdi2)
        self.radii[0] = np.sqrt(self.g*self.H)/np.abs(self.f) # barotropic def. radius
        self.radii[1:] = 1./np.sqrt(self.kdi2[1:])

        # eigenstructure
        self.pmodes = evecs[:,asort]

        # normalize to have unit L2-norm
        Ai = (self.H / (self.Hi[:,np.newaxis]*(self.pmodes**2)).sum(axis=0))**0.5
        self.pmodes = Ai[np.newaxis,:]*self.pmodes

    def modal_projection(self,p,forward=True):
        """ Performs a field p into modal amplitudes pn
                using the basis [pmodes]. The inverse
                transform calculates p from pn"""

        if forward:
            pt = np.linalg.solve(self.pmodes[np.newaxis,np.newaxis],p.T).T
        else:
            pt = np.einsum("ik,k...->i...",self.pmodes,p)

        return pt

    def stability_analysis(self,bottom_friction=False):
        """ Performs the baroclinic linear instability analysis given
                given the base state velocity :math: `(U, V)` and
                the stretching matrix  :math: `S`:

        .. math:: A \Phi = \omega B \Phi,

        where

        .. math:: A = B (U k + V l) + I (k Q_y - l Q_x) +
                        1j \delta_{N N} r_{ek} I  \kappa^2

        where :math:`\delta_{N N} = [0,0,\dots,0,1] ,`

        and

        .. math:: B =  S - I \kappa^2 .


        The eigenstructure is

        .. math:: \Phi

        and the eigenvalue is

        .. math:: `\omega`

        The growth rate is Im\ :math:`\{\omega\}`.


        Parameters
        ----------
        bottom_friction: optional inclusion linear bottom drag
                         in the linear stability calculation
                         (default is False, as if :math: `r_{ek} = 0`)

        Returns
        -------
        omega: complex array
             The eigenvalues with largest complex part (units: inverse model time)
        phi: complex array
             The eigenvectors associated associated with \omega (unitless)

        """

        omega = np.zeros_like(self.wv)+0.j
        phi = np.zeros_like(self.qh)

        I = np.eye(self.nz)

        L2 = self.S[:,:,np.newaxis,np.newaxis] - self.wv2*I[:,:,np.newaxis,np.newaxis]

        Q =  I[:,:,np.newaxis,np.newaxis]*(self.ikQy - self.ilQx).imag

        Uk =(self.Ubg*I[:,:,np.newaxis])[:,:,:,np.newaxis]*self.k
        Vl =(self.Vbg*I)[:,:,np.newaxis,np.newaxis]*self.l
        L3 = np.einsum('ij...,jk...->ik...',L2,Uk+Vl) + 0j

        if bottom_friction:
            L3[-1,-1,:,:] += 1j*self.rek*self.wv2

        L4 = self.a.T

        M = np.einsum('...ij,...jk->...ik',L4,(L3+Q).T)

        evals,evecs = np.linalg.eig(M)
        evals, evecs = evals.T, evecs.T

        # sorting things this way proved way
        #  more faster than using numpy's argsort() !
        imax = evals.imag.argmax(axis=0)
        for i in range(self.nl):
            for j in range(self.nk):
                omega[i,j] = evals[imax[i,j],i,j]
                phi[:,i,j] = evecs[imax[i,j],:,i,j]

        return omega, phi

    ### PRIVATE METHODS - not meant to be called by user ###

    def _step_forward(self):
        self._invert()
        # find streamfunction from pv

        self._do_advection()
        # use streamfunction to calculate advection tendency

        self._do_friction()
        # apply friction
        self._do_viscosity()

        self._do_external_forcing()
        # apply external forcing

        self._calc_diagnostics()
        # do what has to be done with diagnostics

        self._forward_timestep()
        # apply tendencies to step the model forward
        # (filter gets called here)

        # the basic steps are
        self._print_status()

    def _initialize_time(self):
        """Set up timestep stuff"""
        #self.t=0        # actual time
        #self.tc=0       # timestep number
        self.taveints = np.ceil(self.taveint/self.dt)

    ### initialization routines, only called once at the beginning ###
    # TODO: clean up and simplify this whole routine
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
        # set in kernel
        #self.nl = self.ny
        #self.nk = int(self.nx/2+1)
        self.ll = self.dl*np.append( np.arange(0.,self.nx/2),
            np.arange(-self.nx/2,0.) )
        self.kk = self.dk*np.arange(0.,self.nk)

        self.k, self.l = np.meshgrid(self.kk, self.ll)
        self.ik = 1j*self.k
        self.il = 1j*self.l
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

    def _initialize_forcing(self):
        raise NotImplementedError(
            'needs to be implemented by Model subclass')

    def _initialize_filter(self):
        """Set up frictional filter."""
        # this defines the spectral filter (following Arbic and Flierl, 2003)
        cphi=0.65*pi
        wvx=np.sqrt((self.k*self.dx)**2.+(self.l*self.dy)**2.)
        filtr = np.exp(-self.filterfac*(wvx-cphi)**4.)
        filtr[wvx<=cphi] = 1.
        self.filtr = filtr

    def _filter(self, q):
        return self.filtr * q

    def _do_external_forcing(self):
        pass

    # logger
    def _initialize_logger(self):

        self.logger = logging.getLogger(__name__)


        if not (self.logfile is None):
            fhandler = logging.FileHandler(filename=self.logfile, mode='w')
        else:
            fhandler = logging.StreamHandler()

        formatter = logging.Formatter('%(levelname)s: %(message)s')

        fhandler.setFormatter(formatter)

        if not self.logger.handlers:
            self.logger.addHandler(fhandler)

        self.logger.setLevel(self.log_level*10)

        # this prevents the logger to propagate into the ipython notebook log
        self.logger.propagate = False

        self.logger.info(' Logger initialized')


    # compute advection in grid space (returns qdot in fourier space)
    # *** don't remove! needed for diagnostics (but not forward model) ***
    def _advect(self, q, u, v):
        """Given real inputs q, u, v, returns the advective tendency for
        q in spectal space."""
        uq = u*q
        vq = v*q
        # this is a hack, since fft now requires input to have shape (nz,ny,nx)
        # it does an extra unnecessary fft
        is_2d = (uq.ndim==2)
        if is_2d:
            uq = np.tile(uq[np.newaxis,:,:], (self.nz,1,1))
            vq = np.tile(vq[np.newaxis,:,:], (self.nz,1,1))
        tend = self.ik*self.fft(uq) + self.il*self.fft(vq)
        if is_2d:
            return tend[0]
        else:
            return tend

    # def _filter(self, q):
    #     """Apply filter to field q."""
    #     return q

    def _print_status(self):
        """Output some basic stats."""
        if (self.log_level) and ((self.tc % self.twrite)==0):
            self.ke = self._calc_ke()
            self.cfl = self._calc_cfl()
            #print 't=%16d, tc=%10d: cfl=%5.6f, ke=%9.9f' % (
            #       self.t, self.tc, cfl, ke)
            self.logger.info('Step: %i, Time: %3.2e, KE: %3.2e, CFL: %4.3f'
                    , self.tc,self.t,self.ke,self.cfl )

            assert self.cfl<1., self.logger.error('CFL condition violated')


    def _calc_diagnostics(self):
        # here is where we calculate diagnostics
        if (self.t>=self.dt) and (self.t>=self.tavestart) and (self.tc%self.taveints==0):
            self._increment_diagnostics()

    # def _forward_timestep(self):
    #     """Step forward based on tendencies"""
    #
    #     #self.dqhdt = self.dqhdt_adv + self.dqhdt_forc
    #
    #     # Note that Adams-Bashforth is not self-starting
    #     if self.tc==0:
    #         # forward Euler at the first step
    #         qtend = tendency_forward_euler(self.dt, self.dqhdt)
    #     elif (self.tc==1) or (self.useAB2):
    #         # AB2 at step 2
    #         qtend = tendency_ab2(self.dt, self.dqhdt, self.dqhdt_p)
    #     else:
    #         # AB3 from step 3 on
    #         qtend = tendency_ab3(self.dt, self.dqhdt,
    #                     self.dqhdt_p, self.dqhdt_pp)
    #
    #     # add tendency and filter
    #     self.set_qh(self._filter(self.qh + qtend))
    #
    #     # remember previous tendencies
    #     self.dqhdt_pp[:] = self.dqhdt_p.copy()
    #     self.dqhdt_p[:] = self.dqhdt.copy()
    #     #self.dqhdt[:] = 0.
    #
    #     # augment timestep and current time
    #     self.tc += 1
    #     self.t += self.dt

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

    def _initialize_diagnostics(self, diagnostics_list):
        # Initialization for diagnotics
        self.diagnostics = dict()

        self._initialize_core_diagnostics()
        self._initialize_model_diagnostics()

        if diagnostics_list == 'all':
            pass # by default, all diagnostics are active
        elif diagnostics_list == 'none':
            self.set_active_diagnostics([])
        else:
            self.set_active_diagnostics(diagnostics_list)

    def _initialize_core_diagnostics(self):
        """Diagnostics common to all models."""
        self.add_diagnostic('Ensspec',
            description='enstrophy spectrum',
            function= (lambda self: np.abs(self.qh)**2/self.M**2)
        )

        self.add_diagnostic('KEspec',
            description=' kinetic energy spectrum',
            function=(lambda self: self.wv2*np.abs(self.ph)**2/self.M**2)
        )      # factor of 2 to account for the fact that we have only half of
               #    the Fourier coefficients.

        self.add_diagnostic('q',
            description='QGPV',
            function= (lambda self: self.q)
        )

        self.add_diagnostic('EKEdiss',
            description='total energy dissipation by bottom drag',
            function= (lambda self: self.Hi[-1]/self.H*self.rek*(self.v[-1]**2 + self.u[-1]**2).mean())
        )

        self.add_diagnostic('EKE',
            description='mean eddy kinetic energy',
            function= (lambda self: 0.5*(self.v**2 + self.u**2).mean(axis=-1).mean(axis=-1))
        )

    def _calc_derived_fields(self):
        """Should be implemented by subclass."""
        pass

    def _initialize_model_diagnostics(self):
        """Should be implemented by subclass."""
        pass

    def _set_active_diagnostics(self, diagnostics_list):
        for d in self.diagnostics:
            self.diagnostics[d]['active'] == (d in diagnostics_list)

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

    def describe_diagnostics(self):
        """Print a human-readable summary of the available diagnostics."""
        diag_names = list(self.diagnostics.keys())
        diag_names.sort()
        print('NAME               | DESCRIPTION')
        print(80*'-')
        for k in diag_names:
            d = self.diagnostics[k]
            print('{:<10} | {:<54}'.format(
                 *(k,  d['description'])))

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

    def spec_var(self, ph):
        """ compute variance of p from Fourier coefficients ph """
        var_dens = 2. * np.abs(ph)**2 / self.M**2
        # only half of coefs [0] and [nx/2+1] due to symmetry in real fft2
        var_dens[...,0] = var_dens[...,0]/2.
        var_dens[...,-1] = var_dens[...,-1]/2.
        return var_dens.sum()


    def set_qh(self, qh):
        warnings.warn("Method deprecated. Set model.qh directly instead. ",
            DeprecationWarning)
        self.qh = qh


    def set_q(self, q):
        warnings.warn("Method deprecated. Set model.q directly instead. ",
            DeprecationWarning)
        self.q = q
