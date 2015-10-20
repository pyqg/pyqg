#cython: profile=True
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#from __future__ import division
import numpy as np
import warnings
import cython
cimport numpy as np
from cython.parallel import prange, threadid

# see if we got a compile time flag
include '.compile_time_use_pyfftw.pxi'
IF PYQG_USE_PYFFTW:
    import pyfftw
    pyfftw.interfaces.cache.enable()
ELSE:
    import numpy.fft as npfft
    warnings.warn('No pyfftw detected. Using numpy.fft')

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE_real = np.float64
DTYPE_com = np.complex128
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.float64_t DTYPE_real_t
ctypedef np.complex128_t DTYPE_com_t

cdef class PseudoSpectralKernel:
    # array shapes
    cdef public int Nx, Ny, Nz
    cdef public int Nk, Nl

    ### the main state variables (memory views to numpy arrays) ###
    # pv
    cdef DTYPE_real_t [:, :, :] q
    cdef DTYPE_com_t [:, :, :] qh
    # streamfunction
    cdef DTYPE_com_t [:, :, :] ph
    # velocities
    cdef DTYPE_real_t [:, :, :] u
    cdef DTYPE_real_t [:, :, :] v
    cdef DTYPE_com_t [:, :, :] uh
    cdef DTYPE_com_t [:, :, :] vh
    # pv fluxes
    cdef DTYPE_real_t [:, :, :] uq
    cdef DTYPE_real_t [:, :, :] vq
    cdef DTYPE_com_t [:, :, :] uqh
    cdef DTYPE_com_t [:, :, :] vqh
    # the tendencies
    cdef DTYPE_com_t [:, :, :] dqhdt
    cdef DTYPE_com_t [:, :, :] dqhdt_p
    cdef DTYPE_com_t [:, :, :] dqhdt_pp

    # dummy variables for diagnostic ffts
    cdef DTYPE_real_t [:, :, :] _dummy_fft_in
    cdef DTYPE_com_t [:, :, :] _dummy_fft_out
    cdef DTYPE_real_t [:, :, :] _dummy_ifft_out
    cdef DTYPE_com_t [:, :, :] _dummy_ifft_in

    # the variables needed for inversion and advection
    # store a as complex so we don't have to typecast in inversion
    cdef DTYPE_com_t [:, :, :, :] a
    cdef DTYPE_com_t [:] _ik
    cdef DTYPE_com_t [:] _il
    cdef public DTYPE_real_t [:,:] _k2l2
    # background state constants (functions of z only)
    cdef DTYPE_real_t [:] Ubg
    cdef DTYPE_real_t [:] Vbg
    cdef DTYPE_com_t [:, :] _ikQy
    cdef DTYPE_com_t [:, :] _ilQx
    # topography
    cdef public DTYPE_real_t [:, :] _hb
    # spectral filter
    cdef public DTYPE_real_t [:, :] _filtr

    # friction parameter
    cdef public DTYPE_real_t _rek

    # time
    cdef public int tc
    cdef public DTYPE_real_t _dt
    cdef public DTYPE_real_t t

    # threading
    cdef int num_threads
    # number of elements per work group in the y / l direction
    cdef int chunksize

    # pyfftw objects (callable)
    cdef object fft_q_to_qh
    cdef object ifft_qh_to_q
    cdef object ifft_uh_to_u
    cdef object ifft_vh_to_v
    cdef object fft_uq_to_uqh
    cdef object fft_vq_to_vqh
    cdef object _dummy_fft
    cdef object _dummy_ifft

    def _kernel_init(self, int Nz, int Ny, int Nx,
                    np.ndarray[DTYPE_real_t, ndim=4] a,
                    np.ndarray[DTYPE_real_t, ndim=1] k,
                    np.ndarray[DTYPE_real_t, ndim=1] l,
                    np.ndarray[DTYPE_real_t, ndim=1] Ubg,
                    np.ndarray[DTYPE_real_t, ndim=1] Vbg,
                    np.ndarray[DTYPE_real_t, ndim=1] Qy,
                    np.ndarray[DTYPE_real_t, ndim=1] Qx,
                    np.ndarray[DTYPE_real_t, ndim=2] hb,
                    np.ndarray[DTYPE_real_t, ndim=2] filtr,
                    DTYPE_real_t dt=1.0,
                    DTYPE_real_t rek=0.0,
                    fftw_num_threads=1,
    ):
        self.Nz = Nz
        self.Ny = Ny
        self.Nx = Nx
        self.Nl = Ny
        self.Nk = Nx/2 + 1

        self._rek = rek

        ### none of this shape checking works
        #assert a.shape == (self.Nz, self.Nz self.Nl, self.Nk):
        #assert k.shape == (self.Nk,):
        #assert l.shape == (self.Nl,):
        #assert Ubg.shape == (self.Nz,), 'Ubg is the wrong shape'
        #assert Vbg.shape == (self.Nz,), 'Vbg is the wrong shape'
        #assert Qx.shape == (self.Nz,), 'Qx is the wrong shape'
        #assert Qy.shape == (self.Nz,), 'Qy is the wrong shape'

        # assign a, _ik, _il
        self.a = a.astype(DTYPE_com)
        self._ik = 1j*k
        self._il = 1j*l

        self._k2l2 = np.zeros((self.Nl, self.Nk), DTYPE_real)
        for j in range(self.Nl):
            for i in range(self.Nk):
                self._k2l2[j,i] = k[i]**2 + l[j]**2

        # assign Ubg, Vbg, _ilQx, _ikQy
        self.Ubg = Ubg
        self.Vbg = Vbg
        self._ikQy = 1j * k[np.newaxis, :] * Qy[:, np.newaxis]
        self._ilQx = 1j * l[np.newaxis, :] * Qx[:, np.newaxis]

        # assign topography
        self._hb = hb

        # initialize FFT inputs / outputs as byte aligned by pyfftw
        q = self._empty_real()
        self.q = q # assign to memory view
        qh = self._empty_com()
        self.qh = qh

        ph = self._empty_com()
        self.ph = ph

        u = self._empty_real()
        self.u = u
        uh = self._empty_com()
        self.uh = uh

        v = self._empty_real()
        self.v = v
        vh = self._empty_com()
        self.vh = vh

        uq = self._empty_real()
        self.uq = uq
        uqh = self._empty_com()
        self.uqh = uqh

        vq = self._empty_real()
        self.vq = vq
        vqh = self._empty_com()
        self.vqh = vqh

        # dummy variables for diagnostic ffts
        dfftin = self._empty_real()
        self._dummy_fft_in = dfftin
        dfftout = self._empty_com()
        self._dummy_fft_out = dfftout
        difftin = self._empty_com()
        self._dummy_ifft_in = difftin
        difftout = self._empty_real()
        self._dummy_ifft_out = difftout

        # the tendency
        self.dqhdt = self._empty_com()
        self.dqhdt_p = self._empty_com()
        self.dqhdt_pp = self._empty_com()

        # spectral filter
        self._filtr = filtr

        # time stuff
        self._dt = dt
        self.tc = 0
        self.t = 0.0

        # for threading
        self.num_threads = fftw_num_threads
        self.chunksize = self.Nl/self.num_threads

        IF PYQG_USE_PYFFTW:
            # set up FFT plans
            # Note that the Backwards Real transform for the case
            # in which the dimensionality of the transform is greater than 1
            # will destroy the input array. This is inherent to FFTW and the only
            # general work-around for this is to copy the array prior to
            # performing the transform.
            self.fft_q_to_qh = pyfftw.FFTW(q, qh, threads=fftw_num_threads,
                             direction='FFTW_FORWARD', axes=(-2,-1))
            self.ifft_qh_to_q = pyfftw.FFTW(qh, q, threads=fftw_num_threads,
                             direction='FFTW_BACKWARD', axes=(-2,-1))
            self.ifft_uh_to_u = pyfftw.FFTW(uh, u, threads=fftw_num_threads,
                             direction='FFTW_BACKWARD', axes=(-2,-1))
            self.ifft_vh_to_v = pyfftw.FFTW(vh, v, threads=fftw_num_threads,
                             direction='FFTW_BACKWARD', axes=(-2,-1))
            self.fft_uq_to_uqh = pyfftw.FFTW(uq, uqh, threads=fftw_num_threads,
                             direction='FFTW_FORWARD', axes=(-2,-1))
            self.fft_vq_to_vqh = pyfftw.FFTW(vq, vqh, threads=fftw_num_threads,
                             direction='FFTW_FORWARD', axes=(-2,-1))
            # dummy ffts for diagnostics
            self._dummy_fft = pyfftw.FFTW(dfftin, dfftout, threads=fftw_num_threads,
                             direction='FFTW_FORWARD', axes=(-2,-1))
            self._dummy_ifft = pyfftw.FFTW(difftin, difftout, threads=fftw_num_threads,
                             direction='FFTW_BACKWARD', axes=(-2,-1))

    # otherwise define those functions using numpy
    IF PYQG_USE_PYFFTW==0:
        def fft_q_to_qh(self):
            self.qh = npfft.rfftn(self.q, axes=(-2,-1))
        def ifft_qh_to_q(self):
            self.q = npfft.irfftn(self.qh, axes=(-2,-1))
        def ifft_uh_to_u(self):
            self.u = npfft.irfftn(self.uh, axes=(-2,-1))
        def ifft_vh_to_v(self):
            self.v = npfft.irfftn(self.vh, axes=(-2,-1))
        def fft_uq_to_uqh(self):
            self.uqh = npfft.rfftn(self.uq, axes=(-2,-1))
        def fft_vq_to_vqh(self):
            self.vqh = npfft.rfftn(self.vq, axes=(-2,-1))
        def _dummy_fft(self):
            self._dummy_fft_out = npfft.rfftn(self._dummy_fft_in, axes=(-2,-1))
        def _dummy_ifft(self):
            self._dummy_ifft_out = npfft.irfftn(self._dummy_ifft_in, axes=(-2,-1))

    def _empty_real(self):
        """Allocate a space-grid-sized variable for use with fftw transformations."""
        shape = (self.Nz, self.Ny, self.Ny)
        IF PYQG_USE_PYFFTW:
            return pyfftw.n_byte_align_empty(shape,
                                 pyfftw.simd_alignment, dtype=DTYPE_real)
        ELSE:
            return np.empty(shape, dtype=DTYPE_real)

    def _empty_com(self):
        """Allocate a Fourier-grid-sized variable for use with fftw transformations."""
        shape = (self.Nz, self.Nl, self.Nk)
        IF PYQG_USE_PYFFTW:
            return pyfftw.n_byte_align_empty(shape,
                                 pyfftw.simd_alignment, dtype=DTYPE_com)
        ELSE:
            return np.empty(shape, dtype=DTYPE_com)

    def fft(self, np.ndarray[DTYPE_real_t, ndim=3] v):
        """"Generic FFT function for real grid-sized variables.
        Not used for actual model ffs."""
        cdef  DTYPE_real_t [:, :, :] v_view = v
        # copy input into memory view
        self._dummy_fft_in[:] = v_view
        self._dummy_fft()
        # return a copy of the output
        return np.asarray(self._dummy_fft_out).copy()

    def ifft(self, np.ndarray[DTYPE_com_t, ndim=3] v):
        """"Generic IFFT function for complex grid-sized variables.
        Not used for actual model ffs."""
        cdef  DTYPE_com_t [:, :, :] v_view = v
        # copy input into memory view
        self._dummy_ifft_in[:] = v_view
        self._dummy_ifft()
        # return a copy of the output
        return np.asarray(self._dummy_ifft_out).copy()

    # the only way to set q and qh
    def set_qh(self, np.ndarray[DTYPE_com_t, ndim=3] b):
        cdef  DTYPE_com_t [:, :, :] b_view = b
        self.qh[:] = b_view
        self.ifft_qh_to_q()
        # input might have been destroyed, have to re-assign
        self.qh[:] = b_view

    def set_q(self, np.ndarray[DTYPE_real_t, ndim=3] b):
        cdef  DTYPE_real_t [:, :, :] b_view = b
        self.q[:] = b_view
        self.fft_q_to_qh()

    def _invert(self):
        self.__invert()

    cdef void __invert(self) nogil:
        ### algorithm
        # invert ph = a * qh
        # uh, vh = -_il * ph, _ik * ph
        # u, v, = ifft(uh), ifft(vh)

        cdef Py_ssize_t k, k1, k2, j, i
        # set ph to zero
        for k in range(self.Nz):
            for j in prange(self.Nl, nogil=True, schedule='static',
                      chunksize=self.chunksize,
                      num_threads=self.num_threads):
                for i in range(self.Nk):
                    self.ph[k,j,i] = (0. + 0.*1j)

        # invert qh to find ph
        for k2 in range(self.Nz):
            for k1 in range(self.Nz):
                for j in prange(self.Nl, nogil=True, schedule='static',
                          chunksize=self.chunksize,
                          num_threads=self.num_threads):
                    for i in range(self.Nk):
                        self.ph[k2,j,i] = ( self.ph[k2,j,i] +
                            self.a[k2,k1,j,i] * self.qh[k1,j,i] )

        # calculate spectral velocities
        for k in range(self.Nz):
            for j in prange(self.Nl, nogil=True, schedule='static',
                      chunksize=self.chunksize,
                      num_threads=self.num_threads):
                for i in range(self.Nk):
                    self.uh[k,j,i] = -self._il[j] * self.ph[k,j,i]
                    self.vh[k,j,i] =  self._ik[i] * self.ph[k,j,i]

        # transform to get u and v
        with gil:
            #self.ifft_qh_to_q() # necessary now that timestepping is inside kernel
            self.ifft_uh_to_u()
            self.ifft_vh_to_v()

        return

    def _do_advection(self):
        self.__do_advection()

    cdef void __do_advection(self) nogil:
        ### algorithm
        # uq, vq = (u+Ubg)*q, (v+Vbg)*q
        # uqh, vqh, = fft(uq), fft(vq)
        # tend = kj*uqh + _ilQx*ph + lj*vqh + _ilQy*ph

        # the output array: spectal representation of advective tendency
        #cdef np.ndarray tend = np.zeros((self.Nz, self.Nl, self.Nk), dtype=DTYPE_com)

        cdef Py_ssize_t k, j, i

        # multiply to get advective flux in space
        for k in range(self.Nz):
            for j in prange(self.Ny, nogil=True, schedule='static',
                      chunksize=self.chunksize,
                      num_threads=self.num_threads):
                for i in range(self.Nx):
                    self.uq[k,j,i] = (self.u[k,j,i]+self.Ubg[k]) * self.q[k,j,i]
                    self.vq[k,j,i] = (self.v[k,j,i]+self.Vbg[k]) * self.q[k,j,i]


        # add topographic term
        for j in prange(self.Ny, nogil=True, schedule='static',
                  chunksize=self.chunksize,
                  num_threads=self.num_threads):
            for i in range(self.Nx):
                self.uq[self.Nz-1,j,i] += (self.u[self.Nz-1,j,i] +
                        self.Ubg[self.Nz-1]) * self._hb[j,i]
                self.vq[self.Nz-1,j,i] += (self.v[self.Nz-1,j,i] +
                        self.Vbg[self.Nz-1]) * self._hb[j,i]

        # transform to get spectral advective flux
        with gil:
            self.fft_uq_to_uqh()
            self.fft_vq_to_vqh()

        # spectral divergence
        for k in range(self.Nz):
            for j in prange(self.Nl, nogil=True, schedule='static',
                      chunksize=self.chunksize,
                      num_threads=self.num_threads):
                for i in range(self.Nk):
                    # overwrite the tendency, since the forcing gets called after
                    self.dqhdt[k,j,i] = -( self._ik[i] * self.uqh[k,j,i] +
                                    self._il[j] * self.vqh[k,j,i] +
                                    self._ikQy[k,i] * self.ph[k,j,i] )
        return

    def _do_friction(self):
        self.__do_friction()

    cdef void __do_friction(self) nogil:
        """Apply Ekman friction to lower layer tendency"""
        cdef Py_ssize_t k = self.Nz-1
        cdef Py_ssize_t j, i
        if self._rek:
            for j in prange(self.Nl, nogil=True, schedule='static',
                      chunksize=self.chunksize,
                      num_threads=self.num_threads):
                for i in range(self.Nk):
                    self.dqhdt[k,j,i] = (
                     self.dqhdt[k,j,i] +
                             (self._rek *
                             self._k2l2[j,i] *
                             self.ph[k,j,i]) )
        return

    def _forward_timestep(self):
        """Step forward based on tendencies"""
        self.__forward_timestep()

    cdef void __forward_timestep(self) nogil:

        #self.dqhdt = self.dqhdt_adv + self.dqhdt_forc
        cdef DTYPE_real_t dt1
        cdef DTYPE_real_t dt2
        cdef DTYPE_real_t dt3
        cdef Py_ssize_t k, j, i
        cdef DTYPE_com_t [:, :, :] qh_new
        with gil:
            qh_new = self.qh.copy()

        # Note that Adams-Bashforth is not self-starting
        if self.tc==0:
            # forward euler
            dt1 = self._dt
            dt2 = 0.0
            dt3 = 0.0
        elif self.tc==1:
            # AB2 at step 2
            dt1 = 1.5*self._dt
            dt2 = -0.5*self._dt
            dt3 = 0.0
        else:
            # AB3 from step 3 on
            dt1 = 23./12.*self._dt
            dt2 = -16./12.*self._dt
            dt3 = 5./12.*self._dt

        for k in range(self.Nz):
            for j in prange(self.Nl, nogil=True, schedule='static',
                      chunksize=self.chunksize,
                      num_threads=self.num_threads):
                for i in range(self.Nk):
                    qh_new[k,j,i] = self._filtr[j,i] * (
                        self.qh[k,j,i] +
                        dt1 * self.dqhdt[k,j,i] +
                        dt2 * self.dqhdt_p[k,j,i] +
                        dt3 * self.dqhdt_pp[k,j,i]
                    )
                    self.qh[k,j,i] = qh_new[k,j,i]
                    self.dqhdt_pp[k,j,i] = self.dqhdt_p[k,j,i]
                    self.dqhdt_p[k,j,i] = self.dqhdt[k,j,i]
                    #self.dqhdt[k,j,i] = 0.0

        # do FFT of new qh
        with gil:
            self.ifft_qh_to_q() # this destroys qh, need to assign again

        for k in range(self.Nz):
            for j in prange(self.Nl, nogil=True, schedule='static',
                      chunksize=self.chunksize,
                      num_threads=self.num_threads):
                for i in range(self.Nk):
                    self.qh[k,j,i] = qh_new[k,j,i]

        self.tc += 1
        self.t += self._dt
        return

    # attribute aliases: return numpy ndarray views of memory views
    property q:
        def __get__(self):
            return np.asarray(self.q)
    property qh:
        def __get__(self):
            return np.asarray(self.qh)
    property dqhdt:
        def __get__(self):
            return np.asarray(self.dqhdt)
    property dqhdt_p:
        def __get__(self):
            return np.asarray(self.dqhdt_p)
    property dqhdt_pp:
        def __get__(self):
            return np.asarray(self.dqhdt_pp)
    property ph:
        def __get__(self):
            return np.asarray(self.ph)
    property u:
        def __get__(self):
            return np.asarray(self.u)
    property v:
        def __get__(self):
            return np.asarray(self.v)
    property ufull:
        def __get__(self):
            return np.asarray(self.u) + \
                np.asarray(self.Ubg)[:,np.newaxis,np.newaxis]
    property vfull:
        def __get__(self):
            return np.asarray(self.v)
    property uh:
        def __get__(self):
            return np.asarray(self.uh)
    property vh:
        def __get__(self):
            return np.asarray(self.vh)
    property uq:
        def __get__(self):
            return np.asarray(self.uq)
    property vq:
        def __get__(self):
            return np.asarray(self.vq)


# general purpose timestepping routines
# take only complex values, since that what the state variables are
def tendency_forward_euler(DTYPE_real_t dt,
                    np.ndarray[DTYPE_com_t, ndim=3] dqdt):
    """Compute tendency using forward euler timestepping."""
    return dt * dqdt

def tendency_ab2(DTYPE_real_t dt,
                    np.ndarray[DTYPE_com_t, ndim=3] dqdt,
                    np.ndarray[DTYPE_com_t, ndim=3] dqdt_p):
    """Compute tendency using Adams Bashforth 2nd order timestepping."""
    cdef DTYPE_real_t DT1 = 1.5*dt
    cdef DTYPE_real_t DT2 = -0.5*dt
    return DT1 * dqdt + DT2 * dqdt_p

def tendency_ab3(DTYPE_real_t dt,
                    np.ndarray[DTYPE_com_t, ndim=3] dqdt,
                    np.ndarray[DTYPE_com_t, ndim=3] dqdt_p,
                    np.ndarray[DTYPE_com_t, ndim=3] dqdt_pp):
    """Compute tendency using Adams Bashforth 3nd order timestepping."""
    cdef DTYPE_real_t DT1 = 23/12.*dt
    cdef DTYPE_real_t DT2 = -16/12.*dt
    cdef DTYPE_real_t DT3 = 5/12.*dt
    return DT1 * dqdt + DT2 * dqdt_p + DT3 * dqdt_pp
