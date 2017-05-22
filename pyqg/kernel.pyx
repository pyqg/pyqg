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
    cdef public int nx, ny, nz
    cdef public int nk, nl

    ### the main state variables (memory views to numpy arrays) ###
    # pv
    cdef DTYPE_real_t [:, :, :] q
    cdef DTYPE_com_t [:, :, :] qh
    cdef DTYPE_com_t [:, :, :] Qh
    # streamfunction
    cdef DTYPE_com_t [:, :, :] ph
    cdef DTYPE_com_t [:, :, :] Ph
    # velocities
    cdef DTYPE_real_t [:, :, :] u
    cdef DTYPE_real_t [:, :, :] v
    cdef DTYPE_com_t [:, :, :] uh
    cdef DTYPE_com_t [:, :, :] vh
    # pv fluxes
    cdef DTYPE_real_t [:, :, :] uq
    cdef DTYPE_real_t [:, :, :] vq
    cdef readonly DTYPE_com_t [:, :, :] uqh
    cdef readonly DTYPE_com_t [:, :, :] vqh
    # the tendencies
    cdef DTYPE_com_t [:, :, :] dqhdt
    cdef DTYPE_com_t [:, :, :] dqhdt_p
    cdef DTYPE_com_t [:, :, :] dqhdt_pp

    # dummy variables for diagnostic ffts
    cdef DTYPE_real_t [:, :, :] _dummy_fft_in
    cdef DTYPE_com_t [:, :, :] _dummy_fft_out
    cdef DTYPE_real_t [:, :, :] _dummy_ifft_out
    cdef DTYPE_com_t [:, :, :] _dummy_ifft_in

    # k and l are techinically not needed within the kernel, but it is
    # simpler to keep them here
    cdef DTYPE_real_t [:] kk
    cdef DTYPE_real_t [:] ll
    # the variables needed for inversion and advection
    # store a as complex so we don't have to typecast in inversion
    cdef DTYPE_com_t [:, :, :, :] a
    cdef readonly DTYPE_com_t [:] _ik
    cdef readonly DTYPE_com_t [:] _il
    cdef public DTYPE_real_t [:,:] _k2l2
    # background state constants (functions of z only)
    cdef DTYPE_real_t [:,:] Ubg
    cdef DTYPE_real_t [:,:] Qy
    cdef readonly DTYPE_com_t [:, :, :] _ikQy

    # spectral filter
    # TODO: figure out if this really needs to be public
    cdef public DTYPE_real_t [:, :] filtr

    # friction parameter
    cdef public DTYPE_real_t rek

    # friction parameter
    cdef public DTYPE_real_t rbg

    # time
    # need to have a property to deal with resetting timestep
    cdef DTYPE_real_t dt
    cdef readonly int tc
    cdef readonly DTYPE_real_t t
    cdef readonly int ablevel

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

    def __init__(self, int nz, int ny, int nx, int fftw_num_threads=1):
        self.nz = nz
        self.ny = ny
        self.nx = nx
        self.nl = ny
        self.nk = nx/2 + 1
        self.a = np.zeros((self.nz, self.nz, self.nl, self.nk), DTYPE_com)
        self.kk = np.zeros((self.nk), DTYPE_real)
        self._ik = np.zeros((self.nk), DTYPE_com)
        self.ll = np.zeros((self.nl), DTYPE_real)
        self._il = np.zeros((self.nl), DTYPE_com)
        self._k2l2 = np.zeros((self.nl, self.nk), DTYPE_real)

        # initialize FFT inputs / outputs as byte aligned by pyfftw
        q = self._empty_real()
        self.q = q
        qh = self._empty_com()
        self.qh = qh
        Qh = self._empty_com()
        self.Qh = Qh

        ph = self._empty_com()
        self.ph = ph

        Ph = self._empty_com()
        self.Ph = Ph

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

        # time stuff
        self.dt = 0.0
        self.t = 0.0
        self.tc = 0
        self.ablevel = 0

        # friction
        self.rek = 0.0
        self.rbg = 0.0

        # the tendency
        self.dqhdt = self._empty_com()
        self.dqhdt_p = self._empty_com()
        self.dqhdt_pp = self._empty_com()

        # for threading
        self.num_threads = fftw_num_threads
        self.chunksize = self.nl/self.num_threads

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
        shape = (self.nz, self.ny, self.ny)
        IF PYQG_USE_PYFFTW:
            out = pyfftw.n_byte_align_empty(shape,
                                 pyfftw.simd_alignment, dtype=DTYPE_real)
            out.flat[:] = 0.
            return out
        ELSE:
            return np.zeros(shape, dtype=DTYPE_real)

    def _empty_com(self):
        """Allocate a Fourier-grid-sized variable for use with fftw transformations."""
        shape = (self.nz, self.nl, self.nk)
        IF PYQG_USE_PYFFTW:
            out = pyfftw.n_byte_align_empty(shape,
                                 pyfftw.simd_alignment, dtype=DTYPE_com)
            out.flat[:] = 0.+0.j
            return out
        ELSE:
            return np.zeros(shape, dtype=DTYPE_com)

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


    def _invert(self):
        self.__invert()

    cdef void __invert(self) nogil:
    ### algorithm
        # invert ph = a * qh
        # uh, vh = -_il * ph, _ik * ph
        # u, v, = ifft(uh), ifft(vh)

        cdef Py_ssize_t k, k1, k2, j, i
        # set ph to zero
        for k in range(self.nz):
            for j in prange(self.nl, nogil=True, schedule='static',
                      chunksize=self.chunksize,
                      num_threads=self.num_threads):
                for i in range(self.nk):
                    self.ph[k,j,i] = (0. + 0.*1j)
                    self.Ph[k,j,i] = (0. + 0.*1j)

        # invert qh to find ph
        for k2 in range(self.nz):
            for k1 in range(self.nz):
                for j in prange(self.nl, nogil=True, schedule='static',
                          chunksize=self.chunksize,
                          num_threads=self.num_threads):
                    for i in range(self.nk):
                        self.ph[k2,j,i] = ( self.ph[k2,j,i] +
                            self.a[k2,k1,j,i] * self.qh[k1,j,i] )
                        self.Ph[k2,j,i] = ( self.Ph[k2,j,i] +
                            self.a[k2,k1,j,i] * self.Qh[k1,j,i] )

        # calculate spectral velocities
        for k in range(self.nz):
            for j in prange(self.nl, nogil=True, schedule='static',
                      chunksize=self.chunksize,
                      num_threads=self.num_threads):
                for i in range(self.nk):
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
        #cdef np.ndarray tend = np.zeros((self.nz, self.nl, self.nk), dtype=DTYPE_com)

        cdef Py_ssize_t k, j, i

        # multiply to get advective flux in space
        for k in range(self.nz):
            for j in prange(self.ny, nogil=True, schedule='static',
                      chunksize=self.chunksize,
                      num_threads=self.num_threads):
                for i in range(self.nx):
                    self.uq[k,j,i] = (self.u[k,j,i]+self.Ubg[k,j]) * self.q[k,j,i]
                    self.vq[k,j,i] = self.v[k,j,i] * self.q[k,j,i]

        # transform to get spectral advective flux
        with gil:
            self.fft_uq_to_uqh()
            self.fft_vq_to_vqh()

        # spectral divergence
        for k in range(self.nz):
            for j in prange(self.nl, nogil=True, schedule='static',
                      chunksize=self.chunksize,
                      num_threads=self.num_threads):
                for i in range(self.nk):
                    # overwrite the tendency, since the forcing gets called after
                    self.dqhdt[k,j,i] = -( self._ik[i] * self.uqh[k,j,i] +
                                    self._il[j] * self.vqh[k,j,i] +
                                    self._ikQy[k,j,i] * self.ph[k,j,i] )
        return

    def _do_friction(self):
        self.__do_friction()

    cdef void __do_friction(self) nogil:
        """Apply Ekman friction to lower layer tendency"""
        cdef Py_ssize_t k = self.nz-1
        cdef Py_ssize_t j, i
        if self.rek:
            for j in prange(self.nl, nogil=True, schedule='static',
                      chunksize=self.chunksize,
                      num_threads=self.num_threads):
                for i in range(self.nk):
                    self.dqhdt[k,j,i] = (
                     self.dqhdt[k,j,i] +
                             (self.rek *
                             self._k2l2[j,i] *
                             self.ph[k,j,i]) )
        return

    def _do_viscosity(self):
        self.__do_viscosity()

    cdef void __do_viscosity(self) nogil:
        """Apply viscous restoring between eddy and background flows"""
        cdef Py_ssize_t k, i
        if self.rbg:
            for k in range(self.nz):
                for i in range(self.nk):
                    self.dqhdt[k,0,i] = (
                    self.dqhdt[k,0,i] -
                    self.rbg * self.qh[k,0,i])

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
        if self.ablevel==0:
            # forward euler
            dt1 = self.dt
            dt2 = 0.0
            dt3 = 0.0
            self.ablevel=1
        elif self.ablevel==1:
            # AB2 at step 2
            dt1 = 1.5*self.dt
            dt2 = -0.5*self.dt
            dt3 = 0.0
            self.ablevel=2
        else:
            # AB3 from step 3 on
            dt1 = 23./12.*self.dt
            dt2 = -16./12.*self.dt
            dt3 = 5./12.*self.dt

        for k in range(self.nz):
            for j in prange(self.nl, nogil=True, schedule='static',
                      chunksize=self.chunksize,
                      num_threads=self.num_threads):
                for i in range(self.nk):
                    qh_new[k,j,i] = self.filtr[j,i] * (
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

        for k in range(self.nz):
            for j in prange(self.nl, nogil=True, schedule='static',
                      chunksize=self.chunksize,
                      num_threads=self.num_threads):
                for i in range(self.nk):
                    self.qh[k,j,i] = qh_new[k,j,i]

        self.tc += 1
        self.t += self.dt
        return

    property dt:
        def __get__(self):
            return self.dt
        def __set__(self, dt):
            self.dt = dt
            # reset timestepping to forward Euler
            self.ablevel = 0
    # attribute aliases: return numpy ndarray views of memory views
    property kk:
        def __get__(self):
            return np.asarray(self.kk)
        def __set__(self, np.ndarray[DTYPE_real_t, ndim=1] k):
            # do we really need a view here? I guess not.
            # but why do we need one for Qy
            self.kk = k
            self._ik = 1j*k
            for j in range(self.nl):
                for i in range(self.nk):
                    self._k2l2[j,i] = self.kk[i]**2 + self.ll[j]**2
    property ll:
        def __get__(self):
            return np.asarray(self.ll)
        def __set__(self, np.ndarray[DTYPE_real_t, ndim=1] l):
            # do we reall need a view here
            self.ll = l
            self._il = 1j*l
            for j in range(self.nl):
                for i in range(self.nk):
                    self._k2l2[j,i] = self.kk[i]**2 + self.ll[j]**2
    property a:
        def __get__(self):
            return np.asarray(self.a)
        # inversion matrix should be real
        def __set__(self, np.ndarray[DTYPE_real_t, ndim=4] b):
            cdef  DTYPE_com_t [:, :, :, :] b_view = b.astype(DTYPE_com)
            self.a[:] = b_view
    property Ubg:
        def __get__(self):
            return np.asarray(self.Ubg)
        def __set__(self, np.ndarray[DTYPE_real_t, ndim=2] Ubg):
            self.Ubg = Ubg
    property Qy:
        def __get__(self):
            return np.asarray(self.Qy)
        def __set__(self, np.ndarray[DTYPE_real_t, ndim=2] Qy):
            self.Qy = Qy
            self._ikQy = 1j * (np.asarray(self.kk)[np.newaxis, :] *
                               np.asarray(Qy)[:, :, np.newaxis])
    property q:
        def __get__(self):
            return np.asarray(self.q)
        def __set__(self, np.ndarray[DTYPE_real_t, ndim=3] b):
            cdef  DTYPE_real_t [:, :, :] b_view = b
            self.q[:] = b_view
            self.fft_q_to_qh()
    property qh:
        def __get__(self):
            return np.asarray(self.qh)
        def __set__(self, np.ndarray[DTYPE_com_t, ndim=3] b):
            cdef  DTYPE_com_t [:, :, :] b_view = b
            self.qh[:] = b_view
            self.ifft_qh_to_q()
            # input might have been destroyed, have to re-assign
            self.qh[:] = b_view
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
                np.expand_dims(np.asarray(self.Ubg),axis=2)
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
