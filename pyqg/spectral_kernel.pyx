# cython: profile=True
#from __future__ import division
import numpy as np
cimport numpy as np
import pyfftw
pyfftw.interfaces.cache.enable() 


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
    
    # the variables needed for inversion and advection
    # store a as complex so we don't have to typecast in inversion
    cdef DTYPE_com_t [:, :, :, :] a
    cdef DTYPE_com_t [:] ik
    cdef DTYPE_com_t [:] il
    # background state constants (functions of z only)
    cdef DTYPE_real_t [:] Ubg
    #cdef DTYPE_real_t [:] Vbg
    #cdef DTYPE_com_t [:, :] ilQx
    cdef DTYPE_com_t [:, :] ikQy
        
    # pyfftw objects (callable)
    cdef object fft_q_to_qh
    cdef object ifft_qh_to_q
    cdef object ifft_uh_to_u
    cdef object ifft_vh_to_v
    cdef object fft_uq_to_uqh
    cdef object fft_vq_to_vqh    
    def __init__(self, int Nz, int Ny, int Nx, 
                    np.ndarray[DTYPE_real_t, ndim=4] a,
                    np.ndarray[DTYPE_real_t, ndim=1] k,
                    np.ndarray[DTYPE_real_t, ndim=1] l,
                    np.ndarray[DTYPE_real_t, ndim=1] Ubg,
                    #np.ndarray[DTYPE_real_t, ndim=1] Vbg,
                    #np.ndarray[DTYPE_real_t, ndim=1] Qx,
                    np.ndarray[DTYPE_real_t, ndim=1] Qy,                                       
    ):
        self.Nz = Nz
        self.Ny = Ny
        self.Nx = Nx
        self.Nl = Ny
        self.Nk = Nx/2 + 1
        
        ### none of this shape checking works
        #assert a.shape == (self.Nz, self.Nz self.Nl, self.Nk):
        #assert k.shape == (self.Nk,):
        #assert l.shape == (self.Nl,):
        #assert Ubg.shape == (self.Nz,), 'Ubg is the wrong shape'
        #assert Vbg.shape == (self.Nz,), 'Vbg is the wrong shape'
        #assert Qx.shape == (self.Nz,), 'Qx is the wrong shape'
        #assert Qy.shape == (self.Nz,), 'Qy is the wrong shape'
        
        # assign a, ik, il
        self.a = a.astype(DTYPE_com)
        self.ik = 1j*k
        self.il = 1j*l
             
        # assign Ubg, Vbg, ilQx, ikQy
        self.Ubg = Ubg
        #self.Vbg = Vbg
        self.ikQy = 1j * k[np.newaxis, :] * Qy[:, np.newaxis]
        
        # initialize FFT inputs / outputs as byte aligned by pyfftw
        q = pyfftw.n_byte_align_empty((self.Nz, self.Ny, self.Ny),
                         pyfftw.simd_alignment, dtype=DTYPE_real)
        self.q = q
        qh = pyfftw.n_byte_align_empty((self.Nz, self.Nl, self.Nk),
                         pyfftw.simd_alignment, dtype=DTYPE_com)
        self.qh = qh
        
        ph = pyfftw.n_byte_align_empty((self.Nz, self.Nl, self.Nk),
                         pyfftw.simd_alignment, dtype=DTYPE_com)
        self.ph = ph
        
        u = pyfftw.n_byte_align_empty((self.Nz, self.Ny, self.Nx),
                         pyfftw.simd_alignment, dtype=DTYPE_real)
        self.u = u
        uh = pyfftw.n_byte_align_empty((self.Nz, self.Nl, self.Nk),
                         pyfftw.simd_alignment, dtype=DTYPE_com)
        self.uh = uh
        
        v = pyfftw.n_byte_align_empty((self.Nz, self.Ny, self.Nx),
                         pyfftw.simd_alignment, dtype=DTYPE_real)
        self.v = v
        vh = pyfftw.n_byte_align_empty((self.Nz, self.Nl, self.Nk),
                         pyfftw.simd_alignment, dtype=DTYPE_com)
        self.vh = vh
        
        uq = pyfftw.n_byte_align_empty((self.Nz, self.Ny, self.Nx),
                         pyfftw.simd_alignment, dtype=DTYPE_real)
        self.uq = uq
        uqh = pyfftw.n_byte_align_empty((self.Nz, self.Nl, self.Nk),
                         pyfftw.simd_alignment, dtype=DTYPE_com)
        self.uqh = uqh
        
        vq = pyfftw.n_byte_align_empty((self.Nz, self.Ny, self.Nx),
                         pyfftw.simd_alignment, dtype=DTYPE_real)
        self.vq = vq
        vqh = pyfftw.n_byte_align_empty((self.Nz, self.Nl, self.Nk),
                         pyfftw.simd_alignment, dtype=DTYPE_com)
        self.vqh = vqh
        
        # set up FFT plans
        self.fft_q_to_qh = pyfftw.FFTW(q, qh, 
                         direction='FFTW_FORWARD', axes=(-2,-1))
        self.ifft_qh_to_q = pyfftw.FFTW(qh, q, 
                         direction='FFTW_BACKWARD', axes=(-2,-1))
        self.ifft_uh_to_u = pyfftw.FFTW(uh, u, 
                         direction='FFTW_BACKWARD', axes=(-2,-1))
        self.ifft_vh_to_v = pyfftw.FFTW(vh, v, 
                         direction='FFTW_BACKWARD', axes=(-2,-1))
        self.fft_uq_to_uqh = pyfftw.FFTW(uq, uqh, 
                         direction='FFTW_FORWARD', axes=(-2,-1))
        self.fft_vq_to_vqh = pyfftw.FFTW(vq, vqh, 
                         direction='FFTW_FORWARD', axes=(-2,-1))
    
    
    # the only way to set q and qh
    def set_qh(self, np.ndarray[DTYPE_com_t, ndim=3] b):
        cdef  DTYPE_com_t [:, :, :] b_view = b
        self.qh[:] = b_view
        self.ifft_qh_to_q()
        
    def set_q(self, np.ndarray[DTYPE_real_t, ndim=3] b):
        cdef  DTYPE_real_t [:, :, :] b_view = b
        self.q[:] = b_view
        self.fft_q_to_qh()

    
    def advection_tendency(self):
        ### algorithm
        # invert ph = a * qh
        # uh, vh = -il * ph, ik * ph
        # u, v, = ifft(uh), ifft(vh)
        # uq, vq = (u+Ubg)*q, (v+Vbg)*q
        # uqh, vqh, = fft(uq), fft(vq)
        # tend = kj*uqh + ilQx*ph + lj*vqh + ilQy*ph
        
        # the output array: spectal representation of advective tendency
        cdef np.ndarray tend = np.zeros((self.Nz, self.Nl, self.Nk), dtype=DTYPE_com)
        
        # set ph to zero
        for k in range(self.Nz):
            for j in range(self.Nl):
                for i in range(self.Nk):
                    self.ph[k,j,i] = (0. + 0.*1j) 

        # invert qh to find ph
        for k2 in range(self.Nz):
            for k1 in range(self.Nz):
                for j in range(self.Nl):
                    for i in range(self.Nk):
                        self.ph[k2,j,i] = ( self.ph[k2,j,i] +
                            self.a[k2,k1,j,i] * self.qh[k1,j,i] )

        # calculate spectral velocityies
        for k in range(self.Nz):
            for j in range(self.Nl):
                for i in range(self.Nk):
                    self.uh[k,j,i] = -self.il[j] * self.ph[k,j,i]
                    self.vh[k,j,i] =  self.ik[i] * self.ph[k,j,i]

        # transform to get u and v
        self.ifft_uh_to_u()
        self.ifft_vh_to_v()

        # multiply to get advective flux in space
        for k in range(self.Nz):
            for j in range(self.Ny):
                for i in range(self.Nx):
                    self.uq[k,j,i] = self.u[k,j,i] * self.q[k,j,i]
                    self.vq[k,j,i] = self.v[k,j,i] * self.q[k,j,i]

        # transform to get spectral advective flux
        self.fft_uq_to_uqh()
        self.fft_vq_to_vqh()

        # spectral divergence
        for k in range(self.Nz):
            for j in range(self.Nl):
                for i in range(self.Nk):
                    tend[k,j,i] = ( self.ik[i] * self.uqh[k,j,i] +
                                    self.il[j] * self.vqh[k,j,i] +
                                    self.ikQy[k,i] * self.ph[k,j,i] )

        return tend
                        
    # attribute aliases: return numpy ndarray views of memory views
    property q:
        def __get__(self):
            return np.asarray(self.q)
    property qh:
        def __get__(self):
            return np.asarray(self.qh)
    property ph:
        def __get__(self):
            return np.asarray(self.ph)
    property u:
        def __get__(self):
            return np.asarray(self.u)
    property v:
        def __get__(self):
            return np.asarray(self.u)
    property uh:
        def __get__(self):
            return np.asarray(self.uh)
    property vh:
        def __get__(self):
            return np.asarray(self.vh)


        
