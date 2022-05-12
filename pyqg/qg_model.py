import numpy as np
from numpy import pi
from . import qg_diagnostics

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

class QGModel(qg_diagnostics.QGDiagnostics):
    r"""Two layer quasigeostrophic model.

    This model is meant to representflows driven by baroclinic instabilty of a
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
        #rek=5.787e-7,               # linear drag in lower layer
        rd=15000.0,                 # deformation radius
        delta=0.25,                 # layer thickness ratio (H1/H2)
        H1 = 500,                   # depth of layer 1 (H1)
        U1=0.025,                   # upper layer flow
        U2=0.0,                     # lower layer flow
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
            Upper layer flow. Units: meters seconds :sup:`-1`
        U2 : number
            Lower layer flow. Units: meters seconds :sup:`-1`
        """

        # physical
        self.beta = beta
        #self.rek = rek
        self.rd = rd
        self.delta = delta
        self.Hi = np.array([ H1, H1/delta])
        self.U1 = U1
        self.U2 = U2
        #self.filterfac = filterfac


        super().__init__(nz=2, **kwargs)

        # initial conditions: (PV anomalies)
        self.set_q1q2(
            1e-7*np.random.rand(self.ny,self.nx) + 1e-6*(
                np.ones((self.ny,1)) * np.random.rand(1,self.nx) ),
                np.zeros_like(self.x) )



    ### PRIVATE METHODS - not meant to be called by user ###

    def _initialize_background(self):
        """Set up background state (zonal flow and PV gradients)."""

        # Background zonal flow (m/s):
        self.H = self.Hi.sum()
        self.set_U1U2(self.U1, self.U2)
        self.U = self.U1 - self.U2
        self.Vbg = np.zeros_like(self.Ubg)

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

    @property
    def S(self):
        # Define stretching matrix to be used in diagnostics
        return np.array([[-self.F1, self.F1],
                         [self.F2, -self.F2]]).astype(np.float64)

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
        pass
        #"""Set up frictional filter."""
        # this defines the spectral filter (following Arbic and Flierl, 2003)
        # cphi=0.65*pi
        # wvx=np.sqrt((self.k*self.dx)**2.+(self.l*self.dy)**2.)
        # self.filtr = np.exp(-self.filterfac*(wvx-cphi)**4.)
        # self.filtr[wvx<=cphi] = 1.

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

    def set_U1U2(self, U1, U2):
        """Set background zonal flow.

        Parameters
        ----------

        U1 : number
            Upper layer flow. Units: meters seconds :sup:`-1`
        U2 : number
            Lower layer flow. Units: meters seconds :sup:`-1`
        """
        self.U1 = U1
        self.U2 = U2
        #self.Ubg = np.array([U1,U2])[:,np.newaxis,np.newaxis]
        self.Ubg = np.array([U1,U2])

    ### All the diagnostic stuff follows. ###

    def _calc_derived_fields(self):
        self.p = self.ifft(self.ph)
        self.xi =self.ifft(-self.wv2*self.ph)
        self.Jptpc = -self._advect(
                    (self.p[0] - self.p[1]),
                    (self.del1*self.u[0] + self.del2*self.u[1]),
                    (self.del1*self.v[0] + self.del2*self.v[1]))
        # fix for delta.neq.1
        self.Jpxi = self._advect(self.xi, self.u, self.v)

        self.Jq = self._advect(self.q, self.u, self.v)
        self.Sph = np.einsum("ij,jkl->ikl",self.S,self.ph)

    def _initialize_model_diagnostics(self):
        """Extra diagnostics for two-layer model"""

        super()._initialize_model_diagnostics()

        self.add_diagnostic('APEflux',
            description='spectral flux of available potential energy',
            function= (lambda self:
              self.rd**-2 * self.del1*self.del2 *
              np.real((self.ph[0]-self.ph[1])*np.conj(self.Jptpc))/self.M**2 ),
            units='m^2 s^-3',
            dims=('l','k')
       )

        self.add_diagnostic('KEflux',
            description='spectral flux of kinetic energy',
            function= (lambda self:
              (np.real(self.del1*self.ph[0]*np.conj(self.Jpxi[0])) +
               np.real(self.del2*self.ph[1]*np.conj(self.Jpxi[1])))/self.M**2 ),
            units='m^2 s^-3',
            dims=('l','k')
       )

        self.add_diagnostic('APEgen',
            description='total available potential energy generation',
            function= (lambda self: self.U * self.rd**-2 * self.del1 * self.del2 *
                       np.real((1j*self.k*
                            (self.del1*self.ph[0] + self.del2*self.ph[1]) *
                            np.conj(self.ph[0] - self.ph[1])).sum()
                              +(1j*self.k[:,1:-2]*
                            (self.del1*self.ph[0,:,1:-2] + self.del2*self.ph[1,:,1:-2]) *
                            np.conj(self.ph[0,:,1:-2] - self.ph[1,:,1:-2])).sum()) /
                            (self.M**2) ),
            units='m^2 s^-3',
            dims=('time',)
       )


