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

class LayeredModel(model.Model):
    r"""Layered quasigeostrophic model.

    This model is meant to represent flows driven by baroclinic instabilty of a
    base-state shear. The potential vorticity anomalies qi are related to the
    streamfunction psii through

    .. math::


       {q_i} = \nabla^2\psi_i + \frac{f_0^2}{H_i} \left(\frac{\psi_{i-1}-
       \psi_i}{g'_{i-1}}- \frac{\psi_{i}-\psi_{i+1}}{g'_{i}}\right)\,,
       \qquad i = 2,\textsf{N}-1\,,

     and

    .. math::


       {q_1} = \nabla^2\psi_1 + \frac{f_0^2}{H_1} \left(\frac{\psi_{2}-
       \psi_1}{g'_{1}}\right)\,,  \qquad i =1\,,

    .. math::


       {q_\textsf{N}} = \nabla^2\psi_\textsf{N} +
       \frac{f_0^2}{H_\textsf{N}} \left(\frac{\psi_{\textsf{N}-1}-
       \psi_\textsf{N}}{g'_{\textsf{N}}}\right) + \frac{f_0}{H_\textsf{N}}h_b\,,
       \qquad i =\textsf{N}\,,

     where the reduced gravity, or buoyancy jump, is


    .. math::

       g'_i \equiv g \frac{\rho_{i+1}-\rho_i}{\rho_i}\,.

    The evolution equations are

    .. math::


       \,{q_{i}}_t + \mathsf{J}\left(\psi_i\,, q_i\right) + \textsf{Q}_y {\psi_i}_x
       - \textsf{Q}_x {\psi_i}_y = \text{ssd} -
       r_{ek} \delta_{i\textsf{N}} \nabla^2 \psi_i\,, \qquad i = 1,\textsf{N}\,,


    where the mean potential vorticy gradients are

    .. math::

       \textsf{Q}_x = \textsf{S}\textsf{V}\,,

    and

    .. math::

       \textsf{Q}_y = \beta\,\textsf{I} - \textsf{S}\textsf{U}\,\,,

    where S is the stretching matrix, I is the identity matrix,
    and the background velocity is

    :math:`\vec{\textsf{V}}(z) = \left(\textsf{U},\textsf{V}\right)`.

    For more information see http://pyqg.readthedocs.org/en/stable/equations.html


    """

    def __init__(
        self,
        g = 9.81,
        beta=1.5e-11,               # gradient of coriolis parameter
        nz = 4,                     # number of layers
        rd=15000.0,                 # deformation radius
        H = None,                   # layer thickness
        U=None,                     # zonal base state flow
        V=None,                     # meridional base state flow
        rho = None,
        delta = None,
        **kwargs
        ):
        """
        Parameters
        ----------

        g : number
            Gravitational acceleration. Units: meters second :sup:`-2`
        nz : integer number
             Number of layers (> 1)
        beta : number
            Gradient of coriolis parameter. Units: meters :sup:`-1`
            seconds :sup:`-1`
        rek : number
            Linear drag in lower layer. Units: seconds :sup:`-1`
        rd : number
            Deformation radius. Units: meters. Only necessary for
            the two-layer (nz=2) case.
        delta : number
            Layer thickness ratio (H1/H2). Only necessary for the
            two-layer (nz=2) case. Unitless.
        U : array of size nz
            Base state zonal velocity. Units: meters s :sup:`-1`
        V : array of size nz
            Base state meridional velocity. Units: meters s :sup:`-1`
        H : array of size nz
            Layer thickness. Units: meters
        rho: array of size nz.
            Layer density. Units: kilograms meters :sup:`-3`

        """

        # physical
        self.g = g
        self.beta = beta
        self.rd = rd
        self.delta = delta
        self.nz = nz
        self.U = U
        self.V = V
        self.H = H
        self.rho = rho

        super(LayeredModel, self).__init__(**kwargs)


    ### PRIVATE METHODS - not meant to be called by user ###


    def _initialize_stretching_matrix(self):
        """ Set up the stretching matrix """

        self.S = np.zeros((self.nz, self.nz))

        if (self.nz==2)&(self.rd is not None):

            self.del1 = self.delta/(self.delta+1.)
            self.del2 = (self.delta+1.)**-1
            self.Us = self.U[0]-self.U[1]

            self.F1 = self.rd**-2 / (1.+self.delta)
            self.F2 = self.delta*self.F1
            self.S[0,0], self.S[0,1] = -self.F1,  self.F1
            self.S[1,0], self.S[1,1] =  self.F2, -self.F2

        else:

            for i in range(self.nz):

                if i == 0:
                    self.S[i,i]   = -self.f2/self.Hi[i]/self.gpi[i]
                    self.S[i,i+1] =  self.f2/self.Hi[i]/self.gpi[i]

                elif i == self.nz-1:
                    self.S[i,i]   = -self.f2/self.Hi[i]/self.gpi[i-1]
                    self.S[i,i-1] =  self.f2/self.Hi[i]/self.gpi[i-1]

                else:
                    self.S[i,i-1] = self.f2/self.Hi[i]/self.gpi[i-1]
                    self.S[i,i]   = -(self.f2/self.Hi[i]/self.gpi[i] +
                                        self.f2/self.Hi[i]/self.gpi[i-1])
                    self.S[i,i+1] = self.f2/self.Hi[i]/self.gpi[i]

    def _initialize_background(self):
        """Set up background state (zonal flow and PV gradients)."""

        # Need to figure out a warning to user when
        # these entries are not provided
        self.Hi = self.H
        self.Ubg = self.U
        self.Vbg = self.V
        self.rhoi = self.rho
        self.H = self.Hi.sum()


        if not self.nz==2:
            self.gpi = self.g*(self.rhoi[1:]-self.rhoi[:-1])/self.rhoi[:-1]
            self.f2gpi = (self.f2/self.gpi)[:,np.newaxis,np.newaxis]

            assert self.Hi.size == self.nz, "size of Hi does not match number\
                    of vertical levels nz"

            assert self.rhoi.size == self.nz, "size of rhoi does not match number\
                    of vertical levels nz"

            assert self.Ubg.size == self.nz, "size of Ubg does not match number\
                    of vertical levels nz"

            assert self.Vbg.size == self.nz, "size of Vbg does not match number\
                    of vertical levels nz"

        else:
            self.f2gpi = np.array(self.rd**-2 *
                (self.Hi[0]*self.Hi[1])/self.H)[np.newaxis,np.newaxis]


        self._initialize_stretching_matrix()

        # the meridional PV gradients in each layer
        self.Qy = self.beta - np.dot(self.S,self.Ubg)
        self.Qx = np.dot(self.S,self.Vbg)

        self.hb = self.hb * self.f/self.Hi[-1]

        # complex versions, multiplied by k, speeds up computations to precompute
        self.ikQy = self.Qy[:,np.newaxis,np.newaxis]*1j*self.k
        self.ilQx = self.Qx[:,np.newaxis,np.newaxis]*1j*self.l

    def _initialize_inversion_matrix(self):

        a = np.ma.zeros((self.nz, self.nz, self.nl, self.nk), np.dtype('float64'))

        if (self.nz==2):
            det_inv =  np.ma.masked_equal(
                    ( (self.S[0,0]-self.wv2)*(self.S[1,1]-self.wv2) -\
                            self.S[0,1]*self.S[1,0] ), 0.)**-1
            a[0,0] = (self.S[1,1]-self.wv2)*det_inv
            a[0,1] = -self.S[0,1]*det_inv
            a[1,0] = -self.S[1,0]*det_inv
            a[1,1] = (self.S[0,0]-self.wv2)*det_inv
        else:
            I = np.eye(self.nz)[:,:,np.newaxis,np.newaxis]
            a = np.linalg.inv((self.S[:,:,np.newaxis,np.newaxis]-I*self.wv2).T).T
            a[:,:,0,0] = 0.

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

        # check that it works
        if check:
            np.testing.assert_allclose(self.q1, q1)
            np.testing.assert_allclose(self.q1, self.ifft2(self.qh1))

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
        ke = 0.
        for j in range(self.nz):
            ke += .5*self.Hi[j]*self.spec_var(self.wv*self.ph[j])
        return ke.sum() / self.H

    # calculate eddy turn over time
    # (perhaps should change to fraction of year...)
    def _calc_eddy_time(self):
        """ estimate the eddy turn-over time in days """
        ens = 0.
        for j in range(self.nz):
            ens = .5*self.Hi[j] * self.spec_var(self.wv2*self.ph[j])

        return 2.*pi*np.sqrt( self.H / ens.sum() ) / 86400

    def _calc_derived_fields(self):

        self.p = self.ifft(self.ph)
        self.xi =self.ifft(-self.wv2*self.ph)
        self.Jpxi = self._advect(self.xi, self.u, self.v)
        self.Jq = self._advect(self.q, self.u, self.v)

        self.Sph = np.einsum("ij,jkl->ikl",self.S,self.ph)
        self.Sp = self.ifft(self.Sph)
        self.JSp = self._advect(self.Sp,self.u,self.v)


    def _initialize_model_diagnostics(self):
         """ Extra diagnostics for layered model """

         self.add_diagnostic('entspec',
            description='barotropic enstrophy spectrum',
            function= (lambda self:
                       np.abs((self.Hi[:,np.newaxis,np.newaxis]*
                              self.qh).sum(axis=0))**2/self.H) )

         self.add_diagnostic('APEspec',
            description='available potential energy spectrum',
            function= (lambda self:
                       (self.f2gpi*
                        np.abs(self.ph[:-1]-self.ph[1:])**2).sum(axis=0)/self.H))

         self.add_diagnostic('KEflux',
                description='spectral divergence of flux of kinetic energy',
                function =(lambda self: (self.Hi[:,np.newaxis,np.newaxis]*
                           (self.ph.conj()*self.Jpxi).real).sum(axis=0)/self.H))

         self.add_diagnostic('APEflux',
                description='spectral divergence of flux of available potential energy',
                function =(lambda self: (self.Hi[:,np.newaxis,np.newaxis]*
                           (self.ph.conj()*self.JSp).real).sum(axis=0)/self.H))

         self.add_diagnostic('APEgenspec',
                description='the spectrum of the rate of generation of available potential energy',
                function =(lambda self: (self.Hi[:,np.newaxis,np.newaxis]*
                            (self.Ubg[:,np.newaxis,np.newaxis]*self.k +
                             self.Vbg[:,np.newaxis,np.newaxis]*self.l)*
                            (1j*self.ph.conj()*self.Sph).real).sum(axis=0)/self.H))

         self.add_diagnostic('ENSflux',
            description='barotropic enstrophy flux',
            function = (lambda self: (-self.Hi[:,np.newaxis,np.newaxis]*
                            (self.qh.conj()*self.Jq).real).sum(axis=0)/self.H))

         self.add_diagnostic('ENSgenspec',
            description='the spectrum of the rate of generation of barotropic enstrophy',
            function = (lambda self:
                            -(self.Hi[:,np.newaxis,np.newaxis]*
                              ((self.ikQy - self.ilQx)*
                            (self.Sph.conj()*self.ph)).real).sum(axis=0)/self.H))
