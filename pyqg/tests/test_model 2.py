from __future__ import print_function
import unittest
import numpy as np
import pyqg

class PyqgModelTester(unittest.TestCase):

    def setUp(self):
        # need to eliminate beta and U for tests
        self.m = pyqg.QGModel(beta=0., U1=0., U2=0., filterfac=0.)
        # the maximum wavelengths to use in tests
        # if we go to higher wavelengths, we don't get machine precision
        self.kwavemax = int(self.m.nx/8)
        self.lwavemax = int(self.m.ny/8)

    def test_fft2(self, rtol=1e-15):
        """Check whether pyqg fft functions produce the expected results."""
        # define a field with a known Fourier transform
        # if I go higher than a factor of 8, the tests start failing
        for kwave in range(1, self.kwavemax):
            for lwave in range(1, self.lwavemax):

                k = 2*np.pi*kwave/self.m.L
                l = 2*np.pi*lwave/self.m.W

                # check whether these are the correct wavenumbers
                np.testing.assert_allclose(self.m.kk[kwave], k, rtol,
                    err_msg='Incorrect wavenumber (kwave=%g)' % kwave)
                np.testing.assert_allclose(self.m.ll[lwave], l, rtol,
                    err_msg='Incorrect wavenumber (lwave=%g)' % lwave)
                np.testing.assert_allclose(self.m.ll[-lwave], -l, rtol,
                    err_msg='Incorrect wavenumber (lwave=%g)' % lwave)


                q1 = np.cos(k * self.m.x )
                q2 = np.sin(l * self.m.y)

                # assign it to the PV - FFT should also get taken at this point
                self.m.set_q1q2(q1, q2)

                # amplitude of RFFT
                qhamp = np.real(self.m.qh * self.m.qh.conj())

                # expected amplitude of RFFT
                amp = (self.m.nx/2)**2 * self.m.ny**2

                # only have one Fourier component for k-axis
                np.testing.assert_allclose(qhamp[0,0,kwave], amp, rtol,
                    err_msg='Incorrect wave amplitude from FFT (kwave=%g)' % kwave)
                # two symmetric pairs for l-axis
                np.testing.assert_allclose(qhamp[1,lwave,0], amp, rtol,
                    err_msg='Incorrect wave amplitude from FFT (lwave=%g)' % lwave)
                np.testing.assert_allclose(qhamp[1,-lwave,0], amp, rtol,
                    err_msg='Incorrect wave amplitude from FFT (lwave=%g)' % lwave)

                # now mask those components
                qhamp_mask = np.ma.masked_array(qhamp, np.zeros_like(qhamp))
                qhamp_mask.mask[0,0,kwave] = 1
                qhamp_mask.mask[1,lwave,0] = 1
                qhamp_mask.mask[1,-lwave,0] = 1
                # and make sure everything else is zero
                np.testing.assert_allclose(qhamp_mask.filled(0.), 0.,
                    rtol=0., atol=rtol,
                    err_msg='Incorrect wave amplitude from FFT')


    def test_inversion_barotropic(self, rtol=1e-13):
        """Check whether inverting a barotropic PV gives desired result.
        Can't get it to work with rtol < 1e-13 """
        # for barotropic, $q = \nabla^2 \psi$
        #                 $\hat{q} = -(k^2 + l^2) \hat \psi$
        #
        # velocity: u = -dpsi/dy, v = dpsi/dx

        for kwave in range(1, self.kwavemax):
            for lwave in range(1, self.lwavemax):

                k = 2*np.pi*kwave/self.m.L
                l = 2*np.pi*lwave/self.m.W

                q =  np.cos(k * self.m.x ) + np.sin(l * self.m.y)
                psi = -k**-2 * np.cos(k * self.m.x ) - l**-2 * np.sin(l * self.m.y)
                u = l**-1 * np.cos(l * self.m.y)
                v = k**-1 * np.sin(k * self.m.x)

                self.m.set_q1q2(q, q)
                self.m._invert()

                for nz in range(self.m.nz):
                    np.testing.assert_allclose(self.m.u[nz], u, rtol,
                        err_msg='Incorrect velocity from barotropic pv inversion')

                    np.testing.assert_allclose(self.m.v[nz], v, rtol,
                        err_msg='Incorrect velocity from barotropic pv inversion')

    def test_inversion_baroclinic(self, rtol=1e-13):
        """Check whether inverting a baroclinic PV gives desired result."""
        # need to think about how to implement this
        pass

    def test_change_inversion_matrix(self):
        """Make sure we can change the inversion matrix after kernel has been
        initialized."""
        a_new = np.random.rand(self.m.nz, self.m.nz, self.m.nl, self.m.nk)
        self.m.a = a_new
        np.testing.assert_allclose(a_new, self.m.a)

    def test_advection(self, rtol=1e-14):
        """Check whether calculating advection tendency gives the descired result."""
        # sin(2 a) = 2 sin(a) cos(a)

        for kwave in range(1, self.kwavemax):
            for lwave in range(1, self.lwavemax):
                k = 2*np.pi*kwave/self.m.L
                l = 2*np.pi*lwave/self.m.W

                q1 =  np.cos(k * self.m.x )
                q2 =  np.sin(l * self.m.y )
                self.m.set_q1q2(q1, q2)
                # manually set velocity
                u1 = np.sin(k * self.m.x)
                v1 = np.zeros_like(self.m.y)
                u2 = np.zeros_like(self.m.x)
                v2 = np.cos(l * self.m.y)
                self.m.u[0] = u1
                self.m.v[0] = v1
                self.m.u[1] = u2
                self.m.v[1] = v2
                self.m.set_U1U2(0.,0.)

                # calculate tendency
                #self.m._advection_tendency()
                self.m._do_advection()
                dqhdt_adv = self.m.dqhdt

                # expected amplitude of RFFT
                amp = (self.m.nx/2)**2 * self.m.ny**2
                tabs = np.real(dqhdt_adv * dqhdt_adv.conj())

                # these tests pass, but what about the factor of two?
                np.testing.assert_allclose(tabs[0,0,2*kwave], k**2 * amp, rtol,
                    err_msg='Incorrect advection tendency k (%g,%g)' % (lwave,kwave))
                np.testing.assert_allclose(tabs[1,2*lwave,0], l**2 * amp, rtol,
                    err_msg='Incorrect advection tendency +l (%g,%g)' % (lwave,kwave))
                np.testing.assert_allclose(tabs[1,-2*lwave,0], l**2 * amp, rtol,
                    err_msg='Incorrect advection tendency -l (%g,%g)' % (lwave,kwave))

                # now mask those components
                tabs_mask = np.ma.masked_array(tabs, np.zeros_like(tabs))
                tabs_mask.mask[0,0,2*kwave] = 1
                tabs_mask.mask[1,2*lwave,0] = 1
                tabs_mask.mask[1,-2*lwave,0] = 1
                # and make sure everything else is zero
                if np.any(np.isnan(tabs_mask.filled(0.))):
                    print("Found NaNs")
                np.testing.assert_allclose(tabs_mask.filled(0.), 0.,
                    rtol=0., atol=rtol,
                    err_msg='Incorrect advection tendency (%g,%g)' % (lwave,kwave))

    def test_friction(self, rtol=1e-15):
        """Check whether calculating advection tendency gives the expected result."""
        # sin(2 a) = 2 sin(a) cos(a)

        for kwave in range(1, self.kwavemax):
            for lwave in range(1, self.lwavemax):
                k = 2*np.pi*kwave/self.m.L
                l = 2*np.pi*lwave/self.m.W

                q1 =  np.cos(k * self.m.x )
                q2 =  np.sin(l * self.m.y )
                self.m.set_q1q2(q1, q2)
                self.m._invert()
                # make sure tendency is zero
                self.m.dqhdt[:] = 0.
                self.m._do_friction()

                # from code
                #self.dqhdt[-1] += self.rek * self.wv2 * self.ph[-1]
                expected = self.m.rek * self.m.wv2 * self.m.ph[-1]
                np.testing.assert_allclose(self.m.dqhdt[-1], expected, rtol,
                        err_msg='Ekman friction was wrong.')
                np.testing.assert_allclose(self.m.dqhdt[:-1], 0., rtol,
                        err_msg='Nonzero friction found in upper layers.')


    def test_timestepping(self, rtol=1e-15):
        """Make sure timstepping works properly."""

        # set initial conditions to zero
        self.m.set_q(np.zeros_like(self.m.q))

        # create a random tendency
        dqhdt = np.random.rand(*self.m.dqhdt.shape) + 1j*np.random.rand(*self.m.dqhdt.shape)
        self.m.dqhdt[:] = dqhdt
        # hack filter to be constant
        #self.m.filtr = 1.

        # make sure we are at the zero timestep
        self.assertEqual(self.m.tc, 0)
        # step forward first time (should use forward Euler)
        self.m._forward_timestep()

        np.testing.assert_allclose(self.m.qh, 1*self.m.dt*dqhdt,
            err_msg='First timestep incorrect')
        # step forward second time (should use AB2)
        self.m._forward_timestep()
        np.testing.assert_allclose(self.m.qh, 2*self.m.dt*dqhdt,
            err_msg='Second timestep incorrect')
        # step forward third time (should use AB3)
        self.m._forward_timestep()
        np.testing.assert_allclose(self.m.qh, 3*self.m.dt*dqhdt,
            err_msg='Third timestep incorrect')



if __name__ == '__main__':
    unittest.main()
