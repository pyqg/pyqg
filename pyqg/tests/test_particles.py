import unittest
import pytest
import numpy as np
import pyqg

missing_scipy=False
try:
    import scipy.ndimage
except ImportError:
    missing_scipy=True

def constant_velocity_function(u, v):
    """Return a function that returns a constant velocity field."""
    return (lambda x, y: (u, v))

def solid_body_rotation_velocity_function(om):
    """Return a function that returns solid body rotation at angular
    velocity om."""

    def sbr(x0, y0, om0):
        r = np.sqrt(x0**2 + y0**2)
        umag = om0 * r
        a = np.arctan2(y0, x0)
        u = umag * np.sin(a)
        v = -umag * np.cos(a)
        return u, v

    return lambda x, y: sbr(x, y, om)

class ParticleTester(unittest.TestCase):

    def setUp(self):
        pass

    def test_integration(self, rtol=1e-14, atol=1e-10):
        """Check whether timestepping of particles produces the expected results."""

        x0 = np.array([1.,-1])
        y0 = np.array([0.,0.])

        # define a simple constant velocity field
        DT = 10.  # total time interval
        nt = 100  # number of time steps
        dt = DT/nt
        for u in [-1,0,1]:
            for v in [-1,0,1]:
                # compute analytical solution
                x1 = x0 + DT*u
                y1 = y0 + DT*v

                uvfun = constant_velocity_function(u, v)
                lpa = pyqg.LagrangianParticleArray2D(x0, y0)
                for n in range(nt):
                    lpa.step_forward_with_function(uvfun, uvfun, dt)

                # check solution
                np.testing.assert_allclose(x1, lpa.x, rtol=rtol)
                np.testing.assert_allclose(y1, lpa.y, rtol=rtol)

        # solid body rotation field
        nt = 1000
        for om in [0.1, 1., 10.]:
            # period, time to make a complete revolution around unit circle
            T = 2*np.pi/om
            # pick a timestep
            dt = T / nt

            uvfun = solid_body_rotation_velocity_function(om)
            lpa = pyqg.LagrangianParticleArray2D(x0, y0)
            for n in range(nt):
                lpa.step_forward_with_function(uvfun, uvfun, dt)

            # particles should be back to the same place
            np.testing.assert_allclose(
                np.array([lpa.x, lpa.y]),
                np.array([x0, y0]),
                atol=atol
            )

    @pytest.mark.skipif(missing_scipy, reason="requires scipy")
    def test_interpolation(self, rtol=1e-14, atol=1e-7):
        # set up grid
        Lx = 10.
        Ly = 5.
        Nx = 30
        Ny = 20
        dx = Lx/Nx
        dy = Ly/Ny
        xc = np.arange(Nx)*dx
        yc = np.arange(Ny)*dy
        xg = np.arange(Nx)*dx + dx/2
        yg = np.arange(Ny)*dy + dy/2
        xxg, yyg = np.meshgrid(xg, yg)
        xxc, yyc = np.meshgrid(xc, yc)

        # test field to interpolate
        ffun = lambda x, y: np.sin(2*np.pi*x/Lx)*np.sin(2*np.pi*y/Ly)
        f_at_g = ffun(xxg, yyg)
        f_at_c = ffun(xxc, yyc)

        Npart = 10
        x0 = np.array(np.random.rand(Npart)*Lx)
        y0 = np.array(np.random.rand(Npart)*Ly)
        glpa = pyqg.GriddedLagrangianParticleArray2D(
             x0, y0, Nx, Ny,
             xmin=0, xmax=Lx, ymin=0, ymax=Ly,
             periodic_in_x=True, periodic_in_y=True)

        # check if we interpolate back to the same points, we get same result
        f_at_gi = glpa.interpolate_gridded_scalar(xxg, yyg, f_at_g)
        np.testing.assert_allclose(f_at_g, f_at_gi)

        # now try shifting everything
        # used to use cubic interpolation by default
        # that was slow and produced weird artifacts near boundaries
        # now use linear interpolation
        # this definitely produces errors in the approximation of sine and cosine
        # neglect of curvature causes velocity to be systematically underestimated
        f_at_ci = glpa.interpolate_gridded_scalar(xxc, yyc, f_at_g)
        np.testing.assert_allclose(f_at_c, f_at_ci, atol=1e-1)

        # now try some random points
        # what sort of error do we expect
        ci = ffun(x0, y0)
        np.testing.assert_allclose(ci,
            glpa.interpolate_gridded_scalar(x0, y0, f_at_g),
            atol=1e-1
        )

    @pytest.mark.skipif(missing_scipy, reason="requires scipy")
    def test_gridded_integration(self, atol=1e-10):

        # set up grid
        Lx = 4.
        Ly = 4.
        Nx = 100
        Ny = 100
        dx = Lx/Nx
        dy = Ly/Ny
        # corner points
        xc = np.arange(Nx)*dx - Lx/2
        yc = np.arange(Ny)*dy - Lx/2
        # center points
        xg = xc + dx/2
        yg = xc + dy/2
        xxg, yyg = np.meshgrid(xg, yg)
        xxc, yyc = np.meshgrid(xc, yc)

        # initial particle positions
        x0 = np.array([1.,-1])
        y0 = np.array([0.,0.])

        # solid body rotation field
        nt = 1000
        for om in [0.1, 1., 10.]:
            # period, time to make a complete revolution around unit circle
            T = 2*np.pi/om
            # pick a timestep
            dt = T / nt

            uvfun = solid_body_rotation_velocity_function(om)
            u_at_g, v_at_g = uvfun(xxg, yyg)
            glpa = pyqg.GriddedLagrangianParticleArray2D(
                 x0, y0, Nx, Ny,
                 xmin=-Lx/2, xmax=Lx/2, ymin=-Ly/2, ymax=Ly/2,
                 periodic_in_x=True, periodic_in_y=True)

            for n in range(nt):
                glpa.step_forward_with_gridded_uv(
                    u_at_g, v_at_g, u_at_g, v_at_g, dt)

            # particles should be back to the same place
            np.testing.assert_allclose(
                np.array([glpa.x, glpa.y]),
                np.array([x0, y0]),
                atol=atol
            )






if __name__ == '__main__':
    unittest.main()
