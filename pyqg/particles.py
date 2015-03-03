import numpy as np
from scipy.interpolate import RectBivariateSpline
from regulargrid.cartesiangrid import CartesianGrid

class LagrangianParticleArray2D(object):
    """A class for keeping track of a set of lagrangian particles
    in two-dimensional space. Tried to be as vectorized as possible
    """
    
    def __init__(self, x0, y0,
                       geometry='cartesian',
                       periodic_in_x=False,
                       periodic_in_y=False,
                       xmin=-np.inf, xmax=np.inf,
                       ymin=-np.inf, ymax=np.inf,
                       xgrid=None, ygrid=None,
                       particle_dtype='f8'):
        """Initialize a set of particles with initial positions x0,y0.

        x0 and y0 are arrays (same size) representing the particle
        positions.

        Keyword Arguments:
        geometry -- the type of coordinate system being used
                    (Only accepts 'cartesian' for now.)
        periodic_in_x -- whether the domain 'wraps' in the x direction
        periodic_in_y -- whether the domain 'wraps' in the y direction
        xmin, xmax -- maximum and minimum values of x coordinate
        ymin, ymax -- maximum and minimum values of y coordinate
        xgrid, ygrid -- 1D arrays for use with gridded velocity data
        particle_dtype -- data type to use for particles
        """
        
        self.x = np.array(x0, dtype=np.dtype(particle_dtype)).ravel()
        self.y = np.array(y0, dtype=np.dtype(particle_dtype)).ravel()
        
        assert self.x.shape == self.y.shape
        self.N = len(self.x)
        
        # check that the particles are within the specified boundaries
        assert np.all(self.x >= xmin) and np.all(self.x <= xmax)
        assert np.all(self.y >= ymin) and np.all(self.y <= ymax)
        
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.pix = periodic_in_x
        self.piy = periodic_in_y
        
        if xgrid is not None:
            assert xgrid.ndim == 1
            #assert np.testing.assert_allclose(0., np.diff(xgrid,2))
        if ygrid is not None:
            assert ygrid.ndim == 1
            #assert np.testing.assert_allclose(0., np.diff(ygrid,2))
        
        self.xgrid = xgrid
        self.ygrid = ygrid
        
        self.Lx = self.xmax - self.xmin
        self.Ly = self.ymax - self.ymin
    
    def interpolate_scalar(self, x, y, C):
        """Interpolate scalar C at points X, Y to points x,y"""
        #return RectBivariateSpline(self.ygrid, self.xgrid,
        #                           C, kx=1, ky=1).ev(y,x)
        return CartesianGrid(
                [(self.ygrid[0], self.ygrid[-1]),
                 (self.xgrid[0], self.xgrid[-1])], C)(y,x)
        # rewrite this directly using scipy.ndimage.map_coordinates
                 
    def step_forward_with_function(self, uv0fun, uv1fun, dt):
        dx, dy = self.rk4_integrate(self.x, self.y, uv0fun, uv1fun, dt)
        self.x = self.wrap_x(self.x + dx)
        self.y = self.wrap_y(self.y + dy)
        
    def step_forward_with_gridded_uv(self, U0, V0, U1, V1, dt):       
        # create interpolation functions which return u and v
        uv0fun = (lambda x, y : 
                  (self.interpolate_scalar(x, y, U0),
                  self.interpolate_scalar(x, y, V0)))
        uv1fun = (lambda x, y :  
                  (self.interpolate_scalar(x, y, U1),
                  self.interpolate_scalar(x, y, V1)))
      
        dx, dy = self.rk4_integrate(self.x, self.y, uv0fun, uv1fun, dt)
        self.x = self.wrap_x(self.x + dx)
        self.y = self.wrap_y(self.y + dy)
    
    def rk4_integrate(self, x, y, uv0fun, uv1fun, dt):
        """Integrates positions x,y using velocity functions
           uv0fun, uv1fun. Returns dx and dy, the displacements."""
        u0, v0 = uv0fun(x, y)
        k1u = dt*u0
        k1v = dt*v0
        x11 = self.wrap_x(x + 0.5*k1u)
        y11 = self.wrap_y(y + 0.5*k1v)
        u11, v11 = uv1fun(x11, y11)
        k2u = dt*u11
        k2v = dt*v11
        x12 = self.wrap_x(x + 0.5*k2u)
        y12 = self.wrap_y(y + 0.5*k2v)
        u12, v12 = uv1fun(x12, y12)
        k3u = dt*u12
        k3v = dt*v12
        x13 = self.wrap_x(x + k3u)
        y13 = self.wrap_y(y + k3v)
        u13, v13 = uv1fun(x13, y13)
        k4u = dt*u13
        k4v = dt*v13
        
        # update
        dx = 6**-1*(k1u + 2*k2u + 2*k3u + k4u)
        dy = 6**-1*(k1v + 2*k2v + 2*k3v + k4v)
        return dx, dy
        
    def wrap_x(self, x):
        # wrap positions
        if self.pix:
            return np.mod(x-self.xmin, self.Lx) + self.xmin
        else:
            return x
    
    def wrap_y(self, y):
        # wrap y position
        if self.piy:
            return np.mod(y-self.ymin, self.Ly) + self.ymin
        else:
            return y
            
    def distance(self, x0, y0, x1, y1):
        """Utitlity function to compute distance between points."""
        dx = x1-x0
        dy = y1-y0
        # roll displacements across the borders
        dx[ dx > self.Lx/2 ] -= self.Lx
        dx[ dx < -self.Lx/2 ] += self.Lx
        dy[ dy > self.Ly/2 ] -= self.Ly
        dy[ dy < -self.Ly/2 ] += self.Ly
        return dx, dy
        
        
        
