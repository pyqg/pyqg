import warnings
import numpy as np
#from scipy.interpolate import RectBivariateSpline
#from regulargrid.cartesiangrid import CartesianGrid
#from scipy.ndimage import map_coordinates
# works better with mock on readthedocs

try:
    import scipy.ndimage
except ImportError:
    warnings.warn('Failed to import scipy.ndimage. '
           'Gridded interpolation will not work',
	   ImportWarning)

class LagrangianParticleArray2D:
    """A class for keeping track of a set of lagrangian particles
    in two-dimensional space. Tries to be fast.
    """
    
    def __init__(self, x0, y0,
                       periodic_in_x=False,
                       periodic_in_y=False,
                       xmin=-np.inf, xmax=np.inf,
                       ymin=-np.inf, ymax=np.inf,
                       particle_dtype='f8'):
        """
        Parameters
        ---------- 
        
        x0, y0 : array-like
            Two arrays (same size) representing the particle initial
            positions.
        periodic_in_x : bool
            Whether the domain wraps in the x direction.
        periodic_in_y : bool
            Whether the domain 'wraps' in the y direction.
        xmin, xmax : numbers
            Maximum and minimum values of x coordinate
        ymin, ymax : numbers
            Maximum and minimum values of y coordinate
        particle_dtype : dtype
            Data type to use for particles
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
                        
        self.Lx = self.xmax - self.xmin
        self.Ly = self.ymax - self.ymin
                 
    def step_forward_with_function(self, uv0fun, uv1fun, dt):
        """Advance particles using a function to determine u and v.
        
        Parameters
        ----------
        uv0fun : function
            Called like ``uv0fun(x,y)``. Should return the velocity field
            u, v at time t.
        uv1fun(x,y) : function
            Called like ``uv1fun(x,y)``. Should return the velocity field
            u, v at time t + dt.
        dt : number
            Timestep."""
            
        dx, dy = self._rk4_integrate(self.x, self.y, uv0fun, uv1fun, dt)
        self.x = self._wrap_x(self.x + dx)
        self.y = self._wrap_y(self.y + dy)
    
    def _rk4_integrate(self, x, y, uv0fun, uv1fun, dt):
        """Integrates positions x, y using velocity functions
           uv0fun, uv1fun. Returns dx and dy, the displacements."""
        u0, v0 = uv0fun(x, y)
        k1u = dt*u0
        k1v = dt*v0
        x11 = self._wrap_x(x + 0.5*k1u)
        y11 = self._wrap_y(y + 0.5*k1v)
        u11, v11 = uv1fun(x11, y11)
        k2u = dt*u11
        k2v = dt*v11
        x12 = self._wrap_x(x + 0.5*k2u)
        y12 = self._wrap_y(y + 0.5*k2v)
        u12, v12 = uv1fun(x12, y12)
        k3u = dt*u12
        k3v = dt*v12
        x13 = self._wrap_x(x + k3u)
        y13 = self._wrap_y(y + k3v)
        u13, v13 = uv1fun(x13, y13)
        k4u = dt*u13
        k4v = dt*v13
        
        # update
        dx = 6**-1*(k1u + 2*k2u + 2*k3u + k4u)
        dy = 6**-1*(k1v + 2*k2v + 2*k3v + k4v)
        return dx, dy
        
    def _wrap_x(self, x):
        # wrap positions
        if self.pix:
            return np.mod(x-self.xmin, self.Lx) + self.xmin
        else:
            return x
    
    def _wrap_y(self, y):
        # wrap y position
        if self.piy:
            return np.mod(y-self.ymin, self.Ly) + self.ymin
        else:
            return y
            
    def _distance(self, x0, y0, x1, y1):
        """Utitlity function to compute distance between points."""
        dx = x1-x0
        dy = y1-y0
        # roll displacements across the borders
        if self.pix:
            dx[ dx > self.Lx/2 ] -= self.Lx
            dx[ dx < -self.Lx/2 ] += self.Lx
        if self.piy:
            dy[ dy > self.Ly/2 ] -= self.Ly
            dy[ dy < -self.Ly/2 ] += self.Ly
        return dx, dy
        
        
class GriddedLagrangianParticleArray2D(LagrangianParticleArray2D):
    """Lagrangian particles with velocities given on a regular cartesian grid.
    """
    
    def __init__(self, x0, y0, Nx, Ny, grid_type='A', **kwargs):
        """
        Parameters
        ---------- 
        
        x0, y0 : array-like
            Two arrays (same size) representing the particle initial
            positions.
        Nx, Ny: int
            Number of grid points in the x and y directions
        grid_type: {'A'}
            Arakawa grid type specifying velocity positions.
        """
        
        super().__init__(x0, y0, **kwargs)
        self.Nx = Nx
        self.Ny = Ny
        
        if grid_type != 'A':
            raise ValueError('Only A grid velocities supported at this time.')
            
        if not (self.pix and self.piy):
            raise ValueError(
              'Interpolation only works with doubly periodic grids at this time.')
            
        # figure out grid geometry, assuming velocities are located a cell centers
        
    # def xy_to_ij(self, x, y):
    #     """Convert spatial coords x, y to grid coords i, j"""
    #     i = x/self.Lx*self.Nx - 0.5
    #     j = y/self.Ly*self.Ny - 0.5
    #     return i, j
    #
    # def ij_to_xy(self, i, j):
    #     """Convert grid coords i, j to spatial coords x, y"""
    #     x = (i + 0.5)*self.Lx/self.Nx
    #     y = (j + 0.5)*self.Ly/self.Ny
    
    def interpolate_gridded_scalar(self, x, y, c, order=1, pad=1, offset=0):
        """Interpolate gridded scalar C to points x,y.
        
        Parameters
        ----------
        x, y : array-like
            Points at which to interpolate
        c : array-like
            The scalar, assumed to be defined on the grid.
        order : int
            Order of interpolation
        pad : int
            Number of pad cells added
        offset : int
            ??? 
            
        Returns
        -------
        ci : array-like
            The interpolated scalar
        """
        
        ## no longer necessary because we accept pre-padded arrays
        # assert c.shape == (self.Ny, self.Nx), 'Shape of c needs to be (Ny,Nx)'
        
        # first pad the array to deal with the boundaries
        # (map_coordinates can't seem to deal with this by itself)
        # pad twice so cubic interpolation can be used
        if pad > 0:
            cp = self._pad_field(c, pad=pad)
        else:
            cp = c
        # now the shape is Nx+2, Nx+2
        i = (x - self.xmin)/self.Lx*self.Nx + pad + offset - 0.5
        j = (y - self.ymin)/self.Ly*self.Ny + pad + offset - 0.5
        
        # for some reason this still does not work with high precision near the boundaries
        return scipy.ndimage.map_coordinates(cp, [j,i],
                mode='constant', order=order, cval=np.nan)
    
    def _pad_field(self, c, pad=5):
        return np.pad(c, ((pad,pad),(pad,pad)), mode='wrap')
           
    def step_forward_with_gridded_uv(self, U0, V0, U1, V1, dt, order=1):       
        """Advance particles using a gridded velocity field. Because of the
        Runga-Kutta timestepping, we need two velocity fields at different
        times.
        
        Parameters
        ----------
        U0, V0 : array-like
            Gridded velocity fields at time t - dt.
        U1, V1 : array-like
            Gridded velocity fields at time t.
        dt : number
            Timestep.
        order : int
            Order of interpolation.
        """
        # create interpolation functions which return u and v
        
        # pre-pad arrays so it only has to be done once
        # for linear interpolation (default order=1), only one pad is necessary
        pad = order
        [U0p, V0p, U1p, V1p] = [self._pad_field(c, pad=pad) for c in [U0, V0, U1, V1]]
        
        # pad u and v as necessary
        uv0fun = (lambda x, y : 
                  (self.interpolate_gridded_scalar(x, y, U0p,
                           pad=0, order=order, offset=pad),
                   self.interpolate_gridded_scalar(x, y, V0p,
                           pad=0, order=order, offset=pad)))
        uv1fun = (lambda x, y :  
                  (self.interpolate_gridded_scalar(x, y, U1p, pad=0,
                           order=order, offset=pad),
                   self.interpolate_gridded_scalar(x, y, V1p, pad=0,
                           order=order, offset=pad)))
        
        self.step_forward_with_function(uv0fun, uv1fun, dt)
        #dx, dy = self.rk4_integrate(self.x, self.y, uv0fun, uv1fun, dt)
        #self.x = self.wrap_x(self.x + dx)
        #self.y = self.wrap_y(self.y + dy)
        
        
        
        
        
        
        
        
        
        
        
