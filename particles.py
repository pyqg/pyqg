import numpy as np
from scipy.interpolate import RectBivariateSpline

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
        """
        
        self.x = np.array(x0, dtype=np.dtype(particle_dtype)).ravel()
        self.y = np.array(y0, dtype=np.dtype(particle_dtype)).ravel()
        
        assert self.x.shape == self.y.shape
        
        # check that the particles are within the specified boundaries
        assert np.all(self.x > xmin) and np.all(self.x < xmax)
        assert np.all(self.y > ymin) and np.all(self.y < ymax)
        
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.pix = periodic_in_x
        self.piy = periodic_in_y
        
        self.Lx = self.xmax - self.xmin
        self.Ly = self.ymax - self.ymin
        
    def runge_kutta_step(self, X, Y, U0, V0, U1, V1, dt):
        """Update the particle positions from time t0 to time t0+dt.
        
        Arguments:
        X, Y -- grid for velocities. assumed to be regularly spaced and 1D
        U0, V0 -- velocities at time t0 (in m/s)
        V1, V1 -- velocities at time t0 + dt (in m/s)
        dt -- timestep (in seconds)
        """
        # followed these notes:
        # http://graphics.cs.ucdavis.edu/~joy/ecs277/other-notes/Numerical-Methods-for-Particle-Tracing-in-Vector-Fields.pdf
        
        # capitals are gridded data; lower case is point data
        
        # Interpolate the velocities to the current particle positions.
        # Remember the RectBivariateSpline uses strange conventions
        # for coordinates.
        u0 = RectBivariateSpline(Y, X, U0, kx=1, ky=1,
                #bbox=[self.xmin, self.xmax, self.ymin, self.ymax]
                ).ev(self.y, self.x)
        v0 = RectBivariateSpline(Y, X, V0, kx=1, ky=1,
                #bbox=[self.xmin, self.xmax, self.ymin, self.ymax]
                ).ev(self.y, self.x)
        u1interp = RectBivariateSpline(Y, X, U1, kx=1, ky=1)
                #bbox=[self.xmin, self.xmax, self.ymin, self.ymax])
        v1interp = RectBivariateSpline(Y, X, V1, kx=1, ky=1)
                #bbox=[self.xmin, self.xmax, self.ymin, self.ymax]) 
                
        k1u = dt*u0
        k1v = dt*v0
        x11 = self.wrap_x(self.x + 0.5*k1u)
        y11 = self.wrap_y(self.y + 0.5*k1v)
        u11 = u1interp.ev(y11, x11)
        v11 = v1interp.ev(y11, x11)
        k2u = dt*u11
        k2v = dt*v11
        x12 = self.wrap_x(self.x + 0.5*k2u)
        y12 = self.wrap_y(self.y + 0.5*k2v)
        u12 = u1interp.ev(y12, x12)
        v12 = v1interp.ev(y12, x12)
        k3u = dt*u12
        k3v = dt*v12
        x13 = self.wrap_x(self.x + k3u)
        y13 = self.wrap_y(self.y + k3v)
        u13 = u1interp.ev(y13, x13)
        v13 = v1interp.ev(y13, x13)
        k4u = dt*u13
        k4v = dt*v13
        
        # update
        self.x = self.wrap_x( self.x +
                    6**1*(k1u + 2*k2u + 2*k3u + k4u))
        self.y = self.wrap_y( self.y +
                    6**1*(k1v + 2*k2v + 2*k3v + k4v))
                
    
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
            
        
        
        
        
        