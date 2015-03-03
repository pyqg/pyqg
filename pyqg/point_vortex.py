import numpy as np
from particles import LagrangianParticleArray2D
r_twopi = (2*np.pi)**-1

class PointVortexArray2D(LagrangianParticleArray2D):
    """Keeps track of a set of point vortices.""" 
    
    def __init__(self, x0, y0, s0, **kwargs):
        
        # strength
        self.s = s0
        
        LagrangianParticleArray2D.__init__(self,
            x0, y0, **kwargs)
            
        # need to remember previous position
        self.xprev = self.x.copy()
        self.yprev = self.y.copy()
        
    def step_forward_vortices(self, dt):       
        # create interpolation functions which return u and v
        uv0fun = (lambda x, y : self.calc_uv(x, y, prev=True))
        uv1fun = (lambda x, y : self.calc_uv(x, y))
     
        dx, dy = self.rk4_integrate(self.x, self.y, uv0fun, uv1fun, dt)

        self.xprev = self.x.copy()
        self.yprev = self.y.copy()
        self.x = self.wrap_x(self.x + dx)
        self.y = self.wrap_y(self.y + dy)

    def calc_uv(self, x, y, prev=False):
        """Calculate velocity at x and y points due to vortex velocity field.
        Assumes x and y are vortex positions and are ordered the same as
        x0 and y0. The ordering is used to neglect to vortex self interaction."""
        assert len(x) == self.N
        assert len(y) == self.N
        u = np.zeros(self.N, self.x.dtype)
        v = np.zeros(self.N, self.y.dtype)
        for n in xrange(self.N):
            # don't include self interaction
            if prev:
                x0 = self.xprev[np.r_[:n,n+1:self.N]]
                y0 = self.yprev[np.r_[:n,n+1:self.N]]
            else:
                x0 = self.x[np.r_[:n,n+1:self.N]]
                y0 = self.y[np.r_[:n,n+1:self.N]]
            s0 = self.s[np.r_[:n,n+1:self.N]]
            u0, v0 = self.uv_at_xy(x[n], y[n], x0, y0, s0)
            u[n] = u0.sum()
            v[n] = v0.sum()
        return u, v
        
    def uv_at_xy(self, x, y, x0, y0, s0):
        """Returns two arrays of u, v"""
        dx, dy = self.distance(x0, y0, x, y)
        #print 'dx, dy:', dx, dy
        rr2 = (dx**2 + dy**2)**-1
        u = - s0 * dy * r_twopi * rr2
        v = s0 * dx * r_twopi * rr2
        #print 'u, v', u, v
        return u, v

