import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from pyqg import sqg_model

# the model object
year = 1.
m = sqg_model.SQGModel(L=2.*pi,nx=128, tmax = 100*year,
        beta = 0., Nb = 1., H = 1., rek = 0., rd = None, dt = 0.01,
                     taveint=year, use_fftw=True, ntd=8)

# run the model and plot some figs
plt.rcParams['image.cmap'] = 'RdBu'

# Choose ICs from Held et al. (1996)
# case i) Elliptical vortex
x = np.linspace(m.dx/2,2*np.pi,m.nx) - np.pi
y = np.linspace(m.dy/2,2*np.pi,m.ny) - np.pi
x,y = np.meshgrid(x,y)

qi = 0.01*np.exp(-(x**2 + (4.0*y)**2)/(m.L/6.0)**2)
m.set_q(qi[np.newaxis,:,:])

plt.ion()

for snapshot in m.run_with_snapshots(tsnapstart=0, tsnapint=100*m.dt):
    plt.clf()
    p1 = plt.imshow(m.q.squeeze() + m.beta * m.y)
    #plt.clim([-30., 30.])
    plt.title('t='+str(m.t))
    plt.colorbar()
    plt.clim([0, 0.01])
    plt.xticks([])
    plt.yticks([])
    plt.draw()

    plt.pause(0.01)


plt.ioff()
plt.show()
