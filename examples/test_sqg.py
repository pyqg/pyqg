import matplotlib.pyplot as plt
import numpy as np
import pyqg

m_bqg = pyqg.BTModel( nx = 512, L = 2.*np.pi, tmax = 26.005, dt=0.005, rek=0)
m_sqg = pyqg.SQGModel(nx = 512, L = 2.*np.pi, tmax = 26.005, dt=0.005, rek=0)

x = np.linspace(m_sqg.dx/2,2*np.pi,m_sqg.nx) - np.pi
y = np.linspace(m_sqg.dy/2,2*np.pi,m_sqg.ny) - np.pi
x,y = np.meshgrid(x,y)

bi = (-np.exp(-(x**2 + (4.0*y)**2)/(m_sqg.L/6.0)**2))[np.newaxis,:,:]
m_sqg.b = bi
m_sqg._invert()

# run the SQG model
for snapshot in m_sqg.run_with_snapshots(tsnapstart=0., tsnapint=400*m_sqg.dt):
    plt.rcParams['image.cmap'] = 'RdBu'
    plt.clf()
    p1 = plt.imshow(m_sqg.b.squeeze())
    plt.title('t={:.2f}'.format(m_sqg.t))
    plt.colorbar()
    plt.clim([-1, 0])
    plt.xticks([])
    plt.yticks([])
    plt.show()

qi = (-np.exp(-(x**2 + (4.0*y)**2)/(m_bqg.L/6.0)**2))[np.newaxis,:,:]
m_bqg.q = qi
m_bqg._invert()

# run the BTQG model
for snapshot in m_bqg.run_with_snapshots(tsnapstart=0., tsnapint=400*m_bqg.dt):
    plt.rcParams['image.cmap'] = 'RdBu'
    plt.clf()
    p1 = plt.imshow(m_bqg.q.squeeze())
    plt.title('t={:.2f}'.format(m_bqg.t))
    plt.colorbar()
    plt.clim([-1, 0])
    plt.xticks([])
    plt.yticks([])
    plt.show()

#4) RingForcing
#5) create b_parameterization

print("Script test_sqg.py has terminted.")

# if we define f in model.py, no need to define f_0 in sqg_model.  
# in sqg_model, must advect b not q
# need to define a function to compute q from b, maybe (in kernel,pyx)
# now do we compute velocity in sqg_model?

# normally use qh, dq, etc but if sqg then use bh, db, etc 
# create a function, evalaute PV and buoyancy, given p compute q or b.
