import numpy as np
import pyqg

m_bqg = pyqg.BTModel( nx = 16, L = 2.*np.pi)
m_sqg = pyqg.SQGModel(nx = 16, L = 2.*np.pi)

x = np.linspace(m_sqg.dx/2,2*np.pi,m_sqg.nx) - np.pi
y = np.linspace(m_sqg.dy/2,2*np.pi,m_sqg.ny) - np.pi
x,y = np.meshgrid(x,y)

bi = (-np.exp(-(x**2 + (4.0*y)**2)/(m_sqg.L/6.0)**2))[np.newaxis,:,:]
m_sqg.b = bi

qi = (-np.exp(-(x**2 + (4.0*y)**2)/(m_bqg.L/6.0)**2))[np.newaxis,:,:]
m_bqg.q = qi

#1) set b

#2) compute u,v
#3) compute q
#4) RingForcing
#5) create b_parameterization

#FJP: set pi to be whatever we need to be consistent in both.  We can use fft's.  Might make checking easier.
#6) Time step: use q for most but b for SQG=True.
#6a) first invert
#6b) second forward, must add
#6c) compute velocities, test
#6d) forward Euler step
#6e) AB2 and AB3

print("Script test_sqg.py has terminted.")

# if we define f in model.py, no need to define f_0 in sqg_model.  
# in sqg_model, must advect b not q
# need to define a function to compute q from b, maybe (in kernel,pyx)
# now do we compute velocity in sqg_model?

# normally use qh, dq, etc but if sqg then use bh, db, etc 
# create a function, evalaute PV and buoyancy, given p compute q or b.


SQG = False

if not SQG:
    print("Not an SQG model")
else:
    print("An SQG model")

