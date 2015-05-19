from pylab import *
from pyqg import qg_model
import matplotlib

m = qg_model.QGModel(tavestart=0,  dt=8000)
# set initial conditions
m.set_q1q2(
        (1e-6*np.cos(2*5*np.pi * m.x / m.L) +
         1e-7*np.cos(2*5*np.pi * m.y / m.W)),
        np.zeros_like(m.x) )

for n in range(5000):
    m._step_forward()


try:
    q1 = m.q[0]
    q2 = m.q[1]
    qh1 = m.qh[0]
    qh2 = m.qh[1]
    dqh1dt = m.dqhdt[0]
    dqh2dt = m.dqhdt[1]
    tit = 'New Kernel'
except AttributeError:
    q1 = m.q1
    q2 = m.q2
    qh1 = m.qh1
    qh2 = m.qh2
    dqh1dt = m.dqh1dt
    dqh2dt = m.dqh2dt
    tit = 'Old Kernel'

def masked_power(h):
    return np.ma.masked_invalid(np.log10(np.abs(h)))


fig = figure(figsize=(20,12))

subplot(231)
imshow(q1)
clim([-m.beta1*m.L/100., m.beta1*m.L/100.])

subplot(234)
imshow(q2)
clim([-m.beta1*m.L/100., m.beta1*m.L/100.])

subplot(232)
pcolormesh(masked_power(qh1))
clim([-10,-5])

subplot(235)
pcolormesh(masked_power(qh2))
clim([-10,-5])

subplot2grid((2,6), (0,4))
pcolormesh(np.ma.masked_inside(np.real(dqh1dt),-1e-18,1e-18),
             norm=matplotlib.colors.SymLogNorm(1e-20))
clim([-1e-7,1e-7])
title('real')

subplot2grid((2,6), (0,5))
pcolormesh(np.ma.masked_inside(np.imag(dqh1dt),-1e-18,1e-18),
            norm=matplotlib.colors.SymLogNorm(1e-20))
clim([-1e-7,1e-7])
title('imag')


subplot2grid((2,6), (1,4))
pcolormesh(np.ma.masked_inside(np.real(dqh2dt),-1e-18,1e-18),
            norm=matplotlib.colors.SymLogNorm(1e-20))
clim([-1e-7,1e-7])
title('real')

subplot2grid((2,6), (1,5))
pcolormesh(np.ma.masked_inside(np.imag(dqh2dt),-1e-18,1e-18),
             norm=matplotlib.colors.SymLogNorm(1e-20))
clim([-1e-7,1e-7])
title('imag')


fig.text(0.5, 0.95, tit, ha='center')
show()
