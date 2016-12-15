import numpy as np
from matplotlib import pyplot as plt
import pyqg

a = 0.25
c=5
nx = 256
b=nx/2
U1 = a * np.exp(-(np.arange(nx)-b)**2/(2*c**2))
U2 = U1/2
m = pyqg.QGModel(tavestart=0,  dt=8000, U1=U1,U2=U2, nx=nx)

for snapshot in m.run_with_snapshots(
        tsnapstart=0, tsnapint=1000*m.dt):
    plt.clf()
    # plt.imshow(m.q[0] + m.Qy1 * m.y)
    plt.imshow(m.u[0,:,:])
    # plt.clim([0,  m.Qy1 * m.W])
    plt.colorbar()
    plt.pause(0.01)
    plt.draw()

# now the model is done
