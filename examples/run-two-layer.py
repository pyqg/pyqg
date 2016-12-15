import numpy as np
from matplotlib import pyplot as plt
import pyqg

U1 = 0.25 * np.ones((64))
U2 = 0.3 * np.ones((64))
m = pyqg.QGModel(tavestart=0,  dt=8000, U1=U1,U2=U2)

for snapshot in m.run_with_snapshots(
        tsnapstart=0, tsnapint=1000*m.dt):
    print(m.Qy1)
    # plt.clf()
    # plt.imshow(m.q[0] + m.Qy1 * m.y)
    # plt.clim([0,  m.Qy1 * m.W])
    # plt.pause(0.01)
    # plt.draw()

# now the model is done
