from pylab import *
from pyqg import qg_model

m = qg_model.QGModel(tavestart=0,  dt=8000)

for snapshot in m.run_with_snapshots(
        tsnapstart=0, tsnapint=1000*m.dt):
    clf()
    imshow(m.q[0] + m.beta * m.y)
    clim([0,  m.beta * m.W])
    show()
    pause(0.01)
# now the model is done
