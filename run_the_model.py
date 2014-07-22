from pylab import *
import twolayer_qg

m = twolayer_qg.QGModel(tavestart=0,  dt=8000)

for snapshot in m.run_with_snapshots(
        tsnapstart=155520000, tsnapint=m.dt):
    clf()
    imshow(m.q1)
    show()
    pause(0.01)
# now the model is done
