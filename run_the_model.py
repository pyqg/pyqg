from pylab import *
import twolayer_qg

m = twolayer_qg.QGModel(tavestart=0)

figure()
for snapshot in m.run_with_snapshots(tsnapint=4320000):
    # do something like plot
    clf()
    pcolormesh(m.q1 + m.beta1*m.y)
    colorbar()
    pause(0.01)
    show()

# now the model is done