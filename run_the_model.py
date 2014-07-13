from pylab import *
import twolayer_qg

m = twolayer_qg.QGModel(dt=8000, tavestart=0)

figure()
for snapshot in m.run_with_snapshots(tsnapint=4320000):
    # do something like plot
    print 'Mean EKE1: %g' % (m.get_diagnostic('EKE1'))
    clf()
    contourf(m.q1 + m.beta1*m.y,18)
    colorbar()
    pause(0.01)
    show()

# now the model is done