import numpy as np

import twolayer_qg

m = twolayer_qg.QGModel(tavestart=0,  dt=8000)

for snapshot in m.run_with_snapshots(
        tsnapstart=155520000, tsnapint=m.dt):
        print np.real((m.q1*m.q1.conj()).sum)