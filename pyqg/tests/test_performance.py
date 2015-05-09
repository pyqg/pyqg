from pyqg import qg_model, model
import time
import cProfile
import pstats
import numpy as np

tmax = 104000000

dtfac = (64 * 8000.)

mynx = [32, 64, 128, 256]
res = np.zeros((len(mynx), 5))

for j, nx in enumerate(mynx):

    dt = dtfac / nx

    for i, (use_fftw, nth) in enumerate([(False, 1), (True, 1),
        (True, 2), (True, 4), (True, 8)]):

        m = qg_model.QGModel(nx=64, tmax=tmax, dt=dt,
                             use_fftw=use_fftw, ntd=nth)
    
        tic = time.time()
        m.run()
        toc = time.time()
        tottime = toc-tic
        res[j,i] = tottime
        print 'nx=%3d, fftw=%g, threads=%g: %g' % (nx, use_fftw, nth, tottime)
    


# # profiling
# prof = cProfile.Profile()
# prof.run('m.run()')
# p = pstats.Stats(prof)
# p.sort_stats('cum').print_stats(0.3)
