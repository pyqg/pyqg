import pyqg
import time
import cProfile
import pstats
import numpy as np

tmax = 8000*1000

dtfac = 64 * 8000.

mynx = [32, 64, 128, 256, 512, 1024, 2048]
mynth = [1,2,4,8,16,32]
res = np.zeros((len(mynx), 5))


print 'nx, threads, timesteps, time'
for j, nx in enumerate(mynx):

    dt = dtfac / nx

    #for i, (use_fftw, nth) in enumerate([(False, 1), (True, 1),
    #    (True, 2), (True, 4), (True, 8)]):
    for i, nth in enumerate(mynth):

        m = pyqg.QGModel(nx=nx, tmax=tmax, dt=dt, ntd=nth,
                         # no output    
                         twrite=np.inf,
                         # no time average
                         taveint=np.inf,)


    
        tic = time.time()
        m.run()
        toc = time.time()
        tottime = toc-tic
        #res[j,i] = tottime
        #print 'nx=%3d, fftw=%g, threads=%g: %g' % (nx, use_fftw, nth, tottime)
        print '%3d, %3d, %8d, %10.4f' % (nx, nth, m.tc, tottime)
    


# # profiling
# prof = cProfile.Profile()
# prof.run('m.run()')
# p = pstats.Stats(prof)
# p.sort_stats('cum').print_stats(0.3)
