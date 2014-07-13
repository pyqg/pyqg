from pylab import *
import twolayer_qg
import particles
import time

m = twolayer_qg.QGModel(tavestart=0,  dt=8000)

# set up lagrangian particles
Npart = 100 # number of particles
x0 = rand(Npart)*m.L
y0 = rand(Npart)*m.W

lpa = particles.LagrangianParticleArray2D(x0, y0, 
        periodic_in_x=True, periodic_in_y=True,
        xmin=0, xmax=m.L, ymin=0, ymax=m.W)

# set up extended grid for lagrangian particles
x = np.hstack([m.x[0,0]-m.dx, m.x[0,:], m.x[0,-1]+m.dx])
y = np.hstack([m.y[0,0]-m.dy, m.y[:,0], m.y[-1,0]+m.dy])

close('all')
figure(figsize=(12,12))

# number of particles to save
Nhold = 10
particle_history = zeros((Nhold,2,Npart))

started_advecting = False
tprev = time.time()
for snapshot in m.run_with_snapshots(
        tsnapstart=155520000, tsnapint=m.dt):

    # set up velocities for lagrangian advection
    # need a view with wrapped values
    u = m.u1[r_[-1,0:m.ny,0]][:,r_[-1,0:m.nx,0]]
    v = m.v1[r_[-1,0:m.ny,0]][:,r_[-1,0:m.nx,0]]

    if started_advecting:
        lpa.runge_kutta_step(
            x, y, uprev, vprev, u, v, m.dt)
        particle_history = np.roll(
            particle_history, 1, axis=0
        )
        particle_history[0,0,:] = lpa.x
        particle_history[0,1,:] = lpa.y
    
    # daily plots
    if mod(m.t, 24*60*60*2)==0:
        clf()
        contourf(m.x, m.y,
            m.q1 + m.beta1*m.y, linspace(0, m.beta1*m.W,20),extend='both', cmap='RdBu_r')
        quiver(m.x,m.y,m.u1,m.v1, scale=10)
        plot(particle_history[1:,0,:].ravel(),
                 particle_history[1:,1,:].ravel(), 'ko')                 
        plot(particle_history[0,0,:],
                 particle_history[0,1,:], 'go')
        show()
        pause(0.01)
    uprev = u.copy()
    vprev = v.copy()
    started_advecting = True
# now the model is done