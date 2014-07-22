import numpy as np
import twolayer_qg

def test_the_model():

    year = 360*86400.
    m = twolayer_qg.QGModel(
            nx=32,                      # grid resolution
            ny=None,
            L=1e6,                      # domain size 
            W=None,
            # physical parameters
            beta=1.5e-11,               # gradient of coriolis parameter
            rek=5.787e-7,               # linear drag in lower layer
            rd=30000.0,                 # deformation radius
            delta=0.25,                 # layer thickness ratio (H1/H2)
            U1=0.05,                    # upper layer flow
            U2=0.0,                     # lower layer flow
            # timestepping parameters
            dt=12800.,                   # numerical timstep
            tmax=2*year,          # total time of integration
            tavestart=1*year,     # start time for averaging
            taveint=12800.,
            # diagnostics parameters
            diagnostics_list='all'      # which diagnostics to output)
            )

    # set initial conditions
    m.set_q1q2(
            (1e-6*np.cos(2*5*np.pi * m.x / m.L) +
             1e-7*np.cos(2*5*np.pi * m.y / m.W)),
            np.zeros_like(m.x) )
            
    print (m.q1**2).sum()
    m.run()

    assert m.t == 62208000.0
    np.testing.assert_allclose((m.q1**2).sum(), 1.638280961476609e-08)
    np.testing.assert_allclose(m.get_diagnostic('EKE1'), 0.00156374619577)

if __name__ == "__main__":
    test_the_model()
    
