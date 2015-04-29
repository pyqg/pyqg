import numpy as np
from pyqg import qg_model

def test_the_model(rtol=1e-15):
    """Make sure the results are correct within relative tolerance rtol."""

    year = 360*86400.
    m = qg_model.QGModel(
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
            tmax=3*year,          # total time of integration
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
                
    m.run()

    q1norm = (m.q1**2).sum()

    print 'time:       %g' % m.t
    assert m.t == 93312000.0
    print 'q1norm:     %.15e' % q1norm
    np.testing.assert_allclose(q1norm, 9.723198783759038e-08, rtol)
    print 'EKE1:       %.15e' % m.get_diagnostic('EKE1')
    np.testing.assert_allclose(m.get_diagnostic('EKE1'),
                5.695448642915733e-03, rtol)
    print 'EKE2:       %.15e' % m.get_diagnostic('EKE2')
    np.testing.assert_allclose(m.get_diagnostic('EKE2'),
                1.088253274803528e-04, rtol)
    print 'APEge:     %.15e' % m.get_diagnostic('APEgen')
    np.testing.assert_allclose(m.get_diagnostic('APEgen'),
                8.842056320175081e-08, rtol)
    print 'EKEdiss:    %.15e' % m.get_diagnostic('EKEdiss')
    np.testing.assert_allclose(m.get_diagnostic('EKEdiss'),
                6.368668363708053e-08, rtol)
                
    entspec = abs(m.get_diagnostic('entspec')).sum()
    print 'entspec:    %.15e' % entspec
    np.testing.assert_allclose(entspec,
                5.703438193477885e-07, rtol)
    
    apeflux = abs(m.get_diagnostic('APEflux')).sum()
    print 'apeflux:    %.15e' % apeflux
    np.testing.assert_allclose(apeflux,
                9.192940039964286e-05, rtol)
                
    KEflux = abs(m.get_diagnostic('KEflux')).sum()
    print 'KEflux:     %.15e' % KEflux
    np.testing.assert_allclose(KEflux,
                1.702621259427053e-04, rtol)

    APEgenspec = abs(m.get_diagnostic('APEgenspec')).sum()
    print 'APEgenspec: %.15e' % APEgenspec
    np.testing.assert_allclose(APEgenspec,
                9.058591846403974e-05, rtol)
                
    KE1spec = abs(m.get_diagnostic('KE1spec')).sum()
    print 'KE1spec:    %.15e' % KE1spec
    np.testing.assert_allclose(KE1spec,
                3.338261440237941e+03, rtol)

    KE2spec = abs(m.get_diagnostic('KE2spec')).sum()
    print 'KE2spec:    %.15e' % KE2spec
    np.testing.assert_allclose(KE2spec,
                7.043282793801889e+01, rtol)
                              
               

if __name__ == "__main__":
    test_the_model()
    
