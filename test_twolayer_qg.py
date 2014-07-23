import numpy as np
import twolayer_qg

def test_the_model(rtol=1e-15):
    """Make sure the results are correct within relative tolerance rtol."""

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
    np.testing.assert_allclose(q1norm, 9.561430503712755e-08, rtol)
    print 'EKE1:       %.15e' % m.get_diagnostic('EKE1')
    np.testing.assert_allclose(m.get_diagnostic('EKE1'),
                0.008183776317328265, rtol)
    print 'EKE2:       %.15e' % m.get_diagnostic('EKE2')
    np.testing.assert_allclose(m.get_diagnostic('EKE2'),
                0.00015616609033468579, rtol)
    print 'APEgen:     %.15e' % m.get_diagnostic('APEgen')
    np.testing.assert_allclose(m.get_diagnostic('APEgen'),
                2.5225558013107688e-07, rtol)
    print 'EKEdiss:    %.15e' % m.get_diagnostic('EKEdiss')
    np.testing.assert_allclose(m.get_diagnostic('EKEdiss'),
                1.4806764171539711e-07, rtol)
                
    entspec = abs(m.get_diagnostic('entspec')).sum()
    print 'entspec:    %.15e' % entspec
    np.testing.assert_allclose(entspec,
                1.5015983257921716e-06, rtol)
    
    apeflux = abs(m.get_diagnostic('APEflux')).sum()
    print 'apeflux:    %.15e' % apeflux
    np.testing.assert_allclose(apeflux,
                0.00017889483037254459, rtol)
                
    KEflux = abs(m.get_diagnostic('KEflux')).sum()
    print 'KEflux:     %.15e' % KEflux
    np.testing.assert_allclose(KEflux,
                0.00037067750708912918, rtol)

    APEgenspec = abs(m.get_diagnostic('APEgenspec')).sum()
    print 'APEgenspec: %.15e' % APEgenspec
    np.testing.assert_allclose(APEgenspec,
                0.00025837684260178754, rtol)
                
    KE1spec = abs(m.get_diagnostic('KE1spec')).sum()
    print 'KE1spec:    %.15e' % KE1spec
    np.testing.assert_allclose(KE1spec,
                8581.3114357188006, rtol)

    KE2spec = abs(m.get_diagnostic('KE2spec')).sum()
    print 'KE2spec:    %.15e' % KE2spec
    np.testing.assert_allclose(KE2spec,
                163.75201433878425, rtol)
                              
               

if __name__ == "__main__":
    test_the_model()
    
