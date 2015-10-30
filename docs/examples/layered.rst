
Fully developed baroclinic instability of a 3-layer flow
========================================================

.. code:: python

    import numpy as np
    from numpy import pi
    from matplotlib import pyplot as plt
    %matplotlib inline
    
    import pyqg


.. parsed-literal::

    Vendor:  Continuum Analytics, Inc.
    Package: mkl
    Message: trial mode expires in 24 days
    Vendor:  Continuum Analytics, Inc.
    Package: mkl
    Message: trial mode expires in 24 days
    Vendor:  Continuum Analytics, Inc.
    Package: mkl
    Message: trial mode expires in 24 days


Set up
======

.. code:: python

    L =  1000.e3     # length scale of box    [m]
    Ld = 15.e3       # deformation scale      [m]
    kd = 1./Ld       # deformation wavenumber [m^-1]
    Nx = 64          # number of grid points
    
    H1 = 500.        # layer 1 thickness  [m]
    H2 = 1750.       # layer 2 
    H3 = 1750.       # layer 3 
    
    U1 = 0.05          # layer 1 zonal velocity [m/s]
    U2 = 0.01         # layer 2
    U3 = 0.00         # layer 3
    
    rho1 = 1025.
    rho2 = 1025.275
    rho3 = 1025.640
    
    rek = 1.e-7       # linear bottom drag coeff.  [s^-1]
    f0  = 0.0001236812857687059 # coriolis param [s^-1]
    beta = 1.2130692965249345e-11 # planetary vorticity gradient [m^-1 s^-1]
    
    Ti = Ld/(abs(U1))  # estimate of most unstable e-folding time scale [s]
    dt = Ti/500.   # time-step [s]
    tmax = 300*Ti      # simulation time [s]

.. code:: python

    m = pyqg.LayeredModel(nx=Nx, nz=3, U = [U1,U2,U3],V = [0.,0.,0.],L=L,f=f0,beta=beta,
                             H = [H1,H2,H3], rho=[rho1,rho2,rho3],rek=rek,
                            dt=dt,tmax=tmax, twrite=5000, tavestart=Ti*10)


.. parsed-literal::

    2015-10-29 23:34:04,035 - pyqg.model - INFO -  Logger initialized
    2015-10-29 23:34:04,108 - pyqg.model - INFO -  Kernel initialized


Initial condition
=================

.. code:: python

    sig = 1.e-7
    qi = sig*np.vstack([np.random.randn(m.nx,m.ny)[np.newaxis,],
                        np.random.randn(m.nx,m.ny)[np.newaxis,],
                        np.random.randn(m.nx,m.ny)[np.newaxis,]])
    m.set_q(qi)

Run the model
=============

.. code:: python

    m.run()


.. parsed-literal::

    2015-10-29 23:34:11,836 - pyqg.model - INFO -  Step: 5000, Time: 3.000000e+06, KE: 7.972189e-07, CFL: 0.002109
    2015-10-29 23:34:19,524 - pyqg.model - INFO -  Step: 10000, Time: 6.000000e+06, KE: 1.317866e-05, CFL: 0.002538
    2015-10-29 23:34:27,365 - pyqg.model - INFO -  Step: 15000, Time: 9.000000e+06, KE: 3.587654e-04, CFL: 0.006465
    2015-10-29 23:34:35,212 - pyqg.model - INFO -  Step: 20000, Time: 1.200000e+07, KE: 3.105440e-03, CFL: 0.019155
    2015-10-29 23:34:42,949 - pyqg.model - INFO -  Step: 25000, Time: 1.500000e+07, KE: 7.493171e-03, CFL: 0.029432
    2015-10-29 23:34:50,846 - pyqg.model - INFO -  Step: 30000, Time: 1.800000e+07, KE: 1.680096e-02, CFL: 0.034929
    2015-10-29 23:34:58,651 - pyqg.model - INFO -  Step: 35000, Time: 2.100000e+07, KE: 3.637336e-02, CFL: 0.043485
    2015-10-29 23:35:06,530 - pyqg.model - INFO -  Step: 40000, Time: 2.400000e+07, KE: 6.986884e-02, CFL: 0.068028
    2015-10-29 23:35:14,237 - pyqg.model - INFO -  Step: 45000, Time: 2.700000e+07, KE: 1.186980e-01, CFL: 0.070400
    2015-10-29 23:35:22,136 - pyqg.model - INFO -  Step: 50000, Time: 3.000000e+07, KE: 2.035187e-01, CFL: 0.096087
    2015-10-29 23:35:30,004 - pyqg.model - INFO -  Step: 55000, Time: 3.300000e+07, KE: 2.609871e-01, CFL: 0.088660
    2015-10-29 23:35:37,724 - pyqg.model - INFO -  Step: 60000, Time: 3.600000e+07, KE: 4.323064e-01, CFL: 0.125482
    2015-10-29 23:35:45,668 - pyqg.model - INFO -  Step: 65000, Time: 3.900000e+07, KE: 5.691566e-01, CFL: 0.130641
    2015-10-29 23:35:53,522 - pyqg.model - INFO -  Step: 70000, Time: 4.200000e+07, KE: 5.535315e-01, CFL: 0.100923
    2015-10-29 23:36:01,419 - pyqg.model - INFO -  Step: 75000, Time: 4.500000e+07, KE: 6.143957e-01, CFL: 0.133175
    2015-10-29 23:36:09,190 - pyqg.model - INFO -  Step: 80000, Time: 4.800000e+07, KE: 5.370919e-01, CFL: 0.105108
    2015-10-29 23:36:17,120 - pyqg.model - INFO -  Step: 85000, Time: 5.100000e+07, KE: 4.704621e-01, CFL: 0.081749
    2015-10-29 23:36:25,016 - pyqg.model - INFO -  Step: 90000, Time: 5.400000e+07, KE: 4.406778e-01, CFL: 0.097391
    2015-10-29 23:36:32,784 - pyqg.model - INFO -  Step: 95000, Time: 5.700000e+07, KE: 4.704886e-01, CFL: 0.106692
    2015-10-29 23:36:40,736 - pyqg.model - INFO -  Step: 100000, Time: 6.000000e+07, KE: 4.036577e-01, CFL: 0.093186
    2015-10-29 23:36:48,540 - pyqg.model - INFO -  Step: 105000, Time: 6.300000e+07, KE: 3.358218e-01, CFL: 0.093095
    2015-10-29 23:36:56,344 - pyqg.model - INFO -  Step: 110000, Time: 6.600000e+07, KE: 2.876956e-01, CFL: 0.070564
    2015-10-29 23:37:04,165 - pyqg.model - INFO -  Step: 115000, Time: 6.900000e+07, KE: 2.538065e-01, CFL: 0.080431
    2015-10-29 23:37:11,959 - pyqg.model - INFO -  Step: 120000, Time: 7.200000e+07, KE: 2.515991e-01, CFL: 0.081021
    2015-10-29 23:37:19,823 - pyqg.model - INFO -  Step: 125000, Time: 7.500000e+07, KE: 2.532351e-01, CFL: 0.092687
    2015-10-29 23:37:27,638 - pyqg.model - INFO -  Step: 130000, Time: 7.800000e+07, KE: 2.768969e-01, CFL: 0.115862
    2015-10-29 23:37:35,474 - pyqg.model - INFO -  Step: 135000, Time: 8.100000e+07, KE: 4.391087e-01, CFL: 0.141570
    2015-10-29 23:37:43,214 - pyqg.model - INFO -  Step: 140000, Time: 8.400000e+07, KE: 6.821671e-01, CFL: 0.154537
    2015-10-29 23:37:50,998 - pyqg.model - INFO -  Step: 145000, Time: 8.700000e+07, KE: 9.395894e-01, CFL: 0.152700


A snapshot and some diagnostics
===============================

.. code:: python

    plt.figure(figsize=(18,4))
    
    plt.subplot(131)
    plt.pcolormesh(m.x/m.rd,m.y/m.rd,(m.q[0,]+m.Qy[0]*m.y)/(U1/Ld),cmap='Spectral_r')
    plt.xlabel(r'$x/L_d$')
    plt.ylabel(r'$y/L_d$')
    plt.colorbar()
    plt.title('Layer 1 PV')
    
    plt.subplot(132)
    plt.pcolormesh(m.x/m.rd,m.y/m.rd,(m.q[1,]+m.Qy[1]*m.y)/(U1/Ld),cmap='Spectral_r')
    plt.xlabel(r'$x/L_d$')
    plt.ylabel(r'$y/L_d$')
    plt.colorbar()
    plt.title('Layer 2 PV')
    
    plt.subplot(133)
    plt.pcolormesh(m.x/m.rd,m.y/m.rd,(m.q[2,]+m.Qy[2]*m.y)/(U1/Ld),cmap='Spectral_r')
    plt.xlabel(r'$x/L_d$')
    plt.ylabel(r'$y/L_d$')
    plt.colorbar()
    plt.title('Layer 3 PV')




.. parsed-literal::

    <matplotlib.text.Text at 0x111c50050>



.. parsed-literal::

    /Users/crocha/anaconda/lib/python2.7/site-packages/matplotlib/collections.py:590: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      if self._edgecolors == str('face'):



.. image:: layered_files/layered_10_2.png


pyqg has a built-in method that computes the vertical modes

.. code:: python

    m.vertical_modes()

We can also project the solution onto the modes

.. code:: python

    pn = m.modal_projection(m.p)

.. code:: python

    plt.figure(figsize=(18,4))
    
    plt.subplot(131)
    plt.pcolormesh(m.x/m.rd,m.y/m.rd,pn[0]/(U1*Ld),cmap='Spectral_r')
    plt.xlabel(r'$x/L_d$')
    plt.ylabel(r'$y/L_d$')
    plt.colorbar()
    plt.title('Barotropic streamfunction')
    
    plt.subplot(132)
    plt.pcolormesh(m.x/m.rd,m.y/m.rd,pn[1]/(U1*Ld),cmap='Spectral_r')
    plt.xlabel(r'$x/L_d$')
    plt.ylabel(r'$y/L_d$')
    plt.colorbar()
    plt.title('1st baroclinic streamfunction')
    
    plt.subplot(133)
    plt.pcolormesh(m.x/m.rd,m.y/m.rd,pn[2]/(U1*Ld),cmap='Spectral_r')
    plt.xlabel(r'$x/L_d$')
    plt.ylabel(r'$y/L_d$')
    plt.colorbar()
    plt.title('2nd baroclinic streamfunction')




.. parsed-literal::

    <matplotlib.text.Text at 0x1128c1510>




.. image:: layered_files/layered_15_1.png


.. code:: python

    kespec_1 = m.get_diagnostic('KEspec')[0].sum(axis=0)
    kespec_2 = m.get_diagnostic('KEspec')[1].sum(axis=0)
    kespec_3 = m.get_diagnostic('KEspec')[2].sum(axis=0)
    
    
    plt.loglog( m.kk, kespec_1, '.-' )
    plt.loglog( m.kk, kespec_2, '.-' )
    plt.loglog( m.kk, kespec_3, '.-' )
    
    plt.legend(['layer 1','layer 2', 'layer 3'], loc='lower left')
    plt.ylim([1e-9,1e-0]); plt.xlim([m.kk.min(), m.kk.max()])
    plt.xlabel(r'k (m$^{-1}$)'); plt.grid()
    plt.title('Kinetic Energy Spectrum');



.. image:: layered_files/layered_16_0.png


.. code:: python

    ebud = [ m.get_diagnostic('APEgenspec').sum(axis=0),
             m.get_diagnostic('APEflux').sum(axis=0),
             m.get_diagnostic('KEflux').sum(axis=0),
             -m.rek*(m.Hi[-1]/m.H)*m.get_diagnostic('KEspec')[1].sum(axis=0)*m.M**2 ]
    ebud.append(-np.vstack(ebud).sum(axis=0))
    ebud_labels = ['APE gen','APE flux div.','KE flux div.','Diss.','Resid.']
    [plt.semilogx(m.kk, term) for term in ebud]
    plt.legend(ebud_labels, loc='upper right')
    plt.xlim([m.kk.min(), m.kk.max()])
    plt.xlabel(r'k (m$^{-1}$)'); plt.grid()
    plt.title('Spectral Energy Transfers');




.. image:: layered_files/layered_17_0.png


The dynamics here is similar to the reference experiment of `Larichev &
Held
(1995) <http://journals.ametsoc.org/doi/pdf/10.1175/1520-0485%281995%29025%3C2285%3AEAAFIA%3E2.0.CO%3B2>`__.
The APE generated through baroclinic instability is fluxed towards
deformation length scales, where it is converted into KE. The KE the
experiments and inverse tranfer, cascading up to the scale of the
domain. The mechanical bottom drag essentially removes the large scale
KE.

