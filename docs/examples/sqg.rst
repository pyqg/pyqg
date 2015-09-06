
Surface Quasi-Geostrophic (SQG) Model
=====================================

Here will will use pyqg to reproduce the results of the paper: I. M.
Held, R. T. Pierrehumbert, S. T. Garner and K. L. Swanson (1985).
Surface quasi-geostrophic dynamics. Journal of Fluid Mechanics, 282, pp
1-20 [doi:: http://dx.doi.org/10.1017/S0022112095000012)

.. code:: python

    import matplotlib.pyplot as plt
    import numpy as np
    from numpy import pi
    %matplotlib inline
    from pyqg import sqg_model

Surface quasi-geostrophy (SQG) is a relatively simple model that
describes surface intensified flows due to buoyancy. One of it's
advantages is that it only has two spatial dimensions but describes a
three-dimensional solution.

If we define :math:`b` to be the buoyancy, then the evolution equation
for buoyancy at each the top and bottom surface is

.. math::


   \partial_t b + J(\psi, b) = 0.

The invertibility relation between the streamfunction, :math:`\psi`, and
the buoyancy, :math:`b`, is hydrostatic balance

.. math::


   b = \partial_z \psi. 

 Using the fact that the Potential Vorticity is exactly zero in the
interior of the domain and that the domain is semi-infinite, yields that
the inversion in Fourier space is,

.. math::


   \hat b = K \hat \psi.

Held et al. studied several different cases, the first of which was the
nonlinear evolution of an elliptical vortex. There are several other
cases that they studied and people are welcome to adapt the code to
study those as well. But here we focus on this first example for
pedagogical reasons.

.. code:: python

    # create the model object
    year = 1.
    m = sqg_model.SQGModel(L=2.*pi,nx=512, tmax = 26.005,
            beta = 0., Nb = 1., H = 1., rek = 0., rd = None, dt = 0.005,
                         taveint=1, ntd=4)
    # in this example we used ntd=4, four threads
    # if your machine has more (or fewer) cores available, you could try changing it

Initial condition
-----------------

The initial condition is an elliptical vortex,

.. math::


   b = 0.01 \exp( - (x^2 + (4y)^2)/(L/y)^2

where :math:`L` is the length scale of the vortex in the :math:`x`
direction. The amplitude is 0.01, which sets the strength and speed of
the vortex. The aspect ratio in this example is :math:`4` and gives rise
to an instability. If you reduce this ratio sufficiently you will find
that it is stable. Why don't you try it and see for yourself?

.. code:: python

    # Choose ICs from Held et al. (1995)
    # case i) Elliptical vortex
    x = np.linspace(m.dx/2,2*np.pi,m.nx) - np.pi
    y = np.linspace(m.dy/2,2*np.pi,m.ny) - np.pi
    x,y = np.meshgrid(x,y)
    
    qi = -np.exp(-(x**2 + (4.0*y)**2)/(m.L/6.0)**2)

.. code:: python

    # initialize the model with that initial condition
    m.set_q(qi[np.newaxis,:,:])

.. code:: python

    # Plot the ICs
    plt.rcParams['image.cmap'] = 'RdBu'
    plt.clf()
    p1 = plt.imshow(m.q.squeeze() + m.beta * m.y)
    plt.title('b(x,y,t=0)')
    plt.colorbar()
    plt.clim([-1, 0])
    plt.xticks([])
    plt.yticks([])
    plt.show()



.. image:: sqg_files/sqg_8_0.png


Runing the model
----------------

Here we demonstrate how to use the ``run_with_snapshots`` feature to
periodically stop the model and perform some action (in this case,
visualization).

.. code:: python

    for snapshot in m.run_with_snapshots(tsnapstart=0, tsnapint=400*m.dt):
        plt.clf()
        p1 = plt.imshow(m.q.squeeze() + m.beta * m.y)
        #plt.clim([-30., 30.])
        plt.title('t='+str(m.t))
        plt.colorbar()
        plt.clim([-1, 0])
        plt.xticks([])
        plt.yticks([])
        plt.show()



.. image:: sqg_files/sqg_10_0.png



.. image:: sqg_files/sqg_10_1.png


.. parsed-literal::

    t=               4, tc=      1000: cfl=0.239869, ke=0.005206463



.. image:: sqg_files/sqg_10_3.png



.. image:: sqg_files/sqg_10_4.png



.. image:: sqg_files/sqg_10_5.png


.. parsed-literal::

    t=              10, tc=      2000: cfl=0.267023, ke=0.005206261



.. image:: sqg_files/sqg_10_7.png



.. image:: sqg_files/sqg_10_8.png


.. parsed-literal::

    t=              15, tc=      3000: cfl=0.251901, ke=0.005199422



.. image:: sqg_files/sqg_10_10.png



.. image:: sqg_files/sqg_10_11.png



.. image:: sqg_files/sqg_10_12.png


.. parsed-literal::

    t=              20, tc=      4000: cfl=0.259413, ke=0.005189615



.. image:: sqg_files/sqg_10_14.png



.. image:: sqg_files/sqg_10_15.png


.. parsed-literal::

    t=              24, tc=      5000: cfl=0.255257, ke=0.005176248



.. image:: sqg_files/sqg_10_17.png


Compare these results with Figure 2 of the paper. In this simulation you
see that as the cyclone rotates it develops thin arms that spread
outwards and become unstable because of their strong shear. This is an
excellent example of how smaller scale vortices can be generated from a
mesoscale vortex.

You can modify this to run it for longer time to generate the analogue
of their Figure 3.

