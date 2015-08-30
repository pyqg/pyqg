###########################
Two Layer QG Model Example
###########################

Here is a quick overview of how to use the two-layer model. See the
:py:class:`pyqg.QGModel` api documentation for further details.

First import numpy, matplotlib, and pyqg:

.. ipython:: python

    import numpy as np
    from matplotlib import pyplot as plt
    import pyqg

Initialize and Run the Model
----------------------------

Here we set up a model which will run for 10 years and start averaging
after 5 years. There are lots of parameters that can be specified as
keyword arguments but we are just using the defaults.


.. ipython:: python

    year = 24*60*60*360.
    m = pyqg.QGModel(tmax=10*year, twrite=10000, tavestart=5*year)
    m.run()
    
Visualize Output
----------------

We access the actual pv values through the attribute ``m.q``. The first axis
of ``q`` corresponds with the layer number. (Remeber that in python, numbering
starts at 0.)

.. ipython:: python
    
    q_upper = m.q[0] + m.Qy[0]*m.y
    plt.contourf(m.x, m.y, q_upper, 12, cmap='RdBu_r')
    plt.xlabel('x'); plt.ylabel('y'); plt.title('Upper Layer PV')
    @savefig q_contourf.png width=5in
    plt.colorbar();

Plot Diagnostics
----------------

The model automatically accumulates averages of certain diagnostics. We can 
find out what diagnostics are available by calling

.. ipython:: python

    m.describe_diagnostics()

To look at the wavenumber energy spectrum, we plot the `KEspec` diagnostic.
(Note that summing along the l-axis, as in this example, does not give us
a true *isotropic* wavenumber spectrum.)

.. ipython:: python

    kespec_u = m.get_diagnostic('KEspec')[0].sum(axis=0)
    kespec_l = m.get_diagnostic('KEspec')[1].sum(axis=0)
    plt.loglog( m.kk, kespec_u, '.-' )
    plt.loglog( m.kk, kespec_l, '.-' )
    plt.legend(['upper layer','lower layer'], loc='lower left')
    plt.ylim([1e-9,1e-3]); plt.xlim([m.kk.min(), m.kk.max()])
    plt.xlabel(r'k (m$^{-1}$)'); plt.grid()
    @savefig ke_spectrum.png width=5in
    plt.title('Kinetic Energy Spectrum')

We can also plot the spectral fluxes of energy.

.. ipython:: python

    ebud = [ -m.get_diagnostic('APEgenspec').sum(axis=0),
             -m.get_diagnostic('APEflux').sum(axis=0),
             -m.get_diagnostic('KEflux').sum(axis=0),
             -m.rek*m.del2*m.get_diagnostic('KEspec')[1].sum(axis=0)*m.M**2 ]
    ebud.append(-np.vstack(ebud).sum(axis=0))
    ebud_labels = ['APE gen','APE flux','KE flux','Diss.','Resid.']
    [plt.semilogx(m.kk, term) for term in ebud]
    plt.legend(ebud_labels, loc='upper right')
    plt.xlim([m.kk.min(), m.kk.max()])
    plt.xlabel(r'k (m$^{-1}$)'); plt.grid()
    @savefig ke_spectral_flux.png width=5in
    plt.title('Spectral Energy Transfers')
    
