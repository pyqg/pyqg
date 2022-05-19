Parameterizations
=================

pyqg support parameterizations, which are functions that take a
:code:`pyqg.Model` and return an additional term to add to its potential
vorticity tendency every timestep (or two terms to add to each velocity
tendency, in which case we apply them to PV after taking their curl).
Typically, parameterizations are used to account for the contribution of
phenomena occuring at subgrid scales. This approach can be a computationally
efficient way to improve the physical realism of simulations without needing to
increase their spatial resolution (which can be very expensive).

Using predefined parameterizations
----------------------------------

pyqg implements a number of predefined parameterizations (see
:ref:`parameterizations-api` for a full list). You can use these in a
:code:`pyqg.Model` as follows:

.. code-block:: python

    param = pyqg.BackscatterBiharmonic(smag_constant=0.1, back_constant=0.95)
    model = pyqg.QGModel(parameterization=param)

Note that parameterizations either target the tendencies of potential vorticity
:math:`q` or the velocities :math:`u` and :math:`v`. If you have two
parameterizations with the same target, you can add them together, even as a
weighted sum. If they have different targets, you can still use both, but they
must be passed in as separate :code:`q_parameterization` and
:code:`uv_parameterization` arguments:

.. code-block:: python

    param1 = pyqg.Smagorinsky() # targets uv
    param2 = pyqg.ZannaBolton2020() # also targets uv
    good_model = pyqg.QGModel(parameterization=param1 + 0.25*param2) # this works!

    param3 = pyqg.BackscatterBiharmonic() # targets q
    bad_model = pyqg.QGModel(parameterization=param1 + param3) # this will error!

    # do this instead to combine parameterizations of different types
    good_model2 = pyqg.QGModel(uv_parameterization=param1, q_parameterization=param3)

.. _defining-parameterizations:

Defining new parameterizations
------------------------------

To define a new parameterization, you have two options. The first is just to
define a Python function which takes a single argument (the model) and returns
either a single real array of size :code:`(nz, ny, nz)` if it targets :math:`q`
or an iterable of two such arrays if it targets :math:`u` and :math:`v`. This
can then be passed to the model using the type-specific arguments:

.. code-block:: python

    # These parameterizations just add random noise, but with the right shape
    noisy_q_param = lambda model: np.random.normal(size=model.q.shape) 
    noisy_uv_param = lambda model: np.random.normal(size=(2, *model.u.shape))

    model1 = pyqg.QGModel(q_parameterization=noisy_q_param)
    model2 = pyqg.QGModel(uv_parameterization=noisy_uv_param)

The second (and usually better) option is to define a subclass of
:code:`pyqg.UVParameterization` or :code:`pyqg.QParameterization` with a new
definition of :code:`__call__`:

.. code-block:: python

    class NoisyQParam(pyqg.QParameterization):
        def __init__(self, scale):
            self.scale = scale

        def __call__(self, model):
            return np.random.normal(size=model.q.shape) * self.scale

If you would like to make your parameterization available for others to test,
please consider :ref:`contributing-parameterizations`.

Parameterization diagnostics
----------------------------

Parameterizations of potential vorticity affect how energy is redistributed
across scales according to the following formula:

.. math::

    \left(\frac{\partial E(k, l)}{\partial t}\right)^{\text{param}} = 
        -\frac{1}{H}\sum_{n=1}^N H_n\mathbb{R}\left[\hat{\psi}_n^*\hat{\dot{q}}^{\text{param}}_n\right],

The contribution of velocity parameterizations is analogous, except with
:math:`\hat{\dot{q}}^{\text{param}}` replaced by the curl of the velocity
tendency terms in spectral space. This term is made available in the
diagnostics under :code:`paramspec`.

In the case of a quasi-geostrophic model, the :code:`paramspec` can be
decomposed into two terms which represent its contribution to the kinetic and
available potential energy tendencies:

.. math::
    
    \begin{align}
    \left(\frac{\partial \mathrm{KE}(k, l)}{\partial t}\right)^{\text{param}} &= 
        \frac{1}{H}\sum_{n=1}^N H_n\mathbb{R}\left[\kappa^2\hat{\psi}_n^*
                \left(\mathbf{A}\hat{\dot{\mathbf{q}}}^{\text{param}}\right)_n\right] \\
    \left(\frac{\partial \mathrm{APE}(k, l)}{\partial t}\right)^{\text{param}} &= 
         -\frac{1}{H}\sum_{n=1}^N H_n\mathbb{R}\left[\hat{\psi}_n^*
        \left(\mathbf{SA}\hat{\dot{\mathbf{q}}}^{\text{param}}\right)_n\right]
    \end{align}

where :math:`\mathbf{A}(\mathbf{k}) = (\mathbf{S} - \kappa^2\mathbf{I})^{-1}`
and :math:`\mathbf{S}` is the model's stretching matrix (more details 
:ref:`here<layered-subgrid-contribution>`). 

We make these terms available in the diagnostics under :code:`paramspec_KEflux`
and :code:`paramspec_APEflux`, respectively. When comparing the KE and APE
fluxes of parameterized and unparameterized models, it may make sense to do so
after adding these terms to the raw :code:`KEflux` and :code:`APEflux` values.

Evaluating subgrid parameterizations
------------------------------------

As many parameterizations attempt to account for missing physics due to low
resolution, we provide several helper methods for evaluating them.

Assume we have run a high-resolution model and both parameterized and
unparameterized low-resolution models. We provide helper methods to compare the
root mean squared difference in their resulting diagnostics (properly adding,
e.g., :code:`KEflux` and :code:`paramspec_KEflux`), and even compute similarity
metrics describing how much closer each of the parameterized model's
diagnostics are to those of the high-resolution model as compared to those of
the low-resolution model:

.. code-block:: python

    from pyqg.diagnostic tools import diagnostic_differences, diagnostic_similarities

    m_highres = pyqg.QGModel(nx=256)
    m_lowres = pyqg.QGModel(nx=64)
    m_param = pyqg.QGModel(nx=64, parameterization=pyqg.BackscatterBiharmonic())
    [m.run() for m in [m_highres, m_lowres, m_param]]

    highres_lowres_diffs = diagnostic_differences(m_highres, m_lowres)
    highres_param_diffs = diagnostic_differences(m_highres, m_param)

    param_similarity = diagnostic_similarities(m_param,
                                               target=m_highres,
                                               baseline=m_lowres)

The :code:`target` does not need to be a high-resolution model, but regardless,
similarity scores near 1 indicate that the parameterization's diagnostics are
much closer to the :code:`target` than the :code:`baseline`, while scores below
0 indicate they are further from the :code:`target` than the :code:`baseline`.

.. _contributing-parameterizations:

Contributing your parameterization to pyqg
------------------------------------------

We encourage contributions of parameterizations to pyqg for others to test. To add yours, please:

#. Define it as a subclass of :code:`pyqg.UVParameterization` or :code:`pyqg.QParameterization` :ref:`as described above<defining-parameterizations>`.

#. Add the code either to :code:`pyqg/parameterizations.py` or a new file imported in :code:`pyqg/__init__.py`.

#. Write a test ensuring it can be evaluated for the appropriate model classes.

#. Create or update a notebook in :code:`docs/examples` to illustrate its effects or compare it to other parameterizations (optional but encouraged).

#. Create a pull request following the :ref:`normal development workflow<dev-workflow>`.
