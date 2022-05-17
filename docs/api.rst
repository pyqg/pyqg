API
###

.. automodule:: pyqg
   :members:

Base Model Class
================

This is the base class from which all other models inherit. All of these
initialization arguments are available to all of the other model types.
This class is not called directly.

.. autoclass:: Model
    :members: 

Specific Model Types
======================

These are the actual models which are run.

.. autoclass:: QGModel
    :members: 

.. autoclass:: LayeredModel
    :members:  

.. autoclass:: BTModel
    :members:

.. autoclass:: SQGModel
    :members:

Lagrangian Particles
====================

.. autoclass:: LagrangianParticleArray2D
    :members:
    
.. autoclass:: GriddedLagrangianParticleArray2D
    :members:

Diagnostic Tools
================

.. automodule:: pyqg.diagnostic_tools
    :members:  

.. _parameterizations-api:

Parameterizations
=================

.. autoclass:: pyqg.Parameterization
    :member-order: bysource
    :special-members: __call__, parameterization_type, __add__, __mul__

.. automodule:: pyqg.parameterizations
    :members:
    :member-order: bysource
    :exclude-members: Parameterization, CompositeParameterization, WeightedParameterization
