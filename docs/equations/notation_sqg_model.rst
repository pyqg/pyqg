
Surface Quasi-geostrophic Model
===============================

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
