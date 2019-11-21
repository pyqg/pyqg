
Surface Quasi-geostrophic Model
===============================

Surface quasi-geostrophy (SQG) is a relatively simple model that
describes surface intensified flows due to buoyancy. One of it's
advantages is that it only has two spatial dimensions but describes a
three-dimensional solution.

The evolution equation is

.. math::


   \partial_t b + \mathsf{J}(\psi, b) = 0\,,  \qquad \text{at} \qquad z = 0\,,

where :math:`b = \psi_z` is the buoyancy.

The interior potential vorticity is zero. Hence

.. math::


   \frac{\partial }{\partial z}\left(\frac{f_0^2}{N^2}\frac{\partial \psi}{\partial z}\right) + \nabla^2\psi = 0\,,

where :math:`N` is the buoyancy frequency and :math:`f_0` is the
Coriolis parameter. In the SQG model both :math:`N` and :math:`f_0` are
constants. The boundary conditions for this elliptic problem in a
semi-infinite vertical domain are

.. math::


   b = \psi_z\,,  \qquad \text{and} \qquad z = 0\,,

and

.. math::


   \psi = 0,  \qquad \text{at} \qquad z \rightarrow -\infty\,.

The solutions to the elliptic problem above, in horizontal Fourier
space, gives the inversion relationship between surface buoyancy and
surface streamfunction

.. math::


   \widehat{\psi} = \frac{f_0}{N} \frac{1}{\kappa} \widehat{b}\,,  \qquad \text{at} \qquad z = 0\,.

The SQG evolution equation is marched forward similarly to the two-layer
model.
