
Equations For Equivalent Barotropic QG Model
============================================

The equivalent barotropic quasigeostrophy evolution equations is

.. math::


   \partial_t\,{q} + \mathsf{J}\left(\psi\,, q\right) + \beta\, {\psi}_x = \text{ssd} \,.

The potential vorticity anomaly is

.. math::


   {q} = \nabla^2\psi - \kappa_d^2 \psi\,,

where :math:`\kappa_d^2` is the deformation wavenumber. With
:math:`\kappa_d = \beta = 0` we recover the 2D vorticity equation.

The inversion relationship in Fourier space is

.. math::


   \widehat{q} = -\left(\kappa^2 + \kappa_d^2\right) \widehat{\psi}\,.

The system is marched forward in time similarly to the two-layer model.
