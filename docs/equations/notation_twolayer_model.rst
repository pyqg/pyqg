
Equations For Two-Layer QG Model
================================
The two-layer quasigeostrophic evolution equations are (1)

.. math::


   \partial_t\,{q_1} + \mathsf{J}\left(\psi_1\,, q_1\right) + \beta\, {\psi_1}_x = \text{ssd} \,,

and (2)

.. math::


   \partial_t\,{q_2} + \mathsf{J}\left(\psi_2\,, q_2\right)+ \beta\, {\psi_2}_x = -r_{ek}\nabla^2 \psi_2 + \text{ssd}\,,

where the horizontal Jacobian is
:math:`\mathsf{J}\left(A\,, B\right) = A_x B_y - A_y B_x`. Also in (1)
and (2) ssd denotes small-scale dissipation (in turbulence regimes, ssd
absorbs enstrophy that cascades towards small scales). The linear bottom
drag in (2) dissipates large-scale energy.

The potential vorticities are (3)

.. math::


   {q_1} = \nabla^2\psi_1 + F_1\left(\psi_2 - \psi_1\right)\,,

and (4)

.. math::


   {q_2} = \nabla^2\psi_2 + F_2\left(\psi_1 - \psi_2\right)\,,

where

.. math::


   F_1 \equiv \frac{k_d^2}{1 + \delta}\,, \qquad \text{and} \qquad F_2 \equiv \delta \,F_1\,,

with the deformation wavenumber

.. math::


   k_d^2 \equiv\frac{f_0^2}{g'}\frac{H_1+H_2}{H_1 H_2}\,,

where :math:`H = H_1 + H_2` is the total depth at rest.

Forced-dissipative equations
----------------------------

We are interested in flows driven by baroclinic instabilty of a
base-state shear :math:`U_1-U_2`. In this case the evolution equations
(1) and (2) become (5)

.. math::


   \partial_t\,{q_1} + \mathsf{J}\left(\psi_1\,, q_1\right) + \beta_1\, {\psi_1}_x = \text{ssd} \,,

and (6)

.. math::


   \partial_t\,{q_2} + \mathsf{J}\left(\psi_2\,, q_2\right)+ \beta_2\, {\psi_2}_x = -r_{ek}\nabla^2 \psi_2 + \text{ssd}\,,

where the mean potential vorticity gradients are (9,10)

.. math::


   \beta_1 = \beta + F_1\,\left(U_1 - U_2\right)\,, \qquad \text{and} \qquad \beta_2 = \beta - F_2\,\left( U_1 - U_2\right)\,.

Equations in Fourier space
--------------------------

We solve the two-layer QG system using a pseudo-spectral doubly-peridioc
model. Fourier transforming the evolution equations (5) and (6) gives
(7)

.. math::


   \partial_t\,{\widehat{q}_1} = - \widehat{\mathsf{J}}\left(\psi_1\,, q_1\right) - \text{i}\,k\, \beta_1\, {\widehat{\psi}_1} + \widehat{\text{ssd}} \,,

and

.. math::


   \partial_t\,{\widehat{q}_2} = - \widehat{\mathsf{J}}\left(\psi_2\,, q_2\right)-  \text{i}\,k\, \beta_2\, {\widehat{\psi}_2}  + r_{ek}\,\kappa^2\,\, \widehat{\psi}_2 + \widehat{\text{ssd}}\,,

where, in the pseudo-spectral spirit, :math:`\widehat{\mathsf{J}}` means the
Fourier transform of the Jacobian i.e., we compute the products in
physical space, and then transform to Fourier space.

In Fourier space the "inversion relation" (3)-(4) is

.. math::


   \underbrace{\begin{bmatrix}
   -(\kappa^2 + F_1) \qquad \:\:\:\:F_1\\
   \:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:F_2 \qquad - (\kappa^2 + F_2)\;
   \end{bmatrix}}_{\equiv \,\mathsf{M_2}}
   \begin{bmatrix}
   \widehat{\psi}_1\\
   \widehat{\psi}_2\\
   \end{bmatrix}
   =\begin{bmatrix}
   \widehat{q}_1\\
   \widehat{q}_2\\
   \end{bmatrix}
   \,,

or equivalently

.. math::


   \begin{bmatrix}
   \widehat{\psi}_1\\
   \widehat{\psi}_2\\
   \end{bmatrix}
   =\underbrace{\frac{1}{\text{det}\,\mathrm{M_2}}
   \begin{bmatrix}
   -(\kappa^2 + F_2) \qquad \:\:\:\:-F_1\\
   \:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:-F_2 \qquad - (\kappa^2 + F_1)\;
   \end{bmatrix}}_{=\,\mathsf{M_2}^{-1}}
   \begin{bmatrix}
   \widehat{q}_1\\
   \widehat{q}_2\\
   \end{bmatrix}
   \,,\qquad

where

.. math::


   \qquad \text{det}\,\mathsf{M}_2 = \kappa^2\left(\kappa^2 + F_1 + F_2\right)\,.

Marching forward
~~~~~~~~~~~~~~~~

We use a third-order Adams-Bashford scheme

.. math::


   {\widehat{q}_i}^{n+1} = E_f\times\left[{\widehat{q}_i}^{n} + \frac{\Delta t}{12}\left(23\, \widehat{Q}_i^{n} -  16\widehat{Q}_i^{n-1} + 5 \widehat{Q}_i^{n-2}\right)\right]\,,

where

.. math::


   \widehat{Q}_i^n \equiv - \widehat{\mathsf{J}}\left(\psi_i^n\,, q_i^n\right) - \text{i}\,k\, \beta_i\, {\widehat{\psi}_i^n}, \qquad i = 1,2\,.

The AB3 is initialized with a first-order AB (or forward Euler)

.. math::


   {\widehat{q}_i}^{1} = E_f\times\left[{\widehat{q}_i}^{0} + \Delta t \widehat{Q}_i^{0}\right]\,,

The second step uses a second-order AB scheme

.. math::


   {\widehat{q}_i}^{2} = E_f\times\left[{\widehat{q}_i}^{1} + \frac{\Delta t}{2}\left(3\, \widehat{Q}_i^{1} -  \widehat{Q}_i^0\right)\right]\,.

The small-scale dissipation is achieve by a highly-selective exponential
filter

.. math::


   E_f =\begin{cases} \text{e}^{-23.6\,\left(\kappa^{\star} - \kappa_c\right)^4}: &\qquad \kappa \ge\kappa_c\\
   \,\,\,\,\,\,\,\,\,\,\,1:&\qquad \text{otherwise}\,.
   \end{cases}

where the non-dimensional wavenumber is

.. math::


   \kappa^{\star} \equiv \sqrt{ (k\,\Delta x)^2 + (l\,\Delta y)^2 }\, ,

and :math:`\kappa_c` is a (non-dimensional) wavenumber cutoff here taken
as :math:`65\%` of the Nyquist scale :math:`\kappa^{\star}_{ny} = \pi`.
The parameter :math:`-23.6` is obtained from the requirement that the
energy at the largest wanumber (:math:`\kappa^{\star} = \pi`) be zero
whithin machine double precision:

.. math::


   \frac{\log 10^{-15}}{(0.35\, \pi)^4} \approx -23.5\,.

For experiments with :math:`|\widehat{q_i}|\ll\mathcal{O}(1)` one can use a
smaller constant.

Diagnostics
~~~~~~~~~~~

The kinetic energy is

.. math::


   E = \tfrac{1}{H\,S} \int  \tfrac{1}{2} H_1 \, |\boldsymbol{\nabla} \psi_1|^2 +  \tfrac{1}{2} H_2 \, |\boldsymbol{\nabla} \psi_2|^2 \, d S\,.

The potential enstrophy is

.. math::


   Z = \tfrac{1}{H\,S}\int \tfrac{1}{2}H_1 \, q_1^2 + \tfrac{1}{2} H_2 \, q_2^2 \, d S\,.

We can use the enstrophy to estimate the eddy turn-over timescale

.. math::


   T_e \equiv \frac{2\,\pi}{\sqrt{Z}}\,.

