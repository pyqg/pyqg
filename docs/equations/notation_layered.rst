Layered quasi-geostrophic model
===============================

Consider an :math:`{N}`-layer quasi-geostrophic (QG) model with rigid lid and
flat topography (for reference, see Eq. 5.85 in 
`Vallis, 2017 <http://empslocal.ex.ac.uk/people/staff/gv219/aofd/>`__). 
The :math:`{N}`-layer QG potential vorticity is

.. math::
   \begin{aligned}
   {q_1} &= {\nabla^2}\psi_1 + \frac{f_0^2}{H_1} \left(\frac{\psi_{2}-\psi_1}{g'_{1}}\right)\,, \nonumber \\
   {q_n} &= {\nabla^2}\psi_n + \frac{f_0^2}{H_n} \left(\frac{\psi_{n-1}-\psi_n}{g'_{n-1}}  - \frac{\psi_{n}-\psi_{n+1}}{g'_{n}}\right)\,,  \qquad &n = 2,\dots,{N}-1 {\, ,}\nonumber \\
   {q_N} &= {\nabla^2}\psi_N+ \frac{f_0^2}{H_N} \left(\frac{\psi_{N-1}-\psi_N}{g'_{N-1}}\right) \,,
   \end{aligned}

where :math:`q_n` is the `n`-th layer QG potential vorticity, and
:math:`\psi_n` is the streamfunction, :math:`f_0` is the inertial
frequency, :math:`H_n` is the layer depth. Also the `n`-th buoyancy jump (reduced gravity) is

.. math:: 
   g'_n \equiv g \frac{\rho_{n+1}-\rho_{n}}{\rho_n}{\, ,}

where :math:`g` is the acceleration due to gravity and :math:`\rho_n` is
the layer density. 

The relationship between :math:`{q_n}` and :math:`{\psi_n}` can be 
conveniently written as 

.. math::
   \mathbf{q} = (\mathbf{S} + \nabla^2\mathbf{I})\mathbf{\boldsymbol\psi}

where :math:`\mathbf{q} = (q_1, ..., q_N)^\mathrm{T}`, 
:math:`\boldsymbol\psi = (\psi_1, ..., \psi_N)^\mathrm{T}`,  
:math:`{\mathbf{I}}` is the :math:`N\times N`
identity matrix, and the stretching matrix :math:`{\mathbf{S}}` is

.. math:: 
    \mathbf{S} \equiv  f_0^2
    \begin{bmatrix}
    -\dfrac{1}{H_1g_1'} & \dfrac{1}{H_1g_1'} & 0 & 0 & ... \\[6pt]
    & \vdots & & \vdots & \\[6pt]
    ... & \dfrac{1}{H_ng_{n-1}'} & -\dfrac{1}{H_n}\left(\dfrac{1}{g_{n-1}'}+\dfrac{1}{g_n'}\right) & \dfrac{1}{H_ng_n'} & ...\\[6pt]
    & \vdots & & \vdots & \\[6pt]
    ... & 0 & 0 & \dfrac{1}{H_Ng_{N-1}'} & -\dfrac{1}{H_Ng_{N-1}'} \\
    \end{bmatrix}.

The dynamics of the system is given by the evolution of PV. In
particular, we assume a background flow with background velocity
:math:`\overrightarrow{V} = (U,V)` such that

.. math::

   \begin{aligned}
   \label{eq:Uequiv}
   u_n^{{{\text{tot}}}} = U_n - {\psi_n}_y, \nonumber \\
   v_n^{{\text{tot}}} = V_n + {\psi_n}_x,
   \end{aligned}

and

.. math:: q_n^{{\text{tot}}} = Q_n + q_n, 

where :math:`Q_n` is the `n`-th layer background PV. 
:math:`Q_n` satisfies 

.. math:: {\mathbf{Q}} = \beta + \mathbf{SV} x - \mathbf{SU} y,

where :math:`\mathbf{Q}`, :math:`\mathbf{U}`, :math:`\mathbf{V}` are defined similarly
to :math:`\mathbf{q}` and :math:`\boldsymbol\psi`. We then obtain the evolution equations

.. math::

   \begin{aligned}
   \label{eq:qg_dynamics}
   {q_n}_t + \mathsf{J}(\psi_n,q_n)& + \overrightarrow{V}_n\cdot\nabla q_n + {Q_n}_y {\psi_n}_x - {Q_n}_x {\psi_n}_y = \text{ssd}_n - r_{ek} \delta_{n,N} {\nabla^2}\psi_n, ~ n = 1,\dots,N,
   \end{aligned}

where :math:`{\text{ssd}}` stands for small-scale dissipation, which
is achieved by an spectral exponential filter or hyperviscosity, and
:math:`r_{ek}` is the linear bottom drag coefficient. The Dirac delta,
:math:`\delta_{n,N}`, indicates that the drag is only applied to the
bottom layer. The advection of the background PV by the background 
flow is neglected because in each layer, the contribution of this term
is constant for all locations. 

Equations in spectral space
---------------------------

The evolution equation in spectral space is

.. math::

   \begin{aligned}
       {\hat{q}_n}_t + \mathsf{\hat{J}}(\psi_n, q_n) &+ (\mathrm{i} k U_n + \mathrm{i} l V_n) \hat{q}_n \nonumber \\
       &+ (\mathrm{i} k\, {Q_n}_y - \mathrm{i} l\,{Q_n}_x){\hat{\psi}_n} = \widehat{\text{ssd}}_n + r_{ek}\delta_{n,N}\kappa^2 \hat{\psi}_n, ~~ n = 1,\dots,N,
    \end{aligned}

where :math:`\kappa^2 = k^2 + l^2`. Also, in the pseudo-spectral spirit,
we write the transform of the nonlinear terms and the non-constant
coefficient linear term as the transform of the products, calculated in
physical space, as opposed to double convolution sums. That is, 
:math:`\mathsf{\hat{J}}` is the Fourier transform of Jacobian computed
in the physical space.

The inversion relationship between PV and streamfunction is

.. math:: \hat{\mathbf{q}} = \left({\mathbf{S}}- \kappa^2 {\mathbf{I}}\right) \hat{\boldsymbol\psi}. 

Energy spectrum
---------------

The equation for the energy spectrum is,

.. math:: 
    E(k,l) \equiv \frac{1}{2 H}\sum_{n=1}^{{\mathrm{N}}} H_n \kappa^2 {|\hat{\psi}_n|}^2
    + {\frac{1}{2 H} \sum_{n=1}^{{\mathrm{N}}-1} \frac{f_0^2}{g'_n}|\hat{\psi}_{n}- \hat{\psi}_{n+1}|^2},

To obtain the spectral flux of different components, we take the time derivative of the energy spectrum

.. math::
    \begin{aligned}
    \frac{\partial E(k, l)}{\partial t} &= \frac{1}{H}\mathbb{R}\left[\sum_{n=1}^N H_n\kappa^2\frac{\partial\hat{\psi}_n}{\partial t}\hat{\psi}_n^*
    +\sum_{n=1}^{N-1}\frac{f_0^2}{g_n'}
    \left(\frac{\partial \hat{\psi}_n}{\partial t} - \frac{\partial \hat{\psi}_{n+1}}{\partial t}\right)(\hat{\psi}_n^* - \hat{\psi}_{n+1}^*)\right]\\
    =&-\frac{1}{H}\mathbb{R}\left[
    \sum_{n=1}^N H_n\hat{\psi}_n^*\frac{\partial}{\partial t}\left(-\kappa^2\hat{\psi}_n
    +\frac{f_0^2}{H_n}\frac{\hat{\psi}_{n-1}-\hat{\psi}_{n}}{g_{n-1}'}\mathsf{1}_{n>1}
    -\frac{f_0^2}{H_n}\frac{\hat{\psi}_n - \hat{\psi}_{n+1}}{g_n'}\mathsf{1}_{n<N}
    \right)\right]\\
    =&-\frac{1}{H}\sum_{n=1}^N H_n\mathbb{R}\left[\hat{\psi}_n^*{\hat{q}_n}_t\right],
    \end{aligned}

where :math:`\mathsf{1}` is the indicator function. This suggests that the energy tendency of 
the layered QG system is just the dot product of the layer-weighted streamfunction and the 
tendency of QG potential vorticty. Substituting the expression of :math:`{q_n}_t` from 
above, we have

.. math::

   \begin{aligned}
       \frac{\partial E(k,l)}{\partial t} &= {\frac{1}{H}\sum_{n=1}^N H_n \mathbb{R}[\hat{\psi}_n^* {\mathsf{\hat{J}}}(\psi_n,\nabla^2\psi_n)]}
       +\frac{1}{H}\sum_{n=1}^N H_n\mathbb{R}[\hat{\psi}_n^* \hat{\mathsf{J}} (\psi_n,({\mathbf{S}}\boldsymbol\psi)_n)] \nonumber \\
       &+\frac{1}{H}\sum_{n=1}^N H_n (k U_n + l V_n)\, \mathbb{R}[i \, \hat{\psi}^*_n (\mathbf{S}\hat{\boldsymbol\psi})_n] 
       -r_{ek}\frac{H_N}{H} \kappa^2|\hat{\psi}_N|^2 \nonumber \\
       &-\frac{1}{H}\sum_{n=1}^N H_n\mathbb{R}[\hat{\psi}_n^* \widehat{\text{ssd}}_n],
    \end{aligned}

where :math:`*` stands for complex conjugation. We also used the fact that 
the terms involving background vorticity gradients does not make contribution 
to the real part of the right-hand-side. The right-hand-side terms represent, 
from left to right,

I: The spectral divergence of the kinetic energy flux;

II: The spectral divergence of the potential energy flux;

III: The spectrum of the potential energy generation;

IV: The spectrum of the energy dissipation by linear bottom drag;

V: The spectrum of energy loss due to small scale dissipation.

We assume that the fifth term is relatively small, and that, in statistical
steady state, the budget above is dominated by I through IV.

.. _layered-subgrid-contribution:

Contribution from subgrid parameterization
------------------------------------------

Subgrid-scale parameterizations in terms of tendencies in :math:`q` can be added to 
the dynamical equation, and thus has contribution to the energy spectrum. In spectral
space, let the effect of parameterization be 

.. math::

    \left(\frac{\partial \hat{q}_n}{\partial t}\right)^{\text{sub}} = \hat{\dot{q}}^{\text{sub}}_n

From the derivations above, we have

.. math::

    \left(\frac{\partial E(k, l)}{\partial t}\right)^{\text{sub}} = 
        -\frac{1}{H}\sum_{n=1}^N H_n\mathbb{R}\left[\hat{\psi}_n^*\hat{\dot{q}}^{\text{sub}}_n\right],

which is the spectrum of the energy contribution from parameterizations.

We can further expand the contribution of parameterization into its contribution to 
kinetic energy and potential energy. To see how, we consider again the time derivative
of the total energy in matrix form: 

.. math::

    \frac{\partial E(k, l)}{\partial t} = 
        -\frac{1}{H}\sum_{n=1}^N H_n\mathbb{R}\left[\hat{\psi}_n^*\left((-\kappa^2\mathbf{I} + 
        \mathbf{S})\frac{\partial\hat{\boldsymbol\psi}}{\partial t}\right)_n\right], 

where the first term on the right-hand side is the change in kinetic energy, 
and the second term is the change in potential energy. Considering the streamfunction 
tendency is from parameterizations, and letting 
:math:`\mathbf{A}(\mathbf{k}) = (\mathbf{S} - \kappa^2\mathbf{I})^{-1}`
so that :math:`\hat{\boldsymbol\psi} = \mathbf{A}(\mathbf{k})\hat{\mathbf{q}}`,
we have 

.. math::
    
    \begin{align}
    \left(\frac{\partial E(k, l)}{\partial t}\right)^{\text{sub}} =& 
        \frac{1}{H}\sum_{n=1}^N H_n\mathbb{R}\left[\kappa^2\hat{\psi}_n^*
        \left(\mathbf{A}\hat{\dot{\mathbf{q}}}^{\text{sub}}\right)_n\right] - 
        \frac{1}{H}\sum_{n=1}^N H_n\mathbb{R}\left[\hat{\psi}_n^*
        \left(\mathbf{SA}\hat{\dot{\mathbf{q}}}^{\text{sub}}\right)_n\right]
    \end{align}

where on the right-hand side, the first term is the parameterized contribution towards 
kinetic energy, and the second term is towards potential energy. 

Enstrophy spectrum
------------------

Similarly, the evolution of the barotropic enstrophy spectrum,

.. math:: Z(k,l) \equiv \frac{1}{2H} \sum_{n=1}^{\mathrm{N}} H_n {|\hat{q}_n|}^2, 

is governed by

.. math::
    \begin{aligned}
    \frac{\partial Z(k,l)}{\partial t} &= \frac{1}{H}\sum_{n=1}^{\mathrm{N}}\mathbb{R}\left[\hat{q}_n^* \mathsf{\hat{J}}(\psi_n,q_n)\right]
        +\frac{1}{H}\sum_{n=1}^{\mathrm{N}}(l {Q_n}_x - k {Q_n}_y)\mathbb{R}\left[i({\mathbf{S}}\hat{\boldsymbol\psi}^*)_n\hat{\psi}_n\right]\\
        &+r_{ek}\frac{H_N}{H} \kappa^2\mathbb{R}\left[\hat{q}_N^*\hat{\psi}_N\right] 
        +\frac{1}{H}\sum_{n=1}^N H_n\mathbb{R}[\hat{q}_n^* \widehat{\text{ssd}}_n],
   \end{aligned}

where the terms above on the right represent, from left to right,

I: The spectral divergence of barotropic potential enstrophy flux;

II: The spectrum of barotropic potential enstrophy generation;

III: The spectrum of barotropic potential enstrophy loss due to bottom friction;

IV: The spectrum of barotropic potential enstrophy loss due to small scale dissipation.

The enstrophy dissipation is concentrated at the smallest scales
resolved in the model and, in statistical steady state, we expect the
budget above to be dominated by the balance between I and II.

Special case: two-layer model
=============================

With :math:`N=2` (see :doc:`notation_twolayer_model`), 
an alternative notation for the perturbation of potential vorticities can 
be written as

.. math::

   \begin{aligned}
       q_1 &= {\nabla^2}\psi_1 + F_1 (\psi_2 - \psi_1) \nonumber\\
       q_2 &= {\nabla^2}\psi_2 + F_2 (\psi_1  - \psi_2){\, ,}\end{aligned}

where we use the following definitions where

.. math:: F_1 \equiv \frac{k_d^2}{1 + \delta}\,, \qquad \:\:\text{and} \qquad F_2 \equiv \delta \,F_1\,,

with the deformation wavenumber

.. math:: k_d^2 \equiv \, \frac{f_0^2}{g} \frac{H_1+H_2}{H_1 H_2} {\, .}

With this notation, the stretching matrix is simply

.. math::

   {\mathbf{S}}= \begin{bmatrix}
   -F_1 & F_1\\
   F_2 & -F_2
   \end{bmatrix}.

The inversion relationship in Fourier space is

.. math::

   \begin{pmatrix}
   \widehat{\psi}_1\\
   \widehat{\psi}_2\\
   \end{pmatrix}
   = -\frac{1}{\kappa^2(\kappa^2 + F_1 + F_2)}
   \begin{bmatrix}
   \kappa^2 + F_2 & F_1\\
   F_2 & \kappa^2 + F_1
   \end{bmatrix}
   \begin{pmatrix}
   \widehat{q}_1\\
   \widehat{q}_2\\
   \end{pmatrix}.

Substituting the inversion relationship to the rate of change of the energy
spectrum above, we have

.. math:: 
   \begin{aligned}
    \frac{\partial E(k,l)}{\partial t}
    =&\mathbb{R}\left[(\delta_1\hat{\psi}_1^*, \delta_2\hat{\psi}_2^*)\cdot
    \begin{pmatrix}
     &\hat{J}(\psi_1, q_1) + ik\beta_1\hat{\psi}_1 + ikU_1\hat{q}_1\\[6pt]
     &\hat{J}(\psi_2, q_2) + ik\beta_2\hat{\psi}_2 + ikU_2\hat{q}_2 - r_{ek}\kappa^2\hat{\psi}_2
     \end{pmatrix}\right]\\
    =&\sum_{n=1}^2\delta_n\mathbb{R}\left[\hat{\psi}_n^*\hat{J}(\psi_n, \nabla^2\psi_n)\right]
    +\delta_1F_1\mathbb{R}\left[(\hat{\psi}_1^*-\hat{\psi}_2^*)\hat{J}(\psi_1, \psi_2)\right]\\
    &+\delta_1F_1k(U_1-U_2)\mathbb{R}\left[i\hat{\psi}_1^*\hat{\psi}_2\right] - r_{ek}\delta_2\kappa^2|\hat{\psi}_2|^2, 
    \end{aligned}

in which the right-hand-side terms are, from left to right, the spectral divergence 
of kinetic energy flux, the spectral divergence of potential energy flux, the spectrum 
of available potential energy generation, and the spectral contribution by bottom drag. 
Note that we neglected the contribution from eddy viscosity (spectral filter), but they 
have the same form as the multi-layer case above.

