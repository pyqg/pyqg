
Layered quasigeostrophic model
==============================

.. math::


   \,{q_{i}}_t + \mathsf{J}\left(\psi_i\,, q_i\right) + U_i {q_i}_x + V_i {q_i}_y + {Q_i}_y {\psi_i}_x - {Q_i}_x {\psi_i}_y= \text{ssd} - r_{ek} \delta_{i\textsf{N}} \nabla^2 \psi_i\,, \qquad i = 1,\textsf{N}\,,

where

.. math::


   {q_i} = \nabla^2\psi_i + \frac{f_0^2}{H_i} \left(\frac{\psi_{i-1}-\psi_i}{g'_{i-1}}  - \frac{\psi_{i}-\psi_{i+1}}{g'_{i}}\right)\,,  \qquad i = 2,\textsf{N}-1\,,

 and

.. math::


   {q_1} = \nabla^2\psi_1 + \frac{f_0^2}{H_1} \left(\frac{\psi_{2}-\psi_1}{g'_{1}}\right)\,,  \qquad i =1\,,

.. math::


   {q_\textsf{N}} = \nabla^2\psi_\textsf{N} + \frac{f_0^2}{H_\textsf{N}} \left(\frac{\psi_{\textsf{N}-1}-\psi_\textsf{N}}{g'_{\textsf{N}}}\right) + \frac{f_0}{H_\textsf{N}}h_b\,,  \qquad i =\textsf{N}\,,

 where the reduced gravity, or buoyancy jump, is

.. math::


   g'_i \equiv g \frac{\rho_{i+1}-\rho_i}{\rho_i}\,.

The inversion relationship in spectral space is

.. math::


   \hat{q}_i = \underbrace{\left(\textsf{S} - \kappa^2 \textsf{I}\right)}_{\equiv\textsf{A}}\hat{\psi}_i\,,

where the "stretching matrix" is

.. math::


   \textsf{S} \equiv  f_0^2
   \begin{bmatrix}
   -\frac{1}{g'_1 H_1} & \frac{1}{g'_1 H_1} & 0 & \dots& \\
   0 & & & & & &\\
   \vdots & \ddots& \ddots &\ddots & & & &\\
   & \frac{1}{g'_{i-1} H_i} &  -\left(\frac{1}{g'_{i-1} H_i} + \frac{1}{g'_{i} H_i}\right)& \frac{1}{g'_{i} H_i} \\
   & \ddots& \ddots &\ddots & & & &\\
   & & & & & \\
   & \dots & 0 & \frac{1}{ g'_{\textsf{N}-1} H_\textsf{N}} & -\frac{1}{g'_{\textsf{N}-1} H_\textsf{N}}
   \end{bmatrix}

The forced-dissipative equations in Fourier space are

.. math::


   \,{\hat{q}_{i}}_t + ik\,{\hat{\psi}_i} {Q_y} - il\,{\hat{\psi}_i} {Q_x} + (i k U_i + i l V_i) \hat{q}_i+ \mathsf{\hat{J}}\left(\psi_i\,, q_i +  \delta_{i\textsf{N}} \tfrac{f_0}{H_\textsf{N}} h_b \right)     = \text{ssd} \,, \qquad i = 1,\textsf{N}\,,

where the mean potential vorticy gradients are

.. math::


   \textsf{Q}_x = \textsf{S}\textsf{V}\, \qquad \textsf{Q}_y = \beta\,\textsf{I} - \textsf{S}\textsf{U}\,\,,

 where the background velocity is
:math:`\vec{\textsf{V}}(z) = \left(\textsf{U},\textsf{V}\right)`.

Energy balance
==============

The equation for the energy spectrum,

.. math::


   E(k,l) \equiv \underbrace{\frac{1}{2 H}\sum_{i=1}^{\mathsf{N}} H_i \kappa^2 |\hat{\psi}_i|^2}_{\text{kinetic energy}} \,\,\,\,+ \,\,\,\,\,\, \frac{1}{2 H}\underbrace{\sum_{i=1}^{\mathsf{N-1}} \frac{f_0^2}{g'_i}|\hat{\psi}_{i}- \hat{\psi}_{i+1}|^2}_{\text{potential energy}}\,\,\,\,,

is

.. math::


   \frac{d}{dt} E(k,l) = \underbrace{\frac{1}{H}\sum_{i=1}^{\mathsf{N}} H_i \text{Re}[\hat{\psi}_i^\star \hat{\mathsf{J}}(\psi_i,\nabla^2\psi_i)]}_{I} +
   \underbrace{\frac{1}{H}\sum_{i=1}^{\mathsf{N}} H_i\text{Re}[\hat{\psi}_i^\star \hat{\mathsf{J}}(\psi_i,(\mathsf{S} \psi)_i)]}_{II}
   + \underbrace{\frac{1}{H}\sum_{i=1}^{\mathsf{N}} H_i ( k U_i +  l V_i)\, \text{Re}[i \, \hat{\psi}^\star_i (\mathsf{S}\hat{\psi}_i)]}_{III} \,\,\,\,\,\,\,\underbrace{- r_{ek} \frac{H_\mathsf{N}}{H} \kappa^2 |\hat{\psi}_{\mathsf{N}}|^2}_{IV}\, .

where :math:`\kappa^2 = k^2 + l^2` and the terms above represent

I: Spectral divergence of the kinetic energy flux

II: Spectral divergence of the potential energy flux

III: The rate of potential energy generation

IV: The rate of energy dissipation through bottom friction

Using the notation of the two-layer model, the particular case
:math:`\mathsf{N}=2` is

.. math::


   \frac{d}{dt} E(k,l) = \underbrace{\frac{1}{H}\text{Re}[H_1 \hat{\psi}_1^\star \hat{\mathsf{J}}(\psi_1,\nabla^2\psi_1) + H_2 \hat{\psi}_2^\star \hat{\mathsf{J}}(\psi_2,\nabla^2\psi_2)]}_{I} +
   \underbrace{ \frac{H_1 H_2}{H^2}\text{Re}[(\hat{\psi}_1-\hat{\psi}_2)^\star \hat{\mathsf{J}}(\psi_1,\hat{\psi}_2])}_{II}\nonumber \\
   + \underbrace{ \frac{H_1 H_2}{H^2} \left[( U_1 - U_2 )\, \text{Re}[i k\, (\hat{\psi}^\star_1+\hat{\psi}^{\star}_2) (\hat{\psi}_2-\hat{\psi}_1)]\right] + ( V_1 - V_2 )\, \text{Re}[i l\, (\hat{\psi}^\star_1+\hat{\psi}^{\star}_2) (\hat{\psi}_2-\hat{\psi}_1)}_{III} \,\,\,\,\,\,\,\underbrace{- r_{ek} \frac{H_1}{H}\kappa^2 |\hat{\psi}_{\mathsf{N}}|^2}_{IV}\, .

Vertical modes
==============

Standard vertical modes are the eigenvectors,
:math:`\mathsf{\phi}_n (z)`, of the "stretching matrix"

.. math::


   \textsf{S} \,\mathsf{\phi}_n = -m_n^2\, \mathsf{\phi}_n\,,

where the n'th deformation radius is

.. math::


   R_n \equiv m_n^{-1}\,.

Linear stability analysis
=========================

With :math:`h_b = 0`, the linear eigenproblem is

.. math::


    \mathsf{A}\, \mathsf{\Phi} = \omega \, \mathsf{B}\, \mathsf{\Phi}\,,

where

.. math::


   \mathsf{A} \equiv \mathsf{B}(\mathsf{U}\, k + \mathsf{V}\,l) + \mathsf{I}\left(k\,\mathsf{Q}_y - l\,\mathsf{Q}_x\right) + \mathsf{I}\,\delta_{\mathsf{N}\mathsf{N}}\, i\,r_{ek}\,\kappa^2\,,

where :math:`\delta_{\mathsf{N}\mathsf{N}} = [0,0,\dots,0,1]\,,` and

.. math::


   \mathsf{B} \equiv  \mathsf{S} - \mathsf{I} \kappa^2\,. 

 The growth rate is Im\ :math:`\{\omega\}`.

