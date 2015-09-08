
Semi-infinite surface quasi-geostrophic model
=============================================

The surface geostrophic model is associated with a active surface
boundary condition

.. math::


   \partial_t\,{b} + \mathsf{J}\left(\psi\,, b\right)= \text{ssd} \,,\qquad \text{at} \qquad z = 0\,,

where :math:`b = \psi_z` is the buoyancy.

The interior potential vorticity is constant (typically zero), and this
leads the inversion relationship. We have

.. math::


   \frac{\partial }{\partial z}\left(\frac{f_0^2}{N^2}\frac{\partial \psi}{\partial z}\right) + \nabla^2\psi  = 0\,,

where :math:`N` is the buoyancy frequency and :math:`f_0` is the
Coriolis parameter. The boundary conditions for this elliptic problem
are

.. math::


   b = \psi_z \,,\qquad \text{at} \qquad z = 0\,,

and

.. math::


   \psi = b = 0 \,,\qquad \text{at} \qquad z \rightarrow -\infty\,.

Hence, the inversion relationship in Fourier space is

.. math::


   \hat{\psi} = \frac{f_0}{N} \frac{1}{\kappa} \hat{b}\,,\qquad \text{at} \qquad z = 0\,.

The system is marched forward in time similarly to the two-layer model.
