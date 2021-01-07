.. _gauss:

Magnetic field of Galaxy clusters with Gaussian Turbulence
----------------------------------------------------------

This module implements a divergence-free homogeneous and isotropic Gaussian turbulent magnetic field with zero mean
and variance :math:`\mathcal{B}^2` as described in detail in [Meyer2014]_ and also used, e.g., in [Ajello2016]_.
The power spectrum of the turbulence follows a power law in terms of wave numbers :math:`k`,

.. math::

    M(k) \propto k^q

between :math:`k_L \leqslant k \leqslant k_H` and zero otherwise.

The Fourier transform of the correlation function of the transversal B-field components is

.. math::

    \tilde\epsilon_\perp(k) = \frac{\pi\mathcal{B}^2}{4} F_q(k ; k_L, k_H),

where the function :math:`F_q(k; k_L,k_H)` is defined in the appendix of [Meyer2014]_.

A new single realization of the magnetic field along the line of sight
is computed with `gammaALPs.bfields.gauss.Bgaussian.Bgaus`
and the transversal component (relevant to photon-ALP mixing) for potentially many realizations
is calculated with :py:meth:`gammaALPs.bfields.gauss.Bgaussian.new_Bn`.
There are also two convenience functions to calculate the spatial correlation and the the rotation measure
induced by the field (provided that an electron density is provided as well).

Reference / API
^^^^^^^^^^^^^^^

.. automodule:: gammaALPs.bfields.gauss
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:
