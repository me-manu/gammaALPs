.. _theory:

#################################
Photon-ALP conversion probability
#################################

Theoretical background
----------------------

This page provides a little bit of background about how the mixing of photons and axion-like particles (ALPs)
is calculated by gammaALPs. It draws heavily from the discussion provided in [Meyer2014]_.

Photons and ALPs can mix in the presence of electromagnetic fields.
In gammaALPs, this equations of motions are solved through the transfer matrix approach (see [DeAngelis2011]_ for
a review), which is implemented in the :py:class:`~gammaALPs.base.transfer.GammaALPTransfer` class.
It derives from the effective Lagrangian for the mixing between photons and ALPs, which can be written as [Raffelt1988]_

.. math::
    \mathcal{L} = \mathcal{L}_{a\gamma} + \mathcal{L}_\mathrm{EH} + \mathcal{L}_a.

The first term on the right hand side is given describes the photon-ALP mixing (see below)
the second term is the effective Euler-Heisenberg Lagrangian that accounts for one-loop
corrections of the photon propagator and the ALP mass and kinetic terms are

.. math::
    \mathcal{L}_a = \frac 1 2  \partial_\mu a \partial^\mu a - \frac 1 2 m^2_a a^2.

ALPs only couple to photons in the presence of a magnetic field component :math:`\mathbf{B}_\perp` transversal
to the propagation direction and only to photon polarisation states in the plane spanned by :math:`\mathbf{B}` and
the direction of propagation.
Let :math:`z` be the propagation direction so that :math:`\mathbf{B}_\perp = B \hat{\mathbf{y}}`,
and :math:`A_x`, :math:`A_y` the polarisation states along :math:`\hat{x}` and :math:`\hat{y}`, respectively.
Then the equations of motion for a polarised photon beam of energy $E$ propagating
in a cold plasma filled with a homogeneous magnetic field read

.. math::
    \left( i\frac{\mathrm{d}}{\mathrm{d}z} + E + \mathcal{M}_0 \right)\Psi(x_3) = 0,

with :math:`\Psi(z) = (A_x(z), A_y(z), a(z))^T` and the mixing matrix :math:`\mathcal{M}_0`
(neglecting Faraday rotation),

.. math::
    \mathcal{M}_0 =
    \begin{pmatrix}
        \Delta_{\perp} & 0 & 0\\
        0 & \Delta_{||} & \Delta_{a\gamma} \\
        0 & \Delta_{a\gamma} & \Delta_a
    \end{pmatrix}.

The terms :math:`\Delta_{||,\perp}` arise due to the effects of the propagation of photons in a plasma
and the QED vacuum polarisation effect,
:math:`\Delta_{\perp} = \Delta_\mathrm{pl} + 2\Delta_\mathrm{QED} + \Delta_\mathrm{CMB}`,
and :math:`\Delta_{||} = \Delta_\mathrm{pl} + 7/2\Delta_\mathrm{QED} + \Delta_\mathrm{CMB}`.
The plasma contribution depends on electron density :math:`n_{\mathrm{el}}` through
the plasma frequency :math:`\omega_\mathrm{pl}^2 = 4\pi e^2 n_\mathrm{el} / m_e`,
:math:`\Delta_\mathrm{pl} = -\omega_\mathrm{pl} / (2 E)`.
The QED vacuum polarisation term reads :math:`\Delta_\mathrm{QED} = \alpha E / (45\pi)(B /(B_\mathrm{cr}))^2`,
with the fine-structure constant :math:`\alpha`, and the critical magnetic field
:math:`B_\mathrm{cr} = m^2_e / |e| \sim 4.4\times10^{13}`G.
The term :math:`\Delta_\mathrm{CMB} = 44\alpha^2 / (135 m_e^4)\rho_\mathrm{CMB}` accounts for
photon-photon dispersion with the cosmic microwave background (CMB),
where :math:`\rho_\mathrm{CMB} = (\pi^2 / 15)T^4 \sim 0.261\,\mathrm{eV}\,\mathrm{cm}^{-3}`
is the CMB energy density [Dobrynina2015]_.
The kinetic term for the ALP is :math:`\Delta_a = -m_a^2 / (2E)`
and photon-ALP mixing is the result of the off-diagonal elements :math:`\Delta_{a\gamma} = g_{a\gamma} B / 2`.
Suitable numerical values for the different :math:`\Delta` terms are provided in ref. [Horns2012]_.
If photons are lost due to absorption, the diagonal terms :math:`\Delta_{||,\perp}` get an additional imaginary
contribution that scales with the mean free path :math:`\Gamma` of the photon.
The mean free path and additional terms for photon-photon dispersion :math:`\chi` can be provided to
:py:class:`~gammaALPs.base.transfer.GammaALPTransfer` upon initialization.

The equations of motion lead to photon-ALP oscillations with the wave number

.. math::
    \Delta_\mathrm{osc} = \sqrt{(\Delta_{||} - \Delta_a)^2 + 4\Delta_{a\gamma}^2}.

For an unpolarised photon beam,
the problem has to be reformulated in terms of the density matrix :math:`\rho(z) = \Psi(z)\Psi(z)^\dagger`
that obeys the von-Neumann-like commutator equation

.. math::
    i\frac{\mathrm{d}\rho}{\mathrm{d}x_3} = [\rho,\mathcal{M}_0],

which is solved through :math:`\rho(z) = \mathcal{T}(z,0; E)\rho(0)\mathcal{T}^\dagger(z,0; E)`,
with the transfer matrix :math:`\mathcal T` with :math:`\Psi(z) = \mathcal{T}(z,0;E)\Psi(0)` and initial condition
:math:`\mathcal{T}(0,0;E) = 1`.
In general, :math:`\mathbf B_\perp`lwill not be aligned along the :math:`y` axis
but will form an angle :math:`\psi` with it.
In this case, the solutions have to be modified with a similarity transformation and,
consequently, :math:`\mathcal M` and :math:`\mathcal T` will depend on :math:`\psi`.

In practice, the code split up the the calculation into :math:`N` steps along the :math:`z`, where it is assumed that the
magnetic field is constant for each step length :math:`\mathrm{d}z`.
For the mixing in :math:`N` consecutive steps one finds that the
photon survival probability of an initial polarisation :math:`\rho(0)` is given by

.. math::
    P_{\gamma\gamma} = \mathrm{Tr}\left( (\rho_{11} + \rho_{22})
        \mathcal{T}(z_{N},z_{1};\psi_{N},\ldots,\psi_1;E) \rho(0)
        \mathcal{T}^\dagger(z_{N},z_{1};\psi_{N},\ldots,\psi_1;E)\right),

with :math:`\rho_{11} = \mathrm{diag}(1,0,0), \rho_{22} = \mathrm{diag}(0,1,0)`, and

.. math::
    \mathcal{T}(z_{N},z_{1};\psi_{N},\ldots,\psi_1;E) = \prod\limits_{i = 1}^{N} \mathcal{T}(z_{i+1},z_{i};\psi_{i};E).

For an initially unpolarised photon beam one has :math:`\rho(0) = (1/2) \,\mathrm{diag}(1,1,0)`.
By providing the magnetic field, angle :math:`\psi`, and electron density,
(and optionally a photon mean free path and photon-photon dispersion), the codes pre-calculates all matrices
:math:`\mathcal{T}(z_{i+1},z_{i};\psi_{i};E)` for each requested energy :math:`E` as
4 dimensional arrays (:py:class:`~numpy.ndarray`)
using the :py:meth:`~gammaALPs.base.transfer.GammaALPTransfer.fill_transfer` method.
These are then multiplied together with a matrix multiplication along the :math:`z` axis.

Reference / API
---------------

.. autoclass:: gammaALPs.base.transfer.GammaALPTransfer
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

.. autofunction:: gammaALPs.base.transfer.EminGeV

.. autofunction:: gammaALPs.base.transfer.EmaxGeV
