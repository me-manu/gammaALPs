.. _jet:

Magnetic field models for AGN jets
----------------------------------

The module :py:class:`gammaALPs.bfields.jet.Bjet` models the toroidal (perpendicular to the jet axis) magnetic field
as a coherent field and more details can be found in, e.g., [Meyer2014]_.
The B field decreases with a power-law type behavior with increasing distance from the black hole,

.. math::

    B^\mathrm{jet}(r) = B^\mathrm{jet}_0 \left(\frac{r}{r_0}\right)^{\alpha}

Assuming equipartition between the magnetic and particle energies, the electron density also follows a power law
with an index :math:`2\alpha` for :math:`\alpha < 0`. The electron density is modeled with the :py:class:`~gammaALPs.nel.jet.NelJet`
and is further described in :ref:`jet_nel`.

The magnetic field is supposed to be coherent, so :math:`\psi = 0` (see :ref:`theory`) is assumed over the
entire jet region in the simplest case.
A new magnetic field can be calculated with :py:meth:`gammaALPs.bfields.jet.Bjet.new_Bn`.
The B-field array calculated with
:py:meth:`gammaALPs.bfields.jet.Bjet.new_Bn` can be passed to the static method
:py:meth:`gammaALPs.bfields.jet.Bjet.transversal_component_helical`, to modify it to match
a helical magnetic field structure following [ClausenBrown2011]_.

The above equations hold in the co-moving frame of the jet. The photon energy :math:`E^\prime` in this frame
is related to the energy :math:`E` in the laboratory frame through the Doppler factor,
:math:`E^\prime = E / \delta_\mathrm{D}`, where

.. math::

    \delta_\mathrm{D} = [\Gamma_\mathrm{L}( 1 - \beta_\mathrm{j}\cos\theta_\mathrm{obs})]^{-1}

with the relativistic Lorentz and beta
factors  :math:`\Gamma_\mathrm{L},\beta_\mathrm{j}`
of the bulk plasma movement, respectively, and
:math:`\theta_\mathrm{obs}` is the angle between the jet axis and the line of sight.
The bulk Lorentz factor and :math:`\theta_\mathrm{obs}` can be set with attributes of the
:py:class:`~gammaALPs.core.Source` class, :py:attr:`~gammaALPs.core.Source.bLorentz` and
:py:attr:`~gammaALPs.core.Source.theta_obs`.

A more sophisticated model for the AGN jet magnetic field model is implemented in the
:py:class:`gammaALPs.bfields.jet.BjetHelicalTangled` class in which a poloidal magnetic field transforms
into a toroidal component. The class also includes the possibility that a fraction of the magnetic field
energy is carried in a tangled component. More details on this model are provided in [Davies2021]_.

Reference / API
^^^^^^^^^^^^^^^

.. automodule:: gammaALPs.bfields.jet
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:
