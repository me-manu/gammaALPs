.. _struc:

Large scale structured magnetic field of Galaxy Clusters
----------------------------------------------------------

This module implements the magnetic field derived by [Gourgouliatos2010]_ as a solution to the Grad-Shafranov equation.
The field strength vanishes on the cavity surface whose interior is otherwise filled by a large scale magnetic field.
Field components are given in spherical coordinates. All components can be accessed with
:py:meth:`gammaALPs.bfields.struc.structured_field.b_r`,
:py:meth:`gammaALPs.bfields.struc.structured_field.b_theta` and
:py:meth:`gammaALPs.bfields.struc.structured_field.b_phi`.

The user is able to set an orientation of the field's symmetry axis using :math:`\theta` (inclination to the line of sight)
and :math:`\mathrm{pa}` (position angle of jet axis) in galactic coordinates (zero points to Galactic North).

There is also a convenience functions to calculate the the rotation measure
induced by the field (provided that an electron density is provided as well).

Reference / API
^^^^^^^^^^^^^^^

.. automodule:: gammaALPs.bfields.struc
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:
