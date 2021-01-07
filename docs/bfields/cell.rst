.. _cell:

Magnetic fields with cell-like structure
----------------------------------------

The magnetic fields in this module are of a simple cell-like structure, i.e., the field strength is constant
but the orientation changes randomly from one cell to the next.

The cell size is given by the coherence length and stored in :py:attr:`gammaALPs.bfields.cell.Bcell.Lcoh`,
the magnetic field strength is stored in :py:attr:`gammaALPs.bfields.cell.Bcell.B`.

New random fields are computed by :py:meth:`gammaALPs.bfields.cell.Bcell.new_Bn` and
:py:meth:`gammaALPs.bfields.cell.Bcell.new_Bcosmo` for a cosmological magnetic field.
For a cosmological magnetic field the evolution with redshift of the coherence length and the field
strength are taken into account, see, e.g., [DeAngelis2011]_.
The simple cell model was used in, e.g., [Horns2012]_ and [Meyer2013]_ to model the photon-ALP mixing in galaxy clusters.

Reference / API
^^^^^^^^^^^^^^^

.. automodule:: gammaALPs.bfields.cell
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:
