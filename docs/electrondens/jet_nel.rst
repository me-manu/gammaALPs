.. _jet_nel:

Electron density models for AGN jets
------------------------------------

The :py:class:`gammaALPs.nel.jet.NelJet` class simply models the electron density as a power law with

.. math::
    n^\mathrm{jet}_\mathrm{el}(r) = n^\mathrm{jet}_0 \left(\frac{r}{r_0}\right)^{\beta}.

A more realistic model is provided in the :py:class:`gammaALPs.nel.jet.NelJetHelicalTangled` class,
which takes into account the fact that inside a relativistic AGN jet, we are not dealing with a
cold thermal plasma, see, e.g., [Davies2021]_. This is done by evaluating the effective photon mass.

Reference / API
^^^^^^^^^^^^^^^

.. automodule:: gammaALPs.nel.jet
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:
