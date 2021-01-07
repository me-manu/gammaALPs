.. _environments:

Implemented Astrophysical Environments
======================================

This page summarizes the available astrophysical environments in which the photon-ALP propagation
can be calculated.
The environments can be added to the propagation by invoking the :py:meth:`~gammaALPs.core.ModuleList.add_propagation`
function, see the :ref:`module` page for more information, as well as the :ref:`tutorials`.

The astrophysical environments combine the magnetic field models (see :ref:`bfields`)
and models for the electron density :math:`n_\mathrm{el}(r)` (see :ref:`electrondens`) and are summarized in the table below.

A minimal example for initializing mixing in the intergalactic magnetic field (IGMF) would like this:

.. code-block:: python

    from gammaALPs.core import Source, ALP, ModuleList
    src = Source(z=0.859)  # initialize a source with redshift z=0.536
    alp = ALP(m = 1., g = 1.)
    ml = ModuleList(alp, src)
    ml.add_propagaion('IGMF')
    px, py, pa = ml.run()

The other supported environments are listed in the table below.

+-------------------+-------------------------------------------------------------------+--------------------------------------------+--------------------------------------------------------------+-----------------------------------------------------+
| Environment Name  | Environment Class                                                 | Intended Use                               | Used :math:`B`-field model                                   | Used :math:`n_\mathrm{el}` model                    |
+===================+===================================================================+============================================+==============================================================+=====================================================+
| IGMF              | :py:class:`~gammaALPs.base.environs.MixIGMFCell`                  | Mixing in the intergalactic magnetic field | :py:class:`~gammaALPs.bfields.cell.Bcell`                    | constant (evolves with redshift)                    |
|                   |                                                                   | with a cell-like structure                 |                                                              |                                                     |
+-------------------+-------------------------------------------------------------------+--------------------------------------------+--------------------------------------------------------------+-----------------------------------------------------+
| ICMCell           | :py:class:`~gammaALPs.base.environs.MixICMCell`                   | Mixing in a galaxy cluster magnetic field  | :py:class:`~gammaALPs.bfields.cell.Bcell`                    | :py:class:`~gammaALPs.nel.icm.NelICM`               |
|                   |                                                                   | with a cell-like structure that decreases  |                                                              |                                                     |
|                   |                                                                   | with growing distance from cluster center  |                                                              |                                                     |
|                   |                                                                   | following :math:`n_\mathrm{el}(r)`         |                                                              |                                                     |
+-------------------+-------------------------------------------------------------------+--------------------------------------------+--------------------------------------------------------------+-----------------------------------------------------+
| ICMGaussTurb      | :py:class:`~gammaALPs.base.environs.MixICMGaussTurb`              | Mixing in a galaxy cluster magnetic field  | :py:class:`~gammaALPs.bfields.gauss.Bgaussian`               | :py:class:`~gammaALPs.nel.icm.NelICM`               |
|                   |                                                                   | with Gaussian turbulence that decreases    |                                                              |                                                     |
|                   |                                                                   | with growing distance from cluster center  |                                                              |                                                     |
|                   |                                                                   | following :math:`n_\mathrm{el}(r)`         |                                                              |                                                     |
+-------------------+-------------------------------------------------------------------+--------------------------------------------+--------------------------------------------------------------+-----------------------------------------------------+
| Jet               | :py:class:`~gammaALPs.base.environs.MixJet`                       | Mixing in the toroidal magnet field of an  | :py:class:`~gammaALPs.bfields.jet.Bjet`                      | :py:class:`~gammaALPs.nel.jet.NelJet`               |
|                   |                                                                   | AGN jet, where the B field and             |                                                              |                                                     |
|                   |                                                                   | :math:`n_\mathrm{el}(r)` decrease as a     |                                                              |                                                     |
|                   |                                                                   | power law with increasing distance from the|                                                              |                                                     |
|                   |                                                                   | central supermassive black hole            |                                                              |                                                     |
+-------------------+-------------------------------------------------------------------+--------------------------------------------+--------------------------------------------------------------+-----------------------------------------------------+
| JetHelicalTangled | :py:class:`~gammaALPs.base.environs.MixJetHelicalTangled`         | Mixing in helical and tangled field of an  | :py:class:`~gammaALPs.bfields.jet.BjetHelicalTangled`        | :py:class:`~gammaALPs.nel.jet.NelJetHelicalTangled` |
|                   |                                                                   | AGN jet, where the B field and             |                                                              |                                                     |
|                   |                                                                   | :math:`n_\mathrm{el}(r)` decrease as a     |                                                              |                                                     |
|                   |                                                                   | power law with increasing distance from the|                                                              |                                                     |
|                   |                                                                   | central supermassive black hole.           |                                                              |                                                     |
+-------------------+-------------------------------------------------------------------+--------------------------------------------+--------------------------------------------------------------+-----------------------------------------------------+
| GMF               | :py:class:`~gammaALPs.base.environs.MixGMF`                       | Mixing in the magnetic field of the Milky  | :py:class:`~gammaALPs.bfields.gmf.GMF`                       | constant                                            |
|                   |                                                                   | Way. Currently, no model for               | :py:class:`~gammaALPs.bfields.gmf.GMFPshirkov`               |                                                     |
|                   |                                                                   | :math:`n_\mathrm{el}` is implemented       |                                                              |                                                     |
+-------------------+-------------------------------------------------------------------+--------------------------------------------+--------------------------------------------------------------+-----------------------------------------------------+
| File              | :py:class:`~gammaALPs.base.environs.MixFromFile`                  | Environment initalized from B field and    | ---                                                          | ---                                                 |
|                   |                                                                   | :math:`n_\mathrm{el}` from a file          |                                                              |                                                     |
+-------------------+-------------------------------------------------------------------+--------------------------------------------+--------------------------------------------------------------+-----------------------------------------------------+
| Array             | :py:class:`~gammaALPs.base.environs.MixFromArray`                 | Environment initalized from B field and    | ---                                                          | ---                                                 |
|                   |                                                                   | :math:`n_\mathrm{el}` from numpy arrays    |                                                              |                                                     |
+-------------------+-------------------------------------------------------------------+--------------------------------------------+--------------------------------------------------------------+-----------------------------------------------------+
| EBL               | :py:class:`~ebltable.tau_from_model.OptDepth`                     | No mixing, only absorption on the EBL      | ---                                                          | ---                                                 |
+-------------------+-------------------------------------------------------------------+--------------------------------------------+--------------------------------------------------------------+-----------------------------------------------------+

Reference / API
---------------

.. automodule:: gammaALPs.base.environs
    :members:
    :undoc-members:
    :show-inheritance:
