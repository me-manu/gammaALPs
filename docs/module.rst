.. _module:

Core Modules
============

This page describes the core modules that users are likely to interface most frequently.

The two classes :py:class:`~gammaALPs.core.Source` and :py:class:`~gammaALPs.core.ALP` are helper classes
which store information about the assumed astrophysical source (like its redshift) and about the assumed
ALP, in particular its mass and its coupling to photons.

For instance, a source is initialized in this way, providing the redshift and the sky coordinates:

.. code-block:: python

    src = Source(z=0.859, ra='22h53m57.7s', dec='+16d08m54s')

The :py:class:`~gammaALPs.core.ALP` class is initialized by simply providing values for the ALP mass
(in nano electron volts, :math:`\mathrm{neV}\equiv 10^{-9}\,\mathrm{eV}`) and the coupling to photons,
:math:`g_{a\gamma}` in units of :math:`10^{-11}\,\mathrm{GeV}^{-1}`.

.. code-block:: python

    from gammaALPs.core import ALP
    alp = ALP(m = 1., g = 1.)
    print (alp.m, alp.g)


With the initialized :py:class:`~gammaALPs.core.Source` and :py:class:`~gammaALPs.core.ALP` classes, we can
now initialize the :py:class:`~gammaALPs.core.ModuleList` class. This class stores the astrophysical
environments in which we would like to calculate the photon-ALP propagation in the
:py:meth:`~gammaALPs.core.ModuleList.modules` property. This is essentially a list of environments with its
first entry being the environment closest to the source and the last entry the environment closest to the observer.
The :py:class:`~gammaALPs.core.ModuleList` is initialized like this:

.. code-block:: python

    from gammaALPs.core import ModuleList
    ml = ModuleList(alp, src)

Environments are added with the :py:meth:`~gammaALPs.core.ModuleList.add_propagation` function and the overall
conversion probability is then calculated by running :py:meth:`~gammaALPs.core.ModuleList.run`.
Details on the available environments are available on the :ref:`environments` page.

The usage of these core classes is further demonstrated in the :ref:`tutorials`.


Reference / API
---------------

.. autoclass:: gammaALPs.core.Source
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

.. autoclass:: gammaALPs.core.ALP
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

.. autoclass:: gammaALPs.core.ModuleList
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:
