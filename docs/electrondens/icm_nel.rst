.. _icm_nel:

Electron density models for the intra-cluster medium
----------------------------------------------------

The :py:class:`~gammaALPs.nel.icm.NelICM` implements the typical electron density profiles found in clusters,
whereas the :py:class:`~gammaALPs.nel.icm.NelICMFunction` can be used to model an electron density profile
derived from an arbitrary or interpolated function.

In the simplest case, the electron density in :py:class:`~gammaALPs.nel.icm.NelICM` is modeled with the
:math:`beta` profile,

.. math::

    n_\mathrm{el}(r) = n_0 (1 + r^2/r^2_\mathrm{core})^{-3\beta / 2}.

If the attributes :py:attr:`~gammaALPs.nel.icm.NelICM.r_core2` and :py:attr:`~gammaALPs.nel.icm.NelICM.n2` are set
to values > 0, the profile

.. math::

    n_\mathrm{el}(r) = \left(n_0^2 (1 + r^2/r^2_\mathrm{core})^{-3\beta} +
                        n_2^2 (1 + r^2/r^2_\mathrm{core, 2})^{-3\beta}\right)^{\frac{1}{2}}

is used.
If, additionally, also :py:attr:`~gammaALPs.nel.icm.NelICM.beta2` >0, then the profile is changed to

.. math::

    n_\mathrm{el}(r) = n_0 (1 + r^2/r^2_\mathrm{core})^{-3\beta / 2} +
                        n_2 (1 + r^2/r^2_\mathrm{core, 2})^{-3\beta_2 / 2},

which is used, for instance, to model the electron density in the Perseus cluster [Churazov2003]_.

Furthermore, the classes provide the method :py:meth:`~gammaALPs.nel.icm.NelICM.Bscale`, which can be used
to rescale the intra-cluster magnetic fields (see :ref:`cell` and :ref:`gauss`) to follow the radial dependence of the
electron density according to

.. math::

    B(r) = B_0\left( \frac{n_\mathrm{el}(r)}{n_\mathrm{el}(r=0)}\right)^\eta.


Reference / API
^^^^^^^^^^^^^^^

.. automodule:: gammaALPs.nel.icm
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:
