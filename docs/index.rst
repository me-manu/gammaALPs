.. gammaALPs documentation master file, created by
   sphinx-quickstart on Mon Jan  4 00:39:57 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the gammaALPs documentation!
=======================================

gammaALPs is a python package that calculates the oscillation probability between photons and axion-like particles
(ALPs) in various astrophysical environments. The focus lies on environments relevant to mixing between gamma rays and
ALPs but it can be used for broader applications. The code also implements various models of astrophysical
magnetic fields, which can be useful for applications beyond ALP searches.

You also might find the `gammaALPsPlot Package <https://github.com/me-manu/gammaALPsPlot/>`_ useful which aims to
facilitate creating plots of the ALP parameter space.

Getting Started
---------------

For installing the code, please see the :ref:`installation` page.

If you want a quick start, take a look at the :ref:`Tutorials`.

Background on how the photon-ALP mixing is calculated is provided on the :ref:`theory` page. The core modules that
are required to run the photon-ALP oscillation computation are described in more detail on the :ref:`module` page.
The page :ref:`environments` provides information on the available astrophysical environments which combine
magnetic field models described in the :ref:`bfields` section and electron densities described on the
:ref:`electrondens` pages.

Getting Help
------------

If you encounter problems or if you have suggestions,
please open a `GitHub Issue <https://github.com/me-manu/gammaALPs/issues>`_.

Documentation Contents
======================

.. toctree::
   :caption: Getting Started
   :includehidden:
   :maxdepth: 3

   installation
   tutorials/index
   references


.. toctree::
   :caption: gammaALPs Package
   :includehidden:
   :maxdepth: 3

   theory
   module
   environments
   bfields/index
   electrondens/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
