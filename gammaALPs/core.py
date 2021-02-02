from __future__ import absolute_import, division, print_function
from astropy import units as u
from astropy.coordinates import SkyCoord
import numpy as np
from ebltable.tau_from_model import OptDepth
from .base import environs as env
from .base.transfer import calc_conv_prob, calc_lin_pol
from .utils.interp2d import Interp2D
import logging
import sys


class Source(object):
    """
    Class that stores information about the source:
    redshift
    Sky coordinates: ra, dec, l,b
    As well as jet parameters: bulk Lorentz factor, observation
    and jet opening angle
    """
    def __init__(self, z, **kwargs):
        """
        Initialize the source class

        Parameters
        ----------
        z: float
            source redshift
        ra: float or string
            Right Ascension compatible with :py:class:`~astropy.coordinates.SkyCoord` object
        dec: float or string
            Declination compatible with :py:class:`~astropy.coordinates.SkyCoord` object
        l: float
            Galactic longitude or None. If given, overwrites ra and dec
        b: float
            Galactic latitude or None. If given, overwrites ra and dec
        theta_obs: float
            Angle between l.o.s. and jet axis in degrees (default: 3.)
        bLorentz: float
            bulk lorentz factor of gamma-ray emitting plasma (default: 10.)
        theta_jet: float
            Jet opening angle in degrees. Default: 1/bLorentz
        """
        kwargs.setdefault('ra', None)
        kwargs.setdefault('dec', None)
        kwargs.setdefault('l', None)
        kwargs.setdefault('b', None)
        kwargs.setdefault('theta_obs', 3.)
        kwargs.setdefault('bLorentz', 10.)
        kwargs.setdefault('theta_jet', np.rad2deg(1./kwargs['bLorentz']))

        if kwargs['ra'] is None and kwargs['dec'] is None and \
            kwargs['l'] is None and kwargs['b'] is None:
            raise ValueError("Coordinates cannot all be of None Type!")

        # calculate doppler factor
        self._bLorentz = kwargs['bLorentz']
        self._theta_obs = kwargs['theta_obs']
        self._theta_jet = kwargs['theta_jet']
        self._z = z
        self._doppler = None
        self._c = None
        self._c_gal = None
        self._ra = None
        self._dec = None
        self._l = None
        self._b = None

        self.calc_doppler()
        self.set_ra_dec_l_b(ra=kwargs['ra'], dec=kwargs['dec'],
                            l=kwargs['l'], b= kwargs['b'])

        return

    @property
    def z(self):
        return self._z

    @property
    def ra(self):
        return self._ra

    @property
    def dec(self):
        return self._dec

    @property
    def l(self):
        return self._l

    @property
    def b(self):
        return self._b

    @property
    def c(self):
        return self._c

    @property
    def c_gal(self):
        return self._c_gal

    @property
    def theta_obs(self):
        return self._theta_obs

    @property
    def theta_jet(self):
        return self._theta_jet

    @property
    def bLorentz(self):
        return self._bLorentz

    @theta_obs.setter
    def theta_obs(self, theta_obs):
        if type(theta_obs) == u.Quantity:
            self._theta_obs = theta_obs.to('degree').value
        else:
            self._theta_obs = theta_obs
            self.calc_doppler()
        return

    @z.setter
    def z(self, z):
        self._z = z

    @theta_jet.setter
    def theta_jet(self, theta_jet):
        if type(theta_jet) == u.Quantity:
            self._theta_jet = theta_jet.to('degree').value
        else:
            self._theta_jet = theta_jet
        return

    @bLorentz.setter
    def bLorentz(self, bLorentz):
        self._bLorentz = bLorentz
        self.calc_doppler()
        return

    @ra.setter
    def ra(self, ra):
        if type(ra) == u.Quantity:
            self._ra = ra.to('degree').value
        else:
            self._ra = ra
        self.set_ra_dec_l_b(self._ra,self._dec)
        return

    @dec.setter
    def dec(self, dec):
        if type(dec) == u.Quantity:
            self._dec = dec.to('degree').value
        else:
            self._dec = dec
        self.set_ra_dec_l_b(self._ra,self._dec)
        return

    @l.setter
    def l(self, l):
        if type(l) == u.Quantity:
            self._l = l.to('degree').value
        else:
            self._l = l
        self.set_ra_dec_l_b(self._ra,self._dec, l = self._l, b = self._b)
        return

    @b.setter
    def b(self, b):
        if type(b) == u.Quantity:
            self._b = b.to('degree').value
        else:
            self._b =  b
        self.set_ra_dec_l_b(self._ra, self._dec, l=self._l, b=self._b)
        return

    def calc_doppler(self):
        """Calculate the Doppler factor of the plasma"""
        self._doppler = 1./(self._bLorentz * (1. - np.sqrt(1. - 1./self._bLorentz**2.) *
                            np.cos(np.radians(self.theta_obs))))
        return

    def set_ra_dec_l_b(self,ra,dec,l = None, b = None):
        """Set l and b or ra and dec"""

        if l is None and b is None:
            if type(ra) == str:
                self._c = SkyCoord(ra, dec, frame='icrs')
            elif type(ra) == float:
                self._c = SkyCoord(ra, dec, frame='icrs', unit='deg')
            self._c_gal = self._c.galactic
            self._l = self._c_gal.l.value
            self._b = self._c_gal.b.value
        else:
            self._l = l
            self._b = b
            self._c_gal =  SkyCoord(l, b, frame='galactic', unit='deg')
            self._c = self._c_gal.icrs

        self._ra = self._c.ra.value
        self._dec = self._c.dec.value
        return


class ALP(object):
    """
    Class to store the ALP parameters
    """
    def __init__(self, m, g):
        """
        Initialize the ALP class

        Parameters
        ----------
        m: float
            ALP mass in neV.

        g: float
            photon-ALP coupling in 10^-11 GeV^-1.
        """
        self._m = m
        self._g = g
        return

    @property
    def m(self):
        return self._m

    @property
    def g(self):
        return self._g

    @m.setter
    def m(self, m):
        if type(m) == u.Quantity:
            self._m = m.to('neV').value
        else:
            self._m = m
        return

    @g.setter
    def g(self, g):
        if type(g) == u.Quantity:
            self._g = g.to('10**-11GeV-1').value
        else:
            self._g = g
        return


class NamedClassList(list):
    """
    A list of classes indexable with an integer or class's name.
    As it is a subclasses the built-in class ``list``, it provides all the methods of the
    standard ``list`` class.
    Adapted from
    https://github.com/s3rvac/blog/tree/master/en-2014-10-11-indexing-python-lists-with-integer-or-object-name
    """

    def __getitem__(self, key):
        return self._delegate_to_list('__getitem__', key)

    def __setitem__(self, key, value):
        return self._delegate_to_list('__setitem__', key, value)

    def __delitem__(self, key):
        return self._delegate_to_list('__delitem__', key)

    def keys(self):
        return [type(item).__name__.strip("mix") for item in self]

    def values(self):
        return self

    def items(self):
        return [(self.keys()[i], self[i]) for i in range(len(self))]

    def _delegate_to_list(self, method, key, *args):
        if isinstance(key, str):
            key = self._index_of(key)
        if sys.version_info[0] < 3:
            return getattr(super(NamedClassList, self), method)(key, *args)
        else:
            return getattr(super(), method)(key, *args)

    def _index_of(self, name):
        for index, item in enumerate(self):
            if type(item).__name__ == name or type(item).__name__.strip("Mix") == name:
                return index
        raise IndexError('no object named {!r}'.format(name))


class ModuleList(object):
    """
    Class that collects all environments for photon-ALP mixing
    and manages common parameters such as photon-ALP coupling,
    the ALP mass, the source, and the energies at which
    the photon-ALP oscillation is computed.
    """
    def __init__(self,
                 alp,
                 source,
                 pin=None,
                 EGeV=None,
                 seed=None):
        """
        Initialize the class, energy range, and polarization matrices

        Parameters
        ----------
        alp: :py:class:`~gammaALPs.ALP`
            :py:class:`~gammaALPs.ALP` object with ALP parameters

        source: :py:class:`~gammaALPs.Source`
            :py:class:`~gammaALPs.Source` object with source parameters

        pin: array-like
            3x3 dim matrix with initial polarization.
            Default: un-polarized photon beam

        EGeV: array-like
            n-dim numpy array with gamma-ray energies in GeV
            Default: log-spaced array between 1 GeV and 10 TeV

        seed: int, optional
            Seed for RandomState for numpy random numbers.
            Must be convertible to 32 bit unsigned integers.
        """
        self._alp = alp
        self._source = source
        if EGeV is None:
            self._EGeV = np.logspace(0., 4., 100)
        else:
            self._EGeV = EGeV
        # initialize final polarization states
        self._px = np.diag((1., 0., 0.))
        self._py = np.diag((0., 1., 0.))
        self._pa = np.diag((0., 0., 1.))
        if pin is None:
            self._pin = np.diag((1., 1., 0.)) * 0.5
        else:
            self._pin = pin
        self._modules = NamedClassList()
        self._seed = seed
        self._px_src = None
        self._py_src = None
        self._pa_src = None
        self._px_final = None
        self._py_final = None
        self._pa_final = None
        self._lin_pol = None
        self._circ_pol = None
        self.__nsim_max = None
        self._atten = None
        self._eblnorm = None
        return

    @property
    def alp(self):
        return self._alp

    @property
    def source(self):
        return self._source

    @property
    def EGeV(self):
        return self._EGeV

    @property
    def px(self):
        return self._px

    @property
    def py(self):
        return self._py

    @property
    def pa(self):
        return self._pa

    @property
    def pin(self):
        return self._pin

    @property
    def seed(self):
        return self._seed

    @property
    def modules(self):
        return self._modules

    @pin.setter
    def pin(self,pin):
        self._pin = pin
        return

    @EGeV.setter
    def EGeV(self, EGeV):
        if type(EGeV) == u.Quantity:
            self._EGeV = EGeV.to('GeV').value
        else:
            self._EGeV = EGeV

        # update energies for all modules
        for m in self.modules:
            if type(m) == OptDepth:
                self._atten = np.exp(-self._eblnorm * m.opt_depth(self.source.z,EGeV / 1e3))
            elif type(m) == env.MixIGMFCell or type(m) == env.MixGMF:
                m.EGeV = EGeV
            elif type(m) == env.MixJetHelicalTangled:
                m.EGeV = EGeV * (1. + self.source.z)
                m._ee /= m._gammas
            else:
                m.EGeV = EGeV * (1. + self.source.z)
        return

    @seed.setter
    def seed(self, seed):
        self._seed = seed

    def add_propagation(self, environ, order, **kwargs):
        """
        Add a propagation environment to the module list

        Parameters
        ----------
        environ: str
            identifier for environment, see notes for possibilities
        order: int
            the order of the environment along the line of sight,
            starting at zero, where zero is closest to the source and highest value is closest to the
            observer

        Notes
        -----
        kwargs are passed to the specific environment.

        Available environments are the classes given in :py:class:`~gammaALPs.base.environs`,
        where all the specific options are listed.
        The identifiers for the environments are:

        - IGMF: initializes :py:class:`~gammaALPs.base.environs.MixIGMFCell` for mixing
            in intergalactic magnetic field (IGMF) which is assumed to be of a cell-like structure

        - ICMCell: initializes :py:class:`~gammaALPs.base.environs.MixICMCell` for mixing
            in intra-cluster medium which is assumed to be of a cell-like structure

        - ICMGaussTurb: initializes :py:class:`~gammaALPs.base.environs.MixICMGaussTurb` for mixing
            in intra-cluster medium which is assumed to follow a Gaussian turbulence spectrum

        - Jet: initializes :py:class:`~gammaALPs.base.environs.MixJet` for mixing
            in the AGN jet, where the field is assumed to be coherent

        - JetHelicalTangled: initializes :py:class:`~gammaALPs.base.environs.MixJetHelicalTangled` for mixing
            in the AGN jet with two field components (tangled and helical)

        - GMF: initializes :py:class:`~gammaALPs.base.environs.MixGMF` for mixing
            in the Galactic magnetic field (GMF) of the Milky Way

        - File: initializes :py:class:`~gammaALPs.base.environs.MixFromFile` for mixing
            in a magnetic field given by a data file

        - Array: initializes :py:class:`~gammaALPs.base.environs.MixFromArray` for mixing
            in a magnetic field given by a numpy arrays for B,psi,nel,r, and dL

        - EBL: initializes :py:class:`~ebltable.tau_from_model.OptDepth` for EBL attenuation,
            i.e. no photon-ALP mixing in the intergalactic medium
        """
        kwargs.setdefault('eblmodel', 'dominguez')
        kwargs.setdefault('eblnorm', 1.)
        kwargs.setdefault('seed', self._seed)

        self._eblnorm = kwargs['eblnorm']

        if environ == 'EBL':
            self._modules.insert(order, OptDepth.readmodel(model=kwargs['eblmodel']))
            self._atten = np.exp(-self._eblnorm *\
                self._modules[order].opt_depth(self.source.z, self.EGeV / 1e3))

        elif environ == 'IGMF':
            self._modules.insert(order, env.MixIGMFCell(self.alp, self.source,
                                                        EGeV=self.EGeV,
                                                        **kwargs))
        elif environ == 'ICMCell':
            self._modules.insert(order, env.MixICMCell(self.alp,
                                                       EGeV=self.EGeV * (1. + self.source.z),
                                                       **kwargs))
        elif environ == 'ICMGaussTurb':
            self._modules.insert(order, env.MixICMGaussTurb(self.alp,
                                                            EGeV=self.EGeV * (1. + self.source.z),
                                                            **kwargs))
        elif environ == 'Jet':
            self._modules.insert(order, env.MixJet(self.alp, self.source,
                                                   EGeV=self.EGeV * (1. + self.source.z),
                                                   **kwargs))
        elif environ == 'JetHelicalTangled':
            self._modules.insert(order, env.MixJetHelicalTangled(self.alp, self.source,
                                                                 EGeV=self.EGeV * (1. + self.source.z),
                                                                 **kwargs))
        elif environ == 'GMF':
            self._modules.insert(order, env.MixGMF(self.alp, self.source,
                                                   EGeV=self.EGeV,
                                                   **kwargs))
        elif environ == 'File':
            self._modules.insert(order, env.MixFromFile(self.alp,
                                                        EGeV=self.EGeV,
                                                        **kwargs))
        elif environ == 'Array':
            self._modules.insert(order, env.MixFromArray(self.alp,
                                                         EGeV=self.EGeV,
                                                         **kwargs))
        else:
            raise ValueError("Unkwon Environment chosen")
        return

    def add_disp_abs(self, EGeV, r_kpc, disp, module_id, type_matrix='dispersion', **kwargs):
            """
            Add dispersion, absorption, or extra momentum difference term to a propagation module using
            interpolation of of a 2d dispersion / absorption matrix

            Parameters
            ----------
            EGeV: array-like
                n-dim array with gamma-ray energies in GeV at which dispersion/absorption/momentum difference matrix
                is calculated

            r_kpc: array-like
                m-dim array with distnaces in kpc at which dispersion/absorption/momentum difference matrix
                is calculated

            disp: array-like
                n x m-dim array with dispersion (unitless) / absorption (in kpc^-1) / momentum difference (in kpc^-1)

            module_id: int
                id of module to which dispersion / absorption /momentum difference is added

            type_matrix: str
                either 'dispersion', 'absorption', or 'Delta', specifies type of matrix disp
            """
            if module_id >= len(self.modules):
                raise ValueError("requested module id is out of range")
            if isinstance(self.modules[module_id], OptDepth):
                raise RuntimeError("dispersion / absorption cannot be applied to EBL only module")

            # interpolate matrix
            intp = Interp2D(np.log10(EGeV), r_kpc, disp, **kwargs)
            # cast to values used for propagation
            new_disp = intp(np.log10(self.modules[module_id].EGeV), self.modules[module_id]._r)

            if type_matrix == 'dispersion':
                self.modules[module_id].chi = new_disp
            elif type_matrix == 'absorption':
                self.modules[module_id].Gamma = new_disp
            elif type_matrix == 'Delta':
                self.modules[module_id].Delta = new_disp
            else:
                raise ValueError("type_matrix must either be 'dispersion', 'absorption' or 'Delta'")

    def _check_modules_random_fields(self):
        """
        Check how many modules have n > 1 B field realizations.
        At the moment, only one component is allowed to have multiple realizations.
        """
        self._all_nsim = []
        for im, m in enumerate(self.modules):
            if not type(m) == OptDepth:
                self._all_nsim.append(m.nsim)
        logging.debug(self._all_nsim)
        logging.debug(np.array(self._all_nsim) > 1)
        if np.sum(np.array(self._all_nsim) > 1) > 1:
            logging.error("More than one module has multiple B-field realizations, not supported")
            logging.error("Number of realizations for the ALP modules: {0}".format(self._all_nsim))
            raise RuntimeError
        return

    def _multiply_env(self,istart,istop,n):
        """
        Helper function to multiply the transfer matrices
        of the different traversed environments
        """
        for it, Tenv in enumerate(self._Tenv[istart:istop]):
            if self.__nsim_max == (self._all_nsim[istart:istop])[it]:
                nn = n
            else:
                nn = 0
            if not it:
                Tsrc = Tenv[nn]
            else:
                Tsrc = np.matmul(Tsrc, Tenv[nn])
        return Tsrc

    def run(self, multiprocess=1):
        """
        Run the photon-ALP conversion probability calculation for all modules

        Parameters
        ----------
        multiprocess: int
            number of cores to perform calculation; only used when random B field is requested
            by one of the modules and number of realizations is > 1

        Returns
        -------
        Px, Py, Pa: tuple with :py:class:`numpy.ndarray`
            N x M dim. arrays with final photon and ALP states (Px, Py, Pa).
            Each P_i is of dimension (number of realizations x number of energy bins).
            Flat arrays are returned when number of realizations = 1.
        """
        # update energies of all modules
        # also accounts for changed redshift
        self.EGeV = self._EGeV

        # Calculate the transfer matrices for each environment
        self._Tenv = []

        self._check_modules_random_fields()
        for im, m in enumerate(self.modules):
            if not type(m) == OptDepth:
                logging.info('Running Module {0:n}: {1}'.format(im, type(m)))
                logging.debug('Photon-ALP mixing in {0}: {1:.3e}'.format(type(m), m.alp.g))
                logging.debug('ALP mass in {0}: {1:.3e}'.format(type(m), m.alp.m))
                logging.debug('Energy range in {0}: {1:.3e}-{2:.3e}, n = {2:n}'.format(
                                            type(m), m.EGeV[0], m.EGeV[-1], m.EGeV.size))
                logging.debug('Number of B-field real. in {0}: {1:n}'.format(type(m), m.nsim))
                if multiprocess > 1:
                    T = m.calc_transfer_multi(nprocess=multiprocess)
                else:
                    T = []
                    for i in range(m.nsim):
                        T.append(m.calc_transfer(nsim = i, nprocess=1))
                self._Tenv.append(np.array(T))
        # check if we have simple EBL absorption present
        # in which case we calculate separately the mixing in the source,
        # the EBL absorption, and the mixing near the observer.
        # Otherwise only use transfer matrices to calculate the total
        # oscillation probability
        self.__nsim_max = np.max(self._all_nsim)

        self._px_final, self._py_final, self._pa_final = [],[],[]
        self._lin_pol, self._circ_pol = [],[]
        if OptDepth in [type(t) for t in self.modules]:
            # get the index of the EBL module
            idx = [type(t) for t in self.modules].index(OptDepth)
            self._px_src, self._py_src, self._pa_src = [], [], []
            # step through the simulations of random B fields
            for n in range(self.__nsim_max):
                # mutliply all matrices for source environment
                Tsrc = self._multiply_env(0,idx,n)
                self._px_src.append(calc_conv_prob(self.pin, self.px, Tsrc))
                self._py_src.append(calc_conv_prob(self.pin, self.py, Tsrc))
                self._pa_src.append(calc_conv_prob(self.pin, self.pa, Tsrc))

                if not idx==len(self._Tenv):
                    # new polarization matrix close to observer after traversing EBL
                    pol = np.zeros((self.EGeV.size,3,3))
                    pol[:, 0, 0] = self._px_src[-1] * self._atten
                    pol[:, 1, 1] = self._py_src[-1] * self._atten
                    pol[:, 2, 2] = self._pa_src[-1]

                    # mutliply all matrices for observer environment
                    # all_sim and Tenv have one index less, since EBL not included

                    Tobs = self._multiply_env(idx,len(self._Tenv) + 1,n)
                    self._px_final.append(calc_conv_prob(pol, self.px, Tobs))
                    self._py_final.append(calc_conv_prob(pol, self.py, Tobs))
                    self._pa_final.append(calc_conv_prob(pol, self.pa, Tobs))
                    l, c = calc_lin_pol(pol, Tobs)
                else:
                    # if EBL is the final environment, just apply the attenuation
                    self._px_final.append(self._px_src[-1] * self._atten)
                    self._py_final.append(self._py_src[-1] * self._atten)
                    self._pa_final.append(self._pa_src[-1])
                    l, c = calc_lin_pol(self.pin, Tsrc)

                self._lin_pol.append(l)
                self._circ_pol.append(c)

            self._px_src = np.array(self._px_src)
            self._py_src = np.array(self._py_src)
            self._pa_src = np.array(self._pa_src)

        else:
            # mutlitply all environments for all B-field realizations
            for n in range(self.__nsim_max):
                Tobs = self._multiply_env(0,len(self._Tenv) + 1,n)
                self._px_final.append(calc_conv_prob(self.pin, self.px, Tobs))
                self._py_final.append(calc_conv_prob(self.pin, self.py, Tobs))
                self._pa_final.append(calc_conv_prob(self.pin, self.pa, Tobs))
                l, c = calc_lin_pol(self.pin, Tobs)
                self._lin_pol.append(l)
                self._circ_pol.append(c)

        self._px_final = np.array(self._px_final)
        self._py_final = np.array(self._py_final)
        self._pa_final = np.array(self._pa_final)
        self._lin_pol = np.array(self._lin_pol)
        self._circ_pol = np.array(self._circ_pol)

        return self._px_final, self._py_final, self._pa_final
