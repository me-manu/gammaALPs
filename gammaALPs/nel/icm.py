# --- Imports --------------------- #
import numpy as np
from astropy import units as u
# --------------------------------- #

# ========================================================== #
# === Electron densities for intra-cluster medium ========== #
# ========================================================== #


class NelICM(object):
    """Class to set characteristics of electron density of intracluster medium"""
    def __init__(self, **kwargs):
        """
        Initialize the class

        Parameters
        ----------
        n0: float
            electron density in cm**-3 (default 1e-3)

        r_core: float
            core radius in kpc (default 10.)

        beta: float
            exponent of density profile (default: 2. / 3.)

        eta: float
            exponent for scaling of B field with electron density (default = 1.)

        n2: float
            if > 0., use profile with this second density component 

        r_core2: float
            if > 0., use profile with this second r_core value

        beta2: float
            if > 0., use profile with this second beta value as for NGC1275
        """
        kwargs.setdefault('n0', 1e-3)
        kwargs.setdefault('r_core', 10.)
        kwargs.setdefault('eta', 1.)
        kwargs.setdefault('beta', 2. / 3.)
        kwargs.setdefault('n2', 0.)
        kwargs.setdefault('r_core2', 0.)
        kwargs.setdefault('beta2', 0.)

        self._n0 = kwargs['n0']
        self._r_core = kwargs['r_core']
        self._beta = kwargs['beta']
        self._eta = kwargs['eta']

        self._r_core2 = kwargs['r_core2']
        self._beta2 = kwargs['beta2']
        self._n2 = kwargs['n2']

        return 

    @property
    def n0(self):
        return self._n0

    @property
    def n2(self):
        return self._n2

    @property
    def r_core(self):
        return self._r_core

    @property
    def beta(self):
        return self._beta

    @property
    def r_core2(self):
        return self._r_core2

    @property
    def beta2(self):
        return self._beta2

    @property
    def eta(self):
        return self._eta

    @n0.setter
    def n0(self, n0):
        if type(n0) == u.Quantity:
            self._n0 = n0.to('cm**-3').value
        else:
            self._n0 = n0
        return 

    @n2.setter
    def n2(self, n2):
        if type(n2) == u.Quantity:
            self._n2 = n2.to('cm**-3').value
        else:
            self._n2 = n2
        return 

    @r_core.setter
    def r_core(self,r_core):
        if type(r_core) == u.Quantity:
            self._r_core = r_core.to('kpc').value
        else:
            self._r_core = r_core
        return 

    @beta.setter
    def beta(self, beta):
        self._beta = beta
        return

    @r_core2.setter
    def r_core2(self, r_core2):
        if type(r_core2) == u.Quantity:
            self._r_core2 = r_core2.to('kpc').value
        else:
            self._r_core2 = r_core2
        return 

    @beta2.setter
    def beta2(self, beta2):
        self._beta2 = beta2
        return

    @eta.setter
    def eta(self, eta):
        self._eta = eta
        return

    def __call__(self, r):
        """
        Calculate the electron density as function from cluster center

        Parameters
        ----------
        r: array
            n-dim array with distance from cluster center in kpc

        Returns
        -------
        nel: :py:class:`~numpy.ndarray`
            n-dim array with electron density in 10**-3 cm**-3
        """

        if self._beta2 > 0. and self._r_core2 > 0. and self._n2 > 0:
            res = self._n0 * (1. + r**2./self._r_core**2.)**(-1.5 * self._beta) +\
                    self._n2 * (1. + r**2./self._r_core2**2.)**(-1.5 * self._beta2) 
        elif self._r_core2 > 0. and self._n2 > 0:
            res = np.sqrt(self._n0**2. * (1. + r**2./self._r_core**2.)**(-3. * self._beta) +
                            self._n2**2. * (1. + r**2./self._r_core2**2.)**(-3. * self._beta))
        else:
            res = self._n0 * (1. + r**2./self._r_core**2.)**(-1.5 * self._beta)

        return res

    def constant(self,r):
        """
        return constant electron density 

        Parameters
        ----------
        r: array
            n-dim array with distance from cluster center in kpc

        Returns
        -------
        nel: :py:class:`~numpy.ndarray`
            n-dim array with electron density in 10**-3 cm**-3
        """
        return self._n0 * np.ones(r.shape[0])

    def Bscale(self, r):
        """
        Calculate the scaling of the B field with electron density as function from cluster center

        Parameters
        ----------
        r: array-like
            n-dim array with distance from cluster center in kpc

        Returns
        -------
        nel: :py:class:`~numpy.ndarray`
            n-dim array with scaling of B field with electron density
        """
        if self._beta2 > 0. and self._r_core2 > 0. and self._n2 > 0:
            return (self.__call__(r) / (self._n0 + self._n2) )**self._eta
        elif self._r_core2 > 0. and self._n2 > 0:
            return (self.__call__(r) / np.sqrt(self._n0**2. +self._n2**2.) )**self._eta
        else:
            return (self.__call__(r) / self._n0)**self._eta


class NelICMFunction(object):
    """
    Class to set characteristics of electron density of intracluster medium,
    where electron density is provided by a function
    """
    def __init__(self, func, eta=1.):
        """
        Initialize the class

        Parameters
        ----------
        func: function pointer
            function that takes radius in kpc and returns electron density in cm^-3

        eta: float
            exponent for scaling of B field with electron density (default = 1.)

        """

        self._func = func
        self._eta = eta

        return

    @property
    def func(self):
        return self._func

    @property
    def eta(self):
        return self._eta

    @eta.setter
    def eta(self, eta):
        self._eta = eta
        return

    def __call__(self, r):
        """
        Calculate the electron density as function from cluster center

        Parameters
        ----------
        r: array-like
            n-dim array with distance from cluster center in kpc

        Returns
        -------
        nel: :py:class:`~numpy.ndarray`
            n-dim array with electron density in 10**-3 cm**-3
        """

        return self._func(r) * 1e-3

    def Bscale(self, r, r0=0.):
        """
        Calculate the scaling of the B field with electron density as function from cluster center

        Parameters
        ----------
        r: array-like
            n-dim array with distance from cluster center in kpc

        r0: float
            Normalization for scale factor in kpc. Default: 0.

        Returns
        -------
        nel: :py:class:`~numpy.ndarray`
            n-dim array with scaling of B field with electron density
        """
        return (self.__call__(r) / self.__call__(r0))**self._eta
