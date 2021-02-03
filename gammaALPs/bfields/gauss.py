# --- Imports --------------------- #
import numpy as np
import copy
import math
import sys
from numpy.random import rand, seed, randint
from numpy import log, log10, pi, meshgrid, cos, sum, sqrt, array, isscalar, logspace
from math import ceil
from scipy.integrate import simps
from astropy import units as u
# --------------------------------- #


# ========================================================== #
# === Gaussian turbulent magnetic field ==================== #
# ========================================================== #
class Bgaussian(object):
    """
    Class to calculate a magnetic field with gaussian turbulence and power-law spectrum
    """
    def __init__(self, B, kH, kL, q, **kwargs):
        """
        Initialize gaussian turbulence B field spectrum. 
        Defaults assume values typical for a galaxy cluster.

        Parameters
        ----------
        B: float
            rms B field strength, energy is :math:`B^2 / 4 \pi` (default = 1 muG)

        kH: float
            upper wave number cutoff, 
            should be at at least > 1. / osc. wavelength (default = 1 / (1 kpc))

        kL: float
            lower wave number cutoff,
            should be of same size as the system (default = 1 / (100 kpc))

        q: float
            power-law turbulence spectrum (default: q = 11/3 is Kolmogorov type spectrum)

        kMin: float
            minimum wave number in 1. / kpc,
            defualt 1e-3 * kL (the k interval runs from kMin to kH)

        dkType: string
            either linear, log, or random. Determine the spacing of the dk intervals

        dkSteps: int
            number of dkSteps.
            For log spacing, number of steps per decade / number of decades ~ 10
            should be chosen.

        seed: int or None
            random seed
        """
        self._B = B
        self._kH = kH
        self._kL = kL
        self._q = q

        # --- Set the defaults 
        kwargs.setdefault('kMin', None)
        kwargs.setdefault('dkType', 'log')
        kwargs.setdefault('dkSteps', 0)
        kwargs.setdefault('seed', None)

        self.__dict__.update(kwargs)
        seed(self.seed)

        if self.kMin < 0.:
            self.kMin = self._kL * 1e-3

        if not self.dkSteps:
            self.dkSteps = int(ceil(10. * (log10(self._kH) - log10(self.kMin)) ** 2.))

        # initialize the k values and intervalls.
        if self.dkType == 'linear':
            self.__kn = np.linspace(self.kMin, self._kH, self.dkSteps)
            self.__dk = self.__kn[1:] - self.__kn[:-1]
            self.__kn = self.__kn[:-1]
        elif self.dkType == 'log':
            self.__kn = np.logspace(log10(self.kMin), log10(self._kH), self.dkSteps)
            self.__dk = self.__kn[1:] - self.__kn[:-1]
            self.__kn = self.__kn[:-1]
        elif self.dkType == 'random':
            self.__dk = rand(self.dkSteps)
            self.__dk *= (self._kH - self.kMin) / sum(self.__dk)
            self.__kn = np.array([self.kMin + sum(self.__dk[:n]) for n in range(self.__dk.shape[0])])
        else:
            raise ValueError("dkType has to either 'linear', 'log', or 'random', not {0:s}".format(self.dkType))

        self.__Un = rand(self.__kn.shape[0])
        self.__Vn = rand(self.__kn.shape[0])

        return

    def __init_k_array(self):
        """initialize the k array"""
        seed(self.seed)
        # initialize the k values and intervalls.
        if self.dkType == 'linear':
            self.__kn = np.linspace(self.kMin, self._kH, self.dkSteps)
            self.__dk = self.__kn[1:] - self.__kn[:-1]
            self.__kn = self.__kn[:-1]
        elif self.dkType == 'log':
            self.__kn = np.logspace(log10(self.kMin), log10(self._kH), self.dkSteps)
            self.__dk = self.__kn[1:] - self.__kn[:-1]
            self.__kn = self.__kn[:-1]
        elif self.dkType == 'random':
            self.__dk = rand(self.dkSteps)
            self.__dk *= (self._kH - self.kMin) / sum(self.__dk)
            self.__kn = np.array([self.kMin + sum(self.__dk[:n]) for n in range(self.__dk.shape[0])])
        self.__Un = rand(self.__kn.shape[0])
        self.__Vn = rand(self.__kn.shape[0])
        return 

    @property
    def B(self):
        return self._B

    @property
    def kH(self):
        return self._kH

    @property
    def kL(self):
        return self._kH

    @property
    def q(self):
        return self._kH

    @B.setter
    def B(self,B):
        if type(B) == u.Quantity:
            self._B = B.to('10**-6G').value
        else:
            self._B = B
        return

    @kH.setter
    def kH(self, kH):
        if type(kH) == u.Quantity:
            self._kH = kH.to('kpc**-1').value
        else:
            self._kH = kH
        self.__init_k_array()
        return

    @kL.setter
    def kL(self, kL):
        if type(kL) == u.Quantity:
            self._kL = kL.to('kpc**-1').value
        else:
            self._kL = kL
        self.kMin = self._kL * 1e-3
        self.__init_k_array()
        return

    @q.setter
    def q(self,q):
        self._q = q
        return

    def new_random_numbers(self):
        """Generate new random numbers for Un,Vn, and kn if knType == random"""
        seed(self.seed)

        if self.dkType == 'random':
            self.__dk = rand(self.dkSteps)
            self.__dk *= (self._kH - self.kMin) / sum(self.__dk)
            self.__kn = np.array([self.kMin + sum(self.__dk[:n]) for n in range(self.__dk.shape[0])])

        self.__Un = rand(self.__kn.shape[0])
        self.__Vn = rand(self.__kn.shape[0])
        return

    def Fq(self, x):
        """
        Calculate the :math:`F_q` function for given :math:`x, k_L`, and :math:`k_H`

        Parameters
        ----------
        x: array-like
            Ratio between k and _kH

        Returns
        -------
        F: array-like
            n-dim array with :math:`F_q(x)` values
        """
        if self._q == 0.:
            F = lambda x: 3. * self._kH **2. / (self._kH ** 3. - self._kL ** 3.) * \
                            ( 0.5 * (1. - x*x) - x * x * log(x) )
            F_low = lambda x: 3. * (0.5 * (self._kH ** 2. - self._kL ** 2.) + \
                            (x * self._kH) * (x * self._kH) * log(self._kH / self._kL) ) \
                            / (self._kH ** 3. - self._kL ** 3.)
        elif self._q == -2.:
            F = lambda x: ( 0.5 * (1. - x*x) - log(x) ) / (self._kH  - self._kL )
            F_low = lambda x: ( log(self._kH / self._kL ) + (x * self._kH) * \
                            (x * self._kH) * 0.5 * (self._kL ** (-2) - self._kH ** (-2)) ) \
                            / (self._kH  - self._kL )
        elif self._q == -3.:
            F = lambda x:  1. / log(self._kH / self._kL) / self._kH / x  / 3. * (-x*x*x - 3. * x + 4.)
            F_low = lambda x: ( (self._kL ** (-2) - self._kH ** (-2)) + (x * self._kH) * \
                            (x * self._kH) / 3. * (self._kL ** (-3) - self._kH ** (-3)) ) / \
                            log(self._kH / self._kL )
        else:
            F = lambda x: self._kH ** (self._q + 2.) / (self._kH ** (self._q + 3.) - \
                                    self._kL ** (self._q + 3.)) * \
                                    (self._q + 3.) / (self._q * ( self._q + 2.)) * \
                                    (self._q + x * x * ( 2. + self._q - 2. * (1. + self._q) * x ** self._q))
            F_low = lambda x: (self._q + 3.) * ( (self._kH ** (self._q + 2) - \
                                    self._kL ** (self._q +2)) / (self._q + 2.) + \
                                    (x * self._kH) * (x * self._kH) / self._q * \
                                    (self._kH ** self._q - self._kL ** self._q) ) / \
                                    (self._kH ** (self._q + 3.) - self._kL**(self._q + 3) ) 
        return F(x) * (x >= self._kL / self._kH) + F_low(x) * (x < self._kL / self._kH)

    def __corrTrans(self, k):
        """
        Calculate the transversal correlation function for wave number k

        Parameters
        ----------
        k: array-like
            wave number

        Returns
        -------
        spatial_corr: array-like
            n-dim array with values of the correlation function
        """
        spatial_corr = pi / 4. * self._B * self._B * self.Fq(k / self._kH)
        return spatial_corr
        #return pi / 4. * self._B * self._B * Fq(k / self._kH, self._kL, self._kH, self._q)

    def Bgaus(self, z):
        """
        Calculate the magnetic field for a gaussian turbulence field
        along the line of sight direction, denoted by z.

        Arguments
        ---------
        z: array-like
           m-dim array with distance traversed in magnetic field in kpc

        Return
        -------
        B: :py:class:`numpy.ndarray`
            m-dim array with values of transversal field
        """
        #t0 = time.time()
        zz, kk = meshgrid(z, self.__kn)
        _, dd = meshgrid(z, self.__dk)
        _, uu = meshgrid(z, self.__Un)
        _, vv = meshgrid(z, self.__Vn)

        #t1 = time.time()

        corr_trans = self.__corrTrans(kk)
        #t2 = time.time()
        B = sum(sqrt(corr_trans / pi * dd * 2. * log(1. / uu))
                * cos(kk * zz + 2. * pi * vv), axis=0)
        #t3 = time.time()
        #print ("t3-t2", t3-t2, "t2-t1", t2-t1, "t1-t0", t1-t0)
        return B

    def new_Bn(self, z, Bscale=None, nsim=1):
        """
        Calculate two components of a turbulent magnetic field and 
        the angle between the the two.

        Parameters
        ----------
        z: array-like
           m-dim array with distance traversed in magnetic field

        Bscale: array-like or float or None
           if not None, float or m-dim array with scaling factor for magnetic field 
           along distance travelled 

        Returns
        -------
        B, Psin: tuple with :py:class:`~numpy.ndarray`
            Two squeezed (nsim,m)-dim array with absolute value of transversal field,
            as well as angles between total transversal magnetic field and the :math:`y` direction.
        """
        seed(self.seed)

        B, Psin = [], []

        # if seed is integer
        # create a list of random integers which are then used
        # to create nsim Bfield realizations
        if isinstance(self.seed, int):
            high = 2**int(math.log2(sys.maxsize) / 2) - 1
            seeds = randint(high, size=nsim)
            seed_old = copy.deepcopy(self.seed)
        else:
            seeds = None

        for i in range(nsim):
            # calculate first transverse component, 
            # this is already computed with central B-field strength
            Bt = self.Bgaus(z)

            if seeds is not None:
                self.seed = seeds[i]

            self.new_random_numbers()                # new random numbers
            # calculate second transverse component, 
            # this is already computed with central B-field strength
            Bu = self.Bgaus(z)        
            # calculate total transverse component 
            B.append(np.sqrt(Bt ** 2. + Bu ** 2.))
            # and angle to x2 (t) axis -- use atan2 to get the quadrants right
            Psin.append(np.arctan2(Bt, Bu))

        B = np.squeeze(B)
        Psin = np.squeeze(Psin)

        # restore old seed
        if seeds is not None:
            self.seed = copy.deepcopy(seed_old)

        if np.isscalar(Bscale) or type(Bscale) == np.ndarray:
            B *= Bscale

        return B, Psin

    def spatial_correlation(self, z, steps=10000):
        """
        Calculate the spatial correlation of the turbulent field

        Arguments
        ---------
        z: array-like
            distance traversed in magnetic field

        steps: int
            number of integration steps

        Returns
        -------
        corr: array-like
            array with spatial correlation
        """
        if isscalar(z):
            z = array([z])
        t = logspace(-9., 0., steps)
        tt, zz = meshgrid(t, z)
        kernel = self.Fq(tt) * cos(tt * zz * self._kH)
        # the self._kH factor comes from the substitution t = k / _kH
        corr = self._B * self._B / 4. * simps(kernel * tt, log(tt), axis=1) * self._kH
        return corr

    def rotation_measure(self, z, n_el, Bscale=None, nsim=1):
        """
        Calculate the rotation measure of a
        random Gaussian field.

        Parameters
        ----------
        z: array-like
           m-dim array with distance traversed in magnetic field, in kpc

        n_el: array-like
           m-dim array with electron density in cm^-3

        nsim: int
            number of B-field simulations

        Bscale: array-like or None
            if given, this array is multiplied with the B field for an additional scaling depending on r,
            e.g., (n_el(r) / n_el(0))^eta

        Returns
        -------
        rm: :py:class:`~numpy.ndarray` or float
            Rotation measure for each simulated B field. Returned as array if nsim > 1
        """
        if Bscale is None:
            Bscale = np.ones_like(z)

        seed(self.seed)
        # if seed is integer
        # create a list of random integers which are then used
        # to create nsim Bfield realizations
        if isinstance(self.seed, int):
            seeds = randint(2**32 - 1, size=nsim)
            seed_old = copy.deepcopy(self.seed)
        else:
            seeds = None

        kernel = []
        for i in range(nsim):
            if seeds is not None:
                self.seed = seeds[i]
            # calculate the longitudinal component
            # this is sqrt(2) * one of the transversal components,
            # see Eq. A8 in https://arxiv.org/pdf/1406.5972.pdf
            # and the fact that the correlation factor of long. comp.
            # is twice the one of the trans. component and enters
            # the B field in sqrt
            self.new_random_numbers()
            B = np.sqrt(2.) * self.Bgaus(z)
            kernel.append(B * Bscale * n_el)

        # restore old seed
        if seeds is not None:
            self.seed = copy.deepcopy(seed_old)

        rm = 812. * simps(kernel, z, axis=1)
        return rm

