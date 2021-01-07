# --- Imports --------------------- #
from __future__ import absolute_import, division, print_function
import numpy as np
from astropy import units as u
from astropy import constants as c
from scipy.integrate import quad
# --------------------------------- #

# ========================================================== #
# === Electron densities for AGN jet medium ================ #
# ========================================================== #

m_e_GeV = (c.m_e * c.c**2.).to("GeV").value


class NelJet(object):
    """Class to set characteristics of electron density of AGN Jet"""
    def __init__(self, n0, r0, beta):
        """
        Initialize the class

        Parameters
        ----------
        n0: float
            electron density in cm**-3

        r0: float
            radius where electron density is equal to n0 in pc

        beta: float
            power-law index of distance dependence of electron density
        """
        self._n0 = n0
        self._r0 = r0
        self._beta = beta
        return

    @property
    def n0(self):
        return self._n0

    @property
    def r0(self):
        return self._r0

    @property
    def beta(self):
        return self._beta

    @n0.setter
    def n0(self, n0):
        if type(n0) == u.Quantity:
            self._n0 = n0.to('cm**-3').value
        else:
            self._n0 = n0
        return

    @r0.setter
    def r0(self, r0):
        if type(r0) == u.Quantity:
            self._r0 = r0 .to('pc').value
        else:
            self._r0 = r0
        return

    @beta.setter
    def beta(self, beta):
        self._beta = beta
        return

    def __call__(self,r):
        """
        Calculate the electron density as function from cluster center

        Parameters
        ----------
        r: array-like
            n-dim array with distance from cluster center in pc

        Returns
        -------
        nel: :py:class:`~numpy.ndarray`
            n-dim array with electron density in cm**-3
        """
        return self._n0 * np.power(r / self._r0, self._beta)


class NelJetHelicalTangled(object):
    """
    Class to get effective electron densities in jet, taking into account that
    the jet is not a cold plasma. i.e. the electron distribution is non-thermal.
    """
    def __init__(self, n0, r0, alpha, beta):
        """
        Initialize the class

        Parameters
        ----------
        n0: float
            electron density in cm^-3

        r0: float
            radius where electron density is equal to n0 in pc

        alpha: float
            power-law index of electron energy distribution function

        beta: float
            power-law index of distance dependence of electron density
        """
        self._n0 = n0
        self._r0 = r0
        self._alpha = alpha
        self._beta = beta
        return

    @property
    def n0(self):
        return self._n0

    @property
    def r0(self):
        return self._r0

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    @n0.setter
    def n0(self, n0):
        if type(n0) == u.Quantity:
            self._n0 = n0.to('cm**-3').value
        else:
            self._n0 = n0
        return

    @r0.setter
    def r0(self, r0):
        if type(r0) == u.Quantity:
            self._r0 = r0 .to('pc').value
        else:
            self._r0 = r0
        return

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha
        return

    @beta.setter
    def beta(self, beta):
        self._beta = beta
        return

    def get_photon_mass_ne(self, alpha, ne):
        """
        Function to calculate effective photon mass from electron distribution,
        here derived from the electron density and power-law index.

        Parameters
        ----------
        alpha: float
            power-law index

        ne: array-like
            electron density in cm^-3

        Returns
        -------
        m_T_2: float
            effective photon mass squared
        """
        def integrand(E, alpha, m_e):
            return E**(-alpha)/np.sqrt(E**2 - m_e**2)

        # do integration in GeV so the numbers are nicer for scipy
        I = quad(integrand, m_e_GeV, np.inf, args=(alpha, m_e_GeV))

        I_eV = I[0] / 1.e9 ** alpha  # convert back to eV

        A_V = (alpha - 1.) * ne * 1.9e-12 / ((m_e_GeV * 0.511e6)**(alpha - 1.))  # now in eV

        m_T_2 = (c.alpha.value/np.pi**2) * A_V * I_eV

        return m_T_2

    def __call__(self, r):
        """
        Calculate the effective electron density as function from cluster center.
        Done by finding actual electron density and actual photon effective masses,
        then making effective mass = w_pl(n_eff) to get n_eff.

        Parameters
        ----------
        r: array-like
            n-dim array with distance from cluster center in pc

        Returns
        -------
        nel: :py:class:`~numpy.ndarray`
            n-dim array with electron density in cm**-3
        """
        actual_nes = self._n0 * np.power(r / self._r0, self._beta)
        eff_photon_masses2 = self.get_photon_mass_ne(self._alpha, actual_nes)  # eV^2
        eff_nes = eff_photon_masses2/1.3689e-21  # cm^-3: w_pl^2 = ne * e^2/(e_o m) = ne (cm^-3) * 1.3689e-21 (cm^3 eV^2)
        return eff_nes
