# --- Imports --------------------- #
import numpy as np
from astropy import units as u
# --------------------------------- #

# ========================================================== #
# === Electron densities for AGN jet medium ================ #
# ========================================================== #

class NelJet(object):
    """Class to set characteristics of electron density of AGN Jet"""
    def __init__(self,n0,r0,beta):
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
    def r0(self,r0):
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
	r: `~numpy.ndarray`
	    n-dim array with distance from cluster center in pc

	Returns
	-------
	n-dim `~numpy.ndarray` with electron density in cm**-3
	"""
	return self._n0 * np.power(r / self._r0, self._beta)
