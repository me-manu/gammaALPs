# --- Imports --------------------- #
import numpy as np
from numpy.random import rand
from numpy import log,log10,pi,meshgrid,cos,sum,sqrt,linspace,array,isscalar,logspace
from math import ceil
from scipy.integrate import simps
from astropy import units as u
# --------------------------------- #

# ========================================================== #
# === B field of AGN jet assuming a power-law toroidal field #
# ========================================================== #
class Bjet(object):
    """Class to calculate magnetic field in AGN Jet assuming a toroidal field"""
    def __init__(self,B0,r0,alpha):
	"""
	Initialize the class

	Parameters
	----------
	B0: float
	    Bfield strength in G
	r0: float
	    radius where B field is equal to B0 in pc
	alpha: float
	    power-law index of distance dependence of B field
	"""
	self._B0 = B0
	self._r0 = r0
	self._alpha = alpha 
	return 

    @property
    def B0(self):
	return self._B0

    @property
    def r0(self):
	return self._r0

    @property
    def alpha(self):
	return self._alpha

    @B0.setter
    def B0(self, B0):
	if type(B0) == u.Quantity:
	    self._B0 = B0.to('G').value
	else:
	    self._B0 = B0
	return 

    @r0.setter
    def r0(self,r0):
	if type(r0) == u.Quantity:
	    self._r0 = r0 .to('pc').value
	else:
	    self._r0 = r0
	return 

    @alpha.setter
    def alpha(self, alpha):
	self._alpha = alpha 
	return

    def new_Bn(self,z, psi = np.pi / 4.):
	"""
	Calculate the magnetic field as function of distance

	Parameters
	----------
	z: `~numpy.ndarray`
	    n-dim array with distance from r0 in pc

	{options}

	psi: float
	    angle between transversal magnetic field and x2 direction. Default: pi/4

	Returns
	-------
	tuple with n-dim `~numpy.ndarray` with B field in G and psi angles
	"""
	B = self._B0 * np.power(z / self._r0, self._alpha )
	psi = np.ones(B.shape[0]) * psi
	return B, psi
