# --- Imports --------------------- #
import numpy as np
from numpy.random import rand
from numpy import log,log10,pi,meshgrid,cos,sum,sqrt,linspace,array,isscalar,logspace
from math import ceil
from scipy.integrate import simps
from astropy import units as u
from gammaALPs.utils import trafo
from astropy.cosmology import FlatLambdaCDM
# --------------------------------- #

# ========================================================== #
# === cell-like turbulent magnetic field ================= #
# ========================================================== #
class Bcell(object):
    """
    Class to calculate a turbulent magnetic field with a cell like structure
    """
    def __init__(self,B,Lcoh):
	"""
	Initialize B field with cell like structre. 

	Parameters
	----------
	B:  		float
			rms B field strength in micro Gauss
	Lcoh:		float
			coherence length in kpc
	"""
	# --- Set the defaults 
	self._B = B
	self._Lcoh = Lcoh
	return

    @property
    def B(self):
	return self._B

    @property
    def Lcoh(self):
	return self._Lcoh

    @B.setter
    def B(self,B):
	if type(B) == u.Quantity:
	    self._B = B.to('10**-6G').value
	else:
	    self._B = B
	return

    @Lcoh.setter
    def Lcoh(self,Lcoh):
	if type(B) == u.Quantity:
	    self._Lcoh = Lcoh.to('kpc').value
	else:
	    self._Lcoh = Lcoh
	return 

    def new_random_numbers(self, Nd,nsim = 1):
	"""Generate new random numbers for angle of magnetic field for Nd domains"""
	# angle between photon propagation on B-field in i-th domain 
	return 2. * np.pi * rand(nsim,int(Nd))

    def new_Bn(self,Nd, Bscale = None, nsim = 1):
	"""
	Calculate two components of a turbulent magnetic field and 
	the angle between the the two.

	Parameters
	----------
	Nd: int 
	    number of domains

	{options}

	Bscale: `~numpy.ndarray` or float or None
	   if not None, float or Nd-dim array with scaling factor for magnetic field 
	   along distance travelled 

	nsim: int
	    number of random realizations of the magnetic field. Default: 1

	Returns
	-------
	tuple with two squeezed (nsim,Nd)-dim  `~numpy.ndarray` array with absolute value of transversal field,
	angles between total transversal magnetic field and x2 direction. 
	

	Note
	----
	If z is not a multiple integer of Lcoh, last not fully crossed domain will be discarded
	"""
	# determine number of crossed domains, no expansion assumed

	Psin = self.new_random_numbers(Nd,nsim = nsim)
	if nsim == 1:
	    Psin = np.squeeze(Psin)
	    B = np.ones(Nd) * self._B
	else:
	    B = np.ones(Psin.shape) * self._B
	if np.isscalar(Bscale) or type(Bscale) == np.ndarray:
	    B *= Bscale

	return B,Psin

    def new_Bcosmo(self,z, Bscale = None, cosmo = FlatLambdaCDM(H0 = 70., Om0 = 0.3),nsim = 1):
	"""
	Calculate two components of a cosmological turbulent magnetic field and 
	the angle between the the two. 

	B field will be scaled with (1. + z)^2 and coherence length with (1. + z)

	Parameters
	----------
	z: float
	    redshift to source

	{options}

	Bscale: `~numpy.ndarray` or float or None
	   if not None, float or Nd-dim array with scaling factor for magnetic field 
	   along distance travelled 
	cosmo: `~astropy.cosmology.core.FlatLambdaCDM`
	    chosen cosmology, default is H0 = 70, Om0 = 0.3

	nsim: int
	    number of random realizations of the magnetic field. Default: 1

	Returns
	-------
	tuple with two Nd-dim  `~numpy.ndarray` array with absolute value of transversal field,
	angles between total transversal magnetic field and x2 direction
	"""
	# determine number of crossed domains, no expansion assumed

	dL, zstep = trafo.cosmo_cohlength(z,self._Lcoh * u.kpc, cosmo = cosmo)

	Psin = np.squeeze(self.new_random_numbers(dL.shape[0],nsim = nsim))
	zmean = zstep[:-1]
	B = self._B * (1. + zmean)  ** 2.
	if np.isscalar(Bscale) or type(Bscale) == np.ndarray:
	    B *= Bscale

	return B,Psin,dL,zstep
