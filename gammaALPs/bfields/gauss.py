# --- Imports --------------------- #
import numpy as np
from numpy.random import rand
from numpy import log,log10,pi,meshgrid,cos,sum,sqrt,linspace,array,isscalar,logspace
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
    def __init__(self,B,kH,kL,q,**kwargs):
	"""
	Initialize gaussian turbulence B field spectrum. 
	Defaults assume values typical for a galaxy cluster.

	Parameters
	----------
	B:  		float
			rms B field strength, energy is B^2 / 4pi (default = 1 muG)
	kH:		float
			upper wave number cutoff, 
			should be at at least > 1. / osc. wavelength (default = 1 / (1 kpc))
	kL:		float
			lower wave number cutoff,
			should be of same size as the system (default = 1 / (100 kpc))
	q:  		float
			power-law turbulence spectrum (default: q = 11/3 is Kolmogorov type spectrum)
	
	kwargs
	------
	kMin:		float
			minimum wave number in 1. / kpc,
			defualt 1e-3 * kL (the k interval runs from kMin to kH)
	dkType:		string
			either linear, log, or random. Determine the spacing of the dk intervals 	
	dkSteps: 	int
			number of dkSteps.
			For log spacing, number of steps per decade / number of decades ~ 10
			should be chosen.
	"""
	self._B = B
	self._kH = kH
	self._kL = kL
	self._q = q

	# --- Set the defaults 
	kwargs.setdefault('kMin',-1)
	kwargs.setdefault('dkType','log')
	kwargs.setdefault('dkSteps',0)

	self.__dict__.update(kwargs)

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
    def kH(self,kH):
	if type(kH) == u.Quantity:
	    self._kH = kH.to('kpc**-1').value
	else:
	    self._kH = kH
	self.__init_k_array()
	return

    @kL.setter
    def kL(self,kL):
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
	if self.dkType == 'random':
	    self.__dk = rand(self.dkSteps)
	    self.__dk *= (self._kH - self.kMin) / sum(self.__dk)
	    self.__kn = np.array([self.kMin + sum(self.__dk[:n]) for n in range(self.__dk.shape[0])])

	self.__Un = rand(self.__kn.shape[0])
	self.__Vn = rand(self.__kn.shape[0])
	return

    def Fq(self,x):
	"""
	Calculate the F_q function for given x,kL, and _kH

	Arguments
	---------
	x:	n-dim array, Ratio between k and _kH

	Returns
	-------
	n-dim array with Fq values
	"""
	if self._q == 0.:
	    F		= lambda x: 3. * self._kH **2. / (self._kH ** 3. - self._kL ** 3.) * \
			    ( 0.5 * (1. - x*x) - x * x * log(x) )
	    F_low	= lambda x: 3. * (0.5 * (self._kH ** 2. - self._kL ** 2.) + \
			    (x * self._kH) * (x * self._kH) * log(self._kH / self._kL) ) \
			    / (self._kH ** 3. - self._kL ** 3.)
	elif self._q == -2.:
	    F		= lambda x: ( 0.5 * (1. - x*x) - log(x) ) / (self._kH  - self._kL )
	    F_low	= lambda x: ( log(self._kH / self._kL ) + (x * self._kH) * \
			    (x * self._kH) * 0.5 * (self._kL ** (-2) - self._kH ** (-2)) ) \
			    / (self._kH  - self._kL )
	elif self._q == -3.:
	    F		= lambda x:  1. / log(self._kH / self._kL) / self._kH / x  / 3. * (-x*x*x - 3. * x + 4.)
	    F_low	= lambda x: ( (self._kL ** (-2) - self._kH ** (-2)) + (x * self._kH) * \
			    (x * self._kH) / 3. * (self._kL ** (-3) - self._kH ** (-3)) ) / \
			    log(self._kH / self._kL )
	else:
	    F		= lambda x: self._kH ** (self._q + 2.) / (self._kH ** (self._q + 3.) - \
				    self._kL ** (self._q + 3.)) * \
				    (self._q + 3.) / (self._q * ( self._q + 2.)) * \
				    (self._q + x * x * ( 2. + self._q - 2. * (1. + self._q) * x ** self._q))
	    F_low	= lambda x: (self._q + 3.) * ( (self._kH ** (self._q + 2) - \
				    self._kL ** (self._q +2)) / (self._q + 2.) + \
				    (x * self._kH) * (x * self._kH) / self._q * \
				    (self._kH ** self._q - self._kL ** self._q) ) / \
				    (self._kH ** (self._q + 3.) - self._kL**(self._q + 3) ) 
	return F(x) * (x >= self._kL / self._kH) + F_low(x) * (x < self._kL / self._kH)

    def __corrTrans(self,k):
	"""
	Calculate the transversal correlation function for wave number k

	Arguments
	---------
	k:	n-dim array, wave number

	Returns
	-------
	n-dim array with values of the correlation function
	"""
	return pi / 4. * self._B * self._B * self.Fq(k / self._kH)

    def Bgaus(self, z):
	"""
	Calculate the magnetic field for a gaussian turbulence field
	along the line of sight direction, denoted by z.

	Arguments
	---------
	z: `~numpy.ndarray`
	   m-dim array with distance traversed in magnetic field in kpc

	Return
	-------
	m-dim `~numpy.ndarray` array with values of transversal field
	"""
	zz, kk = meshgrid(z,self.__kn)
	zz, dd = meshgrid(z,self.__dk)
	zz, uu = meshgrid(z,self.__Un)
	zz, vv = meshgrid(z,self.__Vn)

	B = sum(sqrt(self.__corrTrans(kk) / pi * dd * 2. * log(1. / uu)) \
		* cos(kk * zz + 2. * pi * vv), axis = 0)
	return B

    def new_Bn(self,z, Bscale = None, nsim = 1):
	"""
	Calculate two components of a turbulent magnetic field and 
	the angle between the the two.

	Parameters
	----------
	z: `~numpy.ndarray`
	   m-dim array with distance traversed in magnetic field

	{options}

	Bscale: `~numpy.ndarray` or float or None
	   if not None, float or m-dim array with scaling factor for magnetic field 
	   along distance travelled 

	Returns
	-------
	tuple with two m-dim  `~numpy.ndarray` array with absolute value of transversal field
	and angles between total transversal magnetic field and x2 direction
	"""
	B, Psin = [],[]
	for i in range(nsim):
	    # calculate first transverse component, 
	    # this is already computed with central B-field strength
	    Bt	= self.Bgaus(z)
	    self.new_random_numbers()		# new random numbers
	    # calculate second transverse component, 
	    #this is already computed with central B-field strength
	    Bu	= self.Bgaus(z)	
	    # calculate total transverse component 
	    B.append(np.sqrt(Bt ** 2. + Bu ** 2.))
	    # and angle to x2 (t) axis -- use atan2 to get the quadrants right
	    Psin.append(np.arctan2(Bt , Bu))

	B = np.squeeze(B)
	Psin = np.squeeze(Psin)


	if np.isscalar(Bscale) or type(Bscale) == np.ndarray:
	    B *= Bscale

	return B,Psin

    def spatialCorr(self, z, steps = 10000):
	"""
	Calculate the spatial coherence of the turbulent field

	Arguments
	---------
	z:	m-dim array, distance traversed in magnetic field

	kwargs
	------
	steps:	integer, number of integration steps

	Returns
	-------
	m-dim array with spatial coherences 
	"""
	if isscalar(z):
	    z = array([z])
	t	= logspace(-9.,0.,steps)
	tt,zz	= meshgrid(t,z)
	kernel	= self.Fq(tt) * cos(tt * zz * self._kH)
	# the self._kH factor comes from the substitution t = k / _kH
	return self._B * self._B / 4. * simps(kernel * tt,log(tt),axis = 1)  * self._kH	
