# --- Imports ------------- #
import numpy as np
import logging
from astropy import units as u
from astropy.table import Table
from os import path
from multiprocessing import Pool,Process
import os
import copy
# ------------------------- #

"""
Class for the calculation of the transfer 
function for photon-ALPs conversions and an arbitrary 
magnetic field
"""
__version__=0.01
__author__="M. Meyer // mameyer@stanford.edu"

Bcrit 		= 4.414e13 # critical magnetic field in G

# --- Momentum differences in kpc^-1 --- #
Delta_ag = lambda g,B: 1.52e-2*g*B
Delta_a = lambda m,E: -7.8e-2*m**2./E
Delta_pl = lambda n,E: -1.1e-4*n/E
Delta_CMB = lambda E: 0.8e-7*E
			
Delta_QED = lambda B,E: 4.1e-9*E*B**2.
# with correction factors of Perna et al. 2012
#Delta_QED = lambda B,E: 4.1e-9*E*B**2. * (1. + 1.2e-6 * B / Bcrit) / \
#				(1. + 1.33e-6*B / Bcrit + 0.59e-6 * (B / Bcrit)**2.)
chiCMB = 0.511e-42
# --------------------------------------- #
#Plasma freq in 10^-9 eV
#n is electron density in cm^-3
w_pl_e9 = lambda n: 0.00117*np.sqrt(n/1e-3)
# --------------------------------------- #
	


class GammaALPTransfer(object):
    """
    Base clasee to calculate the transfer Function
    of photon-oscillations in arbitrary magnetic fields.
    Does not account for redshift evolution.

    Units used are:
    magnetic field B: micro Gauss
    length scales: kpc
    electron densities: cm^-3
    ALP mass: neV
    photon-ALP coupling: 10^-11 GeV^-1

    Unit conversion is done through astropy.units module
    """

    def __init__(self, EGeV, B, psi, nel, dL, alp, Gamma = None, chi = None, Delta = None):
	"""
	Initialize the transfer base class

	Parameters
	----------
	EGeV : `~numpy.ndarray`
	    n-dim numpy array with gamma-ray energies in GeV

	B : `~numpy.ndarray`
	    m-dim numpy array with magnetic field values along line of sight in micro Gauss

	psi : `~numpy.ndarray`
	    m-dim numpy array with angles between transversal magnetic field in photon polarization
	    direction (along y-axis) along the line of sight.
	    Or (k,m)-dim numpy array with angles for k B-field realizations

	nel: `~numpy.ndarray`
	    m-dim numpy array with electron densities in cm^-3 along the line of sight.

	dL: `~numpy.ndarray`
	    m-dim numpy array with distance step length traveled along line of sight in kpc.
	    In each step length dL, magnetic field is assumed to be constant. 
	    
	alp: `~gammaALPs.ALP`
	    `~gammaALPs.ALP` object with ALP parameters
	

	{options}

	Gamma: `~numpy.ndarray` or None
	    (n x m)-dim numpy array with photon absorption rate at energy E and distance L. 
	    In kpc^-1.
	    If None, no absorption is included.
	    Default is None. 

	Delta: `~numpy.ndarray` or None
	    (n x m)-dim numpy array with additional momentum difference term for 0,0 and 1,1 
	    components of mixing matrix at energy E and distance L. 
	    In kpc^-1.
	    If None, no additional term is included.
	    Default is None. 

	chi: `~numpy.ndarray` or None
	    (n x m)-dim numpy array with photon dispersion rate at energy E and distance L. 
	    If None, no dispersion is included.
	    Default is None. 

	m: float
	    ALP mass in neV. Default in 1.

	g: float
	    photon-ALP couplint in 10^-11 GeV^-1. Default in 1.
	"""
	self._EGeV = EGeV
	self._dL = dL
	self._Gamma = Gamma
	self._Delta = Delta
	self._chi = chi
	self._alp = alp

	if len(psi.shape) > 1:
	    self._psin = psi
	    self._psi = psi[0]
	    logging.debug('Psi shape: {0}:'.format(self._psin.shape))
	    self._nsim = psi.shape[0]
	else:
	    self._psi = psi
	    self._psin = np.array([psi])
	    self._nsim = 1

	if len(B.shape) > 1:
	    self._Bn = B
	    logging.debug('B shape: {0}:'.format(self._Bn.shape))
	    self._B = B[0]
	else:
	    self._B = B

	if len(nel.shape) > 1:
	    self._neln = nel
	    logging.debug('nel shape: {0}:'.format(self._neln.shape))
	    self._nel = nel[0]
	else:
	    self._nel = nel

	# init transfer matrices
	self._T1		= np.zeros(self._EGeV.shape + self._B.shape + (3,3),np.complex)	
	self._T2		= np.zeros(self._EGeV.shape + self._B.shape + (3,3),np.complex)
	self._T3		= np.zeros(self._EGeV.shape + self._B.shape + (3,3),np.complex)

	# init meshgrid arrays
	self._ee, self._bb = np.meshgrid(self._EGeV, self._B, indexing = 'ij')
	self._ll = np.meshgrid(self._EGeV, self._dL, indexing = 'ij')[1]
	self._pp = np.meshgrid(self._EGeV, self._psi, indexing = 'ij')[1]
	self._cpp = np.cos(self._pp)
	self._spp = np.sin(self._pp)
	self._nn = np.meshgrid(self._EGeV, self._nel, indexing = 'ij')[1]
	return

    # --- define properties ---- #
    @property 
    def EGeV(self):
	return self._EGeV

    @property
    def B(self):
	return self._B

    @property
    def psi(self):
	return self._psi

    @property
    def nel(self):
	return self._nel

    @property
    def dL(self):
	return self._dL

    @property
    def Gamma(self):
	return self._Gamma

    @property
    def chi(self):
	return self._chi

    @property
    def Delta(self):
	return self._Delta

    @property
    def alp(self):
	return self._alp

    @property
    def psin(self):
	return self._psin

    @property
    def Bn(self):
	return self._Bn

    @property
    def neln(self):
	return self._neln

    @property
    def nsim(self):
	return self._nsim

    # --- define setters ---- #
    @EGeV.setter
    def EGeV(self, EGeV):
	if type(EGeV) == u.Quantity:
	    self._EGeV = EGeV.to('GeV').value
	else:
	    self._EGeV = EGeV
	self.__init_transfer_matrices()
	self.__init_meshgrids()
	return

    @B.setter
    def B(self, B):
	if type(B) == u.Quantity:
	    self._B = B.to('10**-6G').value
	else:
	    self._B = B
	self.__init_transfer_matrices()
	self.__init_meshgrids()
	return 

    @Bn.setter
    def Bn(self, Bn):
	self._Bn = Bn
	self._B = Bn[0]
	self.__init_transfer_matrices()
	self.__init_meshgrids()
	return

    @neln.setter
    def neln(self, neln):
	self._neln = neln
	self._nel = neln[0]
	return

    @psi.setter
    def psi(self, psi):
	self._psi = psi
	self.__init_transfer_matrices()
	self.__init_meshgrids()
	return

    @psin.setter
    def psin(self, psin):
	self._psin = psin
	self._nsim = psin.shape[0]
	self._psi = psin[0]
	self.__init_transfer_matrices()
	self.__init_meshgrids()
	return

    @nel.setter
    def nel(self, nel):
	if type(nel) == u.Quantity:
	    self._nel = nel.to('cm**-3').value
	else:
	    self._nel = nel
	self.__init_transfer_matrices()
	self.__init_meshgrids()
	return 

    @dL.setter
    def dL(self, dL):
	if type(dL) == u.Quantity:
	    self._dL = dL.to('kpc').value
	else:
	    self._dL = dL
	self.__init_transfer_matrices()
	self.__init_meshgrids()
	return

    @Gamma.setter
    def Gamma(self, Gamma):
	if type(Gamma) == u.Quantity:
	    self._Gamma = Gamma.to('kpc**-1')
	else:
	    self._Gamma = Gamma
	return

    @Delta.setter
    def Delta(self, Delta):
	if type(Delta) == u.Quantity:
	    self._Delta = Delta.to('kpc**-1')
	else:
	    self._Delta = Delta
	return

    @chi.setter
    def chi(self, chi):
	self._chi = chi
	return

    def __init_transfer_matrices(self):
	self._T1		= np.zeros(self._EGeV.shape + self._B.shape + (3,3),np.complex)	
	self._T2		= np.zeros(self._EGeV.shape + self._B.shape + (3,3),np.complex)
	self._T3		= np.zeros(self._EGeV.shape + self._B.shape + (3,3),np.complex)
	return

    def __init_meshgrids(self):
	self._ee, self._bb = np.meshgrid(self._EGeV, self._B, indexing = 'ij')
	self._ll = np.meshgrid(self._EGeV, self._dL, indexing = 'ij')[1]
	self._pp = np.meshgrid(self._EGeV, self._psi, indexing = 'ij')[1]
	self._cpp = np.cos(self._pp)
	self._spp = np.sin(self._pp)
	self._nn = np.meshgrid(self._EGeV, self._nel, indexing = 'ij')[1]
	return

    def __setDeltas(self):
	"""Set Deltas and eigenvalues of mixing matrix for each domain"""

	self._Dperp	= Delta_pl(self._nn,self._ee) + 0.j + 2.*Delta_QED(self._bb,self._ee)
	self._Dpar	= Delta_pl(self._nn,self._ee) + 0.j + 3.5*Delta_QED(self._bb,self._ee)

	self._Dpl	= Delta_pl(self._nn,self._ee)
	self._DQED	= Delta_QED(self._bb,self._ee)
	self._Dag	= Delta_ag(self.alp.g,self._bb)
	self._Da	= Delta_a(self.alp.m,self._ee)

	# add additional terms if absorption rate or dispersion are defined
	# absorption
	if type(self._Gamma) == np.ndarray:
	    #self._Dperp += 0.5j * self._Gamma
	    #self._Dpar += 0.5j * self._Gamma
	    self._Dperp -= 0.5j * self._Gamma
	    self._Dpar -= 0.5j * self._Gamma

	if type(self._chi) == np.ndarray:
	    self._Dperp += Delta_CMB(self._ee) / chiCMB * self._chi
	    self._Dpar += Delta_CMB(self._ee) / chiCMB * self._chi

	if type(self._Delta) == np.ndarray:
	    self._Dperp += self._Delta
	    self._Dpar += self._Delta
	 #no CMB: comment out next three lines
	else:
	    # add CMB term
	   self._Dperp += Delta_CMB(self._ee)
	   self._Dpar += Delta_CMB(self._ee)

	#self._Dosc	= np.sqrt((self._Dpar - self._Da)**2.  + 4.*self._Dag**2.)
	self._Dosc	= ((self._Dpar - self._Da)**2.  + 4.*self._Dag**2.)
	self._Dosc	= np.sqrt(self._Dosc)
	
	#self._saca 	= np.sin(self._alph) * np.cos(self._alph)
	self._saca	= self._Dag / self._Dosc # = 0.5 * sin(2. * alpha)

	# cosine^2 of mixing angle
	#self._caca 	= np.cos(self._alph) * np.cos(self._alph)
	self._caca	= 0.5 * (1. +  (self._Dpar - self._Da) / self._Dosc)

	# sin^2 of mixing angle
	#self._sasa 	= np.sin(self._alph) * np.sin(self._alph)
	self._sasa	= 0.5 * (1. -  (self._Dpar - self._Da) / self._Dosc)

	self._EW1 = self._Dperp
	self._EW2 = 0.5 * (self._Dpar + self._Da - self._Dosc)
	self._EW3 = 0.5 * (self._Dpar + self._Da + self._Dosc)

	return

    def __setT1n(self):
	"""Set T1 in all domains and at all energies"""
	self._T1[...,0,0] = self._cpp * self._cpp
	self._T1[...,0,1] = -1. * self._cpp * self._spp
	self._T1[...,1,0] = self._T1[...,0,1]
	self._T1[...,1,1] = self._spp * self._spp
	return

    def __setT2n(self):
	"""Set T2 in all domains and at all energies"""
	self._T2[...,0,0] = self._spp * self._spp * self._sasa
	self._T2[...,0,1] = self._spp * self._cpp * self._sasa
	self._T2[...,0,2] = -1. * self._spp * self._saca
                   
	self._T2[...,1,0] = self._T2[...,0,1]
	self._T2[...,1,1] = self._cpp * self._cpp * self._sasa
	self._T2[...,1,2] = -1. * self._cpp * self._saca
        
	self._T2[...,2,0] = self._T2[...,0,2]
	self._T2[...,2,1] = self._T2[...,1,2]
	self._T2[...,2,2] = self._caca

	return

    def __setT3n(self):
	"""Set T3 in all domains and at all energies"""

	self._T3[...,0,0] = self._spp * self._spp * self._caca
	self._T3[...,0,1] = self._spp * self._cpp * self._caca
	self._T3[...,0,2] = self._spp * self._saca
                    
	self._T3[...,1,0] = self._T3[...,0,1]
	self._T3[...,1,1] = self._cpp * self._cpp * self._caca
	self._T3[...,1,2] = self._cpp * self._saca
                                        
	self._T3[...,2,0] = self._T3[...,0,2]
	self._T3[...,2,1] = self._T3[...,1,2]
	self._T3[...,2,2] = self._sasa

	return

    def __setTn(self):
	"""Set total Transfer Matrix in all domains and at all energies"""
	ew1ll = (self._EW1 * self._ll)[..., np.newaxis, np.newaxis]
	ew2ll = (self._EW2 * self._ll)[..., np.newaxis, np.newaxis]
	ew3ll = (self._EW3 * self._ll)[..., np.newaxis, np.newaxis]

	self._Tn = np.exp(-1.j * ew1ll) * self._T1 + \
		    np.exp(-1.j * ew2ll) * self._T2 + \
		    np.exp(-1.j * ew3ll) * self._T3
	self._Tn = np.round(self._Tn, 8)
	return
	
    def writeEnviron(self, name, filepath = './'):
	"""
	Save the current magnetic field, psi angles, and electron density to 
	numpy files

	Paramters
	---------
	name: str 
	    name of job

	kwargs
	------
	filepath: str
	    full path to output file


	"""
	np.save(path.join(filepath,name) + '_dL.npy', self.dL)
	np.save(path.join(filepath,name) + '_EGeV.npy', self.EGeV)

	if self.nsim > 1:
	    np.save(path.join(filepath,name) + '_psi.npy', self._psin)
	    try:
		np.save(path.join(filepath,name) + '_B.npy', self._Bn)
	    except AttributeError:
	    	logging.debug("B field assumed constant, only angle Psi is changed")
		np.save(path.join(filepath,name) + '_B.npy', self.B)
		pass
	else:
	    np.save(path.join(filepath,name) + '_B.npy', self.B)
	    np.save(path.join(filepath,name) + '_psi.npy', self.psin)

	try:
	    np.save(path.join(filepath,name) + '_nel.npy', self.neln)
	except AttributeError:
	    np.save(path.join(filepath,name) + '_nel.npy', self.nel)
	     
	if type(self._Gamma) == np.ndarray:
	    np.save(path.join(filepath,name) + '_Gamma.npy', self._Gamma)
	if type(self._chi) == np.ndarray:
	    np.save(path.join(filepath,name) + '_chi.npy', self._chi)
	if type(self._Delta) == np.ndarray:
	    np.save(path.join(filepath,name) + '_Delta.npy', self._Delta)
	return path.join(filepath,name) + '*.npy'

    @staticmethod
    def readEnviron(name, alp, filepath = './'):
	"""
	Load a current magnetic field, psi angles, and electron density, 
	absorption and dispersion rate from a previous cofiguration

	Paramters
	---------
	name: str 
	    name of job
	alp: `~gammaALPs.transer.ALP`
	    `~gammaALPs.transer.ALP` object with ALP parameters

	kwargs
	------
	filepath: str
	    full path to output file

	m: float
	    ALP mass in neV. Default in 1.

	g: float
	    photon-ALP couplint in 10^-11 GeV^-1. Default in 1.

	"""
	nel = np.load(path.join(filepath,name) + '_nel.npy')
	dL = np.load(path.join(filepath,name) + '_dL.npy')
	B = np.load(path.join(filepath,name) + '_B.npy')
	psi = np.load(path.join(filepath,name) + '_psi.npy')
	EGeV = np.load(path.join(filepath,name) + '_EGeV.npy')

	if path.isfile(path.join(filepath,name) + '_Gamma.npy'):
	    Gamma = np.load(path.join(filepath,name) + '_Gamma.npy')
	else:
	    Gamma = None
	if path.isfile(path.join(filepath,name) + '_chi.npy'):
	    chi = np.load(path.join(filepath,name) + '_chi.npy')
	else:
	    chi = None
	if path.isfile(path.join(filepath,name) + '_Delta.npy'):
	    Delta = np.load(path.join(filepath,name) + '_Delta.npy')
	else:
	    Delta = None

	return GammaALPTransfer(EGeV, B, psi, nel, dL, alp, Gamma = Gamma,
				chi = chi, Delta = Delta)

    def __setTransferMatrices(self, nsim = -1):
	"""Set the transfer matrices"""
	if nsim >= 0 and self.nsim > 1:
	    self.psi = self.psin[nsim]

	    try:
		self.B = self._Bn[nsim]
	    except AttributeError:
	    	logging.debug("B field assumed constant, only angle Psi is changed")
		pass

	    try:
		self.nel = self._neln[nsim]
	    except AttributeError:
		pass

	self.__setDeltas()
	self.__setT1n()
	self.__setT2n()
	self.__setT3n()
	self.__setTn()
	return

    def fillTransfer(self):
	"""
	Calculate the transfer matrix for every domain for all 
	requested magnetic field realizations

	Returns
	-------
	list with all transfer functions for all B-field realizations
	"""
	if self.nsim == 1:
	    self.__setTransferMatrices()
	    return [self._Tn]
	else:
	    Tn = []
	    for nsim in range(self.nsim):
		self.__setTransferMatrices(nsim)
		Tn.append(self._Tn)
	    return Tn

    def calcTransferMulti(self, nprocess = 1):
	"""
	Calculate Transfer matrix by performing dot product of all transfer matrices 
	along axis of distance. If multiple realizations for the B-field 
	are to be calculated, parallel processing is used.

	kwargs
	------
	nprocess: int
	    distibute matrix multiplication to n (if n > 1) processes using python's multiprocessing. 

	Returns
	-------
	List with n x 3 x 3 dim `~numpy.ndarray` with transfer matrix for all energies. 
	Length of list is equal to number of requested B-field realizations. 
	"""
	Tn = self.fillTransfer()
	if len(Tn) == 1:
	    return [dotProd(Tn[0])]
	os.system("taskset -p 0xff %d" % os.getpid())
	pool = Pool(processes=nprocess)
	T = pool.map(dotProd, Tn)
	pool.close()
	pool.join()
	return T

    def calcTransfer(self, nsim = -1, nprocess = 1):
	"""
	Calculate Transfer matrix by performing dot product of all transfer matrices 
	along axis of distance.

	kwargs
	------
	nsim: int
	    number of B-field realization to be used. Only works if class was initialized 
	    with multiple B-field realizations.

	nprocess: int
	    distibute matrix multiplication to n (if n > 1) processes using python's multiprocessing. 
	    Testing this feature so far did not show any gain in speed, however!

	Returns
	-------
	n x 3 x 3 dim `~numpy.ndarray` with transfer matrix for all energies
	"""
	if nprocess < 1:
	    raise ValueError("nprocess must be >= 1, not {0:n}".format(nprocess))
	
	self.__setTransferMatrices(nsim)

	if nprocess == 1: # only one process
	    return dotProd(self._Tn)
	else: # split to mutliple processes
	    os.system("taskset -p 0xff %d" % os.getpid())
	    pool = Pool(processes=nprocess)
	    # split up the matrix
	    d = np.linspace(self._Tn.shape[1] / nprocess, self._Tn.shape[1] + 1, nprocess, dtype = np.int)
	    d = np.concatenate(([0],d))
	    # set the index boundaries
	    itr = np.array([d[:-1], d[1:]]).T # start and stop
	    # make a list of split up matrices
	    Tn = [self._Tn[:,i[0]:i[1],...] for i in itr]
	    # perform the multiprocessing
	    T = pool.map(dotProd, Tn)
	    pool.close()
	    pool.join()
	    return dotProd(T)

    def show_conv_prob_vs_r(self, pin, pout):
	"""
	Compute the conversion probability as a function of distance.
	Works only after you have computed the transfer matrices in each domain.

	Parameters
	----------
	pin: `~numpy.ndarray`
	    3 x 3 matrix with initial polarization
	pout: `~numpy.ndarray`
	    3 x 3 matrix with final polarization
	
	Returns
	-------
	Conversion probability for each energy and distance 
	"""
	p_r = []
	for i in range(self._Tn.shape[1]):
	    if not i:
		p_r.append(calcConvProb(pin,pout,self._Tn[:,i]))
	    else:
		p_r.append(calcConvProb(pin,pout,dotProd(self._Tn[:,:i])))
	return np.array(p_r)
	

# ---------------------------------------------------------------------------- #
# --- Global Functions ------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
def dotProd(T):
    """Calculate dot product over last two axis of a multi dimensional matrix"""
    #return np.array([reduce(np.dot, Tn) for Tn in T])
    # reverse along domain axis, see comment in next function
    return np.array([reduce(np.dot, Tn) for Tn in T[:,::-1,...]])

def calcConvProb(pin, pout, T):
    """
    Calculate the conversion probability from an initial to a final state

    Parameters
    ----------
    pin: `~numpy.ndarray`
	3 x 3 matrix with initial polarization
    pout: `~numpy.ndarray`
	3 x 3 matrix with final polarization
    T: `~numpy.ndarray`
	n x 3 x 3 complex transfer matrix for n energies

    Returns 
    -------
    n-dim `~numpy.ndarray` with conversion probabilities for each energy
    """
    return np.squeeze(np.real(np.trace(
	    (np.matmul(pout,
		# first has to be transpose and the second conjugate. 
		# to see this for two domains, with 1 being the closest domain to the beam
		# (farthest away from the observer).
		# T = T1 T2
		# what we don't want:
		# rfinal = T r T^dagger = T1T2 r T2^dagger T1^dagger
		# thus, you have to turn around the multiplication in dotProd along the domain axis, 
		# so that T = T2 T1, and thus
		# rfinal = T r T^dagger = T2T1 r T1^dagger T2^dagger
		np.matmul(T,
		    np.matmul(pin,np.transpose(T.conjugate(), axes = (0,2,1)))
		    )
		)
	    ),
	    axis1=1, axis2=2)))

def calcLinPol(pin, T):
    """
    Calculate the the linear polarization degree of the final transfer matrix

    Parameters
    ----------
    pin: `~numpy.ndarray`
	3 x 3 matrix with initial polarization
    T: `~numpy.ndarray`
	n x 3 x 3 complex transfer matrix for n energies

    Returns 
    -------
    n-dim `~numpy.ndarray` with conversion probabilities for each energy

    Notes
    -----
    See Eq. (44) of Bassan et al. (2010): https://arxiv.org/pdf/1001.5267.pdf
    """
    # rfinal = T^T r (T^T)^dagger = T2T1 r T1^dagger T2^dagger
    rfinal = np.matmul(np.transpose(T, axes = (0,2,1)),
		    np.matmul(pin,T.conjugate())
		    )
    lin_pol = np.sqrt((rfinal[:,0,0] - rfinal[:,1,1])**2. + (rfinal[:,0,1] + rfinal[:,1,0])**2.)
    lin_pol /= (rfinal[:,0,0] + rfinal[:,1,1])
    circ_pol = np.imag(rfinal[:,0,1] - rfinal[:,1,0]) / (rfinal[:,0,0] + rfinal[:,1,1])

# need to check this:
    if not np.all(np.imag(lin_pol) == 0.):
	logging.warning("Not all values of linear polarization are real values!")
    if not np.all(np.imag(circ_pol) == 0.):
	logging.warning("Not all values of circular polarization are real values!")
    return np.real(lin_pol), np.real(circ_pol) # number should be real already, this discards the zero imaginary part
