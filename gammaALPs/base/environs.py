import numpy as np
from gammaALPs.base import transfer as trans
from gammaALPs.bfields import cell, gauss, gmf, jet
from gammaALPs.nel import icm 
from gammaALPs.nel import jet as njet
from gammaALPs.utils import trafo
from astropy import units as u
from astropy import constants as c
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord
from ebltable.tau_from_model import OptDepth
from scipy.special import jv
from scipy.interpolate import UnivariateSpline as USpline
import logging

class MixIGMFCell(trans.GammaALPTransfer):
    def __init__(self, alp, source, **kwargs):
	"""
	Initialize mixing in the intergalactic magnetic field (IGMF).

	Parameters
	----------
	alp: `~gammaALPs.ALP`
	    `~gammaALPs.ALP` object with ALP parameters

	source: `~gammaALPs.Source`
	    `~gammaALPs.Source` object with source parameters

	kwargs
	------
	EGeV: `~numpy.ndarray` 
	    Gamma-ray energies in GeV

	dL: `~numpy.ndarray` 
	    Domain lengths. If not given, they will be automatically 
	    generated.

	restore: str or None
	    if str, identifier for files to restore environment. 
	    If None, initialize mixing with new B field

	restore_path: str
	    full path to environment files

	B0: float
	    IGMF at z = 0 in muG

	L0: float
	    Coherence length at z = 0 in kpc

	n0: float
	    electron density of intergalactic medium at z=0 in cm^-3

	eblmodel: string
	    name of the used EBL model (default: Dominguez et al. 2011 Model)

	cosmo: `~astropy.cosmology.core.FlatLambdaCDM`
	    chosen cosmology, default is H0 = 70, Om0 = 0.3

	nsim: int
	    number of B field realizations
	"""
	kwargs.setdefault('EGeV', np.logspace(0.,4.,100))
	kwargs.setdefault('restore', None)
	kwargs.setdefault('restore_path', './')
	kwargs.setdefault('B0', 1.e-3)
	kwargs.setdefault('L0', 1.e3)
	kwargs.setdefault('n0', 1.e-7)
	kwargs.setdefault('nsim', 1)
	kwargs.setdefault('dL', 'None')
	kwargs.setdefault('cosmo',FlatLambdaCDM(H0 = 70., Om0 = 0.3))
	kwargs.setdefault('eblmodel', 'dominguez')

	self._source = source
	self._t = OptDepth.readmodel(model = kwargs['eblmodel'])
	self._cosmo = kwargs['cosmo']

	if kwargs['restore'] == None:
	    self._b = cell.Bcell(kwargs['B0'],kwargs['L0'])
	    B, psi, dL, self._zstep = self._b.new_Bcosmo(self._source.z, 
				cosmo = kwargs['cosmo'], nsim = kwargs['nsim']) 
	    if not kwargs['dL'].lower() == 'none':
		if type(kwargs['dL']) == list or type(kwargs['dL']) == np.ndarray:
		    dL = kwargs['dL']
		else:
		    raise TypeError("dL kwarg must be list or numpy.ndarray")

	    self._zmean = self._zstep[:-1]
	    self._nel = kwargs['n0'] * (1. + self._zmean)**3.
			    
	    dt = self._t.opt_depth(self._zstep[1:],kwargs['EGeV'] / 1.e3)  - \
		self._t.opt_depth(self._zstep[:-1],kwargs['EGeV'] / 1.e3)

	    # absorption rate in kpc^-1
	    Gamma = dt.T / dL
	    # init the transfer function with absorption
	    super(MixIGMFCell,self).__init__(kwargs['EGeV'], B, psi, self._nel, 
						dL, alp, Gamma = Gamma, chi = None, Delta = None)
	    self._ee *= (1. + self._zmean) # transform energies to comoving frame
	else:
	    dL, self._zstep = trafo.cosmo_cohlength(z,kwargs['L0'] * u.kpc, cosmo = self._cosmo)

	    tra = super(MixIGMFCell,self).readEnviron(
					    kwargs['restore'], alp,
					    filepath = kwargs['restore_path'],
					    )
	    super(MixIGMFCell,self).__init__(tra.EGeV, tra.B, tra.psin, 
			tra.nel, tra.dL, tra.alp, Gamma = tra.Gamma, chi = tra.chi,
			Delta = tra.Delta)
	return

    @property
    def t(self):
	return self._t

class MixICMCell(trans.GammaALPTransfer):
    def __init__(self, alp, **kwargs):
	"""
	Initialize mixing in the intracluster magnetic field, 
	assuming that it follows a domain-like structre.

	Parameters
	----------
	alp: `~gammaALPs.ALP`
	    `~gammaALPs.ALP` object with ALP parameters

	kwargs
	------
	EGeV: `~numpy.ndarray` 
	    Gamma-ray energies in GeV

	restore: str or None
	    if str, identifier for files to restore environment. 
	    If None, initialize mixing with new B field

	restore_path: str
	    full path to environment files
	
	rbounds: list or `~numpy.ndarray`
	    bin bounds for steps along line of sight in kpc, 
	    default: linear range between 0. and r_abell with L0 as 
	    step size

	B0: float
	    ICM at r = 0 in muG

	L0: float
	    Coherence length in kpc

	nsim: int
	    number of B field realizations

	ICM kwargs:

	n0: float
	    electron density in cm**-3 (default 1e-3)
	r_core: float
	    core radius in kpc (default 10.)
	beta: float
	    exponent of density profile (default: 1.)

	eta: float
	    exponent for scaling of B field with electron density (default = 2./3.)

	n2: float
	    if > 0., use profile with this second density component 

	r_core2: float
	    if > 0., use profile with this second r_core value

	beta2: float
	    if > 0., use profile with this second beta value as for NGC1275
	"""
	kwargs.setdefault('EGeV', np.logspace(0.,4.,100))
	kwargs.setdefault('restore', None)
	kwargs.setdefault('restore_path', './')
	kwargs.setdefault('nsim', 1)

	# Bfield kwargs
	kwargs.setdefault('B0', 1.)
	kwargs.setdefault('L0', 1.)

	# ICM kwargs
	kwargs.setdefault('n0', 1e-3)
	kwargs.setdefault('r_core', 10.)
	kwargs.setdefault('r_abell', 500.)
	kwargs.setdefault('eta', 1.)
	kwargs.setdefault('beta', 2. / 3.)
	kwargs.setdefault('n2', 0.)
	kwargs.setdefault('r_core2', 0.)
	kwargs.setdefault('beta2', 0.)

	kwargs.setdefault('rbounds', np.arange(0., kwargs['r_abell'], kwargs['L0']))

	self._rbounds = kwargs['rbounds']
	self._r = 0.5 * (self._rbounds[1:] + self._rbounds[:-1])
	dL = self._rbounds[1:] - self._rbounds[:-1]
	self._nel = icm.NelICM(**kwargs)

	if kwargs['restore'] == None:
	    self._b = cell.Bcell(kwargs['B0'],kwargs['L0'])
	    B, psi = self._b.new_Bn(self._r.shape[0], Bscale = self._nel.Bscale(self._r), 
					nsim = kwargs['nsim']) 

			    
	    # init the transfer function with absorption
	    super(MixICMCell,self).__init__(kwargs['EGeV'], B, psi, self._nel(self._r), 
						dL, alp, Gamma = None, chi = None, Delta = None)
	else:
	    tra = super(MixICMCell,self).readEnviron(kwargs['restore'], alp,
						filepath = kwargs['restore_path'])
	    super(MixICMCell,self).__init__(tra.EGeV, tra.B, tra.psin, 
			tra.nel, tra.dL, tra.alp, Gamma = tra.Gamma,
			chi = tra.chi, Delta = tra.Delta)
	return

class MixICMGaussTurb(trans.GammaALPTransfer):
    def __init__(self, alp, **kwargs):
	"""
	Initialize mixing in the intracluster magnetic field, 
	assuming that it follows a Gaussian turbulence

	Parameters
	----------
	alp: `~gammaALPs.ALP`
	    `~gammaALPs.ALP` object with ALP parameters

	kwargs
	------
	EGeV: `~numpy.ndarray` 
	    Gamma-ray energies in GeV

	restore: str or None
	    if str, identifier for files to restore environment. 
	    If None, initialize mixing with new B field
	restore_path: str
	    full path to environment files

	rbounds: list or `~numpy.ndarray`
	    bin bounds for steps along line of sight in kpc, 
	    default: linear range between 0. and r_abell 
	    with 1/kH (min turbulence scale) as step size

	B field kwargs:

	B0: float
	    ICM at r = 0 in muG
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
	    dkType:string
	    either linear, log, or random. Determine the spacing of the dk intervals 	
	dkSteps: int
	    number of dkSteps.
	    For log spacing, number of steps per decade / number of decades ~ 10
	    should be chosen.
	nsim: int
	    number of B field realizations

	ICM kwargs:

	n0: float
	    electron density in cm**-3 (default 1e-3)
	r_core: float
	    core radius in kpc (default 10.)
	beta: float
	    exponent of density profile (default: 1.)
	eta: float
	    exponent for scaling of B field with electron density (default = 2./3.)
	n2: float
	    if > 0., use profile with this second density component 
	r_core2: float
	    if > 0., use profile with this second r_core value
	beta2: float
	    if > 0., use profile with this second beta value as for NGC1275
	"""
	kwargs.setdefault('EGeV', np.logspace(0.,4.,100))
	kwargs.setdefault('restore', None)
	kwargs.setdefault('restore_path', './')
	kwargs.setdefault('nsim', 1)

	# Bfield kwargs
	kwargs.setdefault('B0', 1.)
	kwargs.setdefault('kH', 1. / 100.)
	kwargs.setdefault('kL', 1.)
	kwargs.setdefault('q', 11. / 3.)
	kwargs.setdefault('kMin', -1.)
	kwargs.setdefault('dkType','log')
	kwargs.setdefault('dkSteps',0)

	# ICM kwargs
	kwargs.setdefault('n0', 1e-3)
	kwargs.setdefault('r_core', 10.)
	kwargs.setdefault('r_abell', 500.)
	kwargs.setdefault('eta', 1.)
	kwargs.setdefault('beta', 2. / 3.)
	kwargs.setdefault('n2', 0.)
	kwargs.setdefault('r_core2', 0.)
	kwargs.setdefault('beta2', 0.)

	# step length is assumed to be 1. / kH -> minimum turbulence length scale
	kwargs.setdefault('rbounds', np.arange(0., kwargs['r_abell'],1. / kwargs['kH']))
	self._rbounds = kwargs['rbounds']

	self._r = 0.5 * (self._rbounds[1:] + self._rbounds[:-1])
	dL = self._rbounds[1:] - self._rbounds[:-1]

	self._nelicm = icm.NelICM(**kwargs)

	if kwargs['restore'] == None:
	    self._b = gauss.Bgaussian(kwargs['B0'], kwargs['kH'], 
	    				kwargs['kL'], kwargs['q'], 
					kMin = kwargs['kMin'],
					dkType = kwargs['dkType'],
					dkSteps = kwargs['dkSteps'])
	    B, psi = self._b.new_Bn(self._r, Bscale = self._nelicm.Bscale(self._r), 
					nsim = kwargs['nsim']) 

			    
	    # init the transfer function with absorption
	    super(MixICMGaussTurb,self).__init__(kwargs['EGeV'], B, psi, self._nelicm(self._r), 
						dL, alp, Gamma = None, chi = None, Delta = None)
	else:
	    tra = super(MixICMGaussTurb,self).readEnviron(kwargs['restore'], alp, 
						filepath = kwargs['restore_path'])
	    super(MixICMGaussTurb,self).__init__(tra.EGeV, tra.Bn, tra.psin, tra.nel, 
						tra.dL, tra.alp, Gamma = tra.Gamma,
						chi = tra.chi, Delta = tra.Delta)
	return

class MixJet(trans.GammaALPTransfer):
    def __init__(self, alp, source, **kwargs):
	"""
	Initialize mixing in the magnetic field of the jet, 
	assumed here to be coherent

	Parameters
	----------
	alp: `~gammaALPs.ALP`
	    `~gammaALPs.ALP` object with ALP parameters

	source: `~gammaALPs.Source`
	    `~gammaALPs.Source` object with source parameters

	kwargs
	------
	EGeV: `~numpy.ndarray` 
	    Gamma-ray energies in GeV

	restore: str or None
	    if str, identifier for files to restore environment. 
	    If None, initialize mixing with new B field
	restore_path: str
	    full path to environment files
	rgam: float
	    distance of gamma-ray emitting region to BH in pc (default: 0.1)
	sens: float
	    sens > 0 and sens < 1., sets the number of domains, 
	    for the B field in the n-th domain, it will have changed by B_n = sens * B_{n-1}
	rbounds: list or `~numpy.ndarray`
	    bin bounds for steps along line of sight in pc, 
	    default: log range between rgam and Rjet
	    with step size chosen such that B field changes 
	    by sens parameter in each step


	B field kwargs:

	B0: float
	    Jet field at r = R0 in G (default: 0.1)
	r0: float
	    distance from BH where B = B0 in pc (default: 0.1)
	alpha: float
	    exponent of toroidal mangetic field (default: -1.)
	psi: float
	    Angle between one photon polarization state and B field. 
	    Assumed constant over entire jet. (default: pi / 4)
	helical: bool
	    if True, use helical magnetic-field model from Clausen-Brown et al. (2011). 
	    In this case, the psi kwarg is treated is as the phi angle 
	    of the photon trajectory in the cylindrical jet coordinate system
	    (default: True)

	Electron density kwargs:

	n0: float
	    electron density at R0 in cm**-3 (default 1e3)
	beta: float
	    exponent of electron density (default = -2.)
	equipartition: bool
	    if true, assume equipartition between electrons and the B field. 
	    This will overwrite beta = 2 * alpha and set n0 given the minimum 
	    electron lorentz factor set with gamma_min
	gamma_min: float
	    minimum lorentz factor of emitting electrons, only used if equipartition = True
	gamma_max: float
	    maximum lorentz factor of emitting electrons, only used if equipartition = True
	    by default assumed to be gamma_min * 1e4

	Jet kwargs:

	Rjet: float
	    maximum jet length in pc (default: 1000.)
	theta_obs: float
	    Angle between l.o.s. and jet axis in degrees (default: 3.) 
	bulk_lorentz: float
	    bulk lorentz factor of gamma-ray emitting plasma (default: 10.)
	theta_jet: float
	    Jet opening angle in degrees. If not given, assumed to be 1./bulk_lorentz
	"""
	kwargs.setdefault('EGeV', np.logspace(0.,4.,100))
	kwargs.setdefault('restore', None)
	kwargs.setdefault('restore_path', './')
	kwargs.setdefault('sens', 0.99)
	kwargs.setdefault('rgam', 0.1)
	# Bfield kwargs
	kwargs.setdefault('helical', True)
	kwargs.setdefault('B0', 0.1)
	kwargs.setdefault('r0', 0.1)
	kwargs.setdefault('alpha', -1.)
	kwargs.setdefault('psi', np.pi / 4.)
	# electron density kwargs
	kwargs.setdefault('n0', 1e3)
	kwargs.setdefault('beta', -2.)
	kwargs.setdefault('equipartition', True)
	kwargs.setdefault('gamma_min', 1. )
	kwargs.setdefault('gamma_max', 1e4 * kwargs['gamma_min'])

	# calculate doppler factor
	self._Rjet = kwargs['Rjet']
	self._psi = kwargs['psi']
	self._source = source

	nsteps = int(np.ceil( kwargs['alpha'] * \
			np.log(self._Rjet /kwargs['rgam'] ) / np.log(kwargs['sens']) ))	
	kwargs.setdefault('rbounds',np.logspace(np.log10(kwargs['rgam']),np.log10(self._Rjet),nsteps))
	self._rbounds = kwargs['rbounds']

	self._r = np.sqrt(self._rbounds[1:] * self._rbounds[:-1])
	dL = self._rbounds[1:] - self._rbounds[:-1]

	if kwargs['restore'] == None:
	    self._b = jet.Bjet(kwargs['B0'], kwargs['r0'], 
	    				kwargs['alpha']
					)
	    B, psi = self._b.new_Bn(self._r, psi = kwargs['psi']) 
	    if kwargs['helical']:
		B, psi = self.Bjet_calc(B,psi)

	    if kwargs['equipartition']:
		kwargs['beta'] = kwargs['alpha'] * 2.

		intp = USpline(np.log10(self._r), np.log10(B), k = 1, s = 0)
		B0 = 10.**intp(np.log10(kwargs['r0']))
		# see e.g.https://arxiv.org/pdf/1307.4100.pdf Eq. 2
		kwargs['n0'] = B0** 2. / 8. / np.pi \
			    / kwargs['gamma_min'] / (c.m_e * c.c ** 2.).to('erg').value / \
			    np.log(kwargs['gamma_max'] / kwargs['gamma_min'])
		logging.info("Assuming equipartion at r0: n0(r0) = {0[n0]:.3e} cm^-3".format(kwargs))

	    self._neljet = njet.NelJet(kwargs['n0'],kwargs['r0'],kwargs['beta'])
			    
	    # init the transfer function with absorption
	    super(MixJet,self).__init__(kwargs['EGeV'], B * 1e6, psi, self._neljet(self._r), 
						dL * 1e-3, alp, Gamma = None, chi = None, Delta = None)

	    # transform energies to stationary frame
	    self._ee /= self._source._doppler
	else:
	    tra = super(MixJet,self).readEnviron(kwargs['restore'],alp,
					    filepath = kwargs['restore_path'])
	    super(MixJet,self).__init__(tra.EGeV, tra.B, tra.psi, tra.nel, 
						tra.dL, tra.alp, Gamma = tra.Gamma,
						chi = tra.chi, Delta = tra.Delta)
	return

	
    @property
    def Rjet(self):
	return self._Rjet

    @Rjet.setter
    def Rjet(self, Rjet):
	if type(Rjet) == u.Quantity:
	    self._Rjet= Rjet.to('pc').value
	else:
	    self._Rjet = Rjet
	return

    def Bjet_calc(self, B0, phi):
	"""
	compute Jet magnetic field along line of sight that 
	forms observation angle theta_obs with jet axis. 
	Model assumes the helical jet structure of 
	Clausen-Brown, E., Lyutikov, M., and Kharb, P. (2011); arXiv:1101.5149

	Parameters
	-----------
	B0: `~numpy.ndarray`
	    N-dim array with magnetic field strength along jet axis 
	phi: float
	    phi angle in degrees along with photons propagate along jet 
	    (in cylindrical jet geometry)

	Returns
	-------
	2-dim tuple containing:
	    N-dim np.array, field strength along line of sight 
	    N-dim np.array, with psi angles between photon polarization states 
	    and Jet Bfield
	"""
	# Calculate normalized rho component, i.e. distance
	# from line of sight to jet axis assuming a self similar jet
	p,tj,to = np.radians(phi), np.radians(self._source.theta_jet), \
		    np.radians(self._source.theta_obs)

	rho_n = np.tan(to) / np.tan(tj)
	k = 2.405 # pinch, set so that Bz = 0 at jet boundary

	# compute bessel functions, see Clausen-Brown Eq. 2
	j0 = jv(0.,rho_n * k)
	j1 = jv(1.,rho_n * k)

	# B-field along l.o.s.
	Bn = np.cos(to) * j0 - np.sin(p)*np.sin(to) * j1
	# B-field transversal to l.o.s.
	Bt = np.cos(p) * j1
	Bu = -(np.cos(to) * np.sin(p) * j1 + np.sin(to) * j0)
	
	Btrans	= B0 * np.sqrt(Bt**2. + Bu**2.)	# Abs value of transverse component in all domains
	Psin	= np.arctan2(B0*Bt,B0*Bu)	# arctan2 selects the right quadrant

	return Btrans,Psin

class MixGMF(trans.GammaALPTransfer):
    def __init__(self, alp, source, **kwargs):
	"""
	Initialize mixing in the coherent component of the Galactic magnetic field

	Parameters
	----------
	alp: `~gammaALPs.ALP`
	    `~gammaALPs.ALP` object with ALP parameters

	source: `~gammaALPs.Source`
	    `~gammaALPs.Source` object with source parameters


	kwargs
	------
	EGeV: `~numpy.ndarray` 
	    Gamma-ray energies in GeV

	restore: str or None
	    if str, identifier for files to restore environment. 
	    If None, initialize mixing with new B field
	restore_path: str
	    full path to environment files
	int_steps: interger (default = 100)
	    Number of integration steps
	rbounds: list or `~numpy.ndarray`
	    bin bounds for steps along line of sight in kpc, 
	    default: lin range between end of Galactic Bfield and 0.
	    with int_step steps 

	Source parameters:

	ra: float
	    R.A. of the source (J2000)
	dec: float
	    Declination of the source (J2000)
	galactic: float
	    Distance of source to sun in kpc. If -1, source is extragalactic

	B field parameters:

	rho_max: float
	    maximal rho of GMF in kpc
	    default: 20 kpc
	zmax: float
	    maximal z of GMF in kpc
	    default: 50 kpc
	model: string (default = jansson)
	    GMF model that is used. Currently the model by Jansson & Farrar (2012)
	    (also with updates from Plack measurements)
	    and Pshirkov et al. (2011) are implemented. 
	    Usage: model=[jansson12, jansson12b, jansson12c, phsirkov]
	model_sym: string (default = ASS)
	    Only applies if pshirkov model is chosen: 
	    you can choose between the axis- and bisymmetric version by setting model_sym to ASS or BSS.

	Electron density parameters:

	n0: float
	    Electron density in cm^-3 (default: 10). NE2001 code implementation still missing.
	"""
	kwargs.setdefault('EGeV', np.logspace(0.,4.,100))
	kwargs.setdefault('restore', None)
	kwargs.setdefault('restore_path', './')
	kwargs.setdefault('int_steps',100)

	# Bfield kwargs
	kwargs.setdefault('galactic',-1.)
	kwargs.setdefault('rho_max',20.)
	kwargs.setdefault('zmax',50.)
	kwargs.setdefault('model','jansson12')
	kwargs.setdefault('model_sym','ASS')
	self._model = kwargs['model']
	self._galactic = kwargs['galactic']

	# Nel kwargs
	kwargs.setdefault('n0', 1e1)

	# for B field calculation
	self.__zmax = kwargs['zmax']
	self.__rho_max = kwargs['rho_max']

	self._source = source
	
	if kwargs['model'].find('jansson') >= 0:
	    self._Bgmf = gmf.GMF(mode = kwargs['model'])	# Initialize the Bfield class
	elif kwargs['model'] == 'pshirkov':
	    self._Bgmf = gmf.GMF_Pshirkov(mode = kwargs['model_sym'])	
	else:
	    raise ValueError("Unknown GMF model chosen")

	# set coordinates
    	self.set_coordinates() # sets self._l, self._b and self._smax

	# step length 
	kwargs.setdefault('rbounds' , np.linspace(self._smax,0., kwargs['int_steps'],endpoint = False))
	self._rbounds = kwargs['rbounds']

	self._r = 0.5 * (self._rbounds[1:] + self._rbounds[:-1])
	dL = self._rbounds[:-1] - self._rbounds[1:] # use other way round since we are beginning from 
							# max distance and propagate to Earth

	# NE2001 code missing!
	self._nelgmf = kwargs['n0'] * np.ones(self._r.shape)


	if kwargs['restore'] == None:
	    B, psi = self.Bgmf_calc()

	    # init the transfer function with absorption
	    super(MixGMF,self).__init__(kwargs['EGeV'], B, psi, self._nelgmf, 
						dL, alp, Gamma = None, chi = None, Delta = None)
	else:
	    tra = super(MixGMF,self).readEnviron(kwargs['restore'], alp,
						filepath = kwargs['restore_path'])
	    super(MixGMF,self).__init__(tra.EGeV, tra.B, tra.psi, tra.nel, 
						tra.dL, tra.alp, Gamma = tra.Gamma,
						chi = tra.chi, Delta = tra.Delta)
	return

    @property
    def galactic(self):
	return self._galactic

    @galactic.setter
    def galactic(self,galactic):
	if type(galactic) == u.Quantity:
	    self._galactic = galactic.to('kpc').value
	else:
	    self._galactic = galactic
	self._B, self._psi = self.Bgmf_calc()
	return 

    def set_coordinates(self):
	"""
	Set the coordinates l,b and the the maximum distance smax where |GMF| > 0


	Sets
	----
	l: float
	    galactic longitude
	b: float
	    galactic latitude
	smax: float
	    maximum distance in kpc from sun considered here where |GMF| > 0
	"""

	# Transformation RA, DEC -> L,B
	self._l = np.radians(self._source.l)
	self._b = np.radians(self._source.b)
	d = self._Bgmf.Rsun

	if self.galactic < 0.:	# if source is extragalactic, calculate maximum distance that beam traverses GMF to Earth
	    cl = np.cos(self._l)
	    cb = np.cos(self._b)
	    sb = np.sin(self._b)
	    self._smax = np.amin([self.__zmax/np.abs(sb),
			1./np.abs(cb) * (-d*cl + np.sqrt(d**2 + cl**2 - d**2*cb + self.__rho_max**2))])
	else:
	    self._smax = self._galactic
	return 

    def Bgmf_calc(self,l=0.,b=0.):
	"""
	compute GMF at (s,l,b) position where origin at self.d along x-axis in GC coordinates is assumed

	Parameters
	-----------
	s: N-dim np.array, distance from sun in kpc for all domains
	l (optional): galactic longitude, scalar or N-dim np.array
	b (optional): galactic latitude, scalar or N-dim np.array

	Returns
	-------
	2-dim tuple containing:
	    (3,N)-dim np.parray containing GMF for all domains in 
	    galactocentric cylindrical coordinates (rho, phi, z) 
	    N-dim np.array, field strength for all domains
	"""
	if np.isscalar(l):
	    if not l:
		l = self._l
	if np.isscalar(b):
	    if not b:
		b = self._b

	rho	= trafo.rho_HC2GC(self._r,l,b,self._Bgmf.Rsun)	# compute rho in GC coordinates for s,l,b
	phi	= trafo.phi_HC2GC(self._r,l,b,self._Bgmf.Rsun)	# compute phi in GC coordinates for s,l,b
	z	= trafo.z_HC2GC(self._r,l,b,self._Bgmf.Rsun)	# compute z in GC coordinates for s,l,b

	B = self._Bgmf.Bdisk(rho,phi,z)[0] 	# add all field components
	B += self._Bgmf.Bhalo(rho,z)[0] 
	if self._model.find('jansson') >= 0:
	    B += self._Bgmf.BX(rho,z)[0] 

	### Single components for debugging ###
	#B = self.Bgmf.Bdisk(rho,phi,z)[0] 	
	#B = self.Bgmf.Bhalo(rho,z)[0] 
	#B = self.Bgmf.BX(rho,z)[0] 

	Babs = np.sqrt(np.sum(B**2., axis = 0))	# compute overall field strength
	Bs, Bt, Bu	= trafo.GC2HCproj(B, self._r, self._l, self._b,self._Bgmf.Rsun)
	
	Btrans	= np.sqrt(Bt**2. + Bu**2.)	# Abs value of transverse component in all domains
	Psin	= np.arctan2(Bt,Bu)	# arctan2 selects the right quadrant

	return Btrans,Psin

class MixFromFile(trans.GammaALPTransfer):
    def __init__(self, alp, filename,**kwargs):
	"""
	Initialize mixing in environment given by a data file.
	Data file has to have 5 columns:
	1: distance along l.o.s. (z-axis) in kpc (bin bounds)
	2: electron density in cm^-3 at bin bounds
	3: Temperature in K
	4: Bx component in muG at bin bounds
	5: By component in muG at bin bounds
	6: Bz component in muG at bin bounds

	Parameters
	----------
	alp: `~gammaALPs.ALP`
	    `~gammaALPs.ALP` object with ALP parameters
	filename: str
	    full path to file with electron density and B field

	kwargs
	------
	EGeV: `~numpy.ndarray` 
	    Gamma-ray energies in GeV

	restore: str or None
	    if str, identifier for files to restore environment. 
	    If None, initialize mixing with new B field
	restore_path: str
	    full path to environment files

	"""
	kwargs.setdefault('EGeV', np.logspace(0.,4.,100))
	kwargs.setdefault('restore', None)
	kwargs.setdefault('restore_path', './')

	data = np.loadtxt(filename)
	self._rbounds = data[:,0]
	self._r = 0.5 * (self._rbounds[1:] + self._rbounds[:-1])
	dL = self._rbounds[1:] - self._rbounds[:-1]

	n = 0.5 * (data[1:,1] + data[:-1,1])
	Bx = 0.5 * (data[1:,3] + data[:-1,3])
	By = 0.5 * (data[1:,4] + data[:-1,4])
	Bz = 0.5 * (data[1:,5] + data[:-1,5])
	self._T = 0.5 * (data[1:,2] + data[:-1,2])

	Btrans = np.sqrt(Bx**2. + By**2.)
	psi = np.arctan2(Bx,By)

	if kwargs['restore'] == None:
			    
	    # init the transfer function 
	    super(MixFromFile,self).__init__(kwargs['EGeV'], Btrans, psi, n, 
						dL, alp, Gamma = None, chi = None, Delta = None)
	else:
	    tra = super(MixFromFile,self).readEnviron(kwargs['restore'], alp, 
						filepath = kwargs['restore_path'])
	    super(MixFromFile,self).__init__(tra.EGeV, tra.Bn, tra.psin, tra.nel, 
						tra.dL, tra.alp, Gamma = tra.Gamma,
						chi = tra.chi, Delta = tra.Delta)
	return

class MixFromArray(trans.GammaALPTransfer):
    def __init__(self, alp, Btrans, psi, nel, r, dL, **kwargs):
	"""
	Initialize mixing in environment given by numpy arrays

	Parameters
	----------
	alp: `~gammaALPs.ALP`
	    `~gammaALPs.ALP` object with ALP parameters
	Btrans: `~numpy.ndarray`
	    n-dim or m x n-dim, absolute value of transversal B field, in muG
	    if m x n-dimensional, m realizations are assumed
	psi: `~numpy.ndarray`
	    n-dim or m x n-dim, Angles between transversal direction and one polarization,
	    if m x n-dimensional, m realizations are assumed
	nel: `~numpy.ndarray`
	    n-dim or m x n-dim, electron density, in cm^-3,
	    if m x n-dimensional, m realizations are assumed
	r: `~numpy.ndarray`
	    n-dim, bin centers along line of sight in kpc, 
	    if m x n-dimensional, m realizations are assumed
	dL: `~numpy.ndarray`
	    n-dim, bin widths along line of sight in kpc, 
	    if m x n-dimensional, m realizations are assumed

	kwargs
	------
	EGeV: `~numpy.ndarray` 
	    Gamma-ray energies in GeV

	restore: str or None
	    if str, identifier for files to restore environment. 
	    If None, initialize mixing with new B field
	restore_path: str
	    full path to environment files

	"""
	kwargs.setdefault('EGeV', np.logspace(0.,4.,100))
	kwargs.setdefault('restore', None)
	kwargs.setdefault('restore_path', './')

	if kwargs['restore'] == None:
			    
	    # init the transfer function 
	    super(MixFromArray,self).__init__(kwargs['EGeV'], Btrans, psi, nel, 
						dL, alp, Gamma = None, chi = None, Delta = None)
	else:
	    tra = super(MixFromFile,self).readEnviron(kwargs['restore'], alp, 
						filepath = kwargs['restore_path'])
	    super(MixFromArray,self).__init__(tra.EGeV, tra.Bn, tra.psin, tra.nel, 
						tra.dL, tra.alp, Gamma = tra.Gamma,
						chi = tra.chi, Delta = tra.Delta)
	return
