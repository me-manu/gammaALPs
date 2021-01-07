from __future__ import absolute_import, division, print_function
import numpy as np
from . import transfer as trans
from ..bfields import cell, gauss, gmf, jet
from ..nel import icm
from ..nel import jet as njet
from ..utils import trafo
from astropy import units as u
from astropy import constants as c
from astropy.cosmology import FlatLambdaCDM
from ebltable.tau_from_model import OptDepth
from scipy.interpolate import UnivariateSpline as USpline
import logging


class MixIGMFCell(trans.GammaALPTransfer):
    def __init__(self, alp, source, **kwargs):
        """
        Initialize mixing in the intergalactic magnetic field (IGMF).

        Parameters
        ----------
        alp: :py:class:`~gammaALPs.ALP`
            :py:class:`~gammaALPs.ALP` object with ALP parameters

        source: :py:class:`~gammaALPs.Source`
            :py:class:`~gammaALPs.Source` object with source parameters

        EGeV: array-like
            Gamma-ray energies in GeV

        dL: arra-like
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
        kwargs.setdefault('EGeV', np.logspace(0., 4., 100))
        kwargs.setdefault('restore', None)
        kwargs.setdefault('restore_path', './')
        kwargs.setdefault('B0', 1.e-3)
        kwargs.setdefault('L0', 1.e3)
        kwargs.setdefault('n0', 1.e-7)
        kwargs.setdefault('nsim', 1)
        kwargs.setdefault('dL', None)
        kwargs.setdefault('cosmo',FlatLambdaCDM(H0 = 70., Om0 = 0.3))
        kwargs.setdefault('eblmodel', 'dominguez')
        kwargs.setdefault('seed', None)

        self._source = source
        self._t = OptDepth.readmodel(model=kwargs['eblmodel'])
        self._cosmo = kwargs['cosmo']

        if kwargs['restore'] is None:
            self._Bfield_model = cell.Bcell(kwargs['B0'], kwargs['L0'], seed=kwargs['seed'])
            B, psi, dL, self._z_step = self._Bfield_model.new_Bcosmo(self._source.z,
                                                                     cosmo=kwargs['cosmo'],
                                                                     nsim=kwargs['nsim'])
            if kwargs['dL'] is not None:
                if isinstance(kwargs['dL'], list) or isinstance(kwargs['dL'], np.ndarray):
                    dL = kwargs['dL']
                else:
                    raise TypeError("dL kwarg must be list or numpy.ndarray")

            self._z_mean = self._z_step[:-1]
            self._nel = kwargs['n0'] * (1. + self._z_mean) ** 3.

            dt = self._t.opt_depth(self._z_step[1:], kwargs['EGeV'] / 1.e3) - \
                 self._t.opt_depth(self._z_step[:-1], kwargs['EGeV'] / 1.e3)

            # absorption rate in kpc^-1
            Gamma = dt.T / dL
            # init the transfer function with absorption
            super(MixIGMFCell, self).__init__(kwargs['EGeV'], B, psi, self._nel,
                                              dL, alp,
                                              Gamma=Gamma,
                                              chi=None,
                                              Delta=None)
            self._ee *= (1. + self._z_mean)  # transform energies to comoving frame
        else:
            dL, self._z_step = trafo.cosmo_cohlength(self._source.z, kwargs['L0'] * u.kpc, cosmo=self._cosmo)

            tra = super(MixIGMFCell, self).read_environ(
                                                kwargs['restore'], alp,
                                                filepath=kwargs['restore_path'],
                                                )

            super(MixIGMFCell, self).__init__(tra.EGeV, tra.B, tra.psin,
                          tra.nel, tra.dL, tra.alp, Gamma=tra.Gamma, chi=tra.chi,
                          Delta=tra.Delta)

    @property
    def t(self):
        return self._t

    @property
    def Bfield_model(self):
        return self._Bfield_model

    @property
    def nel_model(self):
        return self._nel


class MixICMCell(trans.GammaALPTransfer):
    def __init__(self, alp, **kwargs):
        """
        Initialize mixing in the intracluster magnetic field,
        assuming that it follows a domain-like structure.

        Parameters
        ----------
        alp: :py:class:`~gammaALPs.ALP`
            :py:class:`~gammaALPs.ALP` object with ALP parameters

        EGeV: array-like
            Gamma-ray energies in GeV

        restore: str or None
            if str, identifier for files to restore environment.
            If None, initialize mixing with new B field

        restore_path: str
            full path to environment files

        rbounds: array-like
            bin bounds for steps along line of sight in kpc,
            default: linear range between 0. and r_abell with L0 as
            step size

        B0: float
            ICM at r = 0 in muG

        L0: float
            Coherence length in kpc

        nsim: int
            number of B field realizations

        r_abell: float
            Abell radius of cluster (radius until which oscillation probability is computed)

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
        kwargs.setdefault('EGeV', np.logspace(0., 4., 100))
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
        kwargs.setdefault('seed', None)

        kwargs.setdefault('rbounds', np.arange(0., kwargs['r_abell'], kwargs['L0']))
        if kwargs['r_abell'] <= kwargs['L0']:
           logging.warning("r_abell <= L0: assuming one domain from 0. to L0")
           kwargs['rbounds'] = np.array([0., kwargs['L0']])

        self._rbounds = kwargs['rbounds']
        self._r = 0.5 * (self._rbounds[1:] + self._rbounds[:-1])
        dL = self._rbounds[1:] - self._rbounds[:-1]
        self._nel = icm.NelICM(**kwargs)

        if kwargs['restore'] is None:
            self._Bfield_model = cell.Bcell(kwargs['B0'], kwargs['L0'])
            B, psi = self._Bfield_model.new_Bn(self._r.shape[0],
                                               Bscale=self._nel.Bscale(self._r),
                                               nsim=kwargs['nsim'])

            # init the transfer function
            super(MixICMCell, self).__init__(kwargs['EGeV'], B, psi, self._nel(self._r),
                                             dL, alp, Gamma=None, chi=None, Delta=None)
        else:
            tra = super(MixICMCell,self).read_environ(kwargs['restore'], alp,
                                                      filepath=kwargs['restore_path'])
            super(MixICMCell,self).__init__(tra.EGeV, tra.B, tra.psin,
                                            tra.nel, tra.dL, tra.alp,
                                            Gamma=tra.Gamma,
                                            chi=tra.chi,
                                            Delta=tra.Delta)

    @property
    def r(self):
        return self._r

    @property
    def rbounds(self):
        return self._rbounds

    @property
    def Bfield_model(self):
        return self._Bfield_model

    @property
    def nel_model(self):
        return self._nel


class MixICMGaussTurb(trans.GammaALPTransfer):
    def __init__(self, alp, **kwargs):
        """
        Initialize mixing in the intracluster magnetic field,
        assuming that it follows a Gaussian turbulence

        Parameters
        ----------
        alp: :py:class:`~gammaALPs.ALP`
            :py:class:`~gammaALPs.ALP` object with ALP parameters

        EGeV: array-like
            Gamma-ray energies in GeV

        restore: str or None
            if str, identifier for files to restore environment.
            If None, initialize mixing with new B field
        restore_path: str
            full path to environment files

        rbounds: larray-like
            bin bounds for steps along line of sight in kpc.
            Default: linear range between 0. and r_abell
            with 1/kH (min turbulence scale) as step size

        thinning: int
            if > 1, thin out array of r

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
            default 1e-3 * kL (the k interval runs from kMin to kH)

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
        kwargs.setdefault('EGeV', np.logspace(0., 4., 100))
        kwargs.setdefault('restore', None)
        kwargs.setdefault('restore_path', './')
        kwargs.setdefault('nsim', 1)
        kwargs.setdefault('thinning', 1)

        # Bfield kwargs
        kwargs.setdefault('B0', 1.)
        kwargs.setdefault('kH', 1. / 100.)
        kwargs.setdefault('kL', 1.)
        kwargs.setdefault('q', 11. / 3.)
        kwargs.setdefault('kMin', -1.)
        kwargs.setdefault('dkType','log')
        kwargs.setdefault('dkSteps',0)
        kwargs.setdefault('seed', None)

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
        kwargs.setdefault('rbounds', np.arange(0., kwargs['r_abell'], 1. / kwargs['kH']))
        self._rbounds = kwargs['rbounds'][::kwargs['thinning']]

        self._r = 0.5 * (self._rbounds[1:] + self._rbounds[:-1])
        dL = self._rbounds[1:] - self._rbounds[:-1]

        self._nelicm = icm.NelICM(**kwargs)

        if kwargs['restore'] is None:
            self._Bfield_model = gauss.Bgaussian(kwargs['B0'], kwargs['kH'],
                                                 kwargs['kL'], kwargs['q'],
                                                 kMin=kwargs['kMin'],
                                                 dkType=kwargs['dkType'],
                                                 dkSteps=kwargs['dkSteps'],
                                                 seed=kwargs['seed'])

            B, psi = self._Bfield_model.new_Bn(self._r,
                                               Bscale=self._nelicm.Bscale(self._r),
                                               nsim=kwargs['nsim'])

            # init the transfer function with absorption
            super(MixICMGaussTurb, self).__init__(kwargs['EGeV'], B, psi, self._nelicm(self._r),
                                                  dL, alp, Gamma=None, chi=None, Delta=None)
        else:
            tra = super(MixICMGaussTurb, self).read_environ(kwargs['restore'], alp,
                                                            filepath=kwargs['restore_path'])
            super(MixICMGaussTurb, self).__init__(tra.EGeV, tra.Bn, tra.psin, tra.nel,
                                                  tra.dL, tra.alp, Gamma=tra.Gamma,
                                                  chi=tra.chi, Delta=tra.Delta)
        return

    @property
    def r(self):
        return self._r

    @property
    def rbounds(self):
        return self._rbounds

    @property
    def Bfield_model(self):
        return self._Bfield_model

    @property
    def nel_model(self):
        return self._nelicm


class MixJet(trans.GammaALPTransfer):
    def __init__(self, alp, source, **kwargs):
        """
        Initialize mixing in the magnetic field of the jet,
        assumed here to be coherent

        Parameters
        ----------
        alp: :py:class:`~gammaALPs.ALP`
            :py:class:`~gammaALPs.ALP` object with ALP parameters

        source: :py:class:`~gammaALPs.Source`
            :py:class:`~gammaALPs.Source` object with source parameters

        EGeV: array-like
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

        nsteps = int(np.ceil( kwargs['alpha'] *
                     np.log(self._Rjet / kwargs['rgam'] ) / np.log(kwargs['sens'])))

        kwargs.setdefault('rbounds', np.logspace(np.log10(kwargs['rgam']), np.log10(self._Rjet), nsteps))
        self._rbounds = kwargs['rbounds']

        self._r = np.sqrt(self._rbounds[1:] * self._rbounds[:-1])
        dL = self._rbounds[1:] - self._rbounds[:-1]

        if kwargs['restore'] is None:
            self._Bfield_model = jet.Bjet(kwargs['B0'], kwargs['r0'],
                                          kwargs['alpha']
                                          )
            B, psi = self._Bfield_model.new_Bn(self._r, psi=kwargs['psi'])

            if kwargs['helical']:
                B, psi = self._Bfield_model.transversal_component_helical(B, psi,
                                                                          theta_jet=self._source.theta_jet,
                                                                          theta_obs=self._source.theta_obs)

            if kwargs['equipartition']:
                kwargs['beta'] = kwargs['alpha'] * 2.

                intp = USpline(np.log10(self._r), np.log10(B), k = 1, s = 0)
                B0 = 10.**intp(np.log10(kwargs['r0']))

                # see e.g.https://arxiv.org/pdf/1307.4100.pdf Eq. 2
                kwargs['n0'] = B0 ** 2. / 8. / np.pi \
                    / kwargs['gamma_min'] / (c.m_e * c.c ** 2.).to('erg').value / \
                    np.log(kwargs['gamma_max'] / kwargs['gamma_min'])

                logging.info("Assuming equipartion at r0: n0(r0) = {0[n0]:.3e} cm^-3".format(kwargs))

            self._neljet = njet.NelJet(kwargs['n0'], kwargs['r0'], kwargs['beta'])

            # init the transfer function with absorption
            super(MixJet, self).__init__(kwargs['EGeV'], B * 1e6, psi, self._neljet(self._r),
                                         dL * 1e-3, alp, Gamma=None, chi=None, Delta=None)

            # transform energies to stationary frame
            self._ee /= self._source._doppler
        else:
            tra = super(MixJet,self).read_environ(kwargs['restore'], alp,
                                                  filepath = kwargs['restore_path'])
            super(MixJet,self).__init__(tra.EGeV, tra.B, tra.psi, tra.nel,
                                        tra.dL, tra.alp, Gamma=tra.Gamma,
                                        chi=tra.chi, Delta=tra.Delta)
        return

    @property
    def r(self):
        return self._r

    @property
    def rbounds(self):
        return self._rbounds

    @property
    def Rjet(self):
        return self._Rjet

    @property
    def Bfield_model(self):
        return self._Bfield_model

    @property
    def nel_model(self):
        return self._neljet

    @Rjet.setter
    def Rjet(self, Rjet):
         if type(Rjet) == u.Quantity:
             self._Rjet= Rjet.to('pc').value
         else:
             self._Rjet = Rjet
         return


class MixJetHelicalTangled(trans.GammaALPTransfer):
    def __init__(self, alp, source, **kwargs):
        """
        Initialize mixing in the magnetic field of the jet

        Parameters
        ----------
        alp: :py:class:`~gammaALPs.ALP`
            :py:class:`~gammaALPs.ALP` object with ALP parameters

        source: :py:class:`~gammaALPs.Source`
            :py:class:`~gammaALPs.Source` object with source parameters

        EGeV: array-like
            Gamma-ray energies in GeV

        restore: str or None
            if str, identifier for files to restore environment.
            If None, initialize mixing with new B field

        restore_path: str
            full path to environment files

        ndom: Number of domains in jet model. (default 400)

        B-Field kwargs:

        ft: float
            fraction of magnetic field energy density in tangled field

        r_T: float
            radius at which helical field becomes toroidal in pc

        Bt_exp: float
            exponent of the transverse component of the helical field
            at r<=r_T. i.e. sin(pitch angle) ~ r^Bt_exp while r<r_T
            and pitch angle = pi/2 at r=r_T

        B0: float
            Bfield strength in G

        r0: float
            radius where B field is equal to b0 in pc

        gmin: float
            jet lorenz factor at rjet

        rvhe:  float
            distance of gamma-ray emission region from BH in pc

        rjet: float
            jet length in pc

        alpha: float
            power-law index of electron energy distribution function

        l_tcor: float
            tangled field coherence average length in pc

        jwf: float
            jet width factor used when calculating l_tcor = jwf*jetwidth

        jwf_dist: string
            type of distribution for jet width factors (jwf) when
            calculating l_tcor with jwf*jetwidth

        tseed: float
            seed for random tangled domains

        Electron density kwargs:

        n0: float
            electron density at R0 in cm**-3 (default 1e3)

        beta: float
            exponent of electron density (default = -2.)

        Jet kwargs:

        g0: float
            jet lorenz factor at r0
        """
        kwargs.setdefault('EGeV', np.logspace(0.,5.,400))
        kwargs.setdefault('restore', None)
        kwargs.setdefault('restore_path', './')
        kwargs.setdefault('ndom', 400)
        # Bfield kwargs
        kwargs.setdefault('ft', 0.3)
        kwargs.setdefault('r_T', 0.3)
        kwargs.setdefault('Bt_exp', -1.)
        kwargs.setdefault('B0', 0.8)
        kwargs.setdefault('r0', 0.3)
        kwargs.setdefault('l_tcor', 0.1)
        kwargs.setdefault('jwf', 1.)
        kwargs.setdefault('jwf_dist', None)
        kwargs.setdefault('tseed', 0)
        # electron density kwargs
        kwargs.setdefault('n0', 5e4)
        kwargs.setdefault('beta', -2.)
        # jet kwargs
        kwargs.setdefault('gmin', 2.)
        kwargs.setdefault('alpha', 1.68)
        kwargs.setdefault('rjet', 3206.3)
        kwargs.setdefault('rvhe', 0.3)

        self._rbounds = np.logspace(np.log10(kwargs['rvhe']),np.log10(kwargs['rjet']),kwargs['ndom'])

        if kwargs['ft'] > 0. and kwargs['l_tcor'] != 'jetdom' and kwargs['l_tcor'] != 'jetwidth':
            while np.average(np.diff(self._rbounds)) > kwargs['l_tcor']:
                kwargs['ndom']+=50
                self._rbounds = np.logspace(np.log10(kwargs['rvhe']), np.log10(kwargs['rjet']), kwargs['ndom'])
            logging.warning("Not enough jet doms to resolve tangled field. Increased to {}".format(kwargs['ndom']))

        self._r = np.sqrt(self._rbounds[1:] * self._rbounds[:-1])
        dL = self._rbounds[1:] - self._rbounds[:-1]

        self._Bfield_model = jet.BjetHelicalTangled(kwargs['ft'],
                                                    kwargs['r_T'],
                                                    kwargs['Bt_exp'],
                                                    kwargs['B0'],
                                                    kwargs['r0'],
                                                    source.bLorentz,
                                                    kwargs['rvhe'],
                                                    kwargs['rjet'],
                                                    kwargs['alpha'],
                                                    kwargs['l_tcor'],
                                                    kwargs['jwf'],
                                                    kwargs['jwf_dist'],
                                                    kwargs['tseed'])

        B, psi = self._Bfield_model.get_jet_props_gen(self._r)

        # change rs if they were not originally resolving the tangled field
        try:
            if self._Bfield_model.trerun:
                try:
                    self._rbounds = self._Bfield_model.newbounds
                except AttributeError:
                    self._rbounds = self._Bfield_model.tdoms
                self._r = np.sqrt(self._rbounds[1:] * self._rbounds[:-1])
                dL = self._rbounds[1:] - self._rbounds[:-1]
        except AttributeError:
            pass

        self._neljet = njet.NelJetHelicalTangled(kwargs['n0'],
                                                 kwargs['rvhe'],
                                                 kwargs['alpha'],
                                                 kwargs['beta'])

        # init the transfer function with absorption
        super(MixJetHelicalTangled, self).__init__(kwargs['EGeV'], B * 1e6, psi, self._neljet(self._r),
                                                   dL * 1e-3, alp, Gamma=None, chi=None, Delta=None)

        # transform energies to stationary frame
        self._gammas = self.jet_gammas_scaled_gg(self._r,
                                                 kwargs['rvhe'],
                                                 kwargs['rjet'],
                                                 kwargs['gmin'],
                                                 source.bLorentz)

        self._ee /= self._gammas

        return

    @property
    def r(self):
        return self._r

    @property
    def rbounds(self):
        return self._rbounds

    @property
    def Rjet(self):
        return self._Rjet

    @property
    def Bfield_model(self):
        return self._Bfield_model

    @property
    def nel_model(self):
        return self._neljet

    @property
    def gammas(self):
        return self._gammas


    @staticmethod
    def jet_gammas_scaled_gg(rs, rvhe, rjet, gmin, gmax):
        """
        Function to get jet lorentz factors. The shape of the gammas
        vs. r from PC Jet model, scaled to r0, gmin, gmax and rjet.
        Jet accelerates in the parabolic base (up to rvhe),
        then logarithmically decelerates in the conical jet.
        """
        gxs = rs
        gz = 4. * (gmax / 9.)
        gmx = 9. * (gmax / 9.)
        gmn = 2. * (gmin / 2.)
        xcon = 0.3 * (rvhe / 0.3)
        L = 3206.3 * (rjet / 3206.3)
        g1 = (gz + ((gmx - gz) / (xcon ** (1. - 0.68))) * gxs**(1. - 0.68)) * (gxs < xcon)
        g2 = (gmx - ((gmx - gmn) / np.log10(L / xcon)) * np.log10(gxs / xcon)) * (gxs >= xcon)
        return g1 + g2


class MixGMF(trans.GammaALPTransfer):
    def __init__(self, alp, source, **kwargs):
        """
        Initialize mixing in the coherent component of the Galactic magnetic field

        Parameters
        ----------
        alp: :py:class:`~gammaALPs.ALP`
            :py:class:`~gammaALPs.ALP` object with ALP parameters

        source: :py:class:`~gammaALPs.Source`
            :py:class:`~gammaALPs.Source` object with source parameters

        EGeV: array-like
            Gamma-ray energies in GeV

        restore: str or None
            if str, identifier for files to restore environment.
            If None, initialize mixing with new B field
        restore_path: str
            full path to environment files
        int_steps: int (default = 100)
            Number of integration steps
        rbounds: array-like
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

        model: str
            (default = jansson)
            GMF model that is used. Currently the model by Jansson & Farrar (2012)
            (also with updates from Planck measurements)
            and Pshirkov et al. (2011) are implemented.
            Usage: model=[jansson12, jansson12b, jansson12c, pshirkov]

        model_sym: str
            (default = ASS)
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
            self._Bgmf = gmf.GMF(model=kwargs['model'])  # Initialize the Bfield class
        elif kwargs['model'] == 'pshirkov':
            self._Bgmf = gmf.GMFPshirkov(model=kwargs['model_sym'])
        else:
            raise ValueError("Unknown GMF model chosen")

        # set coordinates
        self.set_coordinates() # sets self._l, self._b and self._smax

        # step length
        kwargs.setdefault('rbounds' , np.linspace(self._smax,0., kwargs['int_steps'],endpoint = False))
        self._rbounds = kwargs['rbounds']

        self._r = 0.5 * (self._rbounds[1:] + self._rbounds[:-1])

        # use other way round since we are beginning from
        # max distance and propagate to Earth
        dL = self._rbounds[:-1] - self._rbounds[1:]

        # NE2001 code missing!
        self._nelgmf = kwargs['n0'] * np.ones(self._r.shape)

        if kwargs['restore'] is None:
            B, psi = self.Bgmf_calc()

            # init the transfer function with absorption
            super(MixGMF, self).__init__(kwargs['EGeV'], B, psi, self._nelgmf,
                                         dL, alp, Gamma=None, chi=None, Delta=None)
        else:
            tra = super(MixGMF, self).read_environ(kwargs['restore'], alp,
                                                   filepath = kwargs['restore_path'])
            super(MixGMF, self).__init__(tra.EGeV, tra.B, tra.psi, tra.nel,
                                         tra.dL, tra.alp,
                                         Gamma=tra.Gamma,
                                         chi=tra.chi,
                                         Delta=tra.Delta)

    @property
    def galactic(self):
        return self._galactic

    @galactic.setter
    def galactic(self, galactic):
        if type(galactic) == u.Quantity:
            self._galactic = galactic.to('kpc').value
        else:
            self._galactic = galactic
        self._B, self._psi = self.Bgmf_calc()
        return

    @property
    def r(self):
        return self._r

    @property
    def rbounds(self):
        return self._rbounds

    @property
    def Bfield_model(self):
        return self._Bgmf

    @property
    def nel_model(self):
        return self._nelgmf

    def set_coordinates(self):
        """
        Set the coordinates l,b and the the maximum distance smax where |GMF| > 0
        """

        # Transformation RA, DEC -> L,B
        self._l = np.radians(self._source.l)
        self._b = np.radians(self._source.b)
        d = -1. * np.abs(self._Bgmf.Rsun)

        # if source is extragalactic, calculate maximum distance that beam traverses GMF to Earth
        if self.galactic < 0.:
            cl = np.cos(self._l)
            cb = np.cos(self._b)
            sb = np.sin(self._b)
            self._smax = np.amin([self.__zmax/np.abs(sb),
                          1./np.abs(cb) * (-d*cl + np.sqrt(d**2 + cl**2 - d**2*cb + self.__rho_max**2))])
        else:
            self._smax = self._galactic
        return

    def Bgmf_calc(self, l=0., b=0.):
        """
        compute GMF at (s,l,b) position where origin at self.d along x-axis in GC coordinates is assumed

        Parameters
        -----------
        s: array-like
            N-dim array, distance from sun in kpc for all domains

        l: float
            galactic longitude, scalar or N-dim np.array

        b: float
            galactic latitude, scalar or N-dim np.array

        Returns
        -------
        Btrans, Psin: tuple with :py:class:`~numpy.ndarray`
            (3,N)-dim arrays containing GMF for all domains in
            galactocentric cylindrical coordinates (rho, phi, z)
            and N-dim array with angles between propagation direction and line of sight.
        """
        if np.isscalar(l):
            if not l:
                 l = self._l
        if np.isscalar(b):
            if not b:
                 b = self._b

        # compute rho in GC coordinates for s,l,b
        rho = trafo.rho_HC2GC(self._r, l, b, -1. * np.abs(self._Bgmf.Rsun))

        # compute phi in GC coordinates for s,l,b
        phi = trafo.phi_HC2GC(self._r, l, b, -1. * np.abs(self._Bgmf.Rsun))
        z = trafo.z_HC2GC(self._r, b)  # compute z in GC coordinates for s,l,b

        B = self._Bgmf.Bdisk(rho,phi,z)[0]  # add all field components
        B += self._Bgmf.Bhalo(rho,z)[0]
        if self._model.find('jansson') >= 0:
            B += self._Bgmf.BX(rho,z)[0]

        # Single components for debugging ###
        # B = self.Bgmf.Bdisk(rho,phi,z)[0]
        # B = self.Bgmf.Bhalo(rho,z)[0]
        # B = self.Bgmf.BX(rho,z)[0]

        Babs = np.sqrt(np.sum(B**2., axis=0))         # compute overall field strength
        # Bs, Bt, Bu         = trafo.GC2HCproj(B, self._r, self._l, self._b, d = -1. * np.abs(self._Bgmf.Rsun))
        # TODO: what is correct for the Pshirkov model?
        Bs, Bt, Bu = trafo.GC2HCproj(B, self._r, self._l, self._b, d = self._Bgmf.Rsun)

        Btrans = np.sqrt(Bt**2. + Bu**2.)         # Abs value of transverse component in all domains
        Psin = np.arctan2(Bt, Bu)         # arctan2 selects the right quadrant

        return Btrans, Psin


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
        alp: :py:class:`~gammaALPs.ALP`
            :py:class:`~gammaALPs.ALP` object with ALP parameters

        filename: str
            full path to file with electron density and B field

        EGeV: array-like
            Gamma-ray energies in GeV

        restore: str or None
            if str, identifier for files to restore environment.
            If None, initialize mixing with new B field

        restore_path: str
            full path to environment files

        """
        kwargs.setdefault('EGeV', np.logspace(0., 4., 100))
        kwargs.setdefault('restore', None)
        kwargs.setdefault('restore_path', './')

        data = np.loadtxt(filename)
        self._rbounds = data[:, 0]
        self._r = 0.5 * (self._rbounds[1:] + self._rbounds[:-1])
        dL = self._rbounds[1:] - self._rbounds[:-1]

        n = 0.5 * (data[1:,1] + data[:-1,1])
        Bx = 0.5 * (data[1:,3] + data[:-1,3])
        By = 0.5 * (data[1:,4] + data[:-1,4])
        Bz = 0.5 * (data[1:,5] + data[:-1,5])
        self._T = 0.5 * (data[1:,2] + data[:-1,2])

        Btrans = np.sqrt(Bx**2. + By**2.)
        psi = np.arctan2(Bx,By)

        if kwargs['restore'] is None:

            # init the transfer function
            super(MixFromFile, self).__init__(kwargs['EGeV'], Btrans, psi, n,
                                                     dL, alp, Gamma = None, chi = None, Delta = None)
        else:
            tra = super(MixFromFile, self).read_environ(kwargs['restore'], alp,
                                                        filepath=kwargs['restore_path'])

            super(MixFromFile, self).__init__(tra.EGeV, tra.Bn, tra.psin, tra.nel,
                                              tra.dL, tra.alp,
                                              Gamma=tra.Gamma,
                                              chi=tra.chi,
                                              Delta=tra.Delta)


class MixFromArray(trans.GammaALPTransfer):
    def __init__(self, alp, Btrans, psi, nel, dL, **kwargs):
        """
        Initialize mixing in environment given by numpy arrays

        Parameters
        ----------
        alp: :py:class:`~gammaALPs.ALP`
            :py:class:`~gammaALPs.ALP` object with ALP parameters

        Btrans: array-like
            n-dim or m x n-dim array with absolute value of transversal B field, in muG
            if m x n-dimensional, m realizations are assumed

        psi: array-like
            n-dim or m x n-dim array with angles between transversal direction and one polarization,
            if m x n-dimensional, m realizations are assumed

        nel: array-like
            n-dim or m x n-dim array electron density, in cm^-3,
            if m x n-dimensional, m realizations are assumed

        dL: array-like
            n-dim array with bin widths along line of sight in kpc,
            if m x n-dimensional, m realizations are assumed

        EGeV: array-like
            Gamma-ray energies in GeV

        restore: str or None
            if str, identifier for files to restore environment.
            If None, initialize mixing with new B field

        restore_path: str
            full path to environment files

        """
        kwargs.setdefault('EGeV', np.logspace(0., 4., 100))
        kwargs.setdefault('restore', None)
        kwargs.setdefault('restore_path', './')

        if kwargs['restore'] is None:

            # init the transfer function
            super(MixFromArray, self).__init__(kwargs['EGeV'], Btrans, psi, nel,
                                               dL, alp,
                                               Gamma=None,
                                               chi=None,
                                               Delta=None)
        else:
            tra = super(MixFromArray, self).read_environ(kwargs['restore'], alp,
                                                        filepath=kwargs['restore_path'])
            super(MixFromArray, self).__init__(tra.EGeV, tra.Bn, tra.psin, tra.nel,
                                               tra.dL, tra.alp,
                                               Gamma=tra.Gamma,
                                               chi=tra.chi,
                                               Delta=tra.Delta)
