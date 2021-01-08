# --- Imports --------------------- #
from __future__ import absolute_import, division, print_function
import numpy as np
import scipy.special as special
import logging
from scipy.interpolate import UnivariateSpline as USpline
from astropy import units as u
from scipy.special import jv
# --------------------------------- #


# ========================================================== #
# === B field of AGN jet assuming a power-law toroidal field #
# ========================================================== #
class Bjet(object):
    """Class to calculate magnetic field in AGN Jet assuming a toroidal field"""
    def __init__(self, B0, r0, alpha):
        """
        Initialize the class

        Parameters
        ----------
        B0: float
            B field strength in G
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

    def new_Bn(self, z, psi=np.pi / 4.):
        """
        Calculate the magnetic field as function of distance

        Parameters
        ----------
        z: array-like
            n-dim array with distance from r0 in pc

        psi: float
            angle between transversal magnetic field and x2 direction. Default: pi/4

        Returns
        -------
        B, Psi: tuple with :py:class:`numpy.ndarray`
            N-dim array with field strength along line of sight
            N-dim array with psi angles between photon polarization states
            and jet B field
        """
        B = self._B0 * np.power(z / self._r0, self._alpha)
        psi = np.ones(B.shape[0]) * psi
        return B, psi

    @staticmethod
    def transversal_component_helical(B0, phi, theta_jet=3., theta_obs=0.):
        """
        compute Jet magnetic field along line of sight that
        forms observation angle theta_obs with jet axis.
        Model assumes the helical jet structure of
        Clausen-Brown, E., Lyutikov, M., and Kharb, P. (2011); arXiv:1101.5149

        Parameters
        -----------
        B0: array-like
            N-dim array with magnetic field strength along jet axis

        phi: float
            phi angle in degrees along which photons propagate along jet
            (in cylindrical jet geometry)

        theta_jet: float
            jet opening angle in degrees

        theta_obs: float
            angle between jet axis and line of sight in degrees

        Returns
        -------
        Btrans, Psi: tuple with :py:class:`numpy.ndarray`
            N-dim array with field strength along line of sight
            N-dim array with psi angles between photon polarization states
            and jet B field
        """
        # Calculate normalized rho component, i.e. distance
        # from line of sight to jet axis assuming a self similar jet
        p, tj, to = np.radians(phi), np.radians(theta_jet), np.radians(theta_obs)

        rho_n = np.tan(to) / np.tan(tj)
        k = 2.405  # pinch, set so that Bz = 0 at jet boundary

        # compute bessel functions, see Clausen-Brown Eq. 2
        j0 = jv(0., rho_n * k)
        j1 = jv(1., rho_n * k)

        # B-field along l.o.s.
        Bn = np.cos(to) * j0 - np.sin(p)*np.sin(to) * j1
        # B-field transversal to l.o.s.
        Bt = np.cos(p) * j1
        Bu = -(np.cos(to) * np.sin(p) * j1 + np.sin(to) * j0)

        Btrans = B0 * np.sqrt(Bt**2. + Bu**2.)         # Abs value of transverse component in all domains
        Psin = np.arctan2(B0*Bt,B0*Bu)         # arctan2 selects the right quadrant

        return Btrans, Psin


class BjetHelicalTangled(object):
    """
    Class to calculate magnetic field in AGN Jet assuming a two component field:
       1. A helical component transforming from poloidal to toroidal
       2. A tangled component
    """
    def __init__(self, ft, r_T, Bt_exp, B0, r0, g0, rvhe, rjet, alpha, l_tcor, jwf, jwf_dist, tseed):
        """
        Initialize the class

        Parameters
        ----------
        ft: float
            fraction of magnetic field energy density in tangled field
        r_T: float
            radius at which helical field becomes toroidal in pc
        Bt_exp: float
            exponent of the transverse component of the helical field
            at r<=r_T. i.e. sin(pitch angle) ~ r^Bt_exp while r<r_T
            and pitch angle = pi/2 at r=r_T
        B0: float
            B-field strength in G
        r0: float
            radius where B field is equal to b0 in pc
        g0: float
            jet lorenz factor at r0
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
            calculating l_tcor = jwf*jetwidth
        tseed: float
            seed for random tangled domains
        """
        self._ft = ft
        self._r_T = r_T
        self._Bt_exp = Bt_exp
        self._B0 = B0
        self._r0 = r0
        self._g0 = g0
        self._rvhe = rvhe
        self._rjet = rjet
        self._alpha = alpha
        self._l_tcor = l_tcor
        self._jwf = jwf
        self._jwf_dist = jwf_dist
        self._tseed = tseed
        self._tthes = None
        self._tphis = None
        self._trerun = None
        self._newbounds = None
        return

    @property
    def ft(self):
        return self._ft

    @property
    def r_T(self):
        return self._r_T

    @property
    def Bt_exp(self):
        return self._Bt_exp

    @property
    def B0(self):
        return self._B0

    @property
    def r0(self):
        return self._r0

    @property
    def g0(self):
        return self._g0

    @property
    def rvhe(self):
        return self._rvhe

    @property
    def rjet(self):
        return self._rjet

    @property
    def alpha(self):
        return self._alpha

    @property
    def l_tcor(self):
        return self._l_tcor

    @property
    def jwf(self):
        return self._jwf

    @property
    def jwf_dist(self):
        return self._jwf_dist

    @property
    def tseed(self):
        return self._tseed

    @property
    def tthes(self):
        return self._tthes

    @property
    def tphis(self):
        return self._tphis

    @property
    def trerun(self):
        return self._trerun

    @property
    def newbounds(self):
        return self._newbounds

    @ft.setter
    def ft(self, ft):
        self._ft = ft
        return

    @r_T.setter
    def r_T(self,r_T):
        if type(r_T) == u.Quantity:
            self._r_T = r_T .to('pc').value
        else:
            self._r_T = r_T
        return

    @Bt_exp.setter
    def Bt_exp(self, Bt_exp):
        self._Bt_exp = Bt_exp
        return

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

    @g0.setter
    def g0(self, g0):
        self._g0 = g0
        return

    @rvhe.setter
    def rvhe(self, rvhe):
        if type(rvhe) == u.Quantity:
            self._rvhe = rvhe .to('pc').value
        else:
            self._rvhe = rvhe
        return

    @rjet.setter
    def rjet(self,rjet):
        if type(rjet) == u.Quantity:
            self._rjet = rjet .to('pc').value
        else:
            self._rjet = rjet
        return

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha
        return

    @l_tcor.setter
    def l_tcor(self,l_tcor):
        if type(l_tcor) == u.Quantity:
            self._l_tcor = l_tcor .to('pc').value
        else:
            self._l_tcor = l_tcor
        return

    @jwf.setter
    def jwf(self, jwf):
        self._jwf = jwf
        return

    @jwf_dist.setter
    def jwf_dist(self, jwf_dist):
        self._jwf_dist = jwf_dist
        return

    @tseed.setter
    def tseed(self, tseed):
        self._tseed = tseed
        return

    def jet_bfield_scaled_old(self,rs,r0,b0):
        """
        Function to get jet B-field strength. Shape of function (defined by the constants)
        from PC Jet model, scaled to r0 and B0.
        """
        xs=rs
        tr1 = np.log10((0.104778867386/0.3)*r0)
        tr2 = np.log10((0.763434306576/0.3)*r0)
        tt1 = np.log10((0.0656583839948/0.3)*r0)
        tt2 = np.log10((0.312675309121/0.3)*r0)
        sc1 = 1.35770127215
        sc2 = 2.9067727141
        st1 = 1.32933857554
        st2 = 9.99999939106
        f = 0.815193652746
        bs = 0.06*(xs/r0)**-0.85
        bps = 1.2*(xs/r0)**-0.68
        btr = .75*(xs/r0)**-3. * (xs>=(0.27/0.3)*r0) + ((xs<=(0.27/0.3)*r0)*bps*f)
        b_erftr = 0.5*special.erfc(-st1*(np.log10(xs)-tt1))*btr*0.5*special.erfc(st2*(np.log10(xs)-tt2))
        b_erfcs = 0.5*special.erfc(sc1*(np.log10(xs)-tr1))*bps + b_erftr + 0.5*special.erfc(-sc2*(np.log10(xs)-tr2))*bs
        b_erfcs *= (b0/0.8)
        return b_erfcs

    def jet_bfield_scaled(self,rs,rvhe,r0,b0):
        """
        Function to get jet B-field strength. The function is an analytic
        approximation, defined by the constants, to the shape of the B-field
        vs. r from PC Jet model, scaled to rvhe. Strength scaled to r0 and B0.
        """
        xs=rs
        tr1 = np.log10((0.104778867386/0.3)*rvhe)
        tr2 = np.log10((0.763434306576/0.3)*rvhe)
        tt1 = np.log10((0.0656583839948/0.3)*rvhe)
        tt2 = np.log10((0.312675309121/0.3)*rvhe)
        sc1 = 1.35770127215
        sc2 = 2.9067727141
        st1 = 1.32933857554
        st2 = 9.99999939106
        f = 0.815193652746
        bs = 0.06*(xs/rvhe)**-0.85
        bps = 1.2*(xs/rvhe)**-0.68
        btr = .75*(xs/rvhe)**-3. * (xs>=(0.27/0.3)*rvhe) + ((xs<=(0.27/0.3)*rvhe)*bps*f)
        b_erftr = 0.5*special.erfc(-st1*(np.log10(xs)-tt1))*btr*0.5*special.erfc(st2*(np.log10(xs)-tt2))
        b_erfcs = 0.5*special.erfc(sc1*(np.log10(xs)-tr1))*bps + b_erftr + 0.5*special.erfc(-sc2*(np.log10(xs)-tr2))*bs
        #find b_erfcs(r_0) to find scaling factor to make it b0
        interp = USpline(xs,b_erfcs,k=1,s=0)
        ber_r0 = interp(r0)
        b_erfcs *= (b0/ber_r0)
        return b_erfcs

    def jet_gammas_scaled(self,rs,r0,g0,rjet):
        """
        Function to get jet lorentz factors. The shape of the gammas
        vs. r from PC Jet model, scaled to r0, g0 and rjet.
        Jet accelerates in the parabolic base (up to rvhe),
        then logarithmically decelerates in the conical jet.
        """
        gxs = rs
        gz = 4. * (g0/9.)
        gmx = 9. * (g0/9.)
        gmn = 2. * (g0/9.)
        xcon = 0.3 * (r0/0.3)
        L = 3206.3 * (rjet/3206.3)
        g1 = (gz + ((gmx - gz)/(xcon**(1-0.68)))* gxs**(1-0.68)) * (gxs<xcon)
        g2 = (gmx - ((gmx-gmn)/np.log10(L/xcon))*np.log10(gxs/xcon)) * (gxs>=xcon)
        return g1+g2

    def get_jet_props_gen(self, z, tdoms_done=False):
        """
        Calculate the magnetic field as function of distance in the jet frame

        Parameters
        ----------
        z: array-like
            n-dim array with distance from BH in pc

        Returns
        -------
        B, Psi: tuple with :py:class:`numpy.ndarray`
            N-dim array with field strength in G along line of sight
            N-dim array with psi angles between photon polarization states
            and jet B field
        """
        # t1 = time.time()
        # get Bs from PC shape function
        Bs = self.jet_bfield_scaled(z,self._rvhe,self._r0,self._B0)  # r0 and rvhe in pc, b0 in G
        gammas = self.jet_gammas_scaled(z,self._r0,self._g0,self._rjet)

        if not tdoms_done:
            # t2 = time.time()
            if self._ft > 0 and self._l_tcor != 'jetdom' and self._l_tcor != 'jetwidth':
                # tangled domains
                d = z[0]
                self._tdoms=[]
                tdom_seeds = np.arange(6007+self._tseed,6007+(2*len(z))+self._tseed,1)
                np.random.seed(tdom_seeds)
                while d <= z[-1]:
                    self._tdoms.append(d)
                    d += np.random.uniform(self._l_tcor/20.,self._l_tcor*20.)
                self._tdoms = np.array(self._tdoms)

            elif self._l_tcor == 'jetwidth': #self._ft > 0 and

                theta_m1_interp = USpline(np.log10(z),gammas)

                p = lambda r,rsw,c,a: c*(rsw+r)**a
                con = lambda r,rvhe,the: np.tan(the)*(r-rvhe)

                rsw = 1.e-5 * self._rvhe
                C = 1.49*rsw**0.42
                A = 0.58

                self._tdoms=[self._rvhe]

                jwf_seeds = np.arange(4000+self._tseed,4000+1000+self._tseed,1)
                np.random.seed(jwf_seeds)

                while self._tdoms[-1]<= z[-1]:  # tdoms in pc here

                    if self._jwf_dist == 'Uniform':
                        self._jwf = np.random.uniform(0.1,1.)
                    elif self._jwf_dist == 'Normal':
                        jwft = 0.
                        while jwft<0.1 or jwft>1.:
                            self._jwf = np.random.normal(0.55,0.15)
                            jwft = self._jwf
                    elif self._jwf_dist == 'Triangular Rise':
                        self._jwf = np.random.triangular(0.1,1.,1.)
                    elif self._jwf_dist == 'Triangular Lower':
                        self._jwf = np.random.triangular(0.1,0.1,1.)

                    theta = 1./theta_m1_interp(np.log10(self._tdoms[-1]))
                    self._tdoms.append(self._tdoms[-1] + self._jwf*(p(self._rvhe, rsw, C, A)
                                       + con(self._tdoms[-1], self._rvhe, theta)))  # doms length is jetwidth

                self._tdoms = np.array(self._tdoms)

                if len(self._tdoms)-1 > len(z) or min(np.diff(z))>min(np.diff(self._tdoms)):
                    logging.warning("Not resolving tangled field: min z step is {}"
                                    "pc but min tangled length is {} pc".format(
                                        min(np.diff(z)),min(np.diff(self._tdoms))))
                    logging.warning("# of z doms is {} but # tangled doms is {}".format(len(z), len(self._tdoms)))
                    self._trerun = True

                    if len(self._tdoms)-1 > len(z):
                        logging.info("rerunning with r = tdoms")
                        return self.get_jet_props_gen(np.sqrt(self._tdoms[1:] * self._tdoms[:-1]), tdoms_done=True)
                    else:
                        self._newbounds = self._tdoms
                        while len(self._newbounds) <= 400:
                            btwn = (self._newbounds[1:] + self._newbounds[:-1])/2.
                            self._newbounds = np.sort(np.concatenate((self._newbounds, btwn)))

                        logging.info("rerunning with {} domains. new min z step is {} pc".format(
                              len(self._newbounds), min(np.diff(self._newbounds))))

                        return self.get_jet_props_gen(np.sqrt(self._newbounds[1:] * self._newbounds[:-1]),
                                                      tdoms_done=True)

            else:
                self._tdoms = z

        # set up tangled field angles
        tthe_seeds = np.arange(0+self._tseed,len(self._tdoms)+self._tseed,1)
        tphi_seeds = np.arange(1007+self._tseed,1007+len(self._tdoms)+self._tseed,1)
        np.random.seed(tthe_seeds)
        self._tthes = np.random.random(size=len(self._tdoms))*np.pi/2.
        np.random.seed(tphi_seeds)
        self._tphis = np.random.random(size=len(self._tdoms))*2.*np.pi

        BTs, phis = [], []

        # t4 = time.time()

        BhelrT = Bs[np.argmin([abs(ll-self._r_T) for ll in z])] * np.sqrt(1. - self._ft)

        for i, l in enumerate(z):
            # if i == int(len(z)/2):
                # t6 = time.time()
            B = Bs[i]  # Gauss
            B_tang = B * np.sqrt(self._ft)
            B_hel = B * np.sqrt(1. - self._ft)

            h_phi = np.pi/2. #just align helix phi with one axis, why not?
            if l <= self._r_T: #Set section size
                # B_hel *= BhelrT*(l/self._r_T)**(self._Bt_exp + 1.) #make B_hel go like Bt_exp (make 1 '-a')

                fact = (BhelrT*(l/self._r_T)**(self._Bt_exp))/B_hel
                B_hel *= np.where(fact>1.,1.,fact)

            # if i == int(len(z)/2):
                # t7 = time.time()

            # tangled field angles
            if self._ft > 0. and len(z) != len(self._tdoms)-1:
                # could probably be faster
                t_phi = self._tphis[np.argmin([l-tl for tl in self._tdoms if l>=tl])]
                t_the = self._tthes[np.argmin([l-tl for tl in self._tdoms if l>=tl])]
            elif self._l_tcor == 'jetwidth' and len(z)!=len(self._tdoms)-1:
                #could probably be faster
                t_phi = self._tphis[np.argmin([l-tl for tl in self._tdoms if l>=tl])]
                t_the = self._tthes[np.argmin([l-tl for tl in self._tdoms if l>=tl])]
            else:

                t_phi = self._tphis[i]
                t_the = self._tthes[i]

            # if i == int(len(z)/2):
            #     t8 = time.time()
            #     print("getting tangled angles for 1 domain out of {} took {}s".format(len(z),t8-t7))



            h_phi = np.pi/2.
            h_the = np.pi/2. # in r-05 section theta is included in B_fact term ^ otherwise = pi/2 by def
            B_tang_t = B_tang * np.sin(t_the)
            B_hel_t = B_hel * np.sin(h_the)

            Bt_x = np.sqrt((B_hel_t*np.cos(h_phi))**2 + (B_tang_t*np.cos(t_phi))**2)
            Bt_y = np.sqrt((B_hel_t*np.sin(h_phi))**2 + (B_tang_t*np.sin(t_phi))**2)

            phi = np.arctan(Bt_y/Bt_x)
            Bt = np.sqrt(Bt_x**2 + Bt_y**2)

            BTs.append(Bt)
            phis.append(phi)
            # if i == int(len(z)/2):
            #     t9 = time.time()
            #     print("total BT for 1 domain out of {} took {}s".format(len(z),t9-t6))
            # if abs(l - 1.) <= 1.e-2:
            #     print("BT at around 1 pc: {} pc {} G".format(l,Bt))

        # t5 = time.time()
        # print("Calculating BTs took {}s".format(t5-t4))
        # print("this run through the get_jet_props function took {}s".format(t5-t1))
        return np.array(BTs), np.array(phis)
