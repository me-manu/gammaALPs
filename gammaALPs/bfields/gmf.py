# --- Imports --------------------- #
from __future__ import absolute_import, division, print_function
import numpy as np
import warnings
# --------------------------------- #
pi = np.pi


def signum(x):
    """Return the sign of each entry of an array"""
    return (x < 0.) * -1. + (x >= 0) * 1.


def sigmoid(x, x0, w):
    """
    Calculates the logistic sigmoid function.
    Arguments:
    x : float
        The input value
    x0 : float
        The midpoint of the sigmoid
    w : float
        The width of the sigmoid
    
    Returns:
    The result of the sigmoid function
    """
    return 1 / (1 + np.exp(-(x - x0) / w))


def delta_phi(phi0, phi1):
    """
    Calculates the angle between v0 = (cos(phi0), sin(phi0)) and v1 = (cos(phi1), sin(phi1)).
    Arguments:
    phi0 : float
        First angle in radians
    phi1 : float
        Second angle in radians
    
    Returns:
    The angle between v0 and v1 in radians.
    """
    return np.arccos(np.cos(phi1) * np.cos(phi0) + np.sin(phi1) * np.sin(phi0))


class GMF(object):
    """
    Class with analytical functions that describe the 
    galactic magnetic field according to the model of Jannson & Farrar (2012)

    Only the regular field components are implemented. 
    The striated field component is absent.

    Attributes
    ----------
    Rsun: float
        Assumed position of the sun along the x axis in kpc

    bring: float
        field strength in ring at 3 kpc < rho < 5 kpc

    bring_unc: float
        associated uncertainty of field strength in ring at 3 kpc < rho < 5 kpc

    hdisk: float hdisk_unc: float,
        disk/halo transition height

    hdisk_unc: float
        associated uncertainty disk/halo transition height

    wdisk: float
        transition width

    wdisk_unc: float, wdisk_unc: floats,
        associated uncertainty of transition width

    b: :py:class:`~numpy.ndarray`
        arrays with field strengths of spiral arms at 5 kpc

    b_unc: :py:class:`~numpy.ndarray`
        associated uncertainties with field strengths of spiral arms at 5 kpc

    f_cov: :py:class:`~numpy.ndarray`
        relative cross-sectional areas of the spirals (for a fixed radius)

    rx: :py:class:`~numpy.ndarray`
        dividing lines of spiral arms, coordinates of neg. x-axes that intersect with arm

    idisk: float
        spiral arms opening angle

    bn, bn_unc: float
        field strength northern halo

    bn_unc: float
        associated uncertainty of field strength northern halo

    Bs: float
        field strength southern halo

    Bs_unc: float
        associated uncertainty of field strength southern halo

    rhon: float
        transition radius in northern hemisphere

    rhon_unc: float
        associated uncertainty of transition radius in northern hemisphere

    rhos: float
        transition radius in southern heisphere

    rhos_unc: float
        associated uncertainty of transition radius in southern hemisphere

    whalo: float
        transition width

    whalo_unc: float
        associated uncertainty of transition width

    z0: float
        vertical scale height

    z0_unc: float
        associated uncertainty of vertical scale height

    BX0: float
        field strength at origin of X component

    BX0_unc: float
        associated uncertainty of field strength at origin of X component

    ThetaX0: float
        elev. angle at z = 0, rho > rhoXc

    ThetaX0_unc: float
        associated uncertainty of elev. angle at z = 0, rho > rhoXc

    rhoXc: float
        radius where thetaX = thetaX0

    rhoXc_unc: float
        associated uncertainty of radius where thetaX = thetaX0

    rhoX: float,
        exponential scale length

    rhoX_unc: float,
        associated uncertainty of exponential scale length

    gamma: float
        striation and / or rel. elec. number dens. rescaling

    gamma_unc: float
        associated uncertainty striation and / or rel. elec. number dens. rescaling

    Notes
    -----
    see http://adsabs.harvard.edu/abs/2012ApJ...757...14J
    Jansson & Farrar (2012)
    """

    def __init__(self, model='jansson12'):
        """
        Init the GMF class, all B-field values are in muG
        
        Parameters
        ----------
        model: str
            either jansson12, jansson12b, or jansson12c, where jansson12 is the original model
            and the other two options are the modifications of the model with Planck data,
            see http://arxiv.org/abs/1601.00546
        """

        self.Rsun = -8.5  # position of the sun in kpc
        # Best fit values, see Table 1 of Jansson & Farrar
        # Disk
        self.bring, self.bring_unc = 0.1, 0.1  # ring at 3 kpc < rho < 5 kpc
        self.hdisk, self.hdisk_unc = 0.4, 0.03  # disk/halo transition
        self.wdisk, self.wdisk_unc = 0.27, 0.08  # transition width
        self.b = np.array([0.1, 3., -0.9, -0.8, -2.0, -4.2, 0., np.nan])  # field strength of spiral arms at 5 kpc
        self.b_unc = np.array([1.8, 0.6, 0.8, 0.3, 0.1, 0.5, 1.8, 1.8])
        self.f_cov = np.array([0.130, 0.165, 0.094, 0.122, 0.13, 0.118, 0.084, 0.156])
        self.rx = np.array([5.1, 6.3, 7.1, 8.3, 9.8, 11.4, 12.7, 15.5])  # dividing lines of spiral lines
        self.idisk = 11.5 * pi/180.  # spiral arms opening angle
        # Halo
        self.Bn, self.Bn_unc = 1.4, 0.1  # northern halo
        self.Bs, self.Bs_unc = -1.1, 0.1  # southern halo
        self.rhon, self.rhon_unc = 9.22, 0.08  # transition radius north
        self.rhos, self.rhos_unc = 16.7, 0.  # transition radius south, lower limit
        self.whalo, self.whalo_unc = 0.2, 0.12  # transition width
        self.z0, self.z0_unc = 5.3, 1.6  # vertical scale height
        # Out of plaxe or "X" component
        self.BX0, self.BX_unc = 4.6,0.3  # field strength at origin
        self.ThetaX0, self.ThetaX0_unc = 49. * pi/180., pi/180.  # elev. angle at z = 0, rho > rhoXc
        self.rhoXc, self.rhoXc_unc = 4.8, 0.2  # radius where thetaX = thetaX0
        self.rhoX, self.rhoX_unc = 2.9, 0.1  # exponential scale length
        # striated field
        self.gamma, self.gamma_unc = 2.92, 0.14  # striation and / or rel. elec. number dens. rescaling

        # updates from planck, however, see caveats of that paper, http://arxiv.org/abs/1601.00546
        if model == 'jansson12b':
            self.b[5] = -3.5
            self.BX0 = 1.8
        if model == 'jansson12c':
            self.Bn = 1.
            self.Bs = -0.8
            self.BX0 = 3.
            self.b[1], self.b[3], self.b[4] = 2., 2., -3.

        # calculate the magnetic field of the 8th spiral arm,
        # see Jansson & Farrar Sec. 5.1.1.
        self.b[-1] = - np.sum(self.f_cov[:-1] * self.b[:-1]) / self.f_cov[-1]

    def L(self, z, h, w):
        """
        Transition function, see Jansson & Farrar Eq. 5

        Parameters
        ----------
        z: float or array-like
            array with positions (height above disk, z; distance from center, rho)
        h: float
            height parameter
        w: float
            width parameter

        Returns
        -------
        L: :py:class:`~numpy.ndarray`
            array or float (depending on z input) with transition function values
        """
        if np.isscalar(z):
            z = np.array([z])
        ones = np.ones(z.shape[0])
        L = np.squeeze(1./(ones + np.exp(-2. * (np.abs(z) - h) / w)))
        return  L

    def r_log_spiral(self, phi):
        """
        return distance from center for angle phi of logarithmic spiral

        Parameters
        ----------
        phi: scalar or array-like
            polar angle values

        Returns
        -------
        result: :py:class:`~numpy.ndarray`
            logarithmic spiral result with r(phi) = rx * exp(b * phi) as an array

        Notes
        -----
        see http://en.wikipedia.org/wiki/Logarithmic_spiral
        """
        if np.isscalar(phi):
            phi = np.array([phi])
        ones = np.ones(phi.shape[0])

        # self.rx.shape = 8
        # phi.shape = p
        # then result is given as (8,p)-dim array, each row stands for one rx

        result = np.tensordot(self.rx, np.exp((phi - 3.*pi*ones) / np.tan(pi/2. - self.idisk)),
                              axes=0)

        result = np.vstack((result,
                            np.tensordot(self.rx, np.exp((phi - pi*ones) / np.tan(pi/2. - self.idisk)),
                                         axes=0)))
        result = np.vstack((result,
                            np.tensordot(self.rx, np.exp((phi + pi*ones) / np.tan(pi/2. - self.idisk)),
                                         axes=0)))
        result = np.vstack((result,
                            np.tensordot(self.rx, np.exp((phi + 3.*pi*ones) / np.tan(pi/2. - self.idisk)),
                            axes=0)))
        return result

    def Bdisk(self, rho, phi, z):
        """
        Disk component of galactic magnetic field 
        in galactocentric cylindrical coordinates (rho,phi,z)

        Parameters
        ----------
        rho: array-like
            N-dim array with distance from origin in GC cylindrical coordinates, is in kpc
        z: array-like
            N-dim array with height in kpc in GC cylindrical coordinates
        phi: array-like
            N-dim array with polar angle in GC cylindircal coordinates, in radian

        Returns
        -------
        Bdisk, Bdisk_abs: tuple of :py:class:`~numpy.ndarray`
            tuple containing the magnetic field of the disk as a (3,N)-dim array with (rho,phi,z)
            components of disk field for each coordinate tuple and absolute value of the field as
            N-dim array
        """
        if (not rho.shape[0] == phi.shape[0]) and (not z.shape[0] == phi.shape[0]):
            warnings.warn("List do not have equal shape!", RuntimeWarning)
            raise ValueError

        # Bdisk vector in rho, phi, z
        # rows: rho, phi and z component
        Bdisk = np.zeros((3, rho.shape[0]))

        ones = np.ones(rho.shape[0])

        m_center = (rho >= 3.) & (rho < 5.)
        m_disk = (rho >= 5.) & (rho <= 20.)

        Bdisk[1, m_center] = self.bring

        # Determine in which arm we are
        # this is done for each coordinate individually, possible to convert into array task?
        if np.sum(m_disk):
            rls = self.r_log_spiral(phi[m_disk])

            rls = rls - rho[m_disk]
            rls[rls < 0.] = 1e10 * np.ones(np.sum(rls < 0.))
            narm = np.argmin(rls, axis=0) % 8

            Bdisk[0, m_disk] = np.sin(self.idisk) * self.b[narm] * (5. / rho[m_disk])
            Bdisk[1, m_disk] = np.cos(self.idisk) * self.b[narm] * (5. / rho[m_disk])

        Bdisk *= (ones - self.L(z, self.hdisk, self.wdisk))

        return Bdisk, np.sqrt(np.sum(Bdisk**2., axis=0))

    def Bhalo(self, rho, z):
        """
        Halo component of galactic magnetic field 
        in galactocentric cylindrical coordinates (rho,phi,z)

        Bhalo is purely azimuthal (toroidal), i.e. has only a phi component

        Parameters
        ----------
        rho: array-like
            N-dim array with distance from origin in GC cylindrical coordinates, is in kpc
        z: array-like
            N-dim array with height in kpc in GC cylindrical coordinates

        Returns
        -------
        Bhalo, Bhalo_abs: tuple of :py:class:`~numpy.ndarray`
            tuple containing the magnetic field of the halo as a (3,N)-dim array with (rho,phi,z)
            components of disk field for each coordinate tuple and absolute value of the field as
            N-dim array
        """

        if not rho.shape[0] == z.shape[0]:
            raise ValueError("List do not have equal shape! returning -1")

        # Bhalo vector in rho, phi, z
        # rows: rho, phi and z component
        Bhalo = np.zeros((3,rho.shape[0]))

        ones = np.ones(rho.shape[0])
        m = (z != 0.)

        Bhalo[1, m] = np.exp(-np.abs(z[m])/self.z0) * self.L(z[m], self.hdisk, self.wdisk) * \
                            (self.Bn * (ones[m] - self.L(rho[m], self.rhon, self.whalo)) * (z[m] > 0.)
                             + self.Bs * (ones[m] - self.L(rho[m], self.rhos, self.whalo)) * (z[m] < 0.))

        return Bhalo, np.sqrt(np.sum(Bhalo**2., axis=0))

    def BX(self, rho, z):
        """
        X (out of plane) component of galactic magnetic field 
        in galactocentric cylindrical coordinates (rho,phi,z)

        BX is purely poloidal, i.e. phi component = 0

        Parameters
        ----------
        rho: array-like
            N-dim array with distance from origin in GC cylindrical coordinates, is in kpc
        z: array-like
            N-dim array with height in kpc in GC cylindrical coordinates

        Returns
        -------
        BX, BX_abs: tuple of :py:class:`~numpy.ndarray`
            tuple containing the magnetic field of the X component as a (3,N)-dim array with (rho,phi,z)
            components of disk field for each coordinate tuple and absolute value of the field as
            N-dim array
        """

        if (not rho.shape[0] == z.shape[0]):
            warnings.warn("List do not have equal shape! returning -1", RuntimeWarning)
            return -1

        # BX vector in rho, phi, z
        # rows: rho, phi and z component
        BX = np.zeros((3, rho.shape[0]))
        m = np.sqrt(rho**2. + z**2.) >= 1.

        bx = lambda rho_p: self.BX0 * np.exp(-rho_p / self.rhoX)
        #tx = lambda rho, z, rho_p: np.arctan2(np.abs(z), (rho - rho_p))
        tx = lambda rho_p: np.arctan2(self.rhoXc * np.tan(self.ThetaX0),
                                      rho_p)  # Another experssion that handle the z=0 (rho=rho_p) case

        rho_p = rho[m] * self.rhoXc/(self.rhoXc + np.abs(z[m] ) / np.tan(self.ThetaX0))

        m_rho_b = rho_p > self.rhoXc  # region with constant elevation angle
        m_rho_l = rho_p <= self.rhoXc  # region with varying elevation angle

        theta = np.zeros(z[m].shape[0])
        b = np.zeros(z[m].shape[0])

        rho_p0 = (rho[m])[m_rho_b] - np.abs( (z[m])[m_rho_b] ) / np.tan(self.ThetaX0)
        b[m_rho_b] = bx(rho_p0) * rho_p0 / (rho[m])[m_rho_b]
        theta[m_rho_b] = self.ThetaX0 * np.ones(theta.shape[0])[m_rho_b]

        b[m_rho_l] = bx(rho_p[m_rho_l]) * (rho_p[m_rho_l] / (rho[m])[m_rho_l])**2.
        theta[m_rho_l] = tx(rho_p[m_rho_l])
        #theta[m_rho_l] = tx((rho[m])[m_rho_l], (z[m])[m_rho_l], rho_p[m_rho_l])
        mr = (rho[m] == 0.)
        theta[mr] = np.pi/2.

        #BX[0, m] = b * (np.cos(theta) * (z[m] >= 0) + np.cos(pi*np.ones(theta.shape[0]) - theta) * (z[m] < 0))
        #BX[2, m] = b * (np.sin(theta) * (z[m] >= 0) + np.sin(pi*np.ones(theta.shape[0]) - theta) * (z[m] < 0))

        BX[0, m] = b * np.cos(theta) * (-1) ** (z[m] < 0)#BX_rho points outward for z>0 and inward for z<0
        BX[2, m] = b * np.sin(theta) #BX_z points to north

        return BX, np.sqrt(np.sum(BX**2., axis=0))


class GMFPshirkov(object):
    """
    Class with analytical functions that describe the 
    galactic magnetic field according to the model of Pshirkov et al. (2011)

    Only the regular field components are implemented. 

    Attributes
    ----------
    Rsun:  float
        position of the sun in kpc along x axis

    p: dict
        pitch angle, dictionary with entries 'ASS' and 'BSS', in radian

    z0: float
        height of disk in kpc

    d: float
        value if field reversal in kpc

    B0: float
        Value of B field at position of the sun, in muG

    z0n: float
        position of northern halo in kpc

    Bn: float
        northern halo field in muG

    r0n: float
        northern halo

    z1n: float
        scale height of halo toward galactic plane, |z| < z0n

    z2n: float
        scale height of northern halo away from galactic plane, |z| >= z0n

    z0s: float
        position of southern halo in kpc

    Bs: float
        southern halo field in muG

    r0s: float
        southern halo

    z1s: float
        scale height of southern halo toward galactic plane, |z| < z0s

    z2s: float
        scale height of southern halo away from galactic plane, |z| >= z0n

    Notes
    -----
    Paper:
    http://adsabs.harvard.edu/abs/2011ApJ...738..192P
    Pshirkov et al. (2011)
    """

    def __init__(self, model='ASS'):
        """
        Init the GMF class,
        all B-field values are in muG

        Parameters
        ----------
        model: string,
            either ASS or BSS for axissymmetric or bisymmetric model, respectively.

        """
        if not (model == 'ASS' or model == 'BSS'):
            ValueError('mode must be either ASS or BSS not {0}.'.format(model))

        self.m = model

        # Best fit values, see Table 3 of Pshirkov et al. 2011
        self.Rsun = 8.5  # position of the sun in kpc
        # Disk
        self.p = {}
        self.p['ASS'] = -5. * pi / 180.  # pitch angle in radian
        self.p['BSS'] = -6. * pi / 180.  # pitch angle in radian
        self.z0 = 1.  # height of disk in kpc
        self.d = -0.6  # value if field reversal in kpc
        self.B0 = 2.  # Value of B field at position of the sun, in muG
        self.Rc = 5.  # Scale radius of disk component in kpc
        # Halo - North
        self.z0n = 1.3  # position of northern halo in kpc
        self.Bn = 4.  # northern halo in muG
        self.Rn = 8.  # northern halo
        self.z1n = 0.25  # scale height of halo toward galactic plane, |z| < z0n
        self.z2n = 0.40  # scale height of halo away from galactic plane, |z| >= z0n
        # Halo - South
        self.z0s = 1.3  # position of northern halo in kpc
        self.Bs = {}  # northern halo in muG
        self.Bs['ASS'] = 2.                                # northern halo in muG
        self.Bs['BSS'] = 4.                                # northern halo in muG
        self.Rs = 8.                                # northern halo
        self.z1s = 0.25                                # scale height of halo toward galactic plane, |z| < z0n
        self.z2s = 0.40                                # scale height of halo away from galactic plane, |z| >= z0n
        return

    def Bdisk(self, rho, phi, z):
        """
        Disk component of galactic magnetic field
        in galactocentric cylindrical coordinates (rho,phi,z)

        Parameters
        ----------
        rho: array-like
            N-dim array with distance from origin in GC cylindrical coordinates, is in kpc

        phi: array-like
            N-dim array with polar angle in GC cylindircal coordinates, in radian

        z: array-like
            N-dim array with height in kpc in GC cylindrical coordinates

        Returns
        -------
        Bdisk, Bdisk_abs: tuple of :py:class:`~numpy.ndarray`
            tuple containing the magnetic field of the disk as a (3,N)-dim array with (rho,phi,z)
            components of disk field for each coordinate tuple and absolute value of the field as
            N-dim array

        Notes
        -----
        See Pshirkov et al. Eq. (3) - (5)
        """
        if (not rho.shape[0] == phi.shape[0]) and (not z.shape[0] == phi.shape[0]):
            ValueError("List do not have equal shape! returning -1")
        # Bdisk vector in rho, phi, z
        # rows: rho, phi and z component
        Bdisk = np.zeros((3, rho.shape[0]))

        # in order to have same coordinates as Jansson model, i.e. Sun is at x = -8.5 kpc
        phi += np.pi
        m_Rc = rho >= self.Rc

        b = 1. / np.tan(self.p[self.m])
        phi_disk = b * np.log(1. + self.d/self.Rsun) - pi / 2.

        B = np.cos(phi - b * np.log(rho / self.Rsun) + phi_disk)
        if self.m == 'ASS':
            B = np.abs(B)

        B *= np.exp(-np.abs(z) / self.z0)
        B[m_Rc] *= self.B0 * self.Rsun / (rho[m_Rc] * np.cos(phi_disk))
        B[~m_Rc] *= self.B0 * self.Rsun / (self.Rc * np.cos(phi_disk))

        Bdisk[0, :] = B * np.sin(self.p[self.m])
        # minus one multiplied here so that magnetic field
        # is orientated clock wise at earth's position
        Bdisk[1, :] = B * np.cos(self.p[self.m]) * (-1.)

        return Bdisk, np.sqrt(np.sum(Bdisk**2., axis=0))

    def Bhalo(self, rho, z):
        """
        Halo component of galactic magnetic field
        in galactocentric cylindrical coordinates (rho,phi,z)

        Bhalo is purely azimuthal (toroidal), i.e. has only a phi component

        Parameters
        ----------
        rho: array-like
            N-dim array with distance from origin in GC cylindrical coordinates, is in kpc

        z: array-like
            N-dim array with height in kpc in GC cylindrical coordinates

        Returns
        -------
        Bhalo, Bhalo_abs: tuple of :py:class:`~numpy.ndarray`
            tuple containing the magnetic field of the halo as a (3,N)-dim array with (rho,phi,z)
            components of disk field for each coordinate tuple and absolute value of the field as
            N-dim array
        """

        if not rho.shape[0] == z.shape[0]:
            ValueError("List do not have equal shape! returning -1")

        # Bhalo vector in rho, phi, z
        # rows: rho, phi and z component
        Bhalo = np.zeros((3, rho.shape[0]))

        m_zn2 = (z > 0) & (z > self.z0n)  # north and above halo
        m_zn1 = (z > 0) & (z <= self.z0n)  # north and below halo

        m_zs2 = (z < 0) & (z < self.z0n)  # south and below halo
        m_zs1 = (z < 0) & (z >= self.z0n)  # south and above halo, i.e., between halo and disk

        Bhalo[1,m_zn2] = self.Bn / (1. + ((np.abs(z[m_zn2]) - self.z0n) / self.z2n) ** 2.) * rho[m_zn2] / self.Rn \
            * np.exp(1. - rho[m_zn2] / self.Rn)
        Bhalo[1,m_zn1] = self.Bn / (1. + ((np.abs(z[m_zn1]) - self.z0n) / self.z1n) ** 2.) * rho[m_zn1] / self.Rn \
            * np.exp(1. - rho[m_zn1] / self.Rn)

        Bhalo[1,m_zs2] = -1. * self.Bs[self.m] / (1. + ((np.abs(z[m_zs2]) - self.z0s) / self.z2s) ** 2.) \
            * rho[m_zs2] / self.Rs \
            * np.exp(1. - rho[m_zs2] / self.Rs)

        Bhalo[1, m_zs1] = -1. * self.Bs[self.m] / (1. + ((np.abs(z[m_zs1]) - self.z0s) / self.z1s) ** 2.) \
            * rho[m_zs1] / self.Rs \
            * np.exp(1. - rho[m_zs1] / self.Rs)  # the minus sign gives the right rotation direction

        return Bhalo, np.sqrt(np.sum(Bhalo**2.,axis = 0))



degree = pi / 180.
kpc = 1
microgauss = 1
megayear = 1
Gpc = 1e6 * kpc
pc = 1e-3 * kpc
second = megayear / (1e6 * 60 * 60 * 24 * 365.25)
kilometer = kpc / 3.0856775807e+16

models = ['base', 'expX', 'neCl', 'twistX', 'nebCor', 'cre10', 'synCG', 'spur']


class UF23(object):
    """
    Class with analytical functions that describe the
    galactic magnetic field according to the model of Unger & Farrar (2023/2024)
    
    2 components are implemented for the spiral field:
    The spur_field is only used in the 'spur' model.
    The other 7 models use the spiral_field.
    
    3 components are implemented for the halo field:
    The twisted_halo_field is only used for the 'twistX' model.
    The other 7 models use the sum of poloidal_halo_field and toroidal_halo_field.
    
    Attributes
    ----------
    poloidal_A: float
        coasting radius for poloidal halo (a_c)

    r_ref: float
        reference radius, for spiral field 5 kpc for spur field sun radius (8.2 kpc) (r_0)

    r_inner: float
        inner boundary (radius) of spiral field (r_1)

    w_inner: float
        inner boundary (transition width) of spiral field (w_1)

    r_outer: float
        outer boundary (radius) of spiral field (r_2)

    w_outer: float
        outer boundary (transition width) of spiral field (w_2)

    disk_b1, disk_b2, disk_b3: floats
        magnetic field strength of mode 1, 2, 3 (B_m)

    disk_h: float
        transition height (z_d)

    disk_phase1, disk_phase2, disk_phase3: floats
        phase of mode 1, 2, 3 (phi_m)

    disk_pitch: float
        pitch angle (alpha)

    disk_w: float
        vertical transition width (w_d)

    poloidal_B: float
        magnetic field strength normalization constant (B_p)

    poloidal_P: float
        field line exponent (p)

    poloidal_R: float
        radial scale or transition radius (r_p)

    poloidal_W: float
        transition width (w_p)

    poloidal_Z: float
        scale height (z_p)

    toroidal_BN: float
        northern magnetic field strength

    toroidal_BS: float
        southern magnetic field strength

    toroidal_R: float
        transition radius (r_t)

    toroidal_W: float
        radial transition width (w_t)

    toroidal_Z: float
        vertical scale height (z_t)
    
    twisting_time: float
        twisting time

    spur_center: float
        central azimuth in degree

    spur_length: float
        half angular length in degree

    spur_width: float
        gaussian width in degrees

    Notes
    -----
    see https://arxiv.org/abs/2311.12120
    Unger & Farrar (2023)
    """

    def __init__(self, model_type='base'):
        """
        Initialize the UF23 class for 8 different models
        
        Parameters
        ----------
        model_type: str
            one of ['base', 'expX', 'neCl', 'twistX', 'nebCor', 'cre10', 'synCG', 'spur']
            see https://arxiv.org/abs/2311.12120
        """
        self.model_type = model_type
        self.poloidal_A = 1 * Gpc
        self.r_ref = 5 * kpc
        self.r_inner = 5 * kpc
        self.w_inner = 0.5 * kpc
        self.r_outer = 20 * kpc
        self.w_outer = 0.5 * kpc

        if model_type == 'base':
            self.disk_b1 = 1.0878565e+00 * microgauss
            self.disk_b2 = 2.6605034e+00 * microgauss
            self.disk_b3 = 3.1166311e+00 * microgauss
            self.disk_h = 7.9408965e-01 * kpc
            self.disk_phase1 = 2.6316589e+02 * degree
            self.disk_phase2 = 9.7782269e+01 * degree
            self.disk_phase3 = 3.5112281e+01 * degree
            self.disk_pitch = 1.0106900e+01 * degree
            self.disk_w = 1.0720909e-01 * kpc
            self.poloidal_B = 9.7775487e-01 * microgauss
            self.poloidal_P = 1.4266186e+00 * kpc
            self.poloidal_R = 7.2925417e+00 * kpc
            self.poloidal_W = 1.1188158e-01 * kpc
            self.poloidal_Z = 4.4597373e+00 * kpc
            self.striation = 3.4557571e-01
            self.toroidal_BN = 3.2556760e+00 * microgauss
            self.toroidal_BS = -3.0914569e+00 * microgauss
            self.toroidal_R = 1.0193815e+01 * kpc
            self.toroidal_W = 1.6936993e+00 * kpc
            self.toroidal_Z = 4.0242749e+00 * kpc
        elif model_type == 'expX':
            self.disk_b1 = 9.9258148e-01 * microgauss
            self.disk_b2 = 2.1821124e+00 * microgauss
            self.disk_b3 = 3.1197345e+00 * microgauss
            self.disk_h = 7.1508681e-01 * kpc
            self.disk_phase1 = 2.4745741e+02 * degree
            self.disk_phase2 = 9.8578879e+01 * degree
            self.disk_phase3 = 3.4884485e+01 * degree
            self.disk_pitch = 1.0027070e+01 * degree
            self.disk_w = 9.8524736e-02 * kpc
            self.poloidal_A = 6.1938701e+00 * kpc
            self.poloidal_B = 5.8357990e+00 * microgauss
            self.poloidal_P = 1.9510779e+00 * kpc
            self.poloidal_R = 2.4994376e+00 * kpc
            self.poloidal_Z = 2.3684453e+00 * kpc
            self.striation = 5.1440500e-01
            self.toroidal_BN = 2.7077434e+00 * microgauss
            self.toroidal_BS = -2.5677104e+00 * microgauss
            self.toroidal_R = 1.0134022e+01 * kpc
            self.toroidal_W = 2.0956159e+00 * kpc
            self.toroidal_Z = 5.4564991e+00 * kpc
        elif model_type == 'neCL':
            self.disk_b1 = 1.4259645e+00 * microgauss
            self.disk_b2 = 1.3543223e+00 * microgauss
            self.disk_b3 = 3.4390669e+00 * microgauss
            self.disk_h = 6.7405199e-01 * kpc
            self.disk_phase1 = 1.9961898e+02 * degree
            self.disk_phase2 = 1.3541461e+02 * degree
            self.disk_phase3 = 6.4909767e+01 * degree
            self.disk_pitch = 1.1867859e+01 * degree
            self.disk_w = 6.1162799e-02 * kpc
            self.poloidal_B = 9.8387831e-01 * microgauss
            self.poloidal_P = 1.6773615e+00 * kpc
            self.poloidal_R = 7.4084361e+00 * kpc
            self.poloidal_W = 1.4168192e-01 * kpc
            self.poloidal_Z = 3.6521188e+00 * kpc
            self.striation = 3.3600213e-01
            self.toroidal_BN = 2.6256593e+00 * microgauss
            self.toroidal_BS = -2.5699466e+00 * microgauss
            self.toroidal_R = 1.0134257e+01 * kpc
            self.toroidal_W = 1.1547728e+00 * kpc
            self.toroidal_Z = 4.5585463e+00 * kpc
        elif model_type == 'twistX':
            self.disk_b1 = 1.3741995e+00 * microgauss
            self.disk_b2 = 2.0089881e+00 * microgauss
            self.disk_b3 = 1.5212463e+00 * microgauss
            self.disk_h = 9.3806180e-01 * kpc
            self.disk_phase1 = 2.3560316e+02 * degree
            self.disk_phase2 = 1.0189856e+02 * degree
            self.disk_phase3 = 5.6187572e+01 * degree
            self.disk_pitch = 1.2100979e+01 * degree
            self.disk_w = 1.4933338e-01 * kpc
            self.poloidal_B = 6.2793114e-01 * microgauss
            self.poloidal_P = 2.3292519e+00 * kpc
            self.poloidal_R = 7.9212358e+00 * kpc
            self.poloidal_W = 2.9056201e-01 * kpc
            self.poloidal_Z = 2.6274437e+00 * kpc
            self.striation = 7.7616317e-01
            self.twisting_time = 5.4733549e+01 * megayear
        elif model_type == 'nebCor':
            self.disk_b1 = 1.4081935e+00 * microgauss
            self.disk_b2 = 3.5292400e+00 * microgauss
            self.disk_b3 = 4.1290147e+00 * microgauss
            self.disk_h = 8.1151971e-01 * kpc
            self.disk_phase1 = 2.6447529e+02 * degree
            self.disk_phase2 = 9.7572660e+01 * degree
            self.disk_phase3 = 3.6403798e+01 * degree
            self.disk_pitch = 1.0151183e+01 * degree
            self.disk_w = 1.1863734e-01 * kpc
            self.poloidal_B = 1.3485916e+00 * microgauss
            self.poloidal_P = 1.3414395e+00 * kpc
            self.poloidal_R = 7.2473841e+00 * kpc
            self.poloidal_W = 1.4318227e-01 * kpc
            self.poloidal_Z = 4.8242603e+00 * kpc
            self.striation = 3.8610837e-10
            self.toroidal_BN = 4.6491142e+00 * microgauss
            self.toroidal_BS = -4.5006610e+00 * microgauss
            self.toroidal_R = 1.0205288e+01 * kpc
            self.toroidal_W = 1.7004868e+00 * kpc
            self.toroidal_Z = 3.5557767e+00 * kpc
        elif model_type == 'cre10':
            self.disk_b1 = 1.2035697e+00 * microgauss
            self.disk_b2 = 2.7478490e+00 * microgauss
            self.disk_b3 = 3.2104342e+00 * microgauss
            self.disk_h = 8.0844932e-01 * kpc
            self.disk_phase1 = 2.6515882e+02 * degree
            self.disk_phase2 = 9.8211313e+01 * degree
            self.disk_phase3 = 3.5944588e+01 * degree
            self.disk_pitch = 1.0162759e+01 * degree
            self.disk_w = 1.0824003e-01 * kpc
            self.poloidal_B = 9.6938453e-01 * microgauss
            self.poloidal_P = 1.4150957e+00 * kpc
            self.poloidal_R = 7.2987296e+00 * kpc
            self.poloidal_W = 1.0923051e-01 * kpc
            self.poloidal_Z = 4.5748332e+00 * kpc
            self.striation = 2.4950386e-01
            self.toroidal_BN = 3.7308133e+00 * microgauss
            self.toroidal_BS = -3.5039958e+00 * microgauss
            self.toroidal_R = 1.0407507e+01 * kpc
            self.toroidal_W = 1.7398375e+00 * kpc
            self.toroidal_Z = 2.9272800e+00 * kpc
        elif model_type == 'synCG':
            self.disk_b1 = 8.1386878e-01 * microgauss
            self.disk_b2 = 2.0586930e+00 * microgauss
            self.disk_b3 = 2.9437335e+00 * microgauss
            self.disk_h = 6.2172353e-01 * kpc
            self.disk_phase1 = 2.2988551e+02 * degree
            self.disk_phase2 = 9.7388282e+01 * degree
            self.disk_phase3 = 3.2927367e+01 * degree
            self.disk_pitch = 9.9034844e+00 * degree
            self.disk_w = 6.6517521e-02 * kpc
            self.poloidal_B = 8.0883734e-01 * microgauss
            self.poloidal_P = 1.5820957e+00 * kpc
            self.poloidal_R = 7.4625235e+00 * kpc
            self.poloidal_W = 1.5003765e-01 * kpc
            self.poloidal_Z = 3.5338550e+00 * kpc
            self.striation = 6.3434763e-01
            self.toroidal_BN = 2.3991193e+00 * microgauss
            self.toroidal_BS = -2.0919944e+00 * microgauss
            self.toroidal_R = 9.4227834e+00 * kpc
            self.toroidal_W = 9.1608418e-01 * kpc
            self.toroidal_Z = 5.5844594e+00 * kpc
        elif model_type == 'spur':
            self.disk_b1 = -4.2993328e+00 * microgauss
            self.disk_h = 7.5019749e-01 * kpc
            self.disk_phase1 = 1.5589875e+02 * degree
            self.disk_pitch = 1.2074432e+01 * degree
            self.disk_w = 1.2263120e-01 * kpc
            self.poloidal_B = 9.9302987e-01 * microgauss
            self.poloidal_P = 1.3982374e+00 * kpc
            self.poloidal_R = 7.1973387e+00 * kpc
            self.poloidal_W = 1.2262244e-01 * kpc
            self.poloidal_Z = 4.4853270e+00 * kpc
            self.spur_center = 1.5718686e+02 * degree
            self.spur_length = 3.1839577e+01 * degree
            self.spur_width = 1.0318114e+01 * degree
            self.striation = 3.3022369e-01
            self.toroidal_BN = 2.9286724e+00 * microgauss
            self.toroidal_BS = -2.5979895e+00 * microgauss
            self.toroidal_R = 9.7536425e+00 * kpc
            self.toroidal_W = 1.4210055e+00 * kpc
            self.toroidal_Z = 6.0941229e+00 * kpc
            self.r_ref = 8.2 * kpc
        else:
            raise ValueError(f'model must be one of: {models}, not {model_type}')

    def Bdisk(self, rho, phi, z):
        """
        Returns disk component of milky way magnetic field depending on model
        in galactocentric cylindrical coordinates (rho, phi, z)
        
        Parameters
        ----------
        rho: array-like
            N-dim array with distance from origin in GC cylindrical coordinates, is in kpc
        phi: array-like
            N-dim array with polar angle in GC cylindircal coordinates, in radian
        z: array-like
            N-dim array with height in kpc in GC cylindrical coordinates
        
        Returns
        -------
        Bdisk, Bdisk_abs: tuple of :py:class:`~numpy.ndarray`
            tuple containing the magnetic field of the disk as a (3,N)-dim array with (rho,phi,z)
            components of disk field for each coordinate tuple and absolute value of the field as
            N-dim array
        """
        if (not rho.shape[0] == phi.shape[0]) or (not z.shape[0] == phi.shape[0]):
            raise ValueError("List do not have equal shape!")

        if self.model_type == 'spur':
            return self.spur_field(rho, phi, z)
        else:
            return self.spiral_field(rho, phi, z)

    def Bhalo(self, rho, z):
        """
        Returns halo component of milky way magnetic field depending on model
        in galactocentric cylindrical coordinates (rho, phi, z)
        
        Parameters
        ----------
        rho: array-like
            N-dim array with distance from origin in GC cylindrical coordinates, is in kpc
        z: array-like
            N-dim array with height in kpc in GC cylindrical coordinates
        
        Returns
        -------
        Bhalo, Bhalo_abs: tuple of :py:class:`~numpy.ndarray`
            tuple containing the magnetic field of the disk as a (3,N)-dim array with (rho,phi,z)
            components of disk field for each coordinate tuple and absolute value of the field as
            N-dim array
        """
        if not rho.shape[0] == z.shape[0]:
            raise ValueError("List do not have equal shape!")

        if self.model_type == 'twistX':
            return self.twisted_halo_field(rho, z)
        else:
            return self.toroidal_halo_field(rho, z) + self.poloidal_halo_field(rho, z)

    def spiral_field(self, rho, phi, z):
        """
        calculates fourier-spiral disk component
        in galactocentric cylindrical coordinates (rho, phi, z)
        
        Parameters
        ----------
        rho: array-like
            N-dim array with distance from origin in GC cylindrical coordinates, is in kpc

        phi: array-like
            N-dim array with polar angle in GC cylindircal coordinates, in radian

        z: array-like
            N-dim array with height in kpc in GC cylindrical coordinates

        Returns
        -------
        b_cyl, b_cyl_abs: tuple of :py:class:`~numpy.ndarray`
            tuple containing the magnetic field of the disk as a (3,N)-dim array with (rho,phi,z)
            components of disk field for each coordinate tuple and absolute value of the field as
            N-dim array

        Notes
        -----
        See Unger & Farrar (2023) Section 5.2.2, p. 10
        """
        # Avoid division by zero
        rho_safe = np.where(rho == 0, 1e-12 * kpc, rho)

        # Eq. (13)
        hdz = 1 - sigmoid(np.abs(z), self.disk_h, self.disk_w)

        # Eq. (14) time r_ref divided by r
        r_fac_i = sigmoid(rho_safe, self.r_inner, self.w_inner)
        r_fac_o = 1 - sigmoid(rho_safe, self.r_outer, self.w_outer)

        # Lim r--> 0 (replace small values of rho to avoid numerical issues)
        r_fac = np.where(rho_safe > 1e-5 * kpc, (1 - np.exp(-rho_safe * rho_safe)) / rho_safe, rho_safe * (1 - rho_safe**2 / 2))

        gdr_times_rref_by_r = self.r_ref * r_fac * r_fac_o * r_fac_i

        # Eq. (12)
        phi0 = phi - np.log(rho_safe / self.r_ref) / np.tan(self.disk_pitch)

        # Eq. (10)
        b = (self.disk_b1 * np.cos(1 * (phi0 - self.disk_phase1)) +
            self.disk_b2 * np.cos(2 * (phi0 - self.disk_phase2)) +
            self.disk_b3 * np.cos(3 * (phi0 - self.disk_phase3)))

        # Eq. (11)
        fac = hdz * gdr_times_rref_by_r
        b_cyl = np.array([b * fac * np.sin(self.disk_pitch),
                        b * fac * np.cos(self.disk_pitch),
                        np.zeros_like(b)])
        b_cyl = np.where(rho==0, np.array([[0], [0], [0]]), b_cyl)

        return b_cyl, np.sqrt(np.sum(b_cyl**2., axis=0))

    def spur_field(self, rho, phi, z):
        """
        calculates spiral-spur disk component
        in galactocentric cylindrical coordinates (rho, phi, z)
        
        Parameters
        ----------
        rho: array-like
            N-dim array with distance from origin in GC cylindrical coordinates, is in kpc

        phi: array-like
            N-dim array with polar angle in GC cylindircal coordinates, in radian

        z: array-like
            N-dim array with height in kpc in GC cylindrical coordinates

        Returns
        -------
        b_cyl, b_cyl_abs: tuple of :py:class:`~numpy.ndarray`
            tuple containing the magnetic field of the disk as a (3,N)-dim array with (rho,phi,z)
            components of disk field for each coordinate tuple and absolute value of the field as
            N-dim array

        Notes
        -----
        See Unger & Farrar (2023) Section 5.2.3, p. 12
        """
        rho = np.where(rho == 0, 1e-12 * kpc, rho)

        # Adjust phi values
        phi = np.where(phi < 0, phi + 2*pi, phi)

        # reference approximately at solar radius
        phi_ref = self.disk_phase1

        # Logarithmic spiral comparison
        P = phi - phi_ref + np.array([-1, 0, 1])[:, np.newaxis] * 2*pi
        R = self.r_ref * np.exp(P * np.tan(self.disk_pitch))
        abs_diff = np.abs(rho - R)
        i_best = np.argmin(abs_diff, axis=0)

        has_spur = (i_best == 1)

        # Eq. 12
        phi0 = phi - np.log(rho / self.r_ref) / np.tan(self.disk_pitch)

        # Eq. (16)
        delta_phi0 = delta_phi(phi_ref, phi0)
        delta = delta_phi0 / self.spur_width
        b = np.where(has_spur, self.disk_b1 * np.exp(-0.5 * delta ** 2), 0)

        # Eq. (18)
        w_s = 5 * degree
        phi_c = self.spur_center
        delta_phi_c = delta_phi(phi_c, phi)
        l_c = self.spur_length
        g_s = 1 - np.where(has_spur, sigmoid(np.abs(delta_phi_c), l_c, w_s), 0)

        # Eq. (13)
        hd = 1 - sigmoid(np.abs(z), self.disk_h, self.disk_w)

        # Eq. (17)
        b_s = np.where(has_spur, self.r_ref / rho * b * hd * g_s, 0)

        b_cyl = np.vstack((b_s * np.sin(self.disk_pitch), b_s * np.cos(self.disk_pitch), np.zeros_like(b_s)))
        b_cyl = np.where(rho==0, np.array([[0], [0], [0]]), b_cyl)

        return b_cyl, np.sqrt(np.sum(b_cyl**2., axis=0))

    def toroidal_halo_field(self, rho, z):
        """
        calculates toroidal halo field component (has only phi component)
        in galactocentric cylindrical coordinates (rho, phi, z)
        
        Parameters
        ----------
        rho: array-like
            N-dim array with distance from origin in GC cylindrical coordinates, is in kpc
        z: array-like
            N-dim array with height in kpc in GC cylindrical coordinates
        
        Returns
        -------
        b_cyl, b_cyl_abs: tuple of :py:class:`~numpy.ndarray`
            tuple containing the magnetic field of the disk as a (3,N)-dim array with (rho,phi,z)
            components of disk field for each coordinate tuple and absolute value of the field as
            N-dim array

        Notes
        -----
        See Unger & Farrar (2023) Section 5.3.1, p. 13
        """
        b0 = np.where(z>=0, self.toroidal_BN, self.toroidal_BS)
        rh = self.toroidal_R
        z0 = self.toroidal_Z
        fwh = self.toroidal_W
        sigmoid_r = sigmoid(rho, rh, fwh)
        sigmoid_z = sigmoid(np.abs(z), self.disk_h, self.disk_w)

        # Eq. (21)
        b_phi = b0 * (1 - sigmoid_r) * sigmoid_z * np.exp(-np.abs(z) / z0)

        b_cyl = np.zeros((3, len(rho)))
        b_cyl[1, :] = b_phi

        return b_cyl, np.sqrt(np.sum(b_cyl**2., axis=0))

    def poloidal_halo_field(self, rho, z):
        """
        calculates poloidal halo field component (phi-component = 0)
        in galactocentric cylindrical coordinates (rho, phi, z)
        
        Parameters
        ----------
        rho: array-like
            N-dim array with distance from origin in GC cylindrical coordinates, is in kpc
        z: array-like
            N-dim array with height in kpc in GC cylindrical coordinates
        
        Returns
        -------
        b_cyl, b_cyl_abs: tuple of :py:class:`~numpy.ndarray`
            tuple containing the magnetic field of the disk as a (3,N)-dim array with (rho,phi,z)
            components of disk field for each coordinate tuple and absolute value of the field as
            N-dim array

        Notes
        -----
        See Unger & Farrar (2023) Section 5.3.2, p. 13
        """
        c = np.power((self.poloidal_A / self.poloidal_Z), self.poloidal_P)
        a0p = np.power(self.poloidal_A, self.poloidal_P)
        rp = np.power(rho, self.poloidal_P)
        abszp = np.power(np.abs(z), self.poloidal_P)
        cabszp = c * abszp

        t0 = a0p + cabszp - rp
        t1 = np.sqrt(t0**2 + 4 * a0p * rp)
        ap = 2 * a0p * rp / (t1 + t0)

        if np.any(ap < 0) and np.any(rho > np.finfo(float).eps):
            # This should never happen
            raise ValueError("ap = {}".format(ap))

        a = np.power(np.maximum(ap, 0), 1 / self.poloidal_P)

        # Eq. 29 and Eq. 32
        if self.model_type == 'expX':
            radial_dependence = np.exp(-a / self.poloidal_R)
        else:
            radial_dependence = 1 - sigmoid(a, self.poloidal_R, self.poloidal_W)

        # Eq. 28
        bzz = self.poloidal_B * radial_dependence

        # r / a
        r_over_a = 1 / np.power((2 * a0p / (t1 + t0)), (1 / self.poloidal_P))

        # Eq. 35 for p=n
        sign_z = np.where(z < 0, -1, 1)
        br = bzz * c * a / r_over_a * sign_z * np.abs(z)**(self.poloidal_P - 1) / t1

        # Eq. 36 for p=n
        bz = bzz * r_over_a**(self.poloidal_P - 2) * (ap + a0p) / t1

        small_rho = rho < np.finfo(float).eps
        br[small_rho] = 0  # Set radial component to zero where rho is small

        b_cyl = np.zeros((3, rho.size))
        b_cyl[0, :] = br
        b_cyl[2, :] = bz

        return b_cyl, np.sqrt(np.sum(b_cyl**2., axis=0))

    def twisted_halo_field(self, rho, z):
        """
        calculates twisted halo field for twistX model with twisting time not 0
        in galactocentric cylindrical coordinates (rho, phi, z)
        
        Parameters
        ----------
        rho: array-like
            N-dim array with distance from origin in GC cylindrical coordinates, is in kpc
        z: array-like
            N-dim array with height in kpc in GC cylindrical coordinates
        
        Returns
        -------
        b_cyl_x, b_cyl_x_abs: tuple of :py:class:`~numpy.ndarray`
            tuple containing the magnetic field of the disk as a (3,N)-dim array with (rho,phi,z)
            components of disk field for each coordinate tuple and absolute value of the field as
            N-dim array

        Notes
        -----
        See Unger & Farrar (2023) Section 5.3.3, p. 14
        """
        b_x_cyl = self.poloidal_halo_field(rho, z)

        b_r, b_phi, b_z = b_x_cyl

        if self.twisting_time != 0:
            # radial rotation curve parameters (fit to Reid et al 2014)
            v0 = -240 * kilometer / second
            r0 = 1.6 * kpc
            # vertical gradient (Levine+08)
            z0 = 10 * kpc

            # Eq. 43
            fr = 1 - np.exp(-rho / r0)
            # Eq. 44
            t0 = np.exp(2 * np.abs(z) / z0)
            gz = 2 / (1 + t0)

            # Eq. 46
            sign_z = np.where(z < 0, -1, 1)
            delta_z = -sign_z * v0 * fr / z0 * t0 * np.power(gz, 2)
            # Eq. 47
            delta_r = v0 * ((1 - fr) / r0 - fr / np.where(rho == 0, np.inf, rho)) * gz  # avoid deviding by zero

            # Eq. 45
            b_phi = (b_z * delta_z + b_r * delta_r) * self.twisting_time


        b_cyl_x = np.vstack((b_r, b_phi, b_z))
        b_cyl_x = np.where(rho==0, b_x_cyl, b_cyl_x)

        return b_cyl_x, np.sqrt(np.sum(b_cyl_x**2., axis=0))


















