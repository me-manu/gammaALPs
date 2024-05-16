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


class UF23(object):
    """
    Docstring
    """
    
    kPi = np.pi
    kTwoPi = 2 * kPi
    degree = kPi / 180.
    kpc = 1
    microgauss = 1
    megayear = 1
    Gpc = 1e6 * kpc
    pc = 1e-3 * kpc
    second = megayear / (1e6 * 60 * 60 * 24 * 365.25)
    kilometer = kpc / 3.0856775807e+16
    models = ['base', 'expX', 'neCl', 'twistX', 'nebCor', 'cre10', 'synCG', 'spur']
    
    def __init__(self, model_type='base'):
        
        self.fModelType = model_type
        self.fPoloidalA = 1
        
        if model_type == 'base':
            self.fDiskB1 = 1.0878565e+00 * microgauss
            self.fDiskB2 = 2.6605034e+00 * microgauss
            self.fDiskB3 = 3.1166311e+00 * microgauss
            self.fDiskH = 7.9408965e-01 * kpc
            self.fDiskPhase1 = 2.6316589e+02 * degree
            self.fDiskPhase2 = 9.7782269e+01 * degree
            self.fDiskPhase3 = 3.5112281e+01 * degree
            self.fDiskPitch = 1.0106900e+01 * degree
            self.fDiskW = 1.0720909e-01 * kpc
            self.fPoloidalB = 9.7775487e-01 * microgauss
            self.fPoloidalP = 1.4266186e+00 * kpc
            self.fPoloidalR = 7.2925417e+00 * kpc
            self.fPoloidalW = 1.1188158e-01 * kpc
            self.fPoloidalZ = 4.4597373e+00 * kpc
            self.fStriation = 3.4557571e-01
            self.fToroidalBN = 3.2556760e+00 * microgauss
            self.fToroidalBS = -3.0914569e+00 * microgauss
            self.fToroidalR = 1.0193815e+01 * kpc
            self.fToroidalW = 1.6936993e+00 * kpc
            self.fToroidalZ = 4.0242749e+00 * kpc
        elif model_type == 'expX':
            self.fDiskB1 = 9.9258148e-01 * microgauss
            self.fDiskB2 = 2.1821124e+00 * microgauss
            self.fDiskB3 = 3.1197345e+00 * microgauss
            self.fDiskH = 7.1508681e-01 * kpc
            self.fDiskPhase1 = 2.4745741e+02 * degree
            self.fDiskPhase2 = 9.8578879e+01 * degree
            self.fDiskPhase3 = 3.4884485e+01 * degree
            self.fDiskPitch = 1.0027070e+01 * degree
            self.fDiskW = 9.8524736e-02 * kpc
            self.fPoloidalA = 6.1938701e+00 * kpc
            self.fPoloidalB = 5.8357990e+00 * microgauss
            self.fPoloidalP = 1.9510779e+00 * kpc
            self.fPoloidalR = 2.4994376e+00 * kpc
            self.fPoloidalZ = 2.3684453e+00 * kpc
            self.fStriation = 5.1440500e-01
            self.fToroidalBN = 2.7077434e+00 * microgauss
            self.fToroidalBS = -2.5677104e+00 * microgauss
            self.fToroidalR = 1.0134022e+01 * kpc
            self.fToroidalW = 2.0956159e+00 * kpc
            self.fToroidalZ = 5.4564991e+00 * kpc
        elif model_type == 'neCL':
            self.fDiskB1 = 1.4259645e+00 * microgauss
            self.fDiskB2 = 1.3543223e+00 * microgauss
            self.fDiskB3 = 3.4390669e+00 * microgauss
            self.fDiskH = 6.7405199e-01 * kpc
            self.fDiskPhase1 = 1.9961898e+02 * degree
            self.fDiskPhase2 = 1.3541461e+02 * degree
            self.fDiskPhase3 = 6.4909767e+01 * degree
            self.fDiskPitch = 1.1867859e+01 * degree
            self.fDiskW = 6.1162799e-02 * kpc
            self.fPoloidalB = 9.8387831e-01 * microgauss
            self.fPoloidalP = 1.6773615e+00 * kpc
            self.fPoloidalR = 7.4084361e+00 * kpc
            self.fPoloidalW = 1.4168192e-01 * kpc
            self.fPoloidalZ = 3.6521188e+00 * kpc
            self.fStriation = 3.3600213e-01
            self.fToroidalBN = 2.6256593e+00 * microgauss
            self.fToroidalBS = -2.5699466e+00 * microgauss
            self.fToroidalR = 1.0134257e+01 * kpc
            self.fToroidalW = 1.1547728e+00 * kpc
            self.fToroidalZ = 4.5585463e+00 * kpc
        elif model_type == 'twistX':
            self.fDiskB1 = 1.3741995e+00 * microgauss
            self.fDiskB2 = 2.0089881e+00 * microgauss
            self.fDiskB3 = 1.5212463e+00 * microgauss
            self.fDiskH = 9.3806180e-01 * kpc
            self.fDiskPhase1 = 2.3560316e+02 * degree
            self.fDiskPhase2 = 1.0189856e+02 * degree
            self.fDiskPhase3 = 5.6187572e+01 * degree
            self.fDiskPitch = 1.2100979e+01 * degree
            self.fDiskW = 1.4933338e-01 * kpc
            self.fPoloidalB = 6.2793114e-01 * microgauss
            self.fPoloidalP = 2.3292519e+00 * kpc
            self.fPoloidalR = 7.9212358e+00 * kpc
            self.fPoloidalW = 2.9056201e-01 * kpc
            self.fPoloidalZ = 2.6274437e+00 * kpc
            self.fStriation = 7.7616317e-01
            self.fTwistingTime = 5.4733549e+01 * megayear
        elif model_type == 'nebCor':
            self.fDiskB1 = 1.4081935e+00 * microgauss
            self.fDiskB2 = 3.5292400e+00 * microgauss
            self.fDiskB3 = 4.1290147e+00 * microgauss
            self.fDiskH = 8.1151971e-01 * kpc
            self.fDiskPhase1 = 2.6447529e+02 * degree
            self.fDiskPhase2 = 9.7572660e+01 * degree
            self.fDiskPhase3 = 3.6403798e+01 * degree
            self.fDiskPitch = 1.0151183e+01 * degree
            self.fDiskW = 1.1863734e-01 * kpc
            self.fPoloidalB = 1.3485916e+00 * microgauss
            self.fPoloidalP = 1.3414395e+00 * kpc
            self.fPoloidalR = 7.2473841e+00 * kpc
            self.fPoloidalW = 1.4318227e-01 * kpc
            self.fPoloidalZ = 4.8242603e+00 * kpc
            self.fStriation = 3.8610837e-10
            self.fToroidalBN = 4.6491142e+00 * microgauss
            self.fToroidalBS = -4.5006610e+00 * microgauss
            self.fToroidalR = 1.0205288e+01 * kpc
            self.fToroidalW = 1.7004868e+00 * kpc
            self.fToroidalZ = 3.5557767e+00 * kpc
        elif model_type == 'cre10':
            self.fDiskB1 = 1.2035697e+00 * microgauss
            self.fDiskB2 = 2.7478490e+00 * microgauss
            self.fDiskB3 = 3.2104342e+00 * microgauss
            self.fDiskH = 8.0844932e-01 * kpc
            self.fDiskPhase1 = 2.6515882e+02 * degree
            self.fDiskPhase2 = 9.8211313e+01 * degree
            self.fDiskPhase3 = 3.5944588e+01 * degree
            self.fDiskPitch = 1.0162759e+01 * degree
            self.fDiskW = 1.0824003e-01 * kpc
            self.fPoloidalB = 9.6938453e-01 * microgauss
            self.fPoloidalP = 1.4150957e+00 * kpc
            self.fPoloidalR = 7.2987296e+00 * kpc
            self.fPoloidalW = 1.0923051e-01 * kpc
            self.fPoloidalZ = 4.5748332e+00 * kpc
            self.fStriation = 2.4950386e-01
            self.fToroidalBN = 3.7308133e+00 * microgauss
            self.fToroidalBS = -3.5039958e+00 * microgauss
            self.fToroidalR = 1.0407507e+01 * kpc
            self.fToroidalW = 1.7398375e+00 * kpc
            self.fToroidalZ = 2.9272800e+00 * kpc
        elif model_type == 'synCG':
            self.fDiskB1 = 8.1386878e-01 * microgauss
            self.fDiskB2 = 2.0586930e+00 * microgauss
            self.fDiskB3 = 2.9437335e+00 * microgauss
            self.fDiskH = 6.2172353e-01 * kpc
            self.fDiskPhase1 = 2.2988551e+02 * degree
            self.fDiskPhase2 = 9.7388282e+01 * degree
            self.fDiskPhase3 = 3.2927367e+01 * degree
            self.fDiskPitch = 9.9034844e+00 * degree
            self.fDiskW = 6.6517521e-02 * kpc
            self.fPoloidalB = 8.0883734e-01 * microgauss
            self.fPoloidalP = 1.5820957e+00 * kpc
            self.fPoloidalR = 7.4625235e+00 * kpc
            self.fPoloidalW = 1.5003765e-01 * kpc
            self.fPoloidalZ = 3.5338550e+00 * kpc
            self.fStriation = 6.3434763e-01
            self.fToroidalBN = 2.3991193e+00 * microgauss
            self.fToroidalBS = -2.0919944e+00 * microgauss
            self.fToroidalR = 9.4227834e+00 * kpc
            self.fToroidalW = 9.1608418e-01 * kpc
            self.fToroidalZ = 5.5844594e+00 * kpc
        elif model_type == 'spur':
            self.fDiskB1 = -4.2993328e+00 * microgauss
            self.fDiskH = 7.5019749e-01 * kpc
            self.fDiskPhase1 = 1.5589875e+02 * degree
            self.fDiskPitch = 1.2074432e+01 * degree
            self.fDiskW = 1.2263120e-01 * kpc
            self.fPoloidalB = 9.9302987e-01 * microgauss
            self.fPoloidalP = 1.3982374e+00 * kpc
            self.fPoloidalR = 7.1973387e+00 * kpc
            self.fPoloidalW = 1.2262244e-01 * kpc
            self.fPoloidalZ = 4.4853270e+00 * kpc
            self.fSpurCenter = 1.5718686e+02 * degree
            self.fSpurLength = 3.1839577e+01 * degree
            self.fSpurWidth = 1.0318114e+01 * degree
            self.fStriation = 3.3022369e-01
            self.fToroidalBN = 2.9286724e+00 * microgauss
            self.fToroidalBS = -2.5979895e+00 * microgauss
            self.fToroidalR = 9.7536425e+00 * kpc
            self.fToroidalW = 1.4210055e+00 * kpc
            self.fToroidalZ = 6.0941229e+00 * kpc
        else:
            raise ValueError(f'model must be one of: {models}, not {model_type}')

        self.fSinPitch = np.sin(self.fDiskPitch)
        self.fCosPitch = np.cos(self.fDiskPitch)
        self.fTanPitch = np.tan(self.fDiskPitch)

    def Bdisk(self, rho, phi, z):
        if (not rho.shape[0] == phi.shape[0]) or (not z.shape[0] == phi.shape[0]):
            raise ValueError("List do not have equal shape!")


        if self.fModelType == 'spur':
            return self.spur_field(rho, phi, z)
        else:
            return self.spiral_field(rho, phi, z)

    def Bhalo(self, rho, z):
        if not rho.shape[0] == z.shape[0]:
            raise ValueError("List do not have equal shape! returning -1")

        if self.fModelType == 'twistX':
            return self.twisted_halo_field(rho, z)
        else:
            return self.toroidal_halo_field(rho, z) + self.poloidal_halo_field(rho, z)





















