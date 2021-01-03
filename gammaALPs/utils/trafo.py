import numpy as np
from numpy import sin,cos
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import interp1d


def cosmo_cohlength(zmax, L0, cosmo=FlatLambdaCDM(H0=70., Om0=0.3)):
    """
    Calculate a grid of coherence lengths given a maximum
    redshift, coherence length, and cosmology

    Parameters
    ----------
    zmax: float
        maximum redshift

    L0: `~astropy.units.quantity.Quantity`
        coherence length at z = 0

    cosmo: `~astropy.cosmology.core.FlatLambdaCDM`
        chosen cosmology, default is H0 = 70, Om0 = 0.3

    Returns
    -------
    tuple with 
    - N-dim `~numpy.ndarray` with coherence lengths
    - (N+1)-dim `~numpy.ndarray` with redshifts corresponding to summed coherence length starting with 0
    Last bin is chosen such that total luminosity distance is equal to zmax.
    """
    # interpolate Lumi distance
    z = np.linspace(0.,zmax * 1.1, 100)
    l = cosmo.luminosity_distance(z)  # l in Mpc
    zfunc = interp1d(l.value, z)

    #z = [0.,z_at_value(cosmo.luminosity_distance, L0)]
    z = [0.,zfunc(L0.to('Mpc').value)]
    dL = [L0.value]

    while (z[-1] < zmax):
        dL.append(L0.value / (1. + z[-1]))
        #z.append(z_at_value(cosmo.luminosity_distance, np.sum(dL) * L0.unit))
        z.append(zfunc((np.sum(dL) * L0.unit).to('Mpc').value))
    dL[-1] = (cosmo.luminosity_distance(zmax) - cosmo.luminosity_distance(z[-2])).to('kpc').value
    z[-1] = zmax

    # Roncadelli's way
    #dz = (1.17e-3 * L0.to('Mpc') / (5. * u.Mpc)).value
    #Nd = int(ceil(0.85e3 * (5. * u.Mpc / L0.to('Mpc')).value * zmax))
    #dL = np.array([4.29e3 * dz / (1. + 1.45 * (n - 1.) * dz) for n in range(1, Nd + 1)]) * 1e3
    #z = np.array(range(0, Nd + 1)) * dz

    return np.array(dL), np.array(z)

# =========================================================== #
# === Transformation function for Galactic Magnetic Field === #
# =========================================================== #


def norm(vec):
    """
    return the norm of vector vec
    """
    return np.sqrt(np.sum(vec)**2.)


class NewBase:
    def __init__(self, d=8.5):
        """ 
        Init new base
        old base in galactic coordinates l (galactic longitude) [0, 2pi]
        and b (galactic latitude) [-pi/2, +pi/2]
        distance sun gc is in kpc, default is 8.5
        and sun is located at -x = 8.5 kpc, y = 0, z = 0 in GC cartesian coordinates
        """
        self.l = 0.
        self.b = 0.
        self.d = d
        self.r = np.zeros(3)                # r unit vector in cartesian coordinates
        self.t = np.zeros(3)                # theta unit vector in cartesian coordinates
        self.p = np.zeros(3)                # phi unit vector in cartesian coordinates

        self.u1 = np.zeros(3)                # new base vector 1
        self.u2 = np.zeros(3)                # new base vector 2
        self.u3 = np.zeros(3)                # new base vector 3
        return

    def set_obase(self, l, b):
        """
        set components of base in cartesian coordinates with angles l and b
        """
        self.r[0] = cos(l)*cos(b)
        self.r[1] = sin(l)*cos(b)
        self.r[2] = sin(b)

        self.t[0] = cos(l)*sin(b)
        self.t[1] = sin(l)*sin(b)
        self.t[2] = -cos(b)

        self.p[0] = -sin(l)
        self.p[1] = cos(l)
        self.p[2] = 0.

        return

    def set_nbase(self, l, b, s):
        """
        Calculate new set of basis vectors from the linear independent vectors -r + (d,0,0), theta and phi
        using the Gram-Schmidt process, see e.g. http://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process

        s is the distance in kpc of the point in galactic coordinates
        """
        self.set_obase(l,b)        # update old basis

        self.u1 = -s*self.r + np.array([self.d,0.,0.])                # first basis vector
        self.u2 = self.t - np.dot(self.t,self.u1)/np.dot(self.u1,self.u1) * self.u1
        self.u3 = self.p - np.dot(self.p,self.u1)/np.dot(self.u1,self.u1) * self.u1 - np.dot(self.p,self.u2)/np.dot(self.u2,self.u2) * self.u2

        self.u1 = self.u1/norm(self.u1)         # norm it
        self.u2 = self.u2/norm(self.u2)         # norm it
        self.u3 = self.u3/norm(self.u3)         # norm it

        return self.u1,self.u2, self.u3


# ---- Transformation of HC spherical galactical coordinates to CG cylindrical coordinates ---- #
# ---- Functions ----- #
def rho_HC2GC(s,l,b,d=-8.5):
    """
    rho in GC cylindircal coordinates for distance s (measured from the sun in kpc) in galactic coordinates l and b
    """
    if isinstance(l, np.ndarray):
        return np.sqrt(s**2*cos(b)**2 + d**2*np.ones(l.shape[0]) + 2.*s*d*cos(l)*cos(b))
    elif isinstance(b, np.ndarray):
        return np.sqrt(s**2*cos(b)**2 + d**2*np.ones(b.shape[0]) + 2.*s*d*cos(l)*cos(b))
    elif isinstance(s, np.ndarray):
        return np.sqrt(s**2*cos(b)**2 + d**2*np.ones(s.shape[0]) + 2.*s*d*cos(l)*cos(b))
    else:
        return np.sqrt(s**2*cos(b)**2 + d**2 + 2.*s*d*cos(l)*cos(b))


def phi_HC2GC(s,l,b,d=-8.5):
    """
    phi in GC cylindircal coordinates for distance s (measured from the sun in kpc) in galactic coordinates l and b
    """
    if isinstance(l, np.ndarray):
        return np.arctan2(s * sin(l)*cos(b),(s * cos(l)*cos(b) + d*np.ones(l.shape[0])))
    elif isinstance(b, np.ndarray):
        return np.arctan2(s * sin(l)*cos(b),(s * cos(l)*cos(b) + d*np.ones(b.shape[0])))
    elif isinstance(s, np.ndarray):
        return np.arctan2(s * sin(l)*cos(b),(s * cos(l)*cos(b) + d*np.ones(s.shape[0])))
    else:
        return np.arctan2(s * sin(l)*cos(b),(s * cos(l)*cos(b) + d))


def z_HC2GC(s, b):
    """
    z in GC cylindircal coordinates for distance s (measured from the sun in kpc) in galactic coordinates l and b
    """
    return s*sin(b)

def x_HC2GC(s, l, b, d=-8.5):
    """
    x in GC cylindircal coordinates for distance s (measured from the sun in kpc) in galactic coordinates l and b
    """
    if isinstance(l, np.ndarray):
        return s*cos(l)*cos(b) + d*np.ones(l.shape[0])
    elif isinstance(b, np.ndarray):
        return s*cos(l)*cos(b) + d*np.ones(b.shape[0])
    elif isinstance(s, np.ndarray):
        return s*cos(l)*cos(b) + d*np.ones(s.shape[0])
    else:
        return s*cos(l)*cos(b) + d

def y_HC2GC(s, l, b):
    """
    y in GC cylindircal coordinates for distance s (measured from the sun in kpc) in galactic coordinates l and b
    """
    return s*sin(l)*cos(b)


#---- (s,l,b) unit vectors ---------#
def HC_base(l, b):
    """
    Computes cartesian basis for galactic coordinates

    Parameters
    ----------
    l: array-like or scalar
        N-dim array with galactic longitude in radians
    b: array-like or scalar,
        N-dim array with galactic latitude in radians

    Returns
    -------
    Three (3,N) np.arrays, unit vectors in s, l, and b direction
    """
    if np.isscalar(l):
        l = np.array([l])
    if np.isscalar(b):
        b = np.array([b])

    unit_s = np.array([cos(l)*cos(b),sin(l)*cos(b), sin(b)])
    unit_b = np.array([cos(l)*sin(b),sin(l)*sin(b),-cos(b)])
    unit_l = np.array([-sin(l),cos(l),np.zeros(l.shape[0])])
    return unit_s,unit_b,unit_l


# ---- Project a vector in GCCC to HCSC in cartesian basis ----- #
def GC2HCproj(V, s, l, b, d=-8.5):
    """
    Project a Vector in GC cylindrical coordinates to HC galactic coordinates in cartesian basis

    Parameters
    ----------
    V: array-like
        (3,N) dim vector with GC components (V_rho , V_phi , V_z)

    s: array-like or scalar
        distance from the sun
    l: array-like or scalar
        galactic longitude
    b: array-like or scalar
        galactic latitude
    d: float
        origin of HC coordinates along x axis in GC coordinates in kpc,
        default is position of the sun at -8.5 kpc

    Returns
    -------
    tuple with vector components:
    - result[0] = < V, s >
    - result[1] = < V, b >
    - result[2] = < V, l >
    """
    if np.isscalar(l):
        l = np.array([l])
    if np.isscalar(b):
        b = np.array([b])
    if np.isscalar(s):
        s = np.array([s])

    p = phi_HC2GC(s,l,b,d)        # calculate phi angle from s,l,b

    #for q in p:
        #logging.debug("p: {0:20.2f}".format(q))
    
    result = np.zeros(V.shape)
    result[0] = cos(b) * (V[0]*cos(l-p) + V[1]*sin(l-p)) + sin(b) * V[2]
    result[1] = sin(b) * (V[0]*cos(l-p) + V[1]*sin(l-p)) - cos(b) * V[2]
    result[2] = V[0]*sin(p-l) + V[1]*cos(p-l)
    return result

