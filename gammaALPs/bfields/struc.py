import numpy as np
import sys
from scipy.integrate import simpson as simp


class structured_field(object):
    """Class definition of structured magnetic field, see 1008.5353 and
    1908.03084 for details.
    """

    #alpha is lowest positive, non-zero root of tan(alpha)=3alpha/(3-alpha**2).
    alpha = 5.7634591968
    F_0 = (alpha * np.cos(alpha) - np.sin(alpha)) * alpha**2

    #orm calculated using simpy, :math:`lim_{r \to 0}` of Bfield components and taking euclidian norm.
    norm = np.sqrt((3 * F_0 + alpha**5)**2) * 2 / (3 * alpha**2)

    def __init__(self, B0, R, theta, theta_rad, pa, pa_rad, cell_num=1000):
        """Initializes structured B field model. Default values are reasonable for galaxy clusters.

        Parameters
        ----------
        B0: float
            Total B field strength at cluster center r=0.

        R: float
            Cavity radius, B field goes to zero at r=R.

        theta: float
            inclination of symmetry axis w.r.t. line of sight.

        theta_rad: bool
            True if theta given in radians.

        pa: float
            Position angle of symmetry axis in galactic coordinates.

        pa_rad: bool
            True if pa given in radians.

        cell_num: int
            Number of cells B field is divided into for propagation of polarization density matrix.
        """
        
        self.theta = theta if theta_rad else np.radians(theta)
        self.pa = pa if pa_rad else np.radians(pa)
        self.B0 = B0
        self.R = R
        self.cell_num = cell_num
        self.dL = R / cell_num
        self.r = self._get_r_points()
        self.dL_vec = np.full(self.r.shape, self.dL)


    def _get_r_points(self):
        return np.linspace(self.dL / 2, self.R - self.dL / 2, self.cell_num)

    
    @property
    def r(self):
        """r needs to be rescaled in field strength expressions to be smaller
            than one, for "external" use _r is multiplied by R in this property wrapper.
        """
        return self._r * self.R

    @r.setter
    def r(self, val=None):
        """When manually setting r (val = some array) dL_vec needs to be set manually according to r.

        Parameters
        ----------
        val: array-like
            New points along line of sight. val=None resets to default values, val=np.ndarray for different choice of points.
        """
        if val is None:
            print('Resetting radial points and dL_vec.')
            self._r = self._get_r_points() / self.R
            self.dL_vec = np.full(self.r.shape, self.dL)
        else:
            if np.max(val) > self.R:
                raise ValueError('You cannot choose r_i > R')
            else:
                #print('You need to manually set dL_vec!')
                self._r = val / self.R

    
    @property
    def rbounds(self):
        return np.linspace(0, self.R, self.cell_num, endpoint=True)

    @property
    def dL_vec(self):
        return self._dL_vec

    @dL_vec.setter
    def dL_vec(self, val):
        self._dL_vec = val

    @property
    def angle(self):
        return self._angle_b_trans(self.b_phi, self.b_theta) - self.pa

    @property
    def b_r(self):
        return self.B0 * self._b_r(self._r, self.theta)

    @property
    def b_phi(self):
        return self.B0 * self._b_phi(self._r, self.theta)

    @property
    def b_theta(self):
        return self.B0 * self._b_theta(self._r, self.theta)

    @property
    def b_par(self):
        return self.B0 * self._b_par(self._r, self.theta)

    @property
    def b_trans(self):
        return self.B0 * self._b_trans(self._r, self.theta)

    @classmethod
    def _b_par(cls, r, theta):
        return cls._b_r(r, theta)

    @classmethod
    def _b_trans(cls, r, theta):
        b_phi = cls._b_phi(r, theta)
        b_theta = cls._b_theta(r, theta)
        return np.sqrt(b_phi**2 + b_theta**2)

    @staticmethod
    def _angle_b_trans(b_phi, b_theta):
        """Calculates angle of transversal field component (psi).
        Conforms to psi definition of GMF models. See definition of
        Bs, Bt, Bu in GMF environs and trafo.py.
        In this case, B_b = -B_theta, B_l = -B_phi,
        b=galactic latitude, l=galactic longitude.
        """
        return np.arctan2(-b_theta, -b_phi)

    @classmethod
    def _b_r(cls, r, theta):
        zero_val = - np.cos(theta) * (6 * cls.F_0 + 2 * cls.alpha**5) \
                   / (3 * cls.alpha**2)
        if np.any(np.isclose(r, 0)):
            try:
                zero_args = np.argwhere(np.isclose(r, 0))
                val = 2 * np.cos(theta) * cls._f(r) / r**2
                val[zero_args] = zero_val
            except TypeError:
                val = zero_val
        else:
            val = 2 * np.cos(theta) * cls._f(r) / r**2
        return val / cls.norm

    @classmethod
    def _b_theta(cls, r, theta):
        zero_val = np.sin(theta) * (6 * cls.F_0 + 2 * cls.alpha**5) \
                   / (3 * cls.alpha**2)
        if np.any(np.isclose(r, 0)):
            try:
                zero_args = np.argwhere(np.isclose(r, 0))
                val = - np.sin(theta) * cls._f_prime(r) / r
                val[zero_args] = zero_val
            except TypeError:
                val = zero_val
        else:
            val = - np.sin(theta) * cls._f_prime(r) / r
        return val / cls.norm

    @classmethod
    def _b_phi(cls, r, theta):
        zero_val = 0
        if np.any(np.isclose(r, 0)):
            try:
                zero_args = np.argwhere(np.isclose(r, 0))
                val = cls.alpha * np.sin(theta) * cls.f(r) / r
                val[zero_args] = zero_val
            except TypeError:
                val = zero_val
        else:
            val = cls.alpha * np.sin(theta) * cls._f(r) / r
        return val / cls.norm

    @classmethod
    def _f(cls, r):
        # should maybe include special case of r=0 here as well.
        # on the other hand, never used explicitely. same with df/dr
        return cls.alpha * np.cos(cls.alpha * r) - \
               np.sin(cls.alpha * r) / r \
               - cls.F_0 * r**2 / cls.alpha**2

    @classmethod
    def _f_prime(cls, r):
        return (- cls.alpha**2 * np.sin(cls.alpha * r)
                - cls.alpha * np.cos(cls.alpha * r) / r
                + np.sin(cls.alpha * r) / r**2) \
                - 2 * cls.F_0 * r / cls.alpha**2

    def rotation_measure(self, nel):
        """Rotation measure (RM) = rad * m^-2 * 812 * integral nel * B dz,
        nel in 1/cm^3, B in muGauss, z in kpc.

        Parameters
        ----------
        nel: array-like
            Array of electron density in cm^-3 along r.

        Returns
        -------
        rm: float
            Rotation measure in units of rad/m^2.
        """
        return 812. * simp(self.b_par * nel, self.r)


if __name__ == "__main__":
    sys.exit()
