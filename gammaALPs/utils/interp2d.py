import numpy as np
from scipy.interpolate import RectBivariateSpline as RBS


class Interp2D(object):
    """
    Do a 2d interpolation using scipy.interpolate.RectBivariateSpline
    and taking care of the array sorting
    """
    def __init__(self, x, y, z, **kwargs):
        self.__rbs = RBS(x, y, z, **kwargs)
        return

    def __call__(self, x, y):
        """
        Perform the interpolation.

        Parameters
        ----------

        x: array-like
            first interpolation coordinates
        y: array-like
            second interpolation coordinates

        Returns
        -------
        array with interpolation results
        """

        if np.isscalar(x):
            x = np.array([x])
        elif type(x) == list:
            x = np.array(x)
        if np.isscalar(y):
            y = np.array([y])
        elif type(y) == list:
            y = np.array(y)

        result = np.zeros((x.shape[0],y.shape[0]))
        tt = np.zeros((x.shape[0],y.shape[0]))

        args_x = np.argsort(x)
        args_y = np.argsort(y)

        # Spline interpolation requires sorted lists
        tt[args_x, :] = self.__rbs(np.sort(x), np.sort(y))
        result[:, args_y] = tt
        return np.squeeze(result)

