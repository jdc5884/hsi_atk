import numpy as np

from skimage.draw import ellipsoid, ellipsoid_stats, ellipse
from skimage.restoration import denoise_wavelet
from skimage.util import random_noise


class Pentara:
    """
    Painter - base class for HSI image generator
    """
    def __init__(self, shape):
        self._s = shape
        self._img = np.zeros(shape)
        self.noisy = False
        # self.add_noise()
        self.obs = {}

    def get_img(self):
        return self._img

    def get_obs(self):
        return self.obs

    def add_noise(self, **kwargs):
        """
        Adds noise to HSI via skimage util

        :param kwargs: arguments for noise generators
        :return: None, alters base image from self
        """
        self.noisy = True
        self._img = random_noise(self._img, **kwargs)

    def add_ellipsoid(self, center, a, b, c, scale, bands=None, rot=0.0, cut=None, stats=False, name=None):
        """
        Adds ellipsoid structure in simulated HSI.

        :param center: tuple of 2 ints centering structure in image
        :param a: length of ellipsoid semi-major x-axis
            - currently used for bands
        :param b: length of ellipsoid semi-major y-axis
            - currently used for x-axis in image
        :param c: length of ellipsoid semi-major z-axis
            - currently used for y-axis in image
            - will be switching to kwargs or other method of creating ellipsoids
        :param scale: tuple of 2 ints/floats to alter reflectance of structure across pixels/bands
            - typically (-a, b) such that levelset of ellipsoid is flipped by X *= (-a)
            - and raised by X += b
        :param bands: tuple of 2 ints the band range to add structure
        :param rot: rotation elliptical structures in image
            - currently only multiples of pi/2 accepted for rotations
        :param cut: cut a subsection of ellipsoid structure
            - not implemented at the moment
        :param stats: bool (T/F) of whether or not to collect ellipsoid stats
        :param name: unique string for dict key of ellipsoid stats
            - must have valid string if stats==True
        :return: None, alters base image from self
        """
        ell = ellipsoid(a, b, c, levelset=True)
        ell.astype('f8')
        ell *= scale[0]
        ell += scale[1]

        d0, d1, d2 = self._s
        ed0, ed1, ed2 = ell.shape

        c1 = int(ed1 / 2) + 1
        c2 = int(ed2 / 2) + 1

        rr, cc = ellipse(c1, c2, b, c, shape=(ed1, ed2))

        rr1, cc1 = ellipse(center[0], center[1], b, c, shape=(d1, d2), rotation=rot)

        if d0 < ed0:
            self._img[:d0, rr1, cc1] += ell[:d0, rr, cc]
        else:
            self._img[:ed0, rr1, cc1] += ell[:ed0, rr, cc]

        if stats:
            stat = ellipsoid_stats(a, b, c)
            try:
                self.obs[name] = [center, stat]
            except ValueError:
                raise Exception("if stats=True, name must be a unique string not in self.obs")

    def add_structure(self, pentiga, rot=0.0, center=None):
        """
        Add Pentiga object to base hsi image

        :param pentiga: Pentiga - image object, including sub-structures
        :param rot: float - multiples of pi/4 to rotate image by 90 degrees
        :param center: tuple of ints - optional forced center of pentiga image

        :return: None - adds values to base image of self Pentara
        """
        name = pentiga.get_name()
        self.obs[name] = pentiga

        if center is None:
            x, y, z = pentiga.get_center()
        else:
            x, y, z = center

        a, b, c = pentiga.get_sma()
        struct = pentiga.compose()

        d0, d1, d2 = self._s
        ed0, ed1, ed2 = struct.shape

        c1, c2 = int(ed1/2) + 1, int(ed2/2)+1

        rr1, cc1 = ellipse(c1, c2, b, c, shape=(ed1, ed2), rotation=rot)
        rr, cc = ellipse(x-1, y-1, b, c, shape=self._s, rotation=rot)

        if d1 > ed1 or d2 > ed2:
            raise Exception("obj structure is larger than image!")

        if d0 < ed0:
            self._img[:d0, rr, cc] += struct[:d0, rr1, cc1]
        else:
            self._img[:ed0, rr, cc] += struct[:ed0, rr1, cc1]


class Tukara:

    def __init__(self):
        pass
