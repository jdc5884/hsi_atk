import numpy as np

from skimage.draw import ellipsoid, ellipsoid_stats, ellipse
from skimage.restoration import denoise_wavelet
from skimage.util import random_noise


class Pentara:
    """
    Painter - base class for HSI image generator
    """
    def __init__(self, shape, noise=False):
        self._s = shape
        self._img = np.zeros(shape)
        self.noise = False
        self.add_noise()
        self.obs = {}


    def add_noise(self, **kwargs):
        '''
        Adds noise to HSI via skimage util

        :param kwargs: arguments for noise generators
        :return: None, alters base image from self
        '''
        if self.noise != False:
            self._img = random_noise(self._img, **kwargs)


    def add_ellipsoid(self, center, a, b, c, scale, bands=None, rot=0.0, cut=None, stats=False, name=None):
        '''
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
        '''
        ell = ellipsoid(a, b, c, levelset=True)
        ell.astype('f8')
        ell *= scale[0]
        ell += scale[1]

        d0, d1, d2 = self._s
        ed0, ed1, ed2 = ell.shape

        c1 = int(ed1/2)+1
        c2 = int(ed2/2)+1

        rr, cc = ellipse(c1, c2, b, c, shape=(ed1, ed2))

        rr1, cc1 = ellipse(center[0], center[1], b, c, shape=(d1, d2), rotation=rot)

        if d0<ed0:
            self._img[:d0, rr1, cc1] += ell[:d0, rr, cc]
        else:
            self._img[:ed0, rr1, cc1] += ell[:ed0, rr, cc]

        if stats:
            stat = ellipsoid_stats(a, b, c)
            try:
                self.obs[name] = [center, stat]
            except ValueError:
                raise Exception("if stats=True, name must be a unique string not in self.obs")


    def add_structure(self, obj, name, rot=0.0):
        self.obs[name] = obj
        x, y, z = obj.get_center()
        a, b, c = obj._sma
        d0, d1, d2 = self._s
        ed0, ed1, ed2 = obj.structure.shape
        c1, c2 = int(ed1/2)+1, int(ed2/2)+1
        rr1, cc1 = ellipse(c1, c2, b, c, shape=(ed1, ed2), rotation=rot)
        rr, cc = ellipse(x-1, y-1, b, c, shape=self._s, rotation=rot)

        if d1>ed1 or d2>ed2:
            raise Exception("obj structure is larger than image!")

        if d0<ed0:
            self._img[:d0, rr, cc] += obj.structure[:d0, rr1, cc1]
        else:
            self._img[:ed0, rr, cc] += obj.structure[:ed0, rr1, cc1]


class Pentiga(object):
    """
    HSI object to store data about image substructure and easily add or remove it
    """

    def __init__(self, ell_sma, stats=False, is_substructure=False):
        self._sma = ell_sma

        self.scale = (1, 0)

        self.center = None
        self.structure = None
        self.stats = None
        self.img_area = 0

        self.gen_ellipsoid(stats=stats)

        self.is_substructure = is_substructure
        self.dist_center = (0, 0)
        self.sub_structures = {}
        self.sub_scales = {}


    def set_center(self, center):
        self.center = center


    def get_center(self):
        return self.center


    def gen_ellipsoid(self, stats=False, **kwargs):
        a, b, c = self._sma
        self.structure = ellipsoid(a, b, c, **kwargs, levelset=True)
        if stats:
            self.stats = ellipsoid_stats(a, b, c)


    def gen_img_area(self, pix_area=1):
        a, b, c = self._sma
        img_area = np.pi*b*c*pix_area
        self.img_area = img_area


    def compose(self, bands, objects=None):
        d0, d1, d2 = self.structure.shape

        base_img = np.zeros((bands, d1, d2))
        if bands < d0:
            base_img += self.structure[bands,:,:]

        if objects is None:
            objects = self.sub_structures.keys()

        for obj in objects:
            od0, od1, od2 = obj.structure.shape


        return base_img


    def add_substructure(self, obj, sub_name, stats=True):

        if stats:
            if obj.stats != None:
                self.sub_structures[sub_name] = obj
            else:
                obj.gen_ell_stats()
                self.sub_structures[sub_name] = obj

        else:
            self.sub_structures[sub_name] = obj


    def gen_sub_ellipsoid(self, ell_sma, sub_name, dist_center=(0, 0),
                          bands=None, stats=False, **kwargs):
        # make use of spacing with **kwargs to generate offset sub structures
        a0, b0, c0 = self._sma
        a, b, c = ell_sma

        if a > a0 or b > b0 or c > c0:
            raise Exception("semi-major axes of sub-structures must be less-than or equal to primary ellipsoid semi-major axes")
        s_ell = Pentiga(ell_sma, stats=stats, is_substructure=True)
        s_ell.dist_center = dist_center

        self.sub_structures[sub_name] = s_ell


    # Basic version
    # Add scaling instructions instead of basic linear tuples
    def scale_substructure(self, names, scales):

        for name in names:
            n_ell = self.sub_structures[name].structure
            scale = scales[name]
            n_ell *= scale[0]
            n_ell += scale[1]
            self.sub_structures[name].structure = n_ell
            self.sub_scales[name] = scale


    def add_structure(self, add):
        self.structure += add


    def mult_structure(self, mult):
        self.structure *= mult


    def scale_structure(self, mult, add):
        self.mult_structure(mult)
        self.add_structure(add)


    def add_linear(self, a, b, bands):
        for i in range(bands[0], bands[1]+1):
            self.structure[:, :, i] += (a*i + b)


    def add_quadratic(self, a, b, c, bands):
        for i in range(bands[0], bands[1]+1):
            self.structure[:, :, i] += (a*i**2 + b*i + c)


    def gen_ell_stats(self):
        a, b, c = self._sma
        self.stats = ellipsoid_stats(a, b, c)


    def set_label(self, label):
        self.label = label


class Tukara:

    def __init__(self):
        pass


