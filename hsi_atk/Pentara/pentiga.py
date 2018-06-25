import numpy as np
from skimage.draw import ellipse, ellipsoid, ellipsoid_stats


class Pentiga(object):
    """
    HSI object to store data about image substructure and easily add or remove it
    """

    def __init__(self, name, ell_sma, stats=False, is_base=True, is_substructure=False):
        '''
        Initializes pentiga object. Meant to store collections of structures representing one object
        in an HSI, and storing label information related to the structures

        :param name: string - name of pentiga object for use with dict-like structures
        :param ell_sma: tuple of 3 ints - lengths of the semi-major axes to produce an ellipsoid
        :param stats: bool - T/F to compute stats of structure (Volume, Surface Area)
        :param is_base: bool - T/F whether this pentiga has children
        :param is_substructure: bool - T/F whether this pentiga has a parent pentiga
        '''
        self._sma = ell_sma
        self.name = name

        self.scale = (1, 0)

        self.center = (0, 0)
        self.dist_center = None
        self.structure = None
        self.stats = None
        self.img_area = 0

        self.gen_ellipsoid(stats=stats)

        self.is_base = is_base
        self.is_substructure = is_substructure
        self.sub_structures = {}
        self.sub_scales = {}


    def set_center(self, center):
        self.center = center


    def get_center(self):
        return self.center


    def set_dist_center(self, dist_center):
        self.dist_center = dist_center


    def get_dist_center(self):
        return self.dist_center


    def gen_ellipsoid(self, stats=False, **kwargs):
        '''
        Wrapper for skimage.draw.ellipsoid function to create base structure of given pentiga object

        :param stats: bool - T/F to compute ellipsoid stats, can always be done later given any sma
        :param kwargs: for passing additional arguments to ellipsoid function such as spacing

        :return: None - data stored in object
        '''
        a, b, c = self._sma
        self.structure = ellipsoid(a, b, c, **kwargs, levelset=True)
        if stats:
            self.stats = ellipsoid_stats(a, b, c)


    def gen_img_area(self, pix_area=1):
        a, b, c = self._sma
        img_area = np.pi*b*c*pix_area
        self.img_area = img_area


    def compose(self, bands, objects=None):
        '''
        Composes all ellipsoids from parent pentiga and all immediate children pentiga of parent
        limiting third axis to specified number of bands

        :param bands: int - bandwidths to span across
        :param objects: list of strings - names of sub-structures to include in composition
                        if None, all children of parent pentiga
                        if empty list, no children are used in composition and only parent structure
                            is included

        :return: base_img - ndarray of composed image
        '''
        d0, d1, d2 = self.structure.shape
        r0, c0 = np.floor(d0/2), np.floor(d1/2)

        # base_img = self.structure[:, :, :bands]
        base_img = np.zeros((d0, d1, bands))

        if bands < d2:
            rr, cc, bb = self._sma
            rr0, cc0 = ellipse(r0, c0, rr, cc, shape=(d0, d1))
            base_img[rr0, cc0, :] += self.structure[rr0, cc0, :bands]

        if objects is None:
            objects = list(self.sub_structures.keys())

        for obj in objects:
            # od0, od1, od2 = obj.structure.shape
            n_obj = self.sub_structures[obj]
            nd0, nd1, nd2 = n_obj.structure.shape
            nr0, nc0 = np.floor(nd0/2), np.floor(nd1/2)

            nrr, ncc, bb = n_obj._sma
            nrr0, ncc0 = ellipse(nr0, nc0, nrr, ncc, shape=(nd0, nd1))

            nr, nc = n_obj.get_dist_center()
            r, c = nr + r0, nc + c0
            rr0, cc0 = ellipse(r, c, nrr, ncc, shape=(d0, d1))

            base_img[rr0, cc0, :] += n_obj.structure[nrr0, ncc0, :bands]

        return base_img


    def add_substructure(self, obj, stats=True):
        '''
        Adds pentiga object to collection of substructures of the self pentiga

        :param obj: Pentiga - ellipsoid object
        :param stats: bool - T/F enforce the ellipsoid stats are stored in object

        :return: None - stores passed obj in self.sub_structures dictionary
        '''
        sub_name = obj.name
        if stats:
            if obj.stats != None:
                self.sub_structures[sub_name] = obj
            else:
                obj.gen_ell_stats()
                self.sub_structures[sub_name] = obj

        else:
            self.sub_structures[sub_name] = obj


    def gen_sub_ellipsoid(self, ell_sma, sub_name, dist_center=(0, 0),
                          bands=None, stats=False):
        '''
        Generates another Pentiga object as a sub-structure and store it

        :param ell_sma: tuple of 3 ints - semi-major axes lengths for new ellipsoid structure
        :param sub_name: string - name of new structure
        :param dist_center: tuple of ints - new structure center distance from parent pentiga
        :param bands: Not used atm, will later be combined with scaling-addition functions
                      to detail structure's effect and brightness across the spectrum
        :param stats: bool - T/F for computing the elipsoid stats, can always be done later

        :return: None - sub structure is added to self.sub_structure dictionary
        '''

        a0, b0, c0 = self._sma
        a, b, c = ell_sma

        if a > a0 or b > b0 or c > c0:
            raise Exception("semi-major axes of sub-structures must be less-than or equal to primary ellipsoid semi-major axes")
        s_ell = Pentiga(sub_name, ell_sma, stats=stats, is_base=False, is_substructure=True)

        s_ell.center = (dist_center[0] + self.center[0], dist_center[1] + self.center[1])
        s_ell.dist_center = dist_center

        self.sub_structures[sub_name] = s_ell

        if not self.is_base:
            self.is_base = True


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


    def add_cubic(self, a, b, c, d, bands):
        for i in range(bands[0], bands[1]+1):
            self.structure[:, :, i] += (a*i**3 + b*i**2 + c*i + d)


    def add_logarithmic(self, a, b, c, d, bands):
        for i in range(bands[0], bands[1]+1):
            self.structure[:, :, i] += (a*np.log(b*(i+1) + c) + d)


    def add_func(self, fx, bands):
        for i in range(bands[0], bands[1]+1):
            self.structure[:, :, i] += fx(i)


    def gen_ell_stats(self):
        a, b, c = self._sma
        self.stats = ellipsoid_stats(a, b, c)


    def set_label(self, label):
        self.label = label


def basic_gen(name, sma, s_sma, dist_center):
    pent = Pentiga(name, sma, stats=True)

    i = 0
    for _sma, dist in s_sma, dist_center:
        s_name = 'sub_ell_'
        s_name += str(i)
        pent.gen_sub_ellipsoid(s_name, _sma, dist, stats=True)
        i += 1

    return pent


smas = [(240, 18, 27), (240, 4, 10), (240, 5, 6)]
cents = [(0, 0), (-3, -4), (2, 5)]


def test_func(x):
    return (100.*np.sin(.02*x - 1)*np.cos(.05*x-5) + 600 + x)


def test_pent():
    # pent0 = basic_gen("pent0", (240, 20, 30), smas, cents)
    import time
    time0 = time.time()
    pent0 = Pentiga('pent0', (24, 30, 240), stats=True)
    pent0.gen_sub_ellipsoid((5, 8, 240), 'sub_ell_0', stats=True, dist_center=(2, 3))
    pent0.scale_structure(-50, 500)
    pent0.add_func(test_func, (0, 240))
    # pent0.add_linear(.5, 200, (0, 240))
    pent0.sub_structures['sub_ell_0'].scale_structure(-25, 200)
    pent0.sub_structures['sub_ell_0'].add_func(test_func, (0, 240))
    n_img = pent0.compose(240)
    time1 = time.time()
    print(time1-time0)
    return pent0, n_img

pent0, n_img = test_pent()