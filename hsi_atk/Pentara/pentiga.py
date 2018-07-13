import numpy as np
from skimage.draw import ellipse, ellipsoid, ellipsoid_stats


class Pentiga(object):
    """
    HSI object to store data about image substructure and easily add or remove it
    """

    def __init__(self, name, ell_sma, bands=None, scale=(1, 0), center=(0, 0), is_base=True, is_substructure=False,
                 stats=False, labels=None):
        """
        Initializes pentiga object. Meant to store collections of structures representing one object
        in an HSI, and storing label information related to the structures

        :param name: string - name of pentiga object for use with dict-like structures
        :param ell_sma: tuple of 3 ints - lengths of the semi-major axes to produce an ellipsoid
        :param bands: array-like of ints or None - integers representing bands occupied by the structure

        :param scale: tuple of ints - scaling factor, structure is multiplied by scale[0]
                                                      and has scale[1] added as well
        :param center: tuple of ints - supposed center in some image (for passing to Pentara image composer)
        :param is_base: bool - whether this pentiga has children
        :param is_substructure: bool - whether this pentiga has a parent pentiga
        :param stats: bool - to compute stats of structure (Volume, Surface Area)
        :param labels: dictionary - assigned labels of kernel
        """
        self.name = name
        self._sma = ell_sma
        self.bands = bands

        self.scale = scale

        self.center = center
        self.dist_center = None

        self.structure = None
        self.stats = None
        self.img_area = 0
        self.n_pixels = 0

        self.labels = {}
        if labels is not None:
            self.labels = labels

        self.gen_ellipsoid(stats=stats)

        self.is_base = is_base
        self.is_substructure = is_substructure
        self.sub_structures = {}
        self.sub_scales = {}

    def get_name(self):
        return self.name

    def set_center(self, center):
        self.center = center

    def get_center(self):
        return self.center

    def get_sma(self):
        return self._sma

    def get_bands(self):
        if self.bands is not None:
            return self.bands
        else:
            return np.arange(self.structure.shape[2])

    def set_bands(self, bands=None):
        """
        Assign bands to self image structure (slicing across bandwidths)
        :param bands: array-like - list of bands, if None, all of structure 3rd axis is used
        """
        if bands is None:
            d2 = self.structure.shape
            self.bands = np.arange(d2)
        else:
            self.bands = bands

    def gen_rand_bands(self, n_bands, max_band=None):
        """
        Generates random shuffling of bands
        :param n_bands: number of bandwidths to sample from
        :param max_band: max bandwidth or "frequency" to sample from
        """
        if max_band is None and self.structure is not None:
            d2 = self.structure.shape
            bands = np.arange(d2)
            np.random.shuffle(bands)
            self.bands = bands[:n_bands-1]
        elif max_band is not None:
            bands = np.arange(max_band)
            np.random.shuffle(bands)
            self.bands = bands[:n_bands-1]
        else:
            raise Exception("Structure is None and max_bands not defined! One or both must be set.")

    def get_npix(self):
        """
        :return: int - number of pixels taken by object in the 2d spacial dimension
        """
        return self.n_pixels

    def set_dist_center(self, dist_center):
        self.dist_center = dist_center

    def get_dist_center(self):
        return self.dist_center

    def get_structure(self):
        if self.bands is None:
            return self.structure
        else:
            return self.structure[:, :, self.bands]

    def gen_ellipsoid(self, stats=False, n_pix=True, **kwargs):
        """
        Wrapper for skimage.draw.ellipsoid function to create base structure of given pentiga object

        :param stats: bool - to compute ellipsoid stats, can always be done later given any sma
        :param n_pix: bool - computes number of pixels taken by structure and assigns self.n_pixels
        :param kwargs: for passing additional arguments to ellipsoid function such as spacing

        :return: None - data stored in object
        """
        a, b, c = self._sma
        self.structure = ellipsoid(a, b, c, **kwargs, levelset=True)
        if stats:
            self.stats = ellipsoid_stats(a, b, c)

        if n_pix:
            # Getting pixel count of parent ellipsoid
            s0 = a*2+1
            s1 = b*2+1
            rr, cc = ellipse(a, b, a, b, (s0, s1))
            base = np.zeros((s0, s1))
            base[rr, cc] += 1
            pix = np.count_nonzero(base)
            self.n_pixels = pix  # setting n_pixels

    def gen_img_area(self, pix_area=1):
        """
        Get's image area by n_pixels of object and area of pixels.
        For approximating real size of objects.
        :param pix_area: int - size of pixels of real image (ex. 20cm, 6ft, etc.)
        """
        img_area = self.n_pixels*pix_area
        self.img_area = img_area

    def compose(self, objects=None, return_labels=False):
        """
        Composes all ellipsoids from parent pentiga and all immediate children pentiga of parent
        limiting third axis to specified number of bands

        :param bands: int - bandwidths to span across
        :param objects: list of strings - names of sub-structures to include in composition
                        if None, all children of parent pentiga
                        if empty list, no children are used in composition and only parent structure
                            is included
        :param return_labels: bool - return labels array with compose image as separate array

        :return: base_img - ndarray of composed image
                 labels_  - 2d array of labels (if return_labels=True)
        """
        d0, d1, d2 = self.structure.shape
        print(self.structure.shape)
        r0, c0 = np.floor(d0/2), np.floor(d1/2)

        # base_img = self.structure[:, :, :bands]
        base_img = np.zeros((d0, d1, d2))

        # if bands < d2:
        rr, cc, bb = self._sma
        rr0, cc0 = ellipse(r0, c0, rr, cc, shape=(d0, d1))

        labels_ = np.empty((d0, d1), dtype=object)

        if return_labels:
            labels_[rr0, cc0] = self.name

        if self.bands is None:
            base_img[rr0, cc0, :] += self.structure[rr0, cc0, :]
        else:
            base_b = self.bands
            base_img[rr0, cc0, base_b] += self.structure[rr0, cc0, base_b]



        if objects is None:
            objects = self.sub_structures.keys()


        for obj in objects:
            # od0, od1, od2 = obj.structure.shape
            n_obj = self.sub_structures[obj]
            n_obj_b = n_obj.bands
            nd0, nd1, nd2 = n_obj.structure.shape
            nr0, nc0 = np.floor(nd0/2), np.floor(nd1/2)

            nrr, ncc, bb = n_obj.get_sma()
            nrr0, ncc0 = ellipse(nr0, nc0, nrr, ncc, shape=(nd0, nd1))

            nr, nc = n_obj.get_dist_center()
            r, c = nr + r0, nc + c0
            rr0, cc0 = ellipse(r, c, nrr, ncc, shape=(d0, d1))

            base_img[rr0, cc0, n_obj_b] += n_obj.structure[nrr0, ncc0, n_obj_b]
            if return_labels:
                n_l = "," + n_obj.get_name()
                labels_[rr0, cc0] += n_l

        if return_labels:
            return base_img, labels_
        else:
            return base_img

    def add_substructure(self, obj, stats=True):
        """
        Adds pentiga object to collection of substructures of the self pentiga

        :param obj: Pentiga - ellipsoid object
        :param stats: bool - enforce the ellipsoid stats are stored in object

        :return: None - stores passed obj in self.sub_structures dictionary
        """
        sub_name = obj.name
        if stats:
            if obj.stats is not None:
                self.sub_structures[sub_name] = obj
            else:
                obj.gen_ell_stats()
                self.sub_structures[sub_name] = obj

        else:
            self.sub_structures[sub_name] = obj

    def gen_sub_structure(self, ell_sma, name=None, dist_center=(0, 0),
                          bands=None, stats=False):
        """
        Generates another Pentiga object as a sub-structure and store it

        :param ell_sma: tuple of 3 ints - semi-major axes lengths for new ellipsoid structure
        :param name: string - name of new structure, if None default name will be generated..
                                                     "sub_0", "sub_1", "sub_2", ...
        :param dist_center: tuple of ints - new structure center distance from parent pentiga
        :param bands: Not used atm, will later be combined with scaling-addition functions
                      to detail structure's effect and brightness across the spectrum
        :param stats: bool - for computing the elipsoid stats, can always be done later

        :return: None - sub structure is added to self.sub_structure dictionary
        """

        a0, b0, c0 = self._sma
        a, b, c = ell_sma
        if name is None:
            s_i = len(self.sub_structures)
            name = "sub_" + str(s_i + 1)

        if a > a0 or b > b0 or c > c0:
            raise Exception("semi-major axes of sub-structures must be less-than or equal to primary ellipsoid axes")
        s_ell = Pentiga(name, ell_sma, stats=stats, is_base=False, is_substructure=True)

        s_ell.center = (dist_center[0] + self.center[0], dist_center[1] + self.center[1])
        s_ell.dist_center = dist_center

        self.sub_structures[name] = s_ell

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

# TODO: remove redundant functions and generalized to
    # add_func, add_func_bandwise, and add_func_coordwise
    # use generalization with scale_substructure above to pass
    # instruction sets for multi-structure change
    def add_structure(self, add):
        """
        Basic addition across structure array

        :param add: int/float - number to add across structure

        :return: None - value added to self.structure
        """
        self.structure += add

    def mult_structure(self, mult):
        """
        Basic multiplication across structure array

        :param mult: int/float - number to multiply across structure

        :return: None - value added to self.structure
        """
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

    def add_func_coordwise(self, func, rr, cc):
        """
        Alter brightness values by adding from specified function arrays of coordinates

        :param func: input func, some f(x, y) = z
        :param rr: ndarray - list of x-coordinates
        :param cc: ndarray - list of y-coordinates
                - x- and y-coordinates must correspond

        :return: None - values added to image arrays
        """
        for r, c in rr, cc:
            self.structure[r, c, :] += func(r, c)

    def add_func_bandwise(self, func, bands):
        """
        Alter brightness values by adding from specified function with band as input

        :param func: input func, some f(x) = y
        :param bands: int, tuple, list, or ndarray representing band indices

        :return: None - values added to image arrays
        """
        if isinstance(bands, tuple):
            for i in range(bands[0], bands[1]+1):
                self.structure[:, :, i] += func(i)

        elif isinstance(bands, int):
            for i in range(0, bands):
                self.structure[:, :, i] += func(i)

        elif isinstance(bands, list) or isinstance(bands, np.ndarray):
            for i in bands:
                self.structure[:, :, i] += func(i)

    def add_func(self, func, rr, cc, bands):
        for band in bands:
            for r, c, in rr, cc:
                self.structure[r, c, band] += func(r, c, band)

    def gen_ell_stats(self):
        a, b, c = self._sma
        self.stats = ellipsoid_stats(a, b, c)

    def add_label(self, key, label):
        self.labels[key] = label

    def set_labels(self, labels):
        self.labels = labels

    #TODO: implement use_pix_avg and use_pix_var
    def gen_wt_label(self, wt_func, structs=None, str_wts=None, use_pix_avg=False, use_pix_var=False):
        """
        Function for generating continuous labels of weight explicitly or within a
        distribution versus structure and substructure stats
        :param wt_func: mathematical function - takes pixel count and weight to generate weight label
        :param structs: list of keys - sub structures to include in weight label computation
        :param str_wts: dict of floats - keyed to sub structures, float of structure
                                         computational weight vs total weight
        """
        base_pix = self.get_npix()
        tot_mod = 0

        if structs is not None:
            modif = []
            for struct in structs:
                n_str = self.sub_structures[struct]
                modif.append(n_str.get_npix() * str_wts[struct])

            tot_mod = sum(modif)

        if not use_pix_avg:
            weight = wt_func(base_pix) + tot_mod

        else:
            weight = wt_func(base_pix) + tot_mod

        self.set_kernelwt(weight)

    #TODO: implement use_pix_avg and use_pix_var and base "weighting"
    def gen_lp_labels(self, lp_func, palm_func, lino_func, olei_func, stea_func,
                      base_wt, structs=None, str_wts=None, use_pix_avg=False, use_pix_var=False):
        """
        Function for generating continuous labels of lipids explicitly or within a
        distribution versus structure and substructure stats.
        Kernel weight must be assigned first. Use gen_wt_label.
        :param lp_func: mathematical func - takes pixel count and weight to generate
                                              lipid label
        :param palm_func: mathematical func - takes pixel count and weight to generate
                                              palmetic label
        :param lino_func: mathematical func - takes pixel count and weight to generate
                                              linoleic label
        :param olei_func: mathematical func - takes pixel count and weight to generate
                                              oleic label
        :param stea_func: mathematical func - takes pixel count and weight to generate
                                              stearic label
        :param base_wt: float - weight of base structure vs lipid labels
        :param structs: list of keys - structures to conside for given label
        :param str_wts: dict of dicts - corresponding weight of structure for each given label
        """
        base_pix = self.get_npix()
        tot_mod = 0
        tot_mod_p = 0
        tot_mod_l = 0
        tot_mod_o = 0
        tot_mod_s = 0

        if structs is not None:
            modif = []
            modif_p = []
            modif_l = []
            modif_o = []
            modif_s = []
            for struct in structs:
                n_str = self.sub_structures[struct]
                modif.append(lp_func(n_str.get_npix() * str_wts[struct]['lp']) / base_pix)
                modif_p.append(palm_func(n_str.get_npix() * str_wts[struct]['palm']) / base_pix)
                modif_l.append(lino_func(n_str.get_npix() * str_wts[struct]['lino']) / base_pix)
                modif_o.append(olei_func(n_str.get_npix() * str_wts[struct]['olei']) / base_pix)
                modif_s.append(stea_func(n_str.get_npix() * str_wts[struct]['stea']) / base_pix)

            tot_mod = sum(modif)
            tot_mod_p = sum(modif_p)
            tot_mod_l = sum(modif_l)
            tot_mod_o = sum(modif_o)
            tot_mod_s = sum(modif_s)

        if not use_pix_avg:
            lp_weight = lp_func(base_pix) + tot_mod
            p_weight = palm_func(base_pix) + tot_mod_p
            l_weight = lino_func(base_pix) + tot_mod_l
            o_weight = olei_func(base_pix) + tot_mod_o
            s_weight = stea_func(base_pix) + tot_mod_s

        else:
            lp_weight = lp_func(base_pix) + tot_mod
            p_weight = palm_func(base_pix) + tot_mod_p
            l_weight = lino_func(base_pix) + tot_mod_l
            o_weight = olei_func(base_pix) + tot_mod_o
            s_weight = stea_func(base_pix) + tot_mod_s

        self.set_lipidwt(lp_weight)
        self.set_palmetic(p_weight)
        self.set_linoleic(l_weight)
        self.set_oleic(o_weight)
        self.set_stearic(s_weight)

    def set_kernelwt(self, weight):
        """
        :param weight: float - representing weight in grams of kernel
        """
        self.labels['kernelwt'] = weight

    def set_lipidwt(self, weight):
        """
        :param weight: float - representing weight of lipids in grams of kernel
        """
        self.labels['lipidwt'] = weight

    def set_wtratio(self):
        self.labels['wtratio'] = self.labels['lipidwt'] / self.labels['kernelwt']

    def set_palmetic(self, density):
        """
        :param density: float - representing mg/mL measure
        """
        self.labels['palmetic'] = density

    def set_linoleic(self, density):
        """
        :param density: float - representing mg/mL measure
        """
        self.labels['linoleic'] = density

    def set_oleic(self, density):
        """
        :param density: float - representing mg/mL measure
        """
        self.labels['oleic'] = density

    def set_stearic(self, density):
        """
        :param density: float - representing mg/mL measure
        """
        self.labels['stearic'] = density


def basic_gen(name, sma, s_sma, dist_center):
    pent = Pentiga(name, sma, stats=True)

    i = 0
    for _sma, dist in s_sma, dist_center:
        s_name = 'sub_ell_'
        s_name += str(i)
        pent.gen_sub_structure(s_name, _sma, dist, stats=True)
        i += 1

    return pent
