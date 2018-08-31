import numpy as np
from skimage.draw import ellipse


class Pentiga_n(object):
    """
    HSI object to store data about image substructure and easily add or remove it
    """

    def __init__(self, name, ell_sma, bands=240, band_dist=2.072, copy=False, center=(0, 0), is_substructure=False, labels=None):
        """
        Initializes pentiga object. Meant to store collections of structures representing one object
        in an HSI, and storing label information related to the structures

        :param name: string - name of pentiga object for use with dict-like structures
        :param ell_sma: tuple of 2 ints - lengths of the semi-major axes to produce an ellipse
        :param bands: int - integers representing bands occupied by the structure
                        int for total number of bands as representation
        :param band_dist: float - separation (in nm) between wavelength
        :param copy: bool - whether to keep image loaded or not
        :param center: tuple of ints - supposed center in some image (for passing to Pentara image composer)
        :param is_substructure: bool - whether this pentiga has a parent pentiga
        :param stats: bool - to compute stats of structure (Volume, Surface Area)
        :param labels: dictionary - assigned labels of kernel
        """
        self.name = name
        self._sma = ell_sma
        self.bands = bands
        self.band_dist = band_dist

        self.copy = copy

        self.center = center
        self.dist_center = None

        self.structure = None
        self.scale_func = None
        self.img_area = 0
        self.n_pixels = 0

        self.label_funcs = {}
        self.labels = {}
        if labels is not None:
            self.labels = labels

        # self.gen_ellipsoid(stats=stats)

        self.is_substructure = is_substructure
        self.sub_structures = {}
        self.sub_scales = {}

    def set_name(self, name):
        """Sets name of image compont
        :param name: string - image component name
        """
        self.name = name

    def get_name(self):
        """Returns image component identifier
        :return: string - image component name
        """
        return self.name

    def set_center(self, center):
        """Sets assigned image coordinates (row, col)
        :param center: tuple of 2 ints - (row, col) coordinates
        """
        self.center = center

    def get_center(self):
        """Gets assigned image coordinates (row, col)
        :return: tuple of 2 ints - (row, col) coordinates
        """
        return self.center

    def set_sma(self, sma):
        """Sets semi-major and semi-minor axis values for (x, y) axes
        :param sma: tuple of 2 ints - length of axes
        """
        self._sma = sma

    def get_sma(self):
        """Gets semi-major and semi-minor axis values for (x, y) axes
        :return: tuple of 2 ints -  length of axes
        """
        return self._sma

    def set_bands(self, bands):
        """Assign bands to self image structure (slicing across bandwidths)
        :param bands: int - number of bands
        """
        self.bands = bands

    def get_bands(self):
        """Gets number of bands for current image component generator
        :return: int - number of bands
        """
        return self.bands

    def set_copy(self, copy):
        """Whether generator actively keeps copy of image array.
        Objects being designed to hold the information that generates the image component
        with images typically only being stored in fully composed forms.
        :param copy: boolean - if true, image component will be composed at parameter updates
        """
        self.copy = copy

    def get_copy(self):
        """Gets copy boolean
        :return: boolean - whether object is actively updating image component
        """
        return self.copy

    def gen_rand_bands(self, n_bands):
        """
        Generates random shuffling of bands
        :param n_bands: int - number of bandwidths to sample from
        """
        bands = self.get_bands()
        rand_bands = np.arange(bands)
        np.random.shuffle(rand_bands)
        return rand_bands[:n_bands-1]

    def get_npix(self):
        """Gets number of pixels occupied by this (self) structure. Based upon the shape and its params.
        :return: int - number of pixels taken by object in the 2d spacial dimension
        """
        return self.n_pixels

    def gen_npix(self):
        """Calculates pixels occupied by this (self) structure. Currently based under the assumption
        of shape as an ellipse.
        """
        # Getting pixel count of parent ellipsoid
        sma = self.get_sma()
        rr, cc = ellipse(sma[0], sma[1], sma[0], sma[1])
        base = np.zeros((sma[0]*2+1, sma[1]*2+1))
        base[rr, cc] += 1
        pix = np.count_nonzero(base)
        self.n_pixels = pix  # setting n_pixels

    def set_dist_center(self, dist_center):
        """Sets distance from center of parent image component object.
        :param dist_center: tuple of 2 ints - pixel distance from parent image component
        """
        self.dist_center = dist_center

    def get_dist_center(self):
        """Gets distance from center of parent image component object.
        :return: tuple of 2 ints - pixel distance from parent image component
        """
        return self.dist_center


    def gen_structure(self, save_str=False, **kwargs):
        """Generates image array for base structure based on self.scale_func and self._sma

        :param save_str: bool - whether or not to save image array to object
        """
        a, b = self.get_sma()
        bands = self.get_bands()
        struct_shape = (a*2+1, b*2+1, bands)
        scale_func = self.get_scale_func()
        structure = np.fromfunction(scale_func, struct_shape)
        rr, cc = ellipse(20, 20, a, b, shape=struct_shape[:2])

        if save_str:
            self.structure = structure[rr, cc, :]
            return structure[rr, cc, :]
        else:
            return structure[rr, cc, :]

    def gen_img_area(self, pix_area=1):
        """Get's image area by n_pixels of object and area of pixels.
        For approximating real size of objects.

        :param pix_area: int - size of pixels of real image (ex. 20cm, 6ft, etc.)
        """
        img_area = self.n_pixels*pix_area
        self.img_area = img_area

    def compose(self, objects=None, return_labels=False):
        """Composes all ellipsoids from parent pentiga and all immediate children pentiga of parent
        limiting third axis to specified number of bands

        :param objects: list of strings - names of sub-structures to include in composition
                        if None, all children of parent pentiga
                        if empty list, no children are used in composition and only parent structure
                            is included
        :param return_labels: bool - return labels array with compose image as separate array

        :return: base_img - ndarray of composed image
                 labels_  - 2d array of labels (if return_labels=True)
        """
        base = self.gen_structure()
        d0, d1, d2 = base.shape
        print(base.shape)
        r0, c0 = np.floor(d0/2), np.floor(d1/2)

        # base_img = self.structure[:, :, :bands]
        base_img = np.zeros((d0, d1, d2))

        # if bands < d2:
        rr, cc = self._sma
        rr0, cc0 = ellipse(r0, c0, rr, cc, shape=(d0, d1))

        labels_ = np.empty((d0, d1), dtype=object)

        if return_labels:
            labels_[rr0, cc0] = self.name

        if self.bands is None:
            base_img[rr0, cc0, :] += base[rr0, cc0, :]
        elif isinstance(self.bands, int):
            base_img[rr0, cc0, :] += base[rr0, cc0, :]
        else:
            base_b = self.bands - 1
            base_img[rr0, cc0, base_b] += base[rr0, cc0, base_b]



        if objects is None:
            objects = self.sub_structures.keys()


        for obj in objects:
            # od0, od1, od2 = obj.structure.shape
            n_obj = self.sub_structures[obj]
            n_obj_b = n_obj.bands
            n_obj_base = n_obj.gen_ellipsoid()
            nd0, nd1, nd2 = n_obj_base.shape
            nr0, nc0 = np.floor(nd0/2), np.floor(nd1/2)

            nrr, ncc = n_obj.get_sma()
            nrr0, ncc0 = ellipse(nr0, nc0, nrr, ncc, shape=(nd0, nd1))

            nr, nc = n_obj.get_dist_center()
            r, c = nr + r0, nc + c0
            rr0, cc0 = ellipse(r, c, nrr, ncc, shape=(d0, d1))

            base_img[rr0, cc0, n_obj_b] += n_obj_base[nrr0, ncc0, n_obj_b]
            if return_labels:
                n_l = "," + n_obj.get_name()
                labels_[rr0, cc0] += n_l

        if return_labels:
            return base_img, labels_
        else:
            return base_img

    def add_substructure(self, obj):
        """Adds pentiga object to collection of substructures of the self pentiga

        :param obj: Pentiga
        """
        sub_name = obj.name
        self.sub_structures[sub_name] = obj

    def gen_sub_structure(self, ell_sma, name=None, dist_center=(0, 0), bands=None):
        """Generates another Pentiga object as a sub-structure and store it

        :param ell_sma: tuple of 2 ints - semi-major axes lengths for new ellipsoid structure
        :param name: string - name of new structure, if None default name will be generated..
                                                     "sub_0", "sub_1", "sub_2", ...
        :param dist_center: tuple of 2 ints - new structure center distance from parent pentiga
        :param bands: Not used atm, will later be combined with scaling-addition functions
                      to detail structure's effect and brightness across the spectrum
        """

        a0, b0 = self._sma
        a, b = ell_sma
        if name is None:
            s_i = len(self.sub_structures)
            name = "sub_" + str(s_i + 1)

        if a > a0 or b > b0:
            raise Exception("semi-major axes of sub-structures must be less-than or equal to primary ellipsoid axes")
        s_ell = Pentiga_n(name, ell_sma, bands=bands, is_substructure=True)

        c_b = self.get_center()
        s_ell.set_center((dist_center[0] + c_b[0], dist_center[1] + c_b[1]))
        s_ell.set_dist_center(dist_center)

        self.add_substructure(s_ell)

    # TODO: replace/repurpose add_func functions
    def set_scale_func(self, func):
        self.scale_func = func

    def get_scale_func(self):
        return self.scale_func

    # TODO: remove redundant functions and generalized to
    # add_func, add_func_bandwise, and add_func_coordwise
    # use generalization with scale_substructure above to pass
    # instruction sets for multi-structure change
    def add_func_coordwise(self, func, rr, cc):
        """Adds to brightness values

        :param func: input func, some f(x, y) = z
        :param rr: ndarray - list of x-coordinates
        :param cc: ndarray - list of y-coordinates
                - x- and y-coordinates must correspond
        """
        for r in rr:
            for c in cc:
                self.structure[r, c, :] += func(r, c)

    def add_func_bandwise(self, func, bands):
        """Alter brightness values by adding from specified function with band as input

        :param func: input func, some f(x) = y
        :param bands: int, tuple, list, or ndarray representing band indices
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

    # def compose_func_coor(self):
    #     func_mats = []
    #     coor_funcs = self.get_coor_funcs()
    #     shape = self.get_shape()
    #     for key in coor_funcs.keys():
    #         func = coor_funcs[key]
    #         func_mat = np.fromfunction(func, shape)

    # TODO: Separate auto-generated stats functions
    # def gen_ell_stats(self):
    #     a, b = self._sma
    #     self.stats = ellipsoid_stats(a, b)

    def add_label(self, key, label):
        self.labels[key] = label

    def remove_label(self, key):
        del self.labels[key]

    def set_labels(self, labels):
        self.labels = labels

    #TODO: implement use_pix_avg and use_pix_var, pix_band_avg, pix_band_var
    def gen_wt_label(self, wt_func, structs=None, str_wts=None, use_pix_avg=False, use_pix_var=False):
        """Function for generating continuous labels of weight explicitly or within a
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
        """Function for generating continuous labels of lipids explicitly or within a
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


# def basic_gen(name, sma, s_sma, dist_center):
#     pent = nPentiga(name, sma, stats=True)
#
#     i = 0
#     for _sma, dist in s_sma, dist_center:
#         s_name = 'sub_ell_'
#         s_name += str(i)
#         pent.gen_sub_structure(s_name, _sma, dist, stats=True)
#         i += 1
#
#     return pent
