import numpy as np

from skimage.draw import circle, ellipsoid, ellipsoid_stats, ellipse, ellipse_perimeter
from skimage.util import random_noise
from skimage.restoration import denoise_wavelet

# from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, label_ranking_average_precision_score
# from sklearn.svm import SVC, LinearSVC
# from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
#
# from mlens.ensemble import BlendEnsemble
# from mlens.metrics import make_scorer


class Gen3d:
    """
    Base class for HSI image generator. Holds base image object
    """

    def __init__(self, shape):
        self._s = shape
        self._img = np.zeros(shape)
        # self._labels = np.zeros(shape[1], shape[2])
        self.obs = {}
        # self.obs_count = 0


    def add_noise(self, **kwargs):
        '''
        Adds noise to HSI via skimage utils

        :param kwargs: arguments for noise generators
        :return: None, alters base image from self
        '''
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
        ell = ellipsoid(a, b, c, levelset=True, )
        ell *= scale[0]
        ell += scale[1]

        d1, d2, d3 = self._s
        ed1, ed2, ed3 = ell.shape

        c1 = int(ed2/2)+1
        c2 = int(ed3/2)+1

        rr, cc = ellipse(c1, c2, b, c, shape=(ed2, ed3))

        rr1, cc1 = ellipse(center[0], center[1], b, c, shape=(d2, d3), rotation=rot)

        if d1<ed1:
            self._img[:d1, rr1, cc1] += ell[:d1, rr, cc]
        else:
            self._img[:ed1, rr1, cc1] += ell[:ed1, rr, cc]

        if stats:
            stat = ellipsoid_stats(a, b, c)
            try:
                self.obs[name] = [center, stat]
            except ValueError:
                raise Exception("if stats=True, name must be a unique string not in self.obs")


    def add_structure(self, obj, name, rot=0.0, scale=(1, 0)):
        self.obs[name] = obj
        x, y, z = obj.get_center()
        a, b, c = obj.axes
        d1, d2, d3 = self._s
        ed1, ed2, ed3 = obj.structure.shape
        c1, c2 = int(ed2/2)+1, int(ed3/2)+1
        rr1, cc1 = ellipse(c1, c2, b, c, shape=(ed2, ed3), rotation=rot)
        rr, cc = ellipse(x-1, y-1, b, c, shape=self._s, rotation=rot)
        if d1<ed1:
            self._img[:d1, rr, cc] += obj.structure[:d1, rr1, cc1]*scale[0]+scale[1]
        else:
            self._img[:ed1, rr, cc] += obj.structure[:ed1, rr1, cc1]*scale[0]+scale[1]


# TODO: Gen3dStructure image padding, splitting+padding, cutting+padding, permuting+padding
class Gen3dStructure(object):
    """
    HSI object to store data about image substructure and easily add or remove it
    """

    def __init__(self):
        pass


    def set_center(self, x, y, z):
        self._x = x
        self._y = y
        self._z = z


    def get_center(self):
        return (self._x, self._y, self._z)


    def gen_ellipsoid(self, a, b, c, stats=False, **kwargs):
        self.axes = (a, b, c)
        self.structure = ellipsoid(a, b, c, **kwargs, levelset=True)
        if stats:
            self.stats = ellipsoid_stats(a, b, c)


    def scale_structure(self, mult, add):
        self.structure *= mult
        self.structure += add


    def gen_ell_stats(self):
        a, b, c = self.axes
        self.stats = ellipsoid_stats(a, b, c)


    def set_label(self, label):
        self.label = label


# gen3 = Gen3d((240, 50, 50))
#
# ell1 = Gen3dStructure()
# ell1.set_center(24, 24, 120)
# ell1.gen_ellipsoid(120, 6, 5, stats=True)
# ell1.scale_structure(-250, 300)
#
# gen3.add_structure(ell1, 'ell1')
#
# ell2 = Gen3dStructure()
# ell2.set_center(10, 11, 120)
# ell2.gen_ellipsoid(120, 6, 5, stats=True)
# ell2.scale_structure(-250, 300)
#
# gen3.add_structure(ell2, 'ell2', rot=np.pi/2)



def gen3d(shape, r_range, h_range, n_ell=1, noise=False, label=False, ell_stats=False):
    # TESTING+REFORMATING
    '''
    3-dimensional image generator for hsi images

    :param shape:
    :param r_range:
    :param h_range:
    :param n_ell:
    :param noise:
    :return:
    '''

    data = np.zeros(shape)

    if noise:
        data = random_noise(data)
        data *= 200

    ells = []
    r1 = np.random.randint(r_range[0], r_range[1], n_ell)
    r2 = np.random.randint(r_range[0], r_range[1], n_ell)
    h = np.random.randint(h_range[0], h_range[1], n_ell)

    # assuming shape is 240, 200, 200
    # and r_range max val < 30
    # and h_range max val < 240
    # and 1 <= n_ell <= 6
    centers = [
        [120, 40, 40],
        [120, 40, 80],
        [120, 40, 120],
        [120, 80, 40],
        [120, 80, 80],
        [120, 80, 120],
        [120, 120, 40],
        [120, 120, 80],
        [120, 120, 120],
    ]

    for i in range(0, n_ell):
        ell = ellipsoid(h[i], r1[i], r2[i], levelset=True)
        ell *= -500
        ell += 500
        # c = centers[i]
        # xL = c[1] - r1[i] - 3
        # xR = c[1] + r1[i] - 3
        # yL = c[1] - r2[i] - 3
        # yR = c[1] + r2[i] - 3
        # bL = c[0] - h[i]
        # bR = c[0] + h[i]
        # data[bL:bR, xL:xR, yL:yR] += ell
        # ells.append(ell)
        data = add_ell(data, ell)

    if ell_stats:
        stats = []
        for i in range(0, n_ell):
            stat = ellipsoid_stats(h[i], r1[i], r2[i])
            stats.append(stat)


    return data


def add_ell(data, ell, r, c, rot=0.):
    """
    adds ellipsoid to image at particular center and rotation
    :param data: ndarray of 3 dimensions, base image
    :param ell: ellipsoid level-set ndarray
    :param r: x-axis center on base image array
    :param c: y-axis center on base image array
    :param rot: rotation applied to ellipsoid
        - currently only accepts multiples of pi/4
    :return: ndarray, base array with added level-set
    """
    d_shape = data.shape
    e_shape = ell.shape

    if len(d_shape) != len(e_shape):
        raise ValueError('base image and ellipse must be of same dimension(shape)')

    # for i in range(0, len(d_shape)):
    #     pass

    band_c = np.random.randint(0, 20)
    print(e_shape[0])
    band_c2 = band_c + e_shape[0]
    x_c = np.random.randint(40, 120)
    x_c2 = x_c + e_shape[1]
    y_c = np.random.randint(40, 120)
    y_c2 = y_c + e_shape[2]

    data[band_c:band_c2, x_c:x_c2, y_c:y_c2] += ell

    return data

# d = gen3d((59, 200, 200), (12, 18), (20, 22), n_ell=1, noise=True)


# test = np.random.rand(200, 200)
# test = test.reshape(10, 80, 50)

# seed = np.random.seed(2018)

def silly_gen(add_noise=True, rem_noise=False):
    """
    A simple 3d generator for simulated HSI images using ellipsoid and ellipses
    for the image and label space

    :param add_noise: boolean, to apply random noise
    :param rem_noise: boolean, to de-noise
    :return:
        - data_pix - 2d ndarray organizing image into two axes, pixel and wavelength
        - spacial_pax - 1d ndarray organizing label space into pixel and label
        - data - 3d ndarray of base image created in form band,line,col
        - spacial_data - 2d ndarray of image label space
    """
    # creating base label space
    spacial_data = np.zeros((50, 50))

    # creating base image
    data = np.zeros((240, 50, 50))

    if add_noise:
        data = random_noise(data)

    # creating label space
    rr, cc = ellipse(14, 19, 12, 18, (27, 39))
    rr1, cc1 = ellipse(15, 20, 5, 8, (27, 39))
    rr2, cc2 = ellipse(12, 13, 3, 3, (27, 39))
    rr3, cc3 = ellipse(18, 23, 3, 3, (27, 39))

    #prr, pcc = ellipse_perimeter(14, 19, 12, 18, shape=(27, 39))
    # prr2, pcc2 = ellipse_perimeter(14, 19, 11, 17, shape=(27, 39))

    # assigning labels to label space
    spacial_data[rr, cc] += 1
    spacial_data[rr1, cc1] *= 2
    spacial_data[rr2, cc2] *= 3
    spacial_data[rr3, cc3] *= 5
    # spacial_data[rr3, cc3] *= 7
    #spacial_data[prr, pcc] = 11
    # spacial_data[prr2,pcc2] = 11

    # create ellipse for spectral values at depth
    ell = ellipsoid(240, 12, 18, levelset=True)

    # making more similar to real data
    ell *= -500
    ell += 2500
    # ell1 *= -250
    # ell1 += 1000
    # adding ellipse to base image
    data[0:240, rr, cc] += ell[0:240, rr, cc]
    data[0:240, rr1, cc1] += ell[0:240, rr1, cc1]
    ell1 = ell*0.25
    data[25:37, rr2, cc2] -= ell1[25:37, rr2, cc2]
    data[50:63, rr3, cc3] += ell[50:63, rr3, cc3]

    if denoise:
        data = denoise_wavelet(data, mode='soft', multichannel=True)
    # Reshaping image for label classification
    data_n = data.swapaxes(0, 2)
    # data_n = data_n.swapaxes(0, 1)
    # Reshape to 2D in form samples*lines,bands
    # data_pix = data_n.transpose(2, 0, 1).reshape(40000, -1)
    d1, d2, d3 = data_n.shape
    data_pix = data_n.transpose(2, 0, 1).reshape(d1*d2, -1)

    spacial_pix = spacial_data.reshape(d1*d2, -1).ravel()

    # Reshaping label space for classification
    # print(spacial_data.shape)
    # sd1, sd2, sd3 = spacial_data.shape
    # spacial_pix = spacial_data.swapaxes(0, 2).transpose(2, 0, 1).reshape(sd2*sd3, -1)
    # print(spacial_pix.shape)
    # split train and test data
    return data_pix, spacial_pix, data, spacial_data

# data_pix, spacial_pix, data, spacial_data = silly_gen()

# X_train, X_test, y_train, y_test = train_test_split(data_pix, spacial_pix, test_size=.23, random_state=seed)
#
# sc = StandardScaler()
# pca = PCA(whiten=True)
#
# pre_cases = {
#     'case-1': [sc],
#     'case-2': []
# }
#
# ests = [
#     ('etc', ExtraTreesClassifier()),
#     ('rtc', RandomForestClassifier()),
#     ('mlp', MLPClassifier()),
#     ('svc', SVC())
# ]
#
# est = {
#     'case-1': ests,
#     'case-2': ests
# }
#
# # fit + pred
# etc = BlendEnsemble(test_size=0.23, shuffle=True, random_state=seed, n_jobs=1)
# etc.add(est, preprocessing=pre_cases)
# etc.add_meta(SVC())
# etc.fit(X_train, y_train)
# y_pred = etc.predict(X_test)
#
# # scores
# print(precision_score(y_test, y_pred, average='micro'))
# print(accuracy_score(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))

#
# data_base = np.random.rand(20, 200, 200)
# data = np.zeros((20, 200, 200))
#
# ell = ellipsoid(10, 10, 10, levelset=True)
# ell *= -500
# ell += 1000
#
# data = data_base
# data = random_noise(data)
# data *= 200
# data[:, 10:33, 12:35] += ell[3:,: ,: ]
#