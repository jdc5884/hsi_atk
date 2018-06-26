import numpy as np
import scipy as sp
from scipy import ndimage as ndi
from skimage.feature import canny
from skimage.filters import threshold_otsu, sobel
from skimage.morphology import closing, watershed


class Moratikara:

    def __init__(self, img):
        self._img = img
        self._gray = np.mean(img, 2)
        self._histo = np.histogram(self._gray, bins=np.arange(0, 256))

        self.edges = None
        self.fill = None

        self.kernels_ = None


    def find_edges_thresh(self, img, bands=None):
        '''
        Apply edge detection to given ndarray

        :param img: 3D ndarray, input image
        :param bands: array-like, list of ints representing bands to apply edge detection

        :return: array of bools representing threshold regions
        '''
        if bands is None:
            gray = np.mean(img[:, :, :], 2)
        else:
            gray = np.mean(img[:, :, bands], 2)

        thresh = threshold_otsu(gray, nbins=240)
        binary = img > thresh

        return binary


    def find_edges_canny(self):
        self.edges = canny(self._gray)


    def fill_edges(self):
        self.fill = ndi.binary_fill_holes(self.edges)


    def label_kernels(self):
        lbl_obs, nb_lbls = ndi.label(self.fill)
        sizes = np.bincount(lbl_obs.ravel())
        mask_sizes = sizes > 20
        mask_sizes[0] = 0
        self.kernels_ = mask_sizes[lbl_obs]


    def region_based(self):
        mrkrs = np.zeros_like(self._gray)
        mrkrs[self._gray < 200] = 1
        mrkrs[self._gray > 300] = 2
        elv_map = sobel(self._gray)
        segmentation = watershed(elv_map, mrkrs)
        segmentation = ndi.binary_fill_holes(segmentation - 1)
        lbld_krnls, _ = ndi.label(segmentation)
        self.kernels_ = lbld_krnls