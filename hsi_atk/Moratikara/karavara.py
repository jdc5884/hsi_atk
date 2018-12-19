import numpy as np
import scipy.ndimage as ndi
from skimage.filters import threshold_otsu, sobel
from skimage.feature import canny
from skimage.morphology import watershed
from skimage.segmentation import chan_vese

from hsi_atk.utils.hsi2color import hsi2color, hsi2gray


class Karavara:

    def __init__(self):
        self._img = None
        self._gray = None
        self._rgb = None
        # self._histo = np.histogram(self._gray, bins=np.arange(0, 256))

        self.edges = None
        self.fill = None

        self.bbox = None

        self.kernels_ = None
        self.hilbert_ = None

    def fit(self, img, copy_img=True):
        if copy_img:
            self._img = img
        self._gray = np.mean(img, axis=2)
        self._rgb = hsi2color(img)
        self.find_edges_seg()

    def get_gray(self):
        return self._gray

    def get_rgb(self):
        return self._rgb

    def get_hsi(self):
        return self._img

    def find_edges_seg(self):
        """
        Apply edge detection by segmentation algorithm to given ndarray
        :return: array of bools representing threshold regions
        """
        gray = self.get_gray()
        gseg = chan_vese(gray, mu=.99)
        return gseg

    def find_edges_thresh(self):
        """
        Apply edge detection by threshold filtering to given ndarray
        :return: array of bools representing threshold regions
        """
        gray = self.get_gray()
        thresh = threshold_otsu(gray, nbins=240)
        binary = gray > thresh
        return binary


    def fill_edges(self):
        self.fill = ndi.binary_fill_holes(self.edges)

    # def label_kernels(self):
    #     lbl_obs, nb_lbls = ndi.label(self.fill)
    #     sizes = np.bincount(lbl_obs.ravel())
    #     mask_sizes = sizes > 20
    #     mask_sizes[0] = 0
    #     self.kernels_ = mask_sizes[lbl_obs]
    #
    # def region_based(self):
    #     mrkrs = np.zeros_like(self._gray)
    #     mrkrs[self._gray < 200] = 1
    #     mrkrs[self._gray > 300] = 2
    #     elv_map = sobel(self._gray)
    #     segmentation = watershed(elv_map, mrkrs)
    #     segmentation = ndi.binary_fill_holes(segmentation - 1)
    #     lbld_krnls, _ = ndi.label(segmentation)
    #     self.kernels_ = lbld_krnls
    #
    # # def hilbert(self, img):
        
