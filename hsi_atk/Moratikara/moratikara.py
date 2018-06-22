import numpy as np
import scipy as sp
from skimage.filters import threshold_otsu
from skimage.morphology import closing


class Moratikara:

    def __init__(self):
        pass


    def find_edges(self, img, bands=None):
        '''
        Apply edge detection to given ndarray

        :param img: 3D ndarray, input image
        :param bands: array-like, list of ints representing bands to apply edge detection

        :return: array of bools representing threshold regions
        '''
        if bands:
            gray = np.mean(img[:, :, bands], 2)
        else:
            gray = np.mean(img[:, :, :], 2)

        thresh = threshold_otsu(gray, nbins=240)
        binary = img > thresh

        return binary
