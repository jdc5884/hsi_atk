import numpy as np
# from skhyper import process


# class Sasatari(process.Process):

    # def __init__(self, X, scale=True):
    #     process.Process.__init__(self, X, scale)
    #     # self.histo = np.histogram(X, bins=np.arange(0, 256))
    #     self.update()

class Sasatari(object):

    def __init__(self, raw, dec_mode="PCA", seg_mode="thresh", seg_shape="rectangle",
                 dset=None, dset_path=None):
        # Analysis data members
        self.RAW = raw              # raw pixel-cube of HSI
        self.RGB = None             # rgb mask of HSI, looking to deprecate in favor of RGBI
        self.RGBI = None            # rgbi mask of HSI (4th layer in image average IR bands)
        self.GRAY = None            # grayscale of HSI
        self.GRAYV = None           # grayscale of visible spectrum
        self.GRAYIR = None          # grayscale of IR bands
        self.SEGS = None            # segmentation maps
        self.OBJ_COUNT = None       # separated objects found by segmentation maps

        # Options setting data members
        self.DEC_MODE = dec_mode    # Decomposition mode (PCA, etc.)
        self.SEG_MODE = seg_mode    # Segmentation mode (contour matching, bounding boxes, etc.)
        self.SEG_SHAPE = seg_shape  # Shape to fit in segmentation (rectangle, ellipse, etc. not required for some modes)
        self.DSET = dset            # Dataset h5py.File object
        self.DSET_PATH = dset_path  # Path to dataset (/.../mydataset.h5)

    def get_raw(self):
        return self.RAW

    def set_raw(self, raw):
        self.RAW = raw

    def get_rgb(self):
        return self.RGB

    def get_rgbv(self):
        return self.RGBV

    def get_rgbir(self):
        return self.RGBIR

    def get_segs(self):
        return self.SEGS

    def get_obj_count(self):
        return self.OBJ_COUNT

    def get_dec_mode(self):
        return self.DEC_MODE

    def set_dec_mode(self, dec_mode):
        self.DEC_MODE = dec_mode

    def get_seg_mode(self):
        return self.SEG_MODE

    def set_seg_mode(self, seg_mode):
        self.SEG_MODE = seg_mode

    def get_seg_shape(self):
        return self.SEG_SHAPE

    def set_seg_shape(self, seg_shape):
        self.SEG_SHAPE = seg_shape

    def get_dset(self):
        return self.DSET

    def set_dset(self, dset):
        self.DSET = dset

    def get_dset_path(self):
        return self.DSET_PATH

    def set_dset_path(self, dset_path):
        self.DSET_PATH = dset_path


def alt_view(X, face, transpose=False, trp_axes=0):
    '''
    Return image with given face
    Expected input shape (lines, cols, bands)

    :param X: 3D ndarray, intput image
    :param face: string, specified face for output image
    :param transpose: bool, to transpose matrix
    :param trp_axes: tuple of ints, axes to transpose if transpose

    :return: 3D ndarray, output image
    '''
    if face == 'left':
        return X.swapaxes(0, 2)

    elif face == 'right':
        return X.swapaxes(0, 2).flip(2)

    elif face == 'top':
        return X.swapaxes(1, 2).swapaxes(0, 1)

    elif face == 'bottom':
        return X.swapaxes(0, 2).swapaxes(0, 1).flip(2)

    elif face == 'back':
        return X.flip(2)

    if transpose:
        X.transpose(trp_axes)

    return X


def image2pixels(img, labels=None):
    '''
    Transform ndarray image matrix to 1D matrix of pixels
    optional labels input to reshape labels according to image reshaping

    :param img: 3D ndarray, image input (lines, cols, bands)
    :param labels: ndarray of pixel labels

    :return: img_pixels, ndarray of pixel values
    '''
    dims = img.shape

    if len(dims) == 3:
        img_pixels = img.reshape(dims[0]*dims[1], -1)

    else:
        raise Exception("input image array is not of 3 dimensions!")

    if labels != None:
        lbl_pixels = labels.reshape(dims[0]*dims[1], -1)
        return img_pixels, lbl_pixels

    return img_pixels
