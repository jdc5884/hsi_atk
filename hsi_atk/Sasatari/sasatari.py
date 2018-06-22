import numpy as np
from skhyper import process


class Sasatari(process.Process):

    def __init__(self, X, scale=True):
        process.Process.__init__(self, X, scale)

        self.update()


    def alt_frame(self, face, scale=True):
        n_X = alt_view(self.data, face)

        sas = Sasatari(n_X, scale=scale)
        return sas


    def img_means(self):
        return np.mean(self.data, 2)


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
