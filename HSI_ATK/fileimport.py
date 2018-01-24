import numpy as np
import rasterio


def loadImage(file):
    """
    Read in .bil HSI to array

    :param file: HSI file
    :return: numpy array of HSI data values
    """
    with rasterio.open(file) as src:
        array = np.array(src.read())

    return array


# all images captured by same remote sensor with same settings
def loadImageInfo(file):
    """
    Read in .bil.hdr for info on HSI data

    :param file: HSI info file
    :return:    array - array  wavelengths
                reflectance - int  reflectance scale factor
                data_size - array  data shape
                label - string  HSI image label
    """
    #TODO: import and return from .bil.hdr