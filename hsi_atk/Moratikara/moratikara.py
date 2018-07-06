import numpy as np
from skimage.draw import ellipsoid


class Moratikara:

    def __init__(self, sma):
        """

        :param sma: tuple of 3 ints - semi-major axes of ellipsoid representing
                    3d structure of kernel

        """

