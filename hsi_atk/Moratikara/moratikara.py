import numpy as np
from skimage.draw import ellipsoid


class Moratikara:

    def __init__(self, shape):
        self.kernel = np.zeros(shape)
