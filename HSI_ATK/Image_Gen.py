import numpy as np

class Image_Gen():
    def __init__(self, s=None, ar=None):
        self.shape = s
        self.array = ar

        if s is None:
            self.shape = ar.shape
        if ar is None:
            self.array = np.zeros(s)

        x = np.linspace(0, s[0], 1)
        y = np.linspace(0, s[1], 1)
        self.mesh = np.meshgrid(x, y, sparse=True)

    # def conic(self, c, r, f):
