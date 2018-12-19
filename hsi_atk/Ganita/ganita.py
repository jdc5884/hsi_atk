import numpy as np


class Ganita:

    def __init__(self, args):
        self._chr = {}
        self._chr_rgs = {}

    def add_chr(self, name, chr_range, chr_length):
        self._chr[name] = np.random.rand(chr_range[0], chr_range[1], chr_length)
        self._chr_rgs[name] = chr_range

