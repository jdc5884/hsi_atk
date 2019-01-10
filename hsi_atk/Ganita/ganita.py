import numpy as np


class Ganita:

    def __init__(self, name):
        self._name = name
        self._chr = {}

    def add_chr(self, name, chr_range, chr_length):
        genes = []

        for i in range(chr_length):
            gene = (chr_range[1]-chr_range[0])*np.random.rand() + chr_range[0]
            genes.append(gene)

        self._chr[name] = {'genes': genes,
                           'length': chr_length,
                           'range': chr_range}

    def get_chr(self, name, return_length=False, return_range=False):
        if not return_length and not return_range:
            return self._chr[name]['genes']


    def gen_chr_ranges(self, n_chr, chr_range):
        chr_ranges = []

        for i in range(n_chr):
            rang = np.random.randint(chr_range[0], chr_range[1], 2)
            chr_ranges.append(rang)

        return chr_ranges


