import numpy as np

__author__ = "David Ruddell"
__credits__ = ["David Ruddell"]
__license__ = "GPL"
__version__ = "0.0.1"
__status__ = "Development"


class Preprocess:

    def __init__(self, data, labels, args):
        self.data = data
        self.labels = labels

        self.filter_lp = False
        self.denoise = False
        self.histo = False
        self.histograms = None

        self.args = args

        self.parseArgs()

        self.getCovar()
        self.getMean()
        self.getStd()

    def parseArgs(self):

        pass

    def getCovar(self):
        pass

    def getMean(self):
        pass

    def getStd(self):
        pass
