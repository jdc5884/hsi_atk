import numpy as np


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
