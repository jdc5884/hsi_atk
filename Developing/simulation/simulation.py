import numpy as np

from Developing.simulation import probabalistic

__author__ = "David Ruddell"
__credits__ = ["David Ruddell"]
__license__ = "GPL"
__version__ = "0.0.1"
__status__ = "Development"


class Simulation:

    def __init__(self, args):
        self.args = args
        self.sym_type = ""

        self.parseArgs()
        self.buildSim()

    def parseArgs(self):
        pass

    def buildSim(self):
        pass
