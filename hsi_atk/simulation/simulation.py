__author__ = "David Ruddell"
__credits__ = ["David Ruddell"]
__license__ = "GPL"
__version__ = "0.0.1"
__status__ = "Development"


class Simulation:

    def __init__(self, n_images=None, shape=None, type=None, as_generator=False):
        self.n_images = n_images  # int number of images
        self.shape = shape  # tuple shape of images (x,y,z)
        self.sym_type = type  # puresim, hybridsim, augmentedsim
        self.as_generator=as_generator  # return generator object producing sim data

        self.buildSim()

    def pureSim(self):
        pass

    def hybridSim(self):
        pass

    def augmentedSim(self):
        pass

    def buildSim(self):
        pass
