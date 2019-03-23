import argparse, os, time

from Developing.simulation import simulation


__author__ = "David Ruddell"
__credits__ = ["David Ruddell", "Michael Suggs"]
__license__ = "GPL"


class Pipeline:

    def __init__(self):
        self.data_file = ""
        self.inputs = {}
        self.params = {}
        self.datasets = {}
        self.running_jobs = {}
        self.finished_jobs = {}
        self.output_files = {}

        self.simulation = False

        self.checkArgs()

        if self.simulation:
            self.buildSimulation()

        self.runAnalysis()

    def checkArgs(self):
        parser = argparse.ArgumentParser()
        # TODO: json builder for these options if desired
        parser.add_argument("-i", "--inputFile", type=str,
                            help="path to hdf5 file. The file should"
                            "contain at least the 'RAW' group containing"
                            "images labeled GENO.PACKET.TREATMENT"
                            "and additional attributes/groups/datasets"
                            "as required by the input options.")
        parser.add_argument("-o", "--outputFile", type=str,
                            help="hdf5 file for output.")
        parser.add_argument("-j", "--jsonInput", type=str,
                            help="Input json with analysis pipeline params")
        parser.add_argument("-H", "--HailMary", type=bool,
                            help="Run all possible options,"
                            "store all outputs.")

        #TODO: add simulation types
        parser.add_argument("-s", "--simulation", type=bool, default=False,
                            help="If the analysis will be a simulation.")
        parser.add_argument("-t", "--simulationType", type=str,
                            help="What type of simulation should be used:\n"
                            "i.e. probablist, hierarchical")
        parser.add_argument("-b", "--bands", type=int,
                            help="Number of bands to simulate")
        parser.add_argument("-d", "--imageDim", type=int, nargs="+",
                            help="Dimensions of the simulated images")
        parser.add_argument("-n", "--numberSim", type=int,
                            help="Number of images to create in the simulation")
        parser.add_argument("-r", "--rolling", type=bool, default=False,
                            help="Create the simulation as a generator."
                                 "Mutually exclusive with \"-n\" argument.")

        #TODO: add preprocessing and exploratory args
        parser.add_argument("-p", "--preprocessing", type=bool,
                            help="If preprocessing should be run")
        parser.add_argument("-e", "--exploratory", type=bool,
                            help="If there should be exploratory analysis run,"
                                 "i.e. histograms, kmeans clustering")

        #TODO: add analysis and evaluation args


        args = parser.parse_args()
        self.data_file = args.inputFile

        if args.simulation:
            self.simulation = args.simulation
            self.simulationType = args.simulationType
            self.imageDim = tuple(args.imageDim)
            self.n_images = args.numberSim
            self.rolling = args.rolling

    def buildSimulation(self):
        sim = simulation.Simulation(n_images=self.n_images, shape=self.imageDim, type=self.simulationType,
                                    as_generator=self.rolling)


    def runAnalysis(self):
        pass