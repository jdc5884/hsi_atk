import argparse, os, time

from Developing.simulation import simulation
from Developing.simulation import probabalistic

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
        parser.add_argument("-f", "--inputFile", type=str,
                            help="path to hdf5 file. The file should"
                            "contain at least the 'RAW' group containing"
                            "images labeled GENO.PACKET.TREATMENT"
                            "and additional attributes/groups/datasets"
                            "as required by the input options.")
        parser.add_argument("-e", "--exploratory", type=bool,
                            help="If there should be exploratory analysis run,"
                            "i.e. histograms, kmeans clustering")
        parser.add_argument("-p", "--preprocessing", type=bool,
                            help="If preprocessing should be run")
        parser.add_argument("-H", "--HailMary", type=bool,
                            help="Run all possible options,"
                            "store all outputs.")
        #TODO: add simulation types
        parser.add_argument("-s", "--simulation", type=bool,
                            help="If the analysis will be a simulation.")
        parser.add_argument("-t", "--simulationType", type=str,
                            help="What type of simulation should be used:\n"
                            "i.e. probablist, hierarchical")
        parser.add_argument("-b", "--bands", type=int,
                            help="Number of bands to simulate")


        args = parser.parse_args()
        self.data_file = args.inputFile
        self.simulation = args.simulation
        self.simulationType = args.simulationType

    def buildSimulation(self):
        # sim = simulation.Simulation()
        sim = probabalistic.Probabalistic()


    def runAnalysis(self):
        pass