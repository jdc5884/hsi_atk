import unittest
import os

from hsi_atk.pipeline import pipeline


class PipelineTest(unittest.TestCase):

    folder = os.getcwd()[:os.getcwd().index("hsi_atk")] + "hsi_atk/Developing/data/TestData"
    data_file = folder + "/TestData.h5"
    output_files = {"out0": folder + "/TestDataOut.h5"}
    args = {'folder': folder, "analysis":["exploratory", "preprocessing", "HailMary"],
            "data_file": data_file, "output_file": output_files}


    def test_run_analysis(self):
        self.pipeline = pipeline.Pipeline(self.args)
        self.pipeline.run_analysis()