from hsi_atk.Pentara.pentara import *
import numpy as np
import unittest as ut


testPent = Pentara((15, 11, 11))
testPent._img.astype('f8')


class PentaraUnittest(ut.TestCase):


    # def __init__(self):
    #     self.testPent = Pentara((15, 11, 11))


    def test_instanciation(self):
        test_arr = np.zeros((15, 11, 11)).astype('f8')
        np.testing.assert_almost_equal(testPent._img, test_arr, 2)



    def test_add_ell(self):
        testPent.add_ellipsoid((5, 5, 5), 5, 3, 3, (1, 0))
        test_img = np.load("../../Data/test_ell.npy")
        np.testing.assert_almost_equal(testPent._img, test_img, 2)


    def run_tests(self):
        self.test_instanciation()
        self.test_add_ell()


if __name__ == "__main__":
    ut.main()
