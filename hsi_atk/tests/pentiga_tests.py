from hsi_atk.Pentara.pentara import *
import numpy as np
import unittest as ut


# testPent = Pentara((11, 11, 15))
# img = testPent._img.copy().astype('f8')
# testPent._img.astype('f8')


class PentigaUnittest(np.testing.TestCase):


    # def __init__(self):
    #     self.testPent = Pentara((15, 11, 11))


    def test_instanciation(self):

        # test_arr = np.zeros((11, 11, 15)).astype('f8')
        # # img = testPent._img.copy().astype('f8')
        #
        # img_shape = np.array(img.shape)
        # test_arr_shape = np.array(test_arr.shape)
        #
        # img_mean = np.mean(img.flatten())
        # test_arr_mean = np.mean(test_arr.flatten())
        #
        # np.testing.assert_array_equal(img.flatten(), test_arr.flatten())
        #
        # np.testing.assert_array_equal(img_shape, test_arr_shape)
        # np.testing.assert_equal(img_mean, test_arr_mean)
        pass


    def test_add_ell(self):

        # testPent.add_ellipsoid((5, 5, 5), 5, 3, 3, (1, 0))
        # img = testPent._img.copy()
        # test_img = np.load("../../Data/test_ell.npy")
        #
        # np.testing.assert_array_equal(img.flatten(), test_img.flatten())
        pass


    # def run_tests(self):
    #     self.test_instanciation()
    #     self.test_add_ell()


if __name__ == "__main__":
    ut.main()
