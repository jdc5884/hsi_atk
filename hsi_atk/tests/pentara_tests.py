from archives.Pentara import Pentara
import numpy as np
import unittest as ut


class PentaraUnittest(np.testing.TestCase):


    def test_instanciation(self):
        testPent = Pentara((11, 11, 15))
        img = testPent._img.copy()

        test_arr = np.zeros((11, 11, 15))

        img_shape = np.array(img.shape)
        test_arr_shape = np.array(test_arr.shape)

        img_mean = np.mean(img.flatten())
        test_arr_mean = np.mean(test_arr.flatten())

        np.testing.assert_array_equal(img.flatten(), test_arr.flatten())

        np.testing.assert_array_equal(img_shape, test_arr_shape)
        np.testing.assert_equal(img_mean, test_arr_mean)


    def test_add_ell(self):
        testPent = Pentara((11, 11, 15))

        testPent.add_ellipsoid((5, 5, 5), 5, 3, 3, (1, 0))
        img = testPent._img.copy()
        test_img = np.load("../test_data/pentara_test_ell.npy")

        np.testing.assert_array_equal(img.flatten(), test_img.flatten())


if __name__ == "__main__":
    ut.main()
