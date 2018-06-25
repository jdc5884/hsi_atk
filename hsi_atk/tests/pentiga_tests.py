from hsi_atk.Pentara.pentiga import Pentiga
import numpy as np
import unittest as ut


pentiga = Pentiga('pentiga', (3, 3, 5))

class PentigaUnittest(np.testing.TestCase):


    def test_instanciation(self):
        test_ell = np.load('../test_data/pentiga_test_ell.npy')

        np.testing.assert_array_equal(pentiga.structure.flatten(), test_ell.flatten())


    def test_add_ell(self):
        pentiga.gen_sub_ellipsoid('sub_0', (2, 2, 5), stats=True)
        test_ell = np.load('../test_data/pentiga_test_sub_ell.npy')
        n_ell = pentiga.sub_structures['sub_0'].structure

        np.testing.assert_array_equal(n_ell.flatten(), test_ell.flatten())


    def test_compose(self):
        comp_img = pentiga.compose(5)
        test_img = np.load('../test_data/pentiga_test_compose.npy')

        np.testing.assert_array_equal(comp_img.flatten(), test_img.flatten())


if __name__ == "__main__":
    ut.main()
