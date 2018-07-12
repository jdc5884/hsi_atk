from hsi_atk.Pentara.pentiga import Pentiga
import numpy as np
import unittest as ut


pentiga = Pentiga('pentiga', (3, 3, 5))

class PentigaUnittest(np.testing.TestCase):


    def test_instanciation(self):
        test_ell = np.load('../test_data/pentiga_test_ell.npy')

        np.testing.assert_array_equal(pentiga.structure.flatten(), test_ell.flatten())


    def test_add_ell(self):
        pentiga.gen_sub_structure('sub_0', (2, 2, 5), stats=True)
        test_ell = np.load('../test_data/pentiga_test_sub_ell.npy')
        n_ell = pentiga.sub_structures['sub_0'].structure

        np.testing.assert_array_equal(n_ell.flatten(), test_ell.flatten())


    def test_scale_ell(self):
        pentiga.gen_sub_structure('sub_1', (2, 2, 5), stats=False)
        test_ell = np.load('../test_data/pentiga_test_sub_scale.npy')
        n_pent = pentiga.sub_structures['sub_1']
        n_pent.scale_structure(-50, 200)

        np.testing.assert_array_equal(n_pent.structure.flatten(), test_ell.flatten())


    def test_scale_func(self):
        pentiga.gen_sub_structure('sub_2', (2, 2, 5), stats=False)
        test_ell = np.load('../test_data/pentiga_test_scale_func.npy')
        n_pent = pentiga.sub_structures['sub_2']
        def nfunc(x): return ((x - 2)**2 + 10)
        n_pent.add_func_bandwise(nfunc, 5)

        np.testing.assert_array_equal(n_pent.structure.flatten(), test_ell.flatten())


    def test_compose(self):
        comp_img = pentiga.compose(5).astype('f8')
        test_img = np.load('../test_data/pentiga_test_compose.npy').astype('f8')

        np.testing.assert_array_equal(comp_img.flatten(), test_img.flatten())


if __name__ == "__main__":
    ut.main()
