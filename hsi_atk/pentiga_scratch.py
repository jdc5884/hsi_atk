from hsi_atk.Pentara.pentiga import Pentiga
import numpy as np


smas = [(240, 18, 27), (240, 4, 10), (240, 5, 6)]
cents = [(0, 0), (-3, -4), (2, 5)]


def nfunc(x):
    return (100.*np.sin(.02*x - 1)*np.cos(.05*x-5) + 600 + x)


def npentiga():
    # pent0 = basic_gen("pent0", (240, 20, 30), smas, cents)
    import time
    time0 = time.time()
    pent0 = Pentiga('pent0', (30, 33, 240), stats=True)
    pent0.gen_sub_ellipsoid('sub_ell_0', (28, 29, 240), stats=True)
    pent0.scale_structure(-50, 500)
    pent0.add_func_bandwise(nfunc, (0, 240))
    # pent0.add_linear(.5, 200, (0, 240))
    pent0.sub_structures['sub_ell_0'].scale_structure(-25, 200)
    pent0.sub_structures['sub_ell_0'].add_func_bandwise(nfunc, (0, 240))
    n_img = pent0.compose(5)
    time1 = time.time()
    print(time1-time0)
    return pent0, n_img

pent0, n_img = npentiga()