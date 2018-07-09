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

    # initialize pentiga object with base structure
    pent0 = Pentiga('pent0', (30, 33, 240), stats=True)

    # generate sub_structure
    pent0.gen_sub_ellipsoid('sub_ell_0', (28, 29, 240), stats=True)
    pent0.scale_structure(-50, 500) # scale brightness of base structure
    pent0.add_func_bandwise(nfunc, (0, 240)) # scale brightness of base structure
    # pent0.add_linear(.5, 200, (0, 240))
    pent0.sub_structures['sub_ell_0'].scale_structure(-25, 200) # scale brightness of sub str
    pent0.sub_structures['sub_ell_0'].add_func_bandwise(nfunc, (0, 240)) # scale sub
    n_img = pent0.compose(5) # create composed image of base and sub structures

    time1 = time.time()
    print(time1-time0)

    return pent0, n_img

pent0, n_img = npentiga()

def wt_func(x):

    return None

structs = ['sub_ell_0']
str_wts = {'sub_ell_0': .01}

pent0.gen_wt_label(wt_func, structs, str_wts)

def lp_func(x):
    return None

def palm_func(x):
    return None

def lino_func(x):
    return None

def olei_func(x):
    return None

def stea_func(x):
    return None

lp_wts = {'sub_ell_0': {'lp': 0.5, 'palm': 0.05, 'lino': 0.2, 'oleic': 0.1, 'stea': 0.075}}

pent0.gen_lp_labels(lp_func, palm_func, lino_func, olei_func, stea_func, 1, structs, lp_wts)
