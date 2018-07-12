from hsi_atk.Pentara.pentiga import Pentiga
import numpy as np


smas = [(240, 18, 27), (240, 4, 10), (240, 5, 6)]
cents = [(0, 0), (-3, -4), (2, 5)]


def nfunc(x):
    return 100.*np.sin(.02*x - 1)*np.cos(.05*x-5) + 600 + x


def npentiga(name, base_sma, sub_smas):
    # pent0 = basic_gen("pent0", (240, 20, 30), smas, cents)
    import time
    time0 = time.time()

    # initialize pentiga object with base structure
    pent = Pentiga(name, base_sma, stats=True)
    pent.add_func_bandwise(nfunc, (0, 240))  # scale brightness of base structure
    # pent0.add_linear(.5, 200, (0, 240))

    # generate sub_structures
    i = 0
    for sma in sub_smas:
        sub_name = name + "_" + str(i)
        pent.gen_sub_structure(sma, name=sub_name, stats=True)

        pent.sub_structures[sub_name].scale_structure(-25, 200)  # scale brightness of sub str
        pent.sub_structures[sub_name].add_func_bandwise(nfunc, (0, 240))  # scale sub
        i += 1

    n_img = pent.compose()  # create composed image of base and sub structures

    time1 = time.time()
    print(time1-time0)

    return pent, n_img


sub_smas = [
    (15, 13, 240),
    (12, 11, 240),
    (20, 25, 240),
]

pent0, n_img0 = npentiga("pent", (35, 37, 240), sub_smas)


def wt_func(x):
    return x ** 0.75


str_wts = {'pent_0': .01, 'pent_1': .02, 'pent_2': .5}

pent0.gen_wt_label(wt_func, str_wts.keys(), str_wts)


def lp_func(x):
    return x ** 0.75


def palm_func(x):
    return x ** 0.25


def lino_func(x):
    return x ** 0.5


def olei_func(x):
    return x ** 0.5


def stea_func(x):
    return x ** 0.5


lp_wts = {'pent_0': {'lp': 0.2, 'palm': 0.05, 'lino': 0.04, 'olei': 0.01, 'stea': 0.15},
          'pent_1': {'lp': 0.2, 'palm': 0.01, 'lino': 0.02, 'olei': 0.001, 'stea': 0.75},
          'pent_2': {'lp': 0.5, 'palm': 0.5, 'lino': 0.2, 'olei': 0.1, 'stea': 0.075}}

pent0.gen_lp_labels(lp_func, palm_func, lino_func, olei_func, stea_func, 1, str_wts.keys(), lp_wts)
