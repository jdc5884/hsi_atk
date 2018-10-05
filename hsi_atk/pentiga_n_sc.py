from hsi_atk.Pentara.pentiga_n import Pentiga_n
import numpy as np


def scale_fn_par(x, y, z, k=100, p=5, q=5, r=5):
    return p*(x-20)**3 + q*(y-20)**3 + r*z**4 + k


def scale_fn(x, y, z):
    return scale_fn_par(x, y, z)


def nfunc(x, y, z):
    return 10. * np.sin(.01 * x - 1) + np.cos(.05 * x - 5) + 10 + x + \
           10. * -np.cos(.01 * y - 1) + np.cos(.05 * y - 5) + 10 + y + \
           10. * np.sin(.05 * z - 1) * np.cos(.05 * z - 5) + 100 + z**1.05


def nfunc_par(x, y, z, a=.02, a0=.04, b=.02, b0=.04, c=.03, c0=.04, h0=0, h1=0, h2=50):
    return 10. * np.sin(a * x - 1) + np.cos(a0 * x - 5) + h0 + x + \
           10. * -np.cos(b * y - 1) + np.cos(b0 * y - 5) + h1 + y + \
           10. * np.sin(c * z - 1) * np.cos(c0 * z - 5) + h2 + z**1.05


def nfunc_fn(x, y, z):
    return nfunc_par(x, y, z)


def wt_func(x):
    return (x ** 0.75) / 1000


def lp_func(x): return x ** 0.95


def palm_func(x): return x ** 0.45


def lino_func(x): return x ** 0.65


def olei_func(x): return x ** 0.55


def stea_func(x): return x ** 0.185


pent_n = Pentiga_n("pent1", (20, 20))
pent_n.set_scale_func(nfunc)
pent_n.set_wt_lbl_fn(wt_func)
pent_n.set_lp_fn(lp_func)
pent_n.set_palm_fn(palm_func)
pent_n.set_lino_fn(lino_func)
pent_n.set_olei_fn(olei_func)
pent_n.set_stea_fn(stea_func)

pent_s1 = Pentiga_n("pents1", (8,6))
pent_s1.set_dist_center((2,2))
pent_s1.set_scale_func(nfunc_fn)

pent_n.add_substructure(pent_s1)

img_comp = pent_n.compose()
wt = pent_n.gen_wt_label()
lp = pent_n.gen_lp_labels()
