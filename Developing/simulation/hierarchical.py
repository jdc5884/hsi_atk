__author__ = "David Ruddell"
__credits__ = ["David Ruddell"]
__license__ = "GPL"
__version__ = "0.0.1"
__status__ = "Development"

import numpy as np


#TODO: mode=linear, add 'poly1','poly2',...,'poly7' and 'randpoly' to mode options
#TODO: der='down' add 'flat', 'up', and 'rand' type options to der
def dist_coef(dict, mode='linear', der='down', dev=3, deg=5, nclass=8):
    # coef_func_degs = np.random.randint(0,5,deg)
    rd = np.random.rand((nclass,deg)) * dev * 2 - dev
    funcs = []

    for i in range(nclass):
        mean = dict[i]['mean']
        std = dict[i]['std']
        rn = rd[i]
        rdev = rn * dev
        if der =='down':
            yint = rdev * mean
            # yeq1 will be for non-constant slopes
            # yeq1 = -rdev * mean
            m = -rdev
            f = lambda r: m*r + yint
            dict[i]['func'] = f
    return dict
