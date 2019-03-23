__author__ = "David Ruddell"
__credits__ = ["David Ruddell"]
__license__ = "GPL"
__version__ = "0.0.1"
__status__ = "Development"

import numpy as np
from sklearn.cluster import KMeans
from skimage.segmentation import chan_vese
from hsi_atk.utils.hsi2color import hsi2gray


def filter_seg(AOI):
    gray = hsi2gray(AOI)
    gseg = chan_vese(gray, mu=.99)
    rr,cc = np.where(gseg)
    return rr,cc


def fit_ply(AOI, rr, cc, bands=240, deg=5):
    plys = []
    t = np.linspace(0,bands-1,bands)
    for i in range(rr.size):
        y = AOI[rr[i],cc[i],:]
        ply = np.polyfit(t,y,deg=deg)
        plys.append(ply)
    plys = np.stack(plys,axis=-1)
    plys = plys.swapaxes(0,1)
    return plys


def cluster_map(AOI, rr, cc, n_clusters=8):
    mdl = KMeans(n_clusters=n_clusters)
    mdl.fit(AOI[rr,cc,:])
    return mdl.labels_


def get_ply_stats(labels, plys, n_clusters=8):
    ply_stats = {}
    for i in range(n_clusters):
        ply_stats[i] = {}
        lbmap = np.where(labels==i)
        ply_i = plys[lbmap,:]
        ply_mean = np.mean(ply_i,axis=1)
        ply_std = np.std(ply_i,axis=1)
        ply_stats[i]['mean'] = ply_mean
        ply_stats[i]['std'] = ply_std

    return ply_stats


def fit_ply_mdl(AOI,deg=5,n_clusters=8):
    rr, cc = filter_seg(AOI)
    plys = fit_ply(AOI, rr, cc, deg=deg)
    labels = cluster_map(AOI, rr, cc, n_clusters=n_clusters)
    ply_stats = get_ply_stats(labels, plys, n_clusters=n_clusters)
    return ply_stats


if __name__ == '__main__':
    from hsi_atk.utils.dataset import open_hsi_bil
    img = open_hsi_bil("../../Data/B73/32.control.bil")
    AOI = img[88:178,461:536,:]
    ply_stats = fit_ply_mdl(AOI)
