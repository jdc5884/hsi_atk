import numpy as np
import scipy.ndimage as ndi
from sklearn.cluster import KMeans
from skimage import color, filters, feature, measure, morphology, segmentation
from skimage.segmentation import chan_vese
from hsi_atk.utils.hsi2color import hsi2gray


def filter_seg(AOI):
    """Runs simple chan_vese segmentaion and return segmented coordinates. May need to apply
    negation depending on the objects in image.
    AOI - ndarray of dim 2 or 3 (image or image subsection)
    returns rr,cc - 1darrays representing coordinate maps"""

    gray = hsi2gray(AOI)
    gseg = chan_vese(gray, mu=.99)
    rr,cc = np.where(gseg)
    return rr,cc


def get_edges(img):
    gray = hsi2gray(img)
    edges = feature.canny(gray, sigma=4)

    dt = ndi.distance_transform_edt(~edges)
    local_max = feature.peak_local_max(dt, indices=False, min_distance=25)

    markers = measure.label(local_max)
    labels = morphology.watershed(-dt, markers)

    regions = measure.regionprops(labels, intensity_image=gray)
    return regions





def fit_ply(AOI, rr, cc, bands=240, deg=5):
    """Fits a polynomial of specified degree across spectral WF of pixels in coordinate map
    specified by rr,cc.
    AOI - ndarray of dim 2 or 3 (image or image subsection)
    rr,cc - 1darrays representing coordinate maps
    bands - int number of bandwidths in pixel WFs
    deg - int degree of polynomial to be fit
    returns plys - 2darray of polynomial coefficients in specified region of image"""

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
    """Applies KMeans clustering to separate segmentation map into n_clusters classes.
    AOI - ndarray of dim 2 or 3 (image or image subsection)
    rr,cc - 1darrays representing coordinate maps
    n_clusters - int number of classes to cluster
    returns 1darray of ints representing classes
    NOTE: labels_ indices will match rr,cc indices as well as plys labels (when both using same rr,cc)"""

    mdl = KMeans(n_clusters=n_clusters)
    mdl.fit(AOI[rr,cc,:])
    return mdl.labels_


def get_ply_stats(labels, plys, n_clusters=8):
    """Finds averages and standard deviations of coefficients
    for a set of polynomials with respect to a class map (cluster labels)"""
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


def fit_ply_mdl(AOI,deg=5,n_clusters=8, return_counts=False):
    rr, cc = filter_seg(AOI)
    plys = fit_ply(AOI, rr, cc, deg=deg)
    labels = cluster_map(AOI, rr, cc, n_clusters=n_clusters)
    ply_stats = get_ply_stats(labels, plys, n_clusters=n_clusters)
    if return_counts:
        for i in range(n_clusters):
            count = (labels == i).sum()
            ply_stats[i]['count'] = count
        return ply_stats
    return ply_stats


if __name__ == '__main__':
    # from hsi_atk.utils.dataset import open_hsi_bil
    import matplotlib.pyplot as plt
    import h5py as h5
    hf = h5.File("/Volumes/RuddellD/hsi/labeled_set.h5", 'r')
    img = hf['RAW/B73.32.control'][:,:,:]
    # img = open_hsi_bil("../../Data/B73/32.control.bil")
    AOI = img[88:178,461:536,:]
    ply_stats = fit_ply_mdl(AOI)
    gray = hsi2gray(img)
    # denoised = filters.median(gray, selem=np.ones((5,5)))
    #
    edges = feature.canny(gray, sigma=4)
    # from scipy.ndimage import distance_transform_edt
    dt = ndi.distance_transform_edt(~edges)
    local_max = feature.peak_local_max(dt, indices=False, min_distance=25)
    #
    # peak_idx = feature.peak_local_max(dt, indices=True, min_distance=5)
    #
    # from skimage import measure, morphology, segmentation, color
    markers = measure.label(local_max)
    labels = morphology.watershed(-dt, markers)
    #
    regions = measure.regionprops(labels, intensity_image=gray)
    # points = np.hstack((regions[:,1], regions[:,0]))
    # region_means = [r.mean_intensity for r in regions]
    # plt.imshow(color.label2rgb(labels,image=gray))
    # plt.hist(region_means)

    # edges = feature.canny(gray, sigma=4)
    # points = np.array(np.nonzero(edges)).T
    # model_robust, inliers = measure.ransac(points, measure.EllipseModel, min_samples=3,
    #                                        residual_threshold=2, max_trials=5000)
    # plt.show()
