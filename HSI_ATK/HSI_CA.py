from time import time

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import rasterio
import sklearn.decomposition as dec

from pprint import pprint

from skimage.io import imread_collection, concatenate_images

# Spectral bands from .hdr file
wavelength= [393.91, 395.982, 398.054, 400.126, 402.198, 404.27, 406.342, 408.414, 410.486, 412.558, 414.63, 416.702, 418.774, 420.846, 422.918, 424.99, 427.062, 429.134, 431.206, 433.278, 435.35, 437.422, 439.494, 441.566, 443.638, 445.71, 447.782, 449.854, 451.926, 453.998, 456.07, 458.142, 460.214, 462.286, 464.358, 466.43, 468.502, 470.574, 472.646, 474.718, 476.79, 478.862, 480.934, 483.006, 485.078, 487.15, 489.222, 491.294, 493.366, 495.438, 497.51, 499.582, 501.654, 503.726, 505.798, 507.87, 509.942, 512.014, 514.086, 516.158, 518.23, 520.302, 522.374, 524.446, 526.518, 528.59, 530.662, 532.734, 534.806, 536.878, 538.95, 541.022, 543.094, 545.166, 547.238, 549.31, 551.382, 553.454, 555.526, 557.598, 559.67, 561.742, 563.814, 565.886, 567.958, 570.03, 572.102, 574.174, 576.246, 578.318, 580.39, 582.462, 584.534, 586.606, 588.678, 590.75, 592.822, 594.894, 596.966, 599.038, 601.11, 603.182, 605.254, 607.326, 609.398, 611.47, 613.542, 615.614, 617.686, 619.758, 621.83, 623.902, 625.974, 628.046, 630.118, 632.19, 634.262, 636.334, 638.406, 640.478, 642.55, 644.622, 646.694, 648.766, 650.838, 652.91, 654.982, 657.054, 659.126, 661.198, 663.27, 665.342, 667.414, 669.486, 671.558, 673.63, 675.702, 677.774, 679.846, 681.918, 683.99, 686.062, 688.134, 690.206, 692.278, 694.35, 696.422, 698.494, 700.566, 702.638, 704.71, 706.782, 708.854, 710.926, 712.998, 715.07, 717.142, 719.214, 721.286, 723.358, 725.43, 727.502, 729.574, 731.646, 733.718, 735.79, 737.862, 739.934, 742.006, 744.078, 746.15, 748.222, 750.294, 752.366, 754.438, 756.51, 758.582, 760.654, 762.726, 764.798, 766.87, 768.942, 771.014, 773.086, 775.158, 777.23, 779.302, 781.374, 783.446, 785.518, 787.59, 789.662, 791.734, 793.806, 795.878, 797.95, 800.022, 802.094, 804.166, 806.238, 808.31, 810.382, 812.454, 814.526, 816.598, 818.67, 820.742, 822.814, 824.886, 826.958, 829.03, 831.102, 833.174, 835.246, 837.318, 839.39, 841.462, 843.534, 845.606, 847.678, 849.75, 851.822, 853.894, 855.966, 858.038, 860.11, 862.182, 864.254, 866.326, 868.398, 870.47, 872.542, 874.614, 876.686, 878.758, 880.83, 882.902, 884.974, 887.046, 889.118]

# Read in HSI data to 3D array in form bands,lines,samples
with rasterio.open("Data/32.control.bil") as src:
    array = np.array(src.read())

# Change array from bands,lines,samples to samples,lines,bands
n_array = array.swapaxes(0,2)
# Reshape to 2D in form samples*lines,bands
array_2d = n_array.transpose(2,0,1).reshape(320000,-1)

# stats = []
rng = np.random.RandomState(30)

band_c = 0


#TODO: architect class
class HSI_CA(object):

    def __init__(self,data,samples=None,lines=None,bands=None):
        self.data_ = data
        self.stats_ = dict()
        self.aoi = None

        if samples != None or lines != None or bands != None:
            self.setAOI(samples,lines,bands)

    def perPixelBandVar(self,n_components,estimator):
        if n_components == int:



    def setAOI(self,samples,lines,bands):
        samp = samples[1]-samples[0]
        line = lines[1]-lines[0]
        bdata = self.data_[samples[0]:samples[1],lines[0]:lines[1],bands[0]:bands[1]]
        self.aoi = bdata.transpose(2,0,1).reshape(samp*line,-1)






image = HSI_CA(n_array)



    #
    # CA = []
    # estimators = [
    #     ('PCA using randomized SVD',
    #      dec.PCA(n_components=n_components, svd_solver='randomized',
    #                        whiten=True)),
    #
    #     ('Independent components - FastICA',
    #      dec.FastICA(n_components=n_components, whiten=True)),
    #
    #     ('Sparse comp. - MiniBatchSparsePCA',
    #      dec.MiniBatchSparsePCA(n_components=n_components, alpha=0.8,
    #                                       n_iter=100, batch_size=3,
    #                                       random_state=rng))
    # ]
    #
    # for name, estimator in estimators:
    #     print("Extracting top %d components via %s..." % (n_components, name))
    #     t0 = time()
    #     estimator.fit(data)
    #     train_time = (time() - t0)
    #     print("train time: %0.3fs" % train_time)
    #     if hasattr(estimator, 'cluster_centers_'):
    #         components_ = estimator.cluster_centers_
    #     else:
    #         components_ = estimator.components_
    #
    #     CA.append((name,components_))

        # if (hasattr(estimator, 'noise_variance_') and
        #         estimator.noise_variance_.ndim > 0):
        #     plot_gallery("Pixelwise variance",
        #                  estimator.noise_variance_.reshape(1,-1), n_col=1,
        #                  n_row=1)
        # plot_gallery('%s - Train time %.1fs' % (name, train_time),
        #              components_[:n_components])

    # return CA


# stats = HSI_CA(15, array_2d)
# print(stats)
print(array.shape)
stats = []

for band in array[:,382:500,5:634]:
    pca = dec.PCA(whiten=True, svd_solver='randomized', random_state=rng, tol=1e4)
    #ica = FastICA(tol=1e-4, whiten=True, random_state=rng, max_iter=100, fun='cube')
    #ica_components_ = ica.fit_transform(band)
    pca_components_ = pca.fit_transform(band)
    stats.append({
        'band': wavelength[band_c],
        'min': band.min(),
        'mean': band.mean(),
        'median': np.median(band),
        'max': band.max(),
      #  'icas': ica_components_,
        'pcas': pca_components_,
        'pca_mean':pca_components_.mean()
    })
    # band_c += 1

#TODO: ensemble classifiers on new columns
#TODO: data output to file

# pca = PCA(whiten=True, svd_solver='randomized', random_state=rng, copy=True)
# for band in array:
#     pca.fit(band)

# pprint(pca.components_)

# pprint(stats)

# print(array.shape)