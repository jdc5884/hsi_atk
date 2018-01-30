from time import time

import numpy as np
import scipy as sp
import pandas as pd
import rasterio
import sklearn.decomposition as dec

from pprint import pprint

from skimage.io import imread_collection, concatenate_images

# Spectral bands from .hdr file
wavelength= [393.91, 395.982, 398.054, 400.126, 402.198, 404.27, 406.342, 408.414, 410.486, 412.558, 414.63, 416.702, 418.774, 420.846, 422.918, 424.99, 427.062, 429.134, 431.206, 433.278, 435.35, 437.422, 439.494, 441.566, 443.638, 445.71, 447.782, 449.854, 451.926, 453.998, 456.07, 458.142, 460.214, 462.286, 464.358, 466.43, 468.502, 470.574, 472.646, 474.718, 476.79, 478.862, 480.934, 483.006, 485.078, 487.15, 489.222, 491.294, 493.366, 495.438, 497.51, 499.582, 501.654, 503.726, 505.798, 507.87, 509.942, 512.014, 514.086, 516.158, 518.23, 520.302, 522.374, 524.446, 526.518, 528.59, 530.662, 532.734, 534.806, 536.878, 538.95, 541.022, 543.094, 545.166, 547.238, 549.31, 551.382, 553.454, 555.526, 557.598, 559.67, 561.742, 563.814, 565.886, 567.958, 570.03, 572.102, 574.174, 576.246, 578.318, 580.39, 582.462, 584.534, 586.606, 588.678, 590.75, 592.822, 594.894, 596.966, 599.038, 601.11, 603.182, 605.254, 607.326, 609.398, 611.47, 613.542, 615.614, 617.686, 619.758, 621.83, 623.902, 625.974, 628.046, 630.118, 632.19, 634.262, 636.334, 638.406, 640.478, 642.55, 644.622, 646.694, 648.766, 650.838, 652.91, 654.982, 657.054, 659.126, 661.198, 663.27, 665.342, 667.414, 669.486, 671.558, 673.63, 675.702, 677.774, 679.846, 681.918, 683.99, 686.062, 688.134, 690.206, 692.278, 694.35, 696.422, 698.494, 700.566, 702.638, 704.71, 706.782, 708.854, 710.926, 712.998, 715.07, 717.142, 719.214, 721.286, 723.358, 725.43, 727.502, 729.574, 731.646, 733.718, 735.79, 737.862, 739.934, 742.006, 744.078, 746.15, 748.222, 750.294, 752.366, 754.438, 756.51, 758.582, 760.654, 762.726, 764.798, 766.87, 768.942, 771.014, 773.086, 775.158, 777.23, 779.302, 781.374, 783.446, 785.518, 787.59, 789.662, 791.734, 793.806, 795.878, 797.95, 800.022, 802.094, 804.166, 806.238, 808.31, 810.382, 812.454, 814.526, 816.598, 818.67, 820.742, 822.814, 824.886, 826.958, 829.03, 831.102, 833.174, 835.246, 837.318, 839.39, 841.462, 843.534, 845.606, 847.678, 849.75, 851.822, 853.894, 855.966, 858.038, 860.11, 862.182, 864.254, 866.326, 868.398, 870.47, 872.542, 874.614, 876.686, 878.758, 880.83, 882.902, 884.974, 887.046, 889.118]

# Read in HSI data to 3D array in form bands,lines,samples
with rasterio.open("../Data/32.control.bil") as src:
    array = np.array(src.read())

# Change array from bands,lines,samples to samples,lines,bands
n_array = array.swapaxes(0,2)
# Reshape to 2D in form samples*lines,bands
array_2d = n_array.transpose(2,0,1).reshape(320000,-1)

# stats = []
rng = np.random.RandomState(30)

band_c = 0

# Generalized class for analysis and data simulation in development..


#TODO: Piecewise function approximation, circle/oval methods

def pixelfilter(data,sample,line,band,n_comp,tmethod=None,r=None,l=None,a=None):
    """
    General function for manual or automatic pixel selection,
    filtering, tesing, and other stats.

    Other geometric and supervised partitioning methods in development

    :param data: 3D array of HSI
    :param line: 1D array of desired image pixel height range
    :param col: 1D array of desired image pixel width range
    :param band: 1D array of band range to perform statistics on
    :param n_comp: 1D array of range of components for CA methods
    :param tmethod: trace method - "circle","lc"
    :param c: tuple of integers (#,#) representing line and column of center pixel
    :param r: integer - desired radius for pixels of interest
    :param l:
    :param a:
    :return:
    """
    varStats = {}

    DATA = data[sample[0]:sample[1],line[0]:line[1],:]
    lines = line[1]-line[0]
    samples = sample[1]-sample[0]

    # print(DATA.shape, lines, samples)
    DATA = DATA.transpose(2,0,1).reshape(lines*samples,-1)

    for n in range(1,n_comp+1,1):
        pca = dec.PCA(n_components=n, svd_solver='randomized',
                      whiten=True)

        pca.fit(DATA)



        varStats[n] = {
            "comps": n,
            "pca": pca,

            #"band":b,
            # "col":c,
            #"mean":mean,
            #"max":max,
            #"min":min,
            # "map":m,
            # "grad":grad,
            # "rbf":rbf
        }

        # grad = np.gradient()

        # m = np.meshgrid(col,line)
        # rbf = sp.interpolate.Rbf(m[0],m[1],grad)

        # if tmethod == "circle":
        #     circle = lambda c=c, r=r, l=l, a=a: (for)
        #
        #
        #
        # elif tmethod == "lc":

        # individual band variance


    return varStats


kernelStats = pixelfilter(n_array,[472,520],[113,166],[1,240],48)

bgStats = pixelfilter(n_array,[1,640],[364,500],[1,240],48)

mixedStats = pixelfilter(n_array,[57,575],[52,402],[1,240],48)

print(kernelStats[10]["pca"])
print(bgStats[10])