# import os
import numpy as np
import pandas as pd

from hsi_atk.utils.dataset import open_hsi_bil
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVR, SVC
# from sklearn.metrics import accuracy_score
# from skhyper.cluster import KMeans as hKMeans
# from skhyper.decomposition import PCA as hPCA
# from skhyper.process import Process
from skimage.filters import threshold_otsu


b73Path = '/Volumes/RuddellD/hsi/Hyperspectral/B73/'
cmlPath = '/Volumes/RuddellD/hsi/Hyperspectral/Cml103/'

cols = ['Packet #', 'Genotype', 'Hormone', 'Kernelwt', 'Lipidwt']

labeled_data = pd.read_csv("../Data/headers3mgperml.csv", sep=",")
files = []
img_rows = []
img_row = 1

for idx, row in labeled_data.iterrows():

    file = str(row['Packet #']) + row['Hormone'].lower() + '.bil'
    if row['Genotype'] == 'B73':
        file = b73Path + file
    elif row['Genotype'] == 'CML103':
        file = cmlPath + file

    files.append(file)
    img_rows.append(img_row)
    img_row += 1

#TODO: Get specified paths from labeled images in separate function
def get_lbld_img_paths(df, folders=None):  #TODO: refactor to get paths to all images in list of folder paths

    b73Path = '/Volumes/RuddellD/hsi/Hyperspectral/B73/'
    cmlPath = '/Volumes/RuddellD/hsi/Hyperspectral/Cml103/'
    packets = df['Packet #'].tolist()
    genos = df['Genotype'].tolist()
    hormones = df['Hormone'].tolist()
    paths = []
    for i in range(len(packets)):
        pack = str(packets[i])
        horm = str(hormones[i]).lower()
        geno = genos[i]
        path = pack + "." + horm + ".bil"
        if geno == "B73":
            path = b73Path + path
            paths.append(path)
        elif geno == "CML103":
            path = cmlPath + path
            paths.append(path)

    return paths


def build_3d_means_img(paths):
    images = []
    count = 0

    for path in paths:
        img = open_hsi_bil(path)
        img = np.mean(img, axis=2)
        images.append(img)

    gray_img3d = np.stack(images, axis=2)
    return gray_img3d


def build_4d_img(paths, minimize_pix=False):
    """ Reads in HSI's from list of paths and returns as stacked array across new axis
    :param paths: list, strings pathing to images
    :param minimize_pix: boolean, to minimize row height and column width of all images
    :return: 4D array, stacked image arrays
    """
    images = []
    count = 0
    rmin = 500
    rmax = 0
    cmin = 640
    cmax = 0
    for path in paths:
        img = open_hsi_bil(path)

        # img = apply_sc(img)
        # if not minimize_pix:
        #     img = filter_kernelspace(img)
        # else:
        #     img, nrmin, nrmax, ncmin, ncmax = filter_kernelspace(img, return_bbox=True)
        #     if nrmin < rmin:
        #         rmin = nrmin
        #     if nrmax > rmax:
        #         rmax = nrmax
        #     if ncmin < cmin:
        #         cmin = ncmin
        #     if ncmax > cmax:
        #         cmax = ncmax
        images.append(img)
        count += 1
        # if count == 1:
        #     break
        print("Processed ", count, " image(s)...")

    images = np.array(images)
    images = images.swapaxes(0, 3).swapaxes(0, 2).swapaxes(0, 1)

    if minimize_pix:
        return images[rmin:rmax, cmin:cmax, :, :]

    return images


def filter_kernelspace(img, return_bbox=False):
    """ Threshold filter zeroing out background with overall bounding box return option
    :param img: ndarray, hsi image to filter
    :param return_bbox: boolean, to return the row min/max and column min/max
    :return: ndarray, filtered image, and optional bbox
    """
    gray = np.mean(img, 2)
    thresh = threshold_otsu(gray, nbins=240)
    kernel_reg = (thresh < gray)
    rows = np.any(kernel_reg, axis=1)
    cols = np.any(kernel_reg, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    n_reg = ~ kernel_reg
    rr, cc = np.where(n_reg)
    img[rr, cc, :] *= 0
    if return_bbox:
        return img, rmin, rmax, cmin, cmax
    else:
        return img


def apply_sc(img, with_mean=True, with_std=True):
    d0, d1, d2 = img.shape
    img2d = img.reshape(-1, d2)
    sc = StandardScaler(copy=False, with_mean=with_mean, with_std=with_std)
    img2d = sc.fit_transform(img2d)
    scaled_img = img2d.reshape(d0, d1, d2)

    return scaled_img


# def load_proc(hsi_img, filter=True):
#     if filter:
#         hsi_img = filter_kernelspace(hsi_img)  # applies filter to image, zeroing out background
#     print('Processing hsi...')
#     X = Process(hsi_img)  # creates process object of image for skhyper clustering
#     return X


# def get_img_pca(hsi_img_proc, n_components):
#     pca = hPCA(n_components=n_components, copy=False)
#     pca.fit_transform(hsi_img_proc)
#     return pca


def get_class_stats(hsi_img, labels, unify):
    """ Gets statistics for clusters of clustered image
    :param hsi_img: 2/3D array, image
    :param labels: 2D list/array, image labels
    :return: dict, cluster statistics for given image
    """
    # img_cl_stats = {}
    img_cl = {}
    uniq, unic = np.unique(labels, return_counts=True)  # gets unique labels and counts
    # print(uniq, "\n\n", unic)
    all_means = []
    # means = []
    # meds = []
    # vars = []
    # stds = []
    # mins = []
    # maxs = []
    # counts = []

    for i in uniq:
        cl = "class_" + str(i)
        img_cl[cl] = {}
        # print(cl)
        rr0, cc0 = np.where(labels == i)
        nhsi = hsi_img[rr0, cc0, :]
        # print(nhsi.shape)
        coun = nhsi.shape[0]

        mean = np.mean(nhsi, 0)  # mean of all spectra
        img_cl[cl]['mean'] = mean
        # means.append(mean)
        all_means.append(mean)
        med = np.median(nhsi, 0)  # median of all spectra
        img_cl[cl]['med'] = med
        # meds.append(med)
        var = np.mean(nhsi, 0)  # variance of all spectra
        img_cl[cl]['var'] = var
        # vars.append(var)
        std = np.std(nhsi, 0)  # standard deviation of all spectra
        img_cl[cl]['std'] = std
        # stds.append(std)
        min_ = nhsi.min(axis=0)  # minimum of all spectra
        img_cl[cl]['min'] = min_
        # mins.append(min_)
        max_ = nhsi.max(axis=0)  # maximum of all spectra
        img_cl[cl]['max'] = max_
        # maxs.append(max_)
        img_cl[cl]['count'] = coun
        # counts.append(coun)

    # img_cl_stats['means'] = means
    # img_cl_stats['meds'] = meds
    # img_cl_stats['vars'] = vars
    # img_cl_stats['stds'] = stds
    # img_cl_stats['mins'] = mins
    # img_cl_stats['maxs'] = maxs
    # img_cl_stats['counts'] = counts

    if unify:
        return img_cl, all_means
        # return img_cl_stats, all_means

    # return img_cl_stats
    return img_cl


# def load_cl(paths, n_clusters=8, s_scale=True, filter=False, unify=False):
#     """ Performs clustering, filtering, and other operations on images
#     :param paths: list, strings pathing to images
#     :param n_clusters: int, number of clusters
#
#     :param filter: boolean, to perform filtering on images
#     :return: dict, cluster statistics for all images in paths
#     """
#     print('Loading hsi...')
#
#     img_stats = {}
#     means_all = []
#
#     image = 1
#     for path in paths:
#         print("Processing image: ", image)
#         hsi = open_hsi_bil(path)
#
#         if filter:
#             hsi = filter_kernelspace(hsi)
#
#         if s_scale:
#             hsi = apply_sc(hsi)
#
#         X = Process(hsi)
#         km = hKMeans(n_clusters, copy_x=False)
#         km.fit(X)
#         labels = km.labels_
#         if unify:
#             img_cl_stats, all_means = get_class_stats(hsi, labels, unify=unify)
#             means_all.extend(all_means)
#
#         else:
#             img_cl_stats = get_class_stats(hsi, labels, unify=unify)
#         img_stats[path] = img_cl_stats
#         print("Finished image: ", image)
#         image +=1
#
#     if unify:
#         u_labels = unify_labels(means_all, n_clusters)
#         return img_stats, u_labels
#
#     return img_stats


# def load_pca(paths, n_components, s_scale=True, filter=False):
#     image = 1
#     pcas = {}
#     for path in paths:
#         print("Processing image: ", image)
#         hsi = open_hsi_bil(path)
#
#         if filter:
#             hsi = filter_kernelspace(hsi)
#
#         if s_scale:
#             hsi = apply_sc(hsi)
#
#         X = Process(hsi, scale=False)
#         pca = hPCA(n_components=n_components, copy=False)
#         pca.fit_transform(X)
#         # pca.plot_statistics()
#         # input("Pause")
#         pcas[path] = pca
#
#     return None


def unify_labels(cl_means, n_clusters, order_f_imgs=True):
    """ Takes list of unordered clustered pixel means and returns list of unified labels.
    Alternatively, returns 2D list of unified labels for running regression on multiple image cluster stats
    :param cl_means: list/array, all clustered pixels means
    :param n_clusters: int, number of clusters
    :param order_f_imgs: boolean, to return array as 2D unified labels if true
    :return: labels-1D array of unified labels
             n_labels-2D array of unified labels (per image)
    """
    print("clustering clusters")
    cl_cl = KMeans(n_clusters)
    cl_cl.fit(cl_means)
    labels = cl_cl.labels_

    if order_f_imgs:
        l_len = len(labels)
        n_labels = []
        for i in range(0, int(l_len/n_clusters)):
            n_labels.append(labels[i*n_clusters:(i+1)*n_clusters])
        return n_labels

    return labels

#TODO: pipeline predictions schema from cl_dict and headers df
#TODO: add accuracy metrics (rmse and acc)
def train_pred(labeled_df, img_stats, uni_labels, clf_reg, label_pred, cl_stats="counts", vars="con", metrics=True):
    """ Trains predictor(s) on specified label(s) from image cluster statistics, and assesses accuracy metric.
    :param labeled_df: DataFrame, labels for images
    :param img_stats: dictionary, contains cluster statistics for each image
    :param uni_labels: list/array, clustered cluster statistics (unified clusters)
    :param clf_reg: classifier or array of clf's, sklearn
        # currently supports one clf
    :param label_pred: string or array of strings, labels to predict (should match column headers of dataframe)
        # currently supports one label
    :param cl_stats: string or array of strings, cluster statistics to be used a X in clf's
    :param vars: string, "cat" or "con" (categorical or continuous) picks between accuracy_score and mean_squared_error
    :param metrics: boolean, whether to get/return scoring
    :return:
    """
    labels = labeled_df[label_pred]
    images = img_stats.keys()  # keys should have been entered in order with labeled_df row order
    # this order will be retained if get_lbld_img_paths is used

    X_train_keys, X_test_keys, y_train, y_test = train_test_split(images, labels, test_size=.2)


    return None


def build_n_var(img_stats, uni_labels):
    images = []
    images.extend(img_stats.keys())
    print(images)
    # i_order = img_stats[images[0]].keys()
    cl_order = uni_labels[0]
    # n_cl = len(cl_order)
    for lbl_order, img_path in zip(uni_labels[1:], images[1:]):
        # if lbl_order == cl_order:
        #     continue
        # else:
        cou = 1
        print(lbl_order)
        for cl in cl_order:
            cl0 = "class_" + str(cou)
            print(cl)
            index = np.where(lbl_order == cl)
            if len(index[0]) > 1 or len(index[0]) == 0:

                continue
            # print(index[0])
            cl1 = "class_" + str(int(index[0]+1))
            # img_stats[img_path]["null_0"] = img_stats[img_path][cl0]
            img_stats[img_path]["null_1"] = img_stats[img_path][cl1]
            img_stats[img_path][cl1] = img_stats[img_path][cl0]
            img_stats[img_path][cl0] = img_stats[img_path]["null_1"]
            cou += 1
        # del img_stats[img_path]["null_1"]

    return img_stats


import h5py
img_file = h5py.File("../Data/img_all_4d.h5", "r")
images = img_file['datatset']
img_file.close()
images = np.mean(images, axis=3)
print(images.shape)

# paths = get_lbld_img_paths(labeled_data)
# b73paths = []
# b73_rows = []
# cml103paths = []
# cml103_rows = []
# for i in range(len(paths)):
#     path = paths[i]
#     if "B73" in path:
#         b73paths.append(path)
#         b73_rows.append(img_rows[i])
#     elif "Cml103" in path:
#         cml103paths.append(path)
#         cml103_rows.append(img_rows[i])

# all_stats = load_cl(paths, n_clusters=8, s_scale=True, filter=False, unify=False)
# all_means = []
# for img in paths:
#     all_means.append(all_stats[img]["class_1"]['mean'])
#     all_means.append(all_stats[img]["class_2"]['mean'])
#     all_means.append(all_stats[img]["class_3"]['mean'])
#     all_means.append(all_stats[img]["class_4"]['mean'])
#     all_means.append(all_stats[img]["class_5"]['mean'])
#     all_means.append(all_stats[img]["class_6"]['mean'])
#     all_means.append(all_stats[img]["class_7"]['mean'])
#     all_means.append(all_stats[img]["class_8"]['mean'])

# img_4d = build_4d_img(paths)
# img_3d = img_4d.reshape(8*500, 640, 240)
# img_4d = img_4d.reshape(-1, 240)
# X = Process(img_4d, scale=False)
# hpca = hPCA(n_components=8, copy=False)
# hpca.fit_transform(X)
# hpca.plot_statistics()

# b73_all = build_4d_img(b73paths)
# b73img_stats, u_labels = load_cl(b73paths, n_clusters=5)
# reordered_img_stats = build_n_var(b73img_stats, u_labels)

# cml103img_stats = load_cl(cml103paths)
# b73df = pd.DataFrame(b73img_stats)
# cml103df = pd.DataFrame(cml103img_stats)
# img_stats, u_labels = load_cl(paths)
# build_n_var(img_stats, u_labels)

# Testing hSVC image labeling versus hKMeans cluster labels
# nhkm = hKMeans(8, copy_x=False)
# b73_0 = filter_kernelspace(open_hsi_bil(b73paths[0]))
# b73_1 = filter_kernelspace(open_hsi_bil(b73paths[1]))
# b73_0_1 = Process(np.stack((b73_0, b73_1), axis=2))
# b73_0p = Process(b73_0)
# b73_1p = Process(b73_1)
# nhkm.fit(b73_0_1)
# labels = nhkm.labels_
# nsvc = hSVC(kernel="linear", degree=5, tol=1e-4)
# nsvc.fit(b73_0p, labels[:,:,0])
# b73_1_preds = nsvc.predict(b73_1p)
# preds_1 = b73_1_preds.ravel()
# labels_1 = labels[:,1].ravel()
# print(accuracy_score(labels_1, preds_1))
# ~95% accuracy.. good enough for now.. improve later

# img_stats = {}
# means_all = []
# meds_all = []
# vars_all = []
# stds_all = []
# mins_all = []
# maxs_all = []
# counts_all = []
# all_stats = []
#
# img_count = 1
# for path in paths:
#     img_cl_st = load_cl(path)
#     for mean, med, var, std, min, max, count in zip(img_cl_st['means'], img_cl_st['meds'], img_cl_st['vars'],\
#         img_cl_st['stds'], img_cl_st['mins'], img_cl_st['maxs'], img_cl_st['counts']):
#         all_stats.append([mean, med, var, std, min, max, count])
#
#     img_stats[path] = img_cl_st
#     print(img_count)
#     img_count += 1

# from sklearn.cluster import KMeans
#
# cl_cl = KMeans(n_clusters=9)
# cl_cl.fit(all_stats)

# img_cl_st = load_cl("../Data/32.control.bil")

# img_4d = build_4d_img(paths)
# mdl0 = KMeans(9, n_jobs=1)
# mdl0.fit(X0)
