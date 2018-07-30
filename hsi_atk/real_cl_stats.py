import os
import numpy as np
import pandas as pd

from hsi_atk.dataset import open_hsi_bil
from sklearn.svm import SVR, SVC
from skhyper.cluster import KMeans
from skhyper.process import Process
# from skhyper.svm import SVC
from skimage.filters import threshold_otsu


b73Path = '/Volumes/RuddellD/hsi/Hyperspectral/B73/'
cmlPath = '/Volumes/RuddellD/hsi/Hyperspectral/Cml103/'

cols = ['Packet #', 'Genotype', 'Hormone', 'Kernelwt', 'Lipidwt']

labeled_data = pd.read_csv("../Data/headers3mgperml.csv", sep=",")
files = []

for idx, row in labeled_data.iterrows():
    file = str(row['Packet #']) + row['Hormone'].lower() + '.bil'
    if row['Genotype'] == 'B73':
        file = b73Path + file
    elif row['Genotype'] == 'CML103':
        file = cmlPath + file

    files.append(file)


def get_lbld_img_paths(df):
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


def build_4d_img(paths):
    images = []

    count = 0
    rmin = 500
    rmax = 0
    cmin = 640
    cmax = 0
    for path in paths:
        img = open_hsi_bil(path)
        img, nrmin, nrmax, ncmin, ncmax = filter_kernelspace(img, return_bbox=True)
        if nrmin < rmin:
            rmin = nrmin
        if nrmax > rmax:
            rmax = nrmax
        if ncmin < cmin:
            cmin = ncmin
        if ncmax > cmax:
            cmax = ncmax
        images.append(img)

        count += 1
        if count == 1:
            break
        print("Processed ", count, " image(s)...")

    images = np.array(images)
    images = images.swapaxes(0, 3).swapaxes(0, 2).swapaxes(0, 1)


    return images[rmin:rmax, cmin:cmax, :, :]


def clf_chain(list_img, labels):

    # for img, lbl_set in list_img, labels:
    #     imgr =
    return None


def filter_kernelspace(img, return_bbox=False):
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


def load_cl(path):
    print('Loading hsi...')
    hsi = open_hsi_bil(path)
    hsi = filter_kernelspace(hsi)
    print('Processing hsi...')
    X = Process(hsi)
    kmeans = KMeans(9, copy_x=False)
    print('Begin Clustering...')
    kmeans.fit(X)
    labels = kmeans.labels_

    img_cl_stats = {}
    uniq, unic = np.unique(labels, return_counts=True)
    print(uniq, "\n\n", unic)
    means = []
    meds = []
    vars = []
    stds = []
    mins = []
    maxs = []
    counts = []

    for i in uniq:
        cl = "class_" + str(i)
        # print(cl)
        rr0, cc0 = np.where(labels == i)
        nhsi = hsi[rr0, cc0, :]
        # print(nhsi.shape)
        coun = nhsi.shape[0]

        mean = np.mean(nhsi, 0)
        med = np.median(nhsi, 0)
        var = np.mean(nhsi, 0)
        std = np.std(nhsi, 0)
        min_ = nhsi.min(axis=0)
        max_ = nhsi.max(axis=0)

        means.append(mean)
        meds.append(med)
        vars.append(var)
        stds.append(std)
        mins.append(min_)
        maxs.append(max_)
        counts.append(coun)

    img_cl_stats['means'] = means
    img_cl_stats['meds'] = meds
    img_cl_stats['vars'] = vars
    img_cl_stats['stds'] = stds
    img_cl_stats['mins'] = mins
    img_cl_stats['maxs'] = maxs
    img_cl_stats['counts'] = counts

    return img_cl_stats


# def getstats(path, geno):
#     # hsi_info = []
#     svc = SVC(kernel='linear', degree=5)
#     img_dt = {}
#     images = os.listdir(path)
#     count = 0
#     for img in images:
#         print(count)
#
#         if img.endswith('.bil'):
#             lbls = load_cl(path + img)
#             uni_v, uni_c = np.unique(lbls, return_counts=True)
#
#             if count == 0:
#                 svc.fit(img, lbls)
#
#
#             # img_dt['mdl'] = km
#             # hsi_info.append(img_dt)
#
#             img_dt[geno+img] = lbls
#             count += 1
#         # if count == 5:
#         #     break
#
#     # return hsi_info
#     return img_dt


paths = get_lbld_img_paths(labeled_data)
img_stats = {}
# for path in paths:
#     img_cl_st = load_cl(path)
#     img_stats[path] = img_cl_st

from sklearn.cluster import KMeans as BKMeans

img_cl_st = load_cl("../Data/32.control.bil")


# img_4d = build_4d_img(paths)
# mdl0 = KMeans(9, n_jobs=1)
# mdl0.fit(X0)
