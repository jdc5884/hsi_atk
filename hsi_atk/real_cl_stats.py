import os
import numpy as np
import pandas as pd

from hsi_atk.dataset import open_hsi_bil
from skhyper.cluster import KMeans
from skhyper.process import Process
from skhyper.svm import SVC
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
        # if count == 5:
        #     break
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
    kmeans = KMeans(8, copy_x=False)
    print('Begin Clustering...')
    kmeans.fit(X)

    return kmeans.labels_


def getstats(path, geno):
    # hsi_info = []
    svc = SVC(kernel='linear', degree=5)
    img_dt = {}
    images = os.listdir(path)
    count = 0
    for img in images:
        print(count)

        if img.endswith('.bil'):
            lbls = load_cl(path + img)
            uni_v, uni_c = np.unique(lbls, return_counts=True)

            if count == 0:
                svc.fit(img, lbls)


            # img_dt['mdl'] = km
            # hsi_info.append(img_dt)

            img_dt[geno+img] = lbls
            count += 1
        # if count == 5:
        #     break

    # return hsi_info
    return img_dt


paths = get_lbld_img_paths(labeled_data)
img_4d = build_4d_img(paths)
print(img_4d.shape)
# X = Process(img_4d)
# mdl = KMeans(9, n_jobs=1)
# mdl.fit(X)