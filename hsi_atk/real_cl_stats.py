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


def filter_kernelspace(img):
    gray = np.mean(img, 2)
    thresh = threshold_otsu(gray, nbins=240)
    kernel_reg = (thresh < gray)
    n_reg = ~ kernel_reg
    rr, cc = np.where(n_reg)
    img[rr, cc, :] *= 0
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


def unify_labels(img, lbls, clf):
    """
    Takes pre-fitted classifier fitted on cluster of image, and applies predictions to given
    image. The predictions are used to reset labels of given image
    :param img: 3d-array - numpy array of hsi
    :param lbls: 2d-array - numpy array of hsi cluster labels
    :param clf: skhyper classifier - pixel classifier for hsi
    :return: 2d-array - numpy array of unified labels (makes labels similar across images)
    """


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


b73_stats = getstats(b73Path, 'B73')
cml_stats = getstats(cmlPath, 'Cml103')
