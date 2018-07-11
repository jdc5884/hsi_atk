import os
import numpy as np
import pandas as pd

from hsi_atk.dataset import open_hsi_bil
from skhyper.cluster import KMeans
from skhyper.process import Process
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

    return kmeans


def getstats(path, geno):
    # hsi_info = []
    img_dt = {}
    images = os.listdir(path)
    count = 0
    for img in images:
        print(count)

        if img.endswith('.bil'):
            # packet, hormone, ext = img.split('.')
            # img_dt['File'] = img
            # img_dt['Packet #'] = int(packet)
            # img_dt['Genotype'] = geno
            # img_dt['Hormone'] = hormone
            km = load_cl(path+img)
            # img_dt['mdl'] = km
            # hsi_info.append(img_dt)
            img_dt[geno+img] = km
            count += 1
        # if count == 5:
        #     break

    # return hsi_info
    return img_dt


b73_stats = getstats(b73Path, 'B73')
cml_stats = getstats(cmlPath, 'Cml103')
