import time
import numpy as np
import pandas as pd
import scipy as sp
from scipy import ndimage as ndi
import rasterio

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv('../Data/headers3mgperml.csv', sep=',')

X = data.values[:, 15:]
y1 = data.values[:, 1]
y2 = data.values[:, 2]
y3 = data.values[:, 3]
y4 = data.values[:, 4]

le = LabelEncoder()
y1_ = le.fit_transform(y=y1)
y2_ = le.fit_transform(y=y2)
y3_ = le.fit_transform(y=y3)
y4_ = le.fit_transform(y=y4)

Y_ = np.stack((y1_, y2_, y3_, y4_), axis=1)

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y_, test_size=.3)
indices = np.random.permutation(X.shape[0])
training_idx, test_idx = indices[:-5], indices[-5:]
X_train, X_test = X[training_idx, :], X[test_idx, :]
Y_train, Y_test = Y_[training_idx, :], Y_[test_idx, :]

etc = ExtraTreesClassifier(n_estimators=100)

etc.fit(X_train, Y_train)
Y_preds = etc.predict(X_test)
print(Y_test, '\n\n', Y_preds)
print(confusion_matrix(Y_test[3], Y_preds[3]))
print(confusion_matrix(Y_test[4], Y_preds[4]))


def load_hsi(file):
    src = rasterio.open(file)
    img = np.array(src.read())

    return img


# from hsi_atk.Sasatari.sasatari import alt_view

# img_ = loadHSI("../Data/32.control.bil")
# img_ = img_.swapaxes(0, 2)
# img_ = img_.swapaxes(0, 1)

# nimg = alt_view(img, "left")

# from skimage.filters import threshold_otsu, rank
# import matplotlib.pyplot as plt
#
# gray = np.mean(img_, 2)
# thresh = threshold_otsu(gray, nbins=240)
#
# n_img = img_[60:350, 100:530, :]
# n_gray = np.mean(n_img, 2)
# img_ii = integral_image(gray)
# feature = haar_like_feature(img_ii, 0, 0, 349, 529)

# gray = np.mean(nimg[:, 464:534, 95:176], 2)
# thresh = threshold_otsu(nimg, nbins=240)
# thresh = 500

# reg0 = (thresh<gray)
# nreg0 = ~ reg0
#
# rr, cc = np.where(reg0)
# rr0, cc0 = np.where(nreg0)
# n_k_pix = np.sum(reg0)
# k_pix = img_[rr, cc, :]
# k_pix_gray = gray[rr, cc, :]

# mean_pix = np.mean(k_pix, 1)
# var_pix = np.var(k_pix, 1)

# k_pix = k_pix.reshape()

# fig, axs = plt.subplots(2, 2)
# axs[0, 0].imshow(gray)
# axs[0, 1].imshow(reg0)
#
# shift = -2
# edgex = (reg0 ^ np.roll(nreg0, shift=shift, axis=0))
# edgey = (reg0 ^ np.roll(nreg0, shift=shift, axis=1))
#
# axs[1, 1].imshow(gray)
# axs[1, 1].contour(edgex, 2, colors='r', lw=2.)
# axs[1, 1].contour(edgey, 2, colors='r', lw=2.)
# plt.show()
# img[rr0, cc0, :] *= 0
#
# from skhyper.process import Process
# from skhyper.cluster import KMeans
# from skhyper.decomposition import PCA
#
# pca = PCA()
# X = Process(img)
# pca.fit_transform(X)
# pca.plot_statistics()
#
# kmean = KMeans(8)
# kmean.fit(X)
#
# rr1 = np.argwhere(kmean.labels_ != 3)
# imgL3 = img[, :]*0
# XL3 = Process(imgL3)
# XL3.view()
# pca.fit_transform(XL3)
# pca.plot_statistics()


# for i in range(95, 176):
#     time0 = time.time()
#     gray = np.mean(nimg[:, 464:600], 2)
#     thresh = threshold_otsu(nimg, nbins=240)
#     reg0 = (thresh<nimg[:, 464:600, i])
#     nreg0 = ~ reg0
#
#     fig, axs = plt.subplots(2, 2)
#     axs[0, 0].imshow(nimg[:, 464:600, i])
#     axs[0, 1].imshow(reg0)
#
#     shift = -2
#     edgex = (reg0 ^ np.roll(nreg0, shift=shift, axis=0))
#     edgey = (reg0 ^ np.roll(nreg0, shift=shift, axis=1))
#
#     axs[1, 1].imshow(nimg[:, 464:600, i])
#
#     axs[1, 1].contour(edgey, 2, colors='r', lw=2.)
#     axs[1, 1].contour(edgex, 2, colors='r', lw=2.)
#     time1 = time.time()
#     print(time1-time0)
#
#     plt.show()
#     input("Press Enter to continue...")