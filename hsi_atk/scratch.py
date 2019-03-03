# import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as cm
from mpl_toolkits.mplot3d import Axes3D
import h5py as h5
from hsi_atk.utils.hsi2color import hsi2color

from hsi_atk.utils.dataset import open_hsi_bil
from hsi_atk.utils.hsi2color import hsi2color, hsi2color4, hsi2gray
from skimage.segmentation import chan_vese as seger
# import pandas as pd
# import scipy as sp
# from scipy import ndimage as ndi
# import rasterio
#
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.linear_model import RidgeClassifier
# from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import train_test_split
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.preprocessing import LabelEncoder


img_ = open_hsi_bil("../Data/B73/32.control.bil")

import random
chars = '0123456789ABCDEF'

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

gray = hsi2gray(img_)

gseg, phi, energies = seger(gray, mu=.99, extended_output=True)
gseg = ~ gseg
rr, cc = np.where(gseg)

for i in range(0, 240):
	ys = img_[rr, cc, i]
	hist, bins = np.histogram(ys, bins=50)
	c = '#'+''.join(random.sample(chars,6))

	xs = (bins[:-1] + bins[1:])/2
	ax.bar(xs[12:], hist[12:], zs=(i*2.042), zdir='y', alpha=0.8, color=c, ec=c)

plt.show()

# data = pd.read_csv('../Data/headers3mgperml.csv', sep=',')
#
# X = data.values[:, 15:]
# y1 = data.values[:, 1]
# y2 = data.values[:, 2]
# y3 = data.values[:, 3]
# y4 = data.values[:, 4]
#
# le = LabelEncoder()
# y1_ = le.fit_transform(y=y1)
# y2_ = le.fit_transform(y=y2)
# y3_ = le.fit_transform(y=y3)
# y4_ = le.fit_transform(y=y4)
#
# Y_ = np.stack((y1_, y2_, y3_, y4_), axis=1)
#
# # X_train, X_test, Y_train, Y_test = train_test_split(X, Y_, test_size=.3)
# indices = np.random.permutation(X.shape[0])
# training_idx, test_idx = indices[:-5], indices[-5:]
# X_train, X_test = X[training_idx, :], X[test_idx, :]
# Y_train, Y_test = Y_[training_idx, :], Y_[test_idx, :]
#
# etc = ExtraTreesClassifier(n_estimators=100)
#
# etc.fit(X_train, Y_train)
# Y_preds = etc.predict(X_test)
# print(Y_test, '\n\n', Y_preds)
# print(confusion_matrix(Y_test[3], Y_preds[3]))
# print(confusion_matrix(Y_test[4], Y_preds[4]))

# hf0 = h5.File("../Data/all_labels.h5", "r")



# stacked_hists = []
# stacked_bwise = []
# for i in range(5):

	# data = hf0["raw4d"][i,:,:,:]

	# plt.hist(data[:,:,220], bins='auto')
	# plt.show()

	# input("Next..")

	# hists = []
	# pband = []

	# for j in range(240):
	# 	plt.hist(data[:,:,i], bins=100)
	# 	plt.show()
	# 	input("Pass")
		# hist, bins = np.histogram(a=data[:,:,j], bins=800, range=(0.0,4095.0))
		# hists.append(hist)

	# hists = np.stack(hists, axis=-1)

	# for i in range(800):
	# 	bcount_per_band, bins = np.histogram(a=hists[i, :], bins=100)
	# 	pband.append(bcount_per_band)

	# pband = np.stack(pband, axis=-1)

	# stacked_hists.append(hists)
	# stacked_bwise.append(pband)



# stacked_hists = np.stack(stacked_hists, axis=-1)
# print(stacked_hists.shape)
# stacked_bwise = np.stack(stacked_bwise, axis=-1)
# print(stacked_bwise.shape)

# for i in range(5):
# 	color = hsi2color(data[i,:,:,:])
# 	plt.imshow(color)
# 	plt.show()
# 	input("Pause..")

### Testing rgb converter and segmentation

# gray = hsi2color(img_, scale=False, out_type=float)
# f = np.fft.fft2(img_[:, 70:570,:])
# fshift = np.fft.fftshift(f)
# magnitude_spectrum = 20*np.log(np.abs(fshift))
# magnitude_spectrum = np.mean(magnitude_spectrum, axis=2)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# x = np.linspace(0,499,500)
# y = np.linspace(0,639,640)
# z = np.linspace(0,239,240)

# X, Y = np.meshgrid(x,x)

# c = np.random.standard_normal(640)

# ax.plot_surface(X,Y,magnitude_spectrum[:500,:500])
# plt.show()

# hsi_rgbp = hsi2color4(img_)

#
# import matplotlib.pyplot as plt
# plt.imshow(hsi_rgb_mask)
# plt.show()
# plt.figure(1)
# plt.imshow(hsi_rgb_mask)
# seg = seger(hsi_rgb)
# seg2 = seger(hsi_rgbp)


# from skhyper.process import Process
# from skhyper.decomposition import PCA
# from skhyper.cluster import KMeans
# mdl = KMeans(8)
#
# from skimage.filters import threshold_otsu
# gray = np.mean(img_, 2)
# thresh = threshold_otsu(gray, nbins=240)
# reg0 = (thresh<gray)
# nreg0 = ~ reg0
# rr, cc = np.where(reg0)
# rr0, cc0 = np.where(nreg0)
# img_[rr0, cc0, :] *= 0
#
# time_p = time.time()
# X = Process(img_)
# X.view()
# time_proc = time.time()
# print("Time to process img... ", (time_proc-time_p))
# print(X.var_image)
# print(X.var_spectrum)
#
# mdl = PCA(50, copy=False)
# time_inf = time.time()
# mdl.fit_transform(X)
# time_fit = time.time()
# print("Time to fit img... ", (time_fit-time_inf))
# mdl.plot_statistics()

# X.view()


#
# n_img = img_[60:350, 100:530, :]
# n_gray = np.mean(n_img, 2)
# img_ii = integral_image(gray)
# feature = haar_like_feature(img_ii, 0, 0, 349, 529)

# gray = np.mean(nimg[:, 464:534, 95:176], 2)
# thresh = threshold_otsu(nimg, nbins=240)
# thresh = 500

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
