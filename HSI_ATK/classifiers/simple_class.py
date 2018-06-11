import rasterio
import numpy as np
import pandas as pd

from skimage import draw

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

rng = np.random.RandomState(33)


def im_collapse(im_array):
    shape = im_array.shape
    coll_array = np.zeros(shape=shape)
    for i in shape[0]:
        coll_array += im_array[i, :, :]
    return coll_array


def in_circle(x,y,h,k,r):
    if ((x-h)**2 + (y-k)**2) <= r**2:
        return True
    return False

#TODO: convert to or add multilayer labeling to hold multiple labels per pixel
def label_circle(shape, r, c, l, l_space=None):
    rr, cc = draw.circle(c[0], c[1], r, shape=shape)
    if l_space is None:
        label_space = np.zeros(shape)
    else:
        label_space = l_space
    label_space[rr, cc] += l
    return label_space

# def in_ellipse(x,y,h,k,r):


def label_ellipse(shape, r, c, r_r, c_r, rot, l):
    rr, cc = draw.ellipse(r, c, r_r, c_r, shape=shape, rotation=rot)
    label_space = np.zeros(shape)
    label_space[rr, cc] += l
    return label_space

def im_part_label(im_array, parts, labels):
    # for part in parts:

    return None

# lspace = label_circle((25,25), 5, (10,10), 1)
# lspace = label_circle((25,25), 5, (10,10), 2, l_space=lspace)
#
# with rasterio.open("../../Data/32.control.bil") as src:
#     hsi_raw = np.array(src.read())
#
# k1 = hsi_raw[:, 86:120, 237:277]
# k2 = hsi_raw[:, 126:200, 191:226]
# a, b, c = k1.shape
# d, e, f = k2.shape
# kL = np.ones(b*c+e*f)
#
# k1 = k1.swapaxes(0, 2).transpose(2, 0, 1).reshape(b*c, -1)
# k2 = k2.swapaxes(0, 2).transpose(2, 0, 1).reshape(e*f, -1)
#
# X1 = np.vstack((k1, k2))
#
# e1 = hsi_raw[:, 1:118, 1:192]
# h, i, j = e1.shape
# eL = np.zeros(i*j)
#
# e1 = e1.swapaxes(0, 2).transpose(2, 0, 1).reshape(i*j, -1)
#
# X = np.vstack((X1, e1))
#
# Y = np.hstack((kL, eL))
#
# gbc = GradientBoostingClassifier()
#
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=rng)
#
# gbc.fit(X_train, y_train)
#
# y_pred = gbc.predict(X_test)
#
# acc = accuracy_score(y_test, y_pred)
# print(acc)