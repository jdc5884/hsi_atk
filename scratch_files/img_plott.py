import rasterio
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d


with rasterio.open("../Data/32.control.bil") as src:
    array = np.array(src.read())

# array in shape bands, lines, samples
grad = np.gradient(array)

print(grad)

# Change array from bands, lines, samples to samples, lines, bands
n_array = array.swapaxes(0,2)
# Reshape to 2D in form samples*lines,bands
array_2d = n_array.transpose(2,0,1).reshape(320000,-1)

idx = np.random.random_integers(0,239,15) # size == n_comp for arbitrary class

X = np.arange(50,500,1)
Y = np.arange(50,500,1)
X, Y = np.meshgrid(X, Y)
Z = np.var(array[idx, 50:500, 50:500],axis=0)

# for i in range(1,240,1):
#     Z += array[i,50:500,50:500]
#
# Z = np.divide(Z, 240)
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# cset = ax.contour(X, Y, Z, cmap=cm.coolwarm)
# ax.clabel(cset, fontsize=9, inline=1)
#
# plt.show()