# author David Ruddell
# contact: dr1236@uncw.edu, dlruddell@gmail.com

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as axes3d
from matplotlib.collections import PolyCollection
from matplotlib import colors as mcolors
import seaborn as sns

hyper_data = pd.read_csv('~/PycharmProjects/HyperSpectralImaging/Data/massaged_data.csv', sep=',')

# TODO: Convert visualization to polygon or bar plot.
# Consider using average signal for wavelength of all in a geno.

# geno = [[0],[0],[0],[0],[1],[1],[1]]
# geno = [[0],[1]]

data = []
for i in range(254):
    data.append([hyper_data[i,1].astype(float), hyper_data[i,13:].astype(float)])

wavelengths = hyper_data.columns.values[13:]

fig = plt.figure()
ax = fig.gca(projection='3d')


def cc(arg):
    return mcolors.to_rgba(arg, alpha=0.6)


xs = data[1]
ys = wavelengths
verts = []
zs = data[0]
for z in zs:
    verts.append(list(zip(xs, ys)))

poly = PolyCollection(verts, facecolors=[])

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# x = hyper_data.values[4:6, 13:].astype(float)
# y = hyper_data.columns.values[13:].astype(float)
# z = geno
# z = hyper_data.values[:, 1].astype(float)
# _ = ax.plot_surface(x, y, z)
# _ = ax.set_xlabel('Signal')
# _ = ax.set_ylabel('Wavelengths')
# _ = ax.set_zlabel('Genotype')
# _ = plt.title('Genotype vs. Wavelengths visualization')
# plt.show()
