import numpy as np
import matplotlib.pyplot as plt


def gradient_plot(data):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xi, yi, varray)
    plt.show()

#TODO: fix visualization

# S = rng.standard_t(1.5, size=(640,2))
# S[:,0] *= 2
#
# A = np.array([[1,1], [0,2]])
#
# X = np.dot(S,A.T)
#
#
# def plot_samples(S, axis_list=None):
#     plt.scatter(S[:, 0], S[:, 1], s=2, marker='o', zorder=10,
#                 color='steelblue', alpha=0.5)
#     if axis_list is not None:
#         colors = ['orange', 'red']
#         for color, axis in zip(colors, axis_list):
#             axis /= axis.std()
#             x_axis, y_axis = axis
#             plt.plot(0.1 * x_axis, 0.1 * y_axis, linewidth=2, color=color)
#             plt.quiver(0, 0, x_axis, y_axis, zorder=11, width=0.01, scale=6,
#                        color=color)
#     plt.hlines(0, -3, 3)
#     plt.vlines(0, -3, 3)
#     plt.xlim(-3, 3)
#     plt.ylim(-3, 3)
#     plt.xlabel('x')
#     plt.ylabel('y')
#
# plt.figure()
# plt.subplot(2, 2, 1)
# plot_samples(S / S.std())
# plt.title('True Independent Sources')
#
# axis_list = [pca.components_.T, ica.mixing_]
# plt.subplot(2, 2, 2)
# plot_samples(X / np.std(X), axis_list=axis_list)
# legend = plt.legend(['PCA', 'ICA'], loc='upper right')
# legend.set_zorder(100)
#
# plt.title('Observations')
#
# plt.subplot(2, 2, 3)
# plot_samples(S_pca_ / np.std(S_pca_, axis=0))
# plt.title('PCA recovered signals')
#
# plt.subplot(2, 2, 4)
# plot_samples(S_ica_ / np.std(S_ica_))
# plt.title('ICA recovered signals')
#
# plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.36)
# plt.show()
#
#