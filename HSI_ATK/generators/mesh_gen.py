import numpy as np
from math import sqrt, hypot
# from scipy import interpolate as itp, integrate as int
# import matplotlib.pyplot as plt


def point_xval(c, xy):
    """
    simply distance compute for generator functions

    :param c: (tuple) center values
    :param xy: (tuple) point values
    :return: distance from center
    """
    x_val = hypot(c, xy)
    return x_val


def image_gen(size, shape=None, center=None, radius=None, height=None, length=None, width=None):
    """
    Generate n_dimensional image with specified shapes individually or by collections with modular parameters

    :param size: tuple specifying image shape (ex. 400x500 = (400,500))
    :param shape: shape function
    :param center:
    :param radius:
    :param height:
    :param length:
    :param width:
    :return:
    """
    x = size[0]
    y = size[1]
    xx, yy = np.mgrid[-x:x,-y:y]
    values = conic(xx, yy, radius, height, center)
    return values


def conic(x, y, r, h, center, hmod=0.0, func='x'):
    """
    Generate conic z_vals based of simple functions

    :param x: (float) - point on func for computing z value
    :param hmod: (float) - z-value displacement
    :return: (float) - computed z-value at point
    """
    c = r/h
    z = sqrt(((x-center[0])**2+(y-center[1])**2)/c**2)
    if z > 0:
        return z
    else:
        return 0

# cone1 = image_gen((500,500), center=(25,25), radius=5, height=5)
#
# plt.plot(cone1)
# plt.show()