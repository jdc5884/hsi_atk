from math import sqrt
import numpy as np

import skimage.draw as draw

import pprint


# Size for each dimension
n = 50
m = 50

r1 = 4-sqrt(10)
r2 = 4-sqrt(5)
r3 = 4-sqrt(2)

sim_data = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, r1, 1, r1, 0, 0, 0],
            [0, 0, 1, r2, 2, r2, r1, 1, 0],
            [0, r1, r2, r3, 3, r3, r2, r1, 0],
            [0, 1, 2, 3, 4, 3, 2, 1, 0],
            [0, r1, r2, r3, 3, r3, r2, r1, 0],
            [0, 0, 1, r2, 2, r2, 1, 0, 0],
            [0, 0, 0, r1, 1, r1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],]


# data_skeleton = [[0 for j in range(m)] for i in range(n)]

# data_skeleton = np.array(data_skeleton)

# class cone():
#     def __init__(self, radius, height, step):
#         self.r, r = radius
#         self.h, h = height
#         self.s, s = step
#         self.z_vals = self.enum(r,s,h)

# TODO: shape gen with input array
def enum_cone(shape, xc, yc, r, h):
    """
    generate 2d array of z values for a conic
    """
    z = [[0 for i in range(shape[0]+1)] for j in range(shape[1]+1)]
    z[xc][yc] = h
    m = h/r
    for i in range(0,shape[0]+1):
        for j in range(0,shape[1]+1):
            zd = -m*np.sqrt(((i-xc)**2+(j-yc)**2)/(r/h**2))+h
            if zd > 0:
                z[i][j] = zd
            else:
                pass
    return z

cone1 = np.array(enum_cone((25, 25), 11, 11, 5, 5))

cone2 = np.array(enum_cone((500,640), 55, 200, 30, 50))


ellip0 = draw.ellipsoid(30, 30, 240, spacing=(1.0, 1.0, 1.0), levelset=True)
ellip0s = draw.ellipsoid_stats(30, 30, 240)

ell = np.array(ellip0)
ell_s = ell.shape
ell = ell.reshape(ell_s[0]*ell_s[1], -1)
