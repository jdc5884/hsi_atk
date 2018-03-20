from math import sqrt
import numpy as np

import skimage.draw as draw

import pprint


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

# TODO: generator for collections of shapes.

# class cone():
#     def __init__(self, radius, height, step):
#         self.r, r = radius
#         self.h, h = height
#         self.s, s = step
#         self.z_vals = self.enum(r,s,h)

def square1d(shape, val=0):
    im = [val]*shape
    return im

def square2d(shape, val=0):
    im = [[val]*shape[0] for i in range(shape[1])]
    return im

def square3d(shape, val=0):
    im = [[[val]*shape[0] for i in range(shape[1])] for j in range(shape[2])]
    return im

def cone_lin_func(r, h, x, y, xC, yC, hmod, rd):
    return round(-(h/r)*np.sqrt(((x-xC)**2+(y-yC)**2)/(r/h)**2)+h+hmod, rd)

def enum_cone(shape, xc, yc, r, h, hmod=0, rd=4):
    """
    generate 2d array of z values for a conic
    """
    z = [[0 for i in range(shape[0]+1)] for j in range(shape[1]+1)]
    # z[xc][yc] = h
    m = h/r
    for i in range(0,shape[0]+1):
        for j in range(0,shape[1]+1):
            zd = cone_lin_func(r, h, i, j, xc, yc, hmod, rd)
            if zd > 0:
                z[i][j] += zd
            else:
                pass
    return z

def cone_gen(im_array, xc, yc, r, h, hmod=0):
    xL = int(np.floor(xc - r))
    yB = int(np.floor(yc - r))
    xR = int(np.ceil(xc + r))
    yT = int(np.ceil(yc + r))
    for i in range(xL, xR):
        for j in range(yB, yT):
            zd = cone_lin_func(r, h, i, j, xc, yc, hmod, 4)
            if zd > 0:
                im_array[i][j] = zd
            else:
                pass
    return im_array


print(cone_lin_func(10,10,35,35,35.5,35.5,0,4))

# sq1 = np.array(square1d(200))
sq2 = square2d((200,200))
# sq3 = np.array(square3d((200,200,10)))
sq2 = np.array(cone_gen(sq2, 35.5, 35.5, 10, 10))
sq2 = np.array(cone_gen(sq2, 50, 50, 11, 12))
sq2 = np.array(cone_gen(sq2, 100, 100, 11, 12))
sq2 = np.array(cone_gen(sq2, 20, 50, 10, 15))
### Testing generation
# cone1 = np.array(enum_cone((25, 25), 11, 11, 5, 5))
#
# cone2 = np.array(enum_cone((500,640), 55, 200, 50, 50))
#
# cone3 = np.array(enum_cone((400,400), 35.5, 35.5, 10, 10))
#
# cone4 = np.array(enum_cone((400,400), 35.2, 35.7, 10, 10))
#
# cone5 = np.array(enum_cone(()))
#
#
# ellip0 = draw.ellipsoid(30, 30, 240, spacing=(1.0, 1.0, 1.0), levelset=True)
# ellip0s = draw.ellipsoid_stats(30, 30, 240)
#
# ell = np.array(ellip0)
# ell_s = ell.shape
# ell = ell.reshape(ell_s[0]*ell_s[1], -1)
