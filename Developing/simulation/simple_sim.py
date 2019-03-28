__author__ = "David Ruddell"
__credits__ = ["David Ruddell"]
__license__ = "GPL"
__version__ = "0.0.2"
__status__ = "Development"

import numpy as np
from sympy import poly
from sympy.abc import z
from skimage.draw import ellipse


def gen_brightness_func(coefs, center, sigma=None):
    deg = coefs.size - 1
    switch = {0: poly0,
              1: poly1,
              2: poly2,
              3: poly3,
              4: poly4,
              5: poly5}
    if sigma is None:
        brightness = switch[deg](coefs, center)
    else:
        def brightness(x,y,z):
            rand_shift = np.add(coefs, sigma/3) #np.multiply(np.random.randn(coefs.size), sigma))
            return sum([rand_shift[i]*z**(deg-i) for i in range(deg+1)])
    return brightness


def poly0(coefs, center):
    def poly(x,y,z):
        return (1+np.cos(np.sqrt((x-center[0])**2 + (y-center[1])**2))/10) * \
               (coefs[0])
    return poly


def poly1(coefs, center):
    def poly(x,y,z):
        return (1+np.cos(np.sqrt((x-center[0])**2 + (y-center[1])**2))/10) * \
               (coefs[0]*z**2 + coefs[1]*z + coefs[2])
    return poly


def poly2(coefs, center):
    def poly(x,y,z):
        return (1+np.cos(np.sqrt((x-center[0])**2 + (y-center[1])**2))/10) * \
               (coefs[0]*z**2 + coefs[1]*z + coefs[2])
    return poly


def poly3(coefs, center):
    def poly(x,y,z):
        return (1+np.cos(np.sqrt((x-center[0])**2 + (y-center[1])**2))/10) * \
               (coefs[0]*z**3 + coefs[1]*z**2 + coefs[2]*z +
                coefs[3])
    return poly


def poly4(coefs, center):
    def poly(x,y,z):
        return (1+np.cos(np.sqrt((x-center[0])**2 + (y-center[1])**2))/10) * \
               (coefs[0]*z**4 + coefs[1]*z**3 + coefs[2]*z**2 +
                coefs[3]*z + coefs[4])
    return poly


def poly5(coefs, center):
    def poly(x,y,z):
        return (1+np.cos(0.25 * np.sqrt((x-center[0])**2 + (y-center[1])**2))/10) * \
               (coefs[0]*z**5 + coefs[1]*z**4 + coefs[2]*z**3 +
                coefs[3]*z**2 + coefs[4]*z + coefs[5])
    return poly


def compose(bfuncs, shape, smas, centers, rots=None):
    base = np.zeros(shape)
    x,y,z = shape
    for i in range(len(bfuncs)):
        bfunc = bfuncs[i]
        RCR = smas[i]
        r_r,c_r = RCR
        RC = centers[i]
        r,c=RC
        if rots is None:
            rr,cc = ellipse(r,c,r_r,c_r,shape)
            rr0,cc0 = ellipse(r_r,c_r,r_r,c_r,(np.ceil(r_r*2).astype(int),np.ceil(c_r*2).astype(int)))
        else:
            rot = rots[i]
            rr, cc = ellipse(r, c, r_r, c_r, shape, rot)
            rr0, cc0 = ellipse(r_r, c_r, r_r, c_r, (np.ceil(r_r * 2).astype(int), np.ceil(c_r * 2).astype(int)), rot)
        subimg = np.fromfunction(bfunc,(np.ceil(r_r*2).astype(int),np.ceil(c_r*2).astype(int),z))
        base[rr,cc,:] = subimg[rr0,cc0,:]
    return base




if __name__ == '__main__':
    # import sys
    # print('Python %s on %s' % (sys.version, sys.platform))
    # sys.path.extend(['/Users/tensorstrings/hsi_atk/hsi_atk'])

    from Developing.exploratory.model_extraction import fit_ply_mdl
    from hsi_atk.utils.hsi2color import hsi2color
    from hsi_atk.utils.dataset import open_hsi_bil
    from Developing.simulation.histogram_matching import match_histograms

    img = open_hsi_bil("../../Data/B73/32.control.bil")
    AOI = img[88:178, 461:536, :]
    ply_stats = fit_ply_mdl(AOI, return_counts=True)
    coefs = ply_stats[3]['mean'].reshape(6)
    sigma = ply_stats[3]['std'].reshape(6)
    # coefs = np.random.rand(6)
    bfunc = gen_brightness_func(coefs, (15,15))
    # bfunc = gen_brightness_func(coefs)
    img = compose([bfunc],(40,40,240),[(15,12)],[(21,25)])
    from skimage.util import random_noise
    img_ = np.random.randn(40,40,240)*350
    rr,cc = ellipse(21,25,15,12,(40,40))
    # rr0,cc0 = ellipse(15,12,15,12,(np.ceil(15*2),np.ceil(12*2)))
    ref_ = AOI[25:65, 25:65, :]
    ref = np.zeros((40,40,240))
    ref[rr,cc,:] = ref_[rr,cc,:]
    ref = np.add(ref,img_.copy())
    img = np.add(img,img_)
    img = match_histograms(img,ref,multichannel=True)
    # img2 = compose([bfunc], (40, 40, 40), [(15, 12)], [(21, 25)], [.3])

    color = hsi2color(img, scale=False, out_type=float)
    import matplotlib.pyplot as plt
    plt.imshow(color)
    plt.show()

