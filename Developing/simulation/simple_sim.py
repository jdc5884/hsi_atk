__author__ = "David Ruddell"
__credits__ = ["David Ruddell"]
__license__ = "GPL"
__version__ = "0.0.2"
__status__ = "Development"

import numpy as np
from skimage.draw import ellipse


def gen_brightness_func(coefs, sigma=None):
    deg = coefs.size - 1
    if sigma is None:
        brightness = lambda x, y, z: sum(coefs[i] * z ** (deg - i) for i in range(deg + 1))
    else:
        def brightness(x,y,z):
            rand_shift = np.add(coefs, sigma/3) #np.multiply(np.random.randn(coefs.size), sigma))
            return sum([rand_shift[i]*z**(deg-i) for i in range(deg+1)])
    return brightness


def gen_class_pix(brightnessfunc, shape):
    return np.fromfunction(brightnessfunc,shape)


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
    from Developing.exploratory.model_extraction import fit_ply_mdl
    from hsi_atk.utils.hsi2color import hsi2color
    from hsi_atk.utils.dataset import open_hsi_bil

    img = open_hsi_bil("../../Data/B73/32.control.bil")
    AOI = img[88:178, 461:536, :]
    ply_stats = fit_ply_mdl(AOI)
    coefs = ply_stats[3]['mean'].reshape(6)
    sigma = ply_stats[3]['std'].reshape(6)
    # coefs = np.random.rand(6)
    bfunc = gen_brightness_func(coefs, sigma)
    # bfunc = gen_brightness_func(coefs)
    img = compose([bfunc],(40,40,240),[(15,12)],[(21,25)])
    from skimage.util import random_noise
    img_ = np.random.randn(40,40,240)*400
    img = np.add(img,img_)
    # img2 = compose([bfunc], (40, 40, 40), [(15, 12)], [(21, 25)], [.3])

    color = hsi2color(img, scale=False, out_type=float)
    import matplotlib.pyplot as plt
    plt.imshow(color)
    plt.show()

