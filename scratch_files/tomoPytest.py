import math
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from tomopy import angles, circ_mask, recon, find_center
import tifffile as tiff

with rasterio.open("../Data/32.control.bil") as src:
    hsi_raw = np.array(src.read())

theta = angles(hsi_raw.shape[0])

center = find_center(hsi_raw[:,110:191,400:461], theta=theta, sinogram_order=True)

recon = recon(hsi_raw[:,110:191,400:461], theta=theta, center=center, algorithm='bart')

recon = circ_mask(recon, axis=0, ratio=0.95)
recon = np.array(recon)

shape = recon.shape

tiff.imsave(file="../Data/tomoRall.tiff", data=recon, shape=shape)

# print(recon[:,20:41,20:41])

# plt.imshow(recon[80,:,:], cmap='Greys_r')
# plt.show()