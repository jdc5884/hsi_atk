import numpy as np
from scipy import interpolate as interp

from HSI_ATK.fileimport import *


hsi_raw = loadImage("../Data/32.control.bil")

x = np.linspace(0, 499, 500)
y = np.linspace(0, 499, 500)

bands, lines, samples = hsi_raw.shape

int_stats = []

# for i in range(0, bands):
#     frame = np.copy(hsi_raw[i, :, 70:570])
#     maxz = int(frame.max())
#     z = np.linspace(0, maxz-1, maxz)
#     int_p = interp.RegularGridInterpolator((x, y), frame)
#     int_stats.append(int_p)

frame = np.copy(hsi_raw[0, :, 70:570])

int_p = interp.RegularGridInterpolator((x, y), frame, method="nearest")

x = np.linspace(0, 499, 1000)
y = np.linspace(0, 499, 1000)
