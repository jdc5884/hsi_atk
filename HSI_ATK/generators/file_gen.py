import os
import numpy as np

from HSI_ATK.Generators.simple_gen import xy_gen, add_noise


image_set = []
label_set = []
l_space = []

for i in range(500):
    im, la, lp = xy_gen((50, 50), 4, 20, 5, pixel_label=True)
    image_set.append(im)
    label_set.append(la)
    l_space.append(lp)

for j in range(50):
    im, la, lp = xy_gen((50, 50), 0, 2, 2, pixel_label=True)
    image_set.append(im)
    label_set.append(la)
    l_space.append(lp)

    im, la, lp = xy_gen((50, 50), 4, 50, 5, pixel_label=True)
    image_set.append(im)
    label_set.append(la)
    l_space.append(lp)

image_set = np.array(image_set)
imS = image_set.shape
image_set = image_set.reshape(imS[0], imS[1]*imS[2])

l_space = np.array(l_space)
imS = l_space.shape
l_space = l_space.reshape(imS[0], imS[1]*imS[2])

# save X data to csv
np.savetxt('../TestData/c1_gn.csv', image_set, delimiter=',', fmt='%d')
# save y data to csv
np.savetxt('../TestData/c1_lb.csv', label_set, delimiter=',', fmt='%d')
# save xy coord labels
np.savetxt('../TestData/c1_xy.csv', l_space, delimiter=',', fmt='%d')
