__author__ = "David Ruddell"
__credits__ = ["David Ruddell"]
__license__ = "GPL"
__version__ = "0.0.2"
__status__ = "Development"

import numpy as np


# wavelengths in HSI
_WAVELENGTHS = [393.91, 395.982, 398.054, 400.126, 402.198, 404.27, 406.342, 408.414, 410.486, 412.558, 414.63, 416.702, 418.774, 420.846, 422.918, 424.99, 427.062, 429.134, 431.206, 433.278, 435.35, 437.422, 439.494, 441.566, 443.638, 445.71, 447.782, 449.854, 451.926, 453.998, 456.07, 458.142, 460.214, 462.286, 464.358, 466.43, 468.502, 470.574, 472.646, 474.718, 476.79, 478.862, 480.934, 483.006, 485.078, 487.15, 489.222, 491.294, 493.366, 495.438, 497.51, 499.582, 501.654, 503.726, 505.798, 507.87, 509.942, 512.014, 514.086, 516.158, 518.23, 520.302, 522.374, 524.446, 526.518, 528.59, 530.662, 532.734, 534.806, 536.878, 538.95, 541.022, 543.094, 545.166, 547.238, 549.31, 551.382, 553.454, 555.526, 557.598, 559.67, 561.742, 563.814, 565.886, 567.958, 570.03, 572.102, 574.174, 576.246, 578.318, 580.39, 582.462, 584.534, 586.606, 588.678, 590.75, 592.822, 594.894, 596.966, 599.038, 601.11, 603.182, 605.254, 607.326, 609.398, 611.47, 613.542, 615.614, 617.686, 619.758, 621.83, 623.902, 625.974, 628.046, 630.118, 632.19, 634.262, 636.334, 638.406, 640.478, 642.55, 644.622, 646.694, 648.766, 650.838, 652.91, 654.982, 657.054, 659.126, 661.198, 663.27, 665.342, 667.414, 669.486, 671.558, 673.63, 675.702, 677.774, 679.846, 681.918, 683.99, 686.062, 688.134, 690.206, 692.278, 694.35, 696.422, 698.494, 700.566, 702.638, 704.71, 706.782, 708.854, 710.926, 712.998, 715.07, 717.142, 719.214, 721.286, 723.358, 725.43, 727.502, 729.574, 731.646, 733.718, 735.79, 737.862, 739.934, 742.006, 744.078, 746.15, 748.222, 750.294, 752.366, 754.438, 756.51, 758.582, 760.654, 762.726, 764.798, 766.87, 768.942, 771.014, 773.086, 775.158, 777.23, 779.302, 781.374, 783.446, 785.518, 787.59, 789.662, 791.734, 793.806, 795.878, 797.95, 800.022, 802.094, 804.166, 806.238, 808.31, 810.382, 812.454, 814.526, 816.598, 818.67, 820.742, 822.814, 824.886, 826.958, 829.03, 831.102, 833.174, 835.246, 837.318, 839.39, 841.462, 843.534, 845.606, 847.678, 849.75, 851.822, 853.894, 855.966, 858.038, 860.11, 862.182, 864.254, 866.326, 868.398, 870.47, 872.542, 874.614, 876.686, 878.758, 880.83, 882.902, 884.974, 887.046, 889.118]

# color wavelengths
_RED_RANGE = (622, 750)
_ORANGE_RANGE = (597, 622)
_YELLOW_RANGE = (577, 597)
_GREEN_RANGE = (492, 577)
_BLUE_RANGE = (455, 492)
_VIOLET_RANGE = (390, 455)
_IFR = (750,900)

_RGB_RNG = [_RED_RANGE, _GREEN_RANGE, _BLUE_RANGE]
_RGB_PLUS = [_RED_RANGE, _GREEN_RANGE, _BLUE_RANGE, _IFR]


def hsi2gray(img):
    gray = np.mean(img, axis=2)
    gray /= np.max(gray.ravel())
    return gray

def hsi2color(img, color_rngs=_RGB_RNG, wavelengths=_WAVELENGTHS, scale_in=4095, scale_out=255, scale=False, out_type=float):
    images = []

    for rng in color_rngs:
        img_slice = color_slice(img, rng, wavelengths)
        img_slice /= scale_in
        images.append(img_slice)
    rgb = np.stack(images, axis=2)

    if scale:
        return rgb*scale_out

    if out_type is int:
        np.floor(rgb)

    rgb.astype(out_type)

    return rgb


def hsi2color4(img, color_rngs=_RGB_PLUS, wavelengths=_WAVELENGTHS, scale_in=4095, scale_out=255, scale=False):
    images = []

    for rng in color_rngs:
        img_slice = color_slice(img, rng, wavelengths)
        img_slice /= scale_in
        images.append(img_slice)
    img_out = np.stack(images, axis=2)

    if scale:
        return img_out * scale_out

    return img_out


def color_slice(img, wave_rng, wavelengths):
    c = 0
    slices = []
    for wl in wavelengths:
        if np.ceil(wl) >= wave_rng[0]-2 and np.floor(wl) <= wave_rng[1]+2:
            slices.append(c)
        c += 1

    return np.mean(img[:, :, slices], axis=2)