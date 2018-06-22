import numpy as np
import h5py
import rasterio

# pass as list of tuples to make 1 for loop generating file
def build_dataset(file, pentaras, pentigas):

    hf = h5py.File(file, 'w')
    g0 = hf.create_group('images')
    g1 = hf.create_group('labels')
    # g2 = g1.create_group('kernels')

    i = 0
    for pentara in pentaras:
        data_s = 'data_' + str(i)
        g0.create_dataset(data_s, data=pentara._img, compression="gzip", compression_opts=9, dtype='f8')
        k_s = 'kernels_' + str(i)
        g2 = g1.create_group(k_s)
        kernels = pentigas[data_s]
        j = 0
        for kernel in kernels:
            kern = 'kernel_' + str(j)
            g2.create_dataset(kern, data=kernel.labels, compression="gzip", compression_opts=9)
            j += 1
        i += 1

    hf.close()


# Don't really need unless one wants to load into separate dicts
def enum_dataset(file):

    hf = h5py.File(file, 'r')

    data_i = hf['images'].keys()
    label_i = hf['labels'].keys()

    data_d = {}
    label_d = {}

    for key in data_i:
        data_d[key] = hf['images'].get(key)

    for key in label_i:
        kernel_i = hf['labels'].keys()
        label_d[key] = {}
        for kkey in kernel_i:
            label_d[key][kkey] = hf['labels'].get(kkey)

    return data_d, label_d


def openBIL(file_path):
    '''
    Grab image array from .bil file
    :param file_path: string, path to .bil file (requires .bil.hdr file in same directory)
    :return: numpy array of image in shape (lines, cols, bands)
    '''
    raw = open(file_path)
    img = np.array(raw.read())
    img = img.swapaxes(0, 2).swapaxes(0, 1)
    return img
