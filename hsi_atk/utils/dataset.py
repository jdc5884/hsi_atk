import h5py as h5
import os
import numpy as np
import pandas as pd
import rasterio


class Dataset():
    """Class for file I/O of stored datasets"""

    def __init__(self, h5_file, mode='r'):
        self.file_name = h5_file
        self.mode = mode
        self.hf = h5.File(h5_file, mode=mode)

    def get_hf(self):
        return self.hf

    def get_mode(self):
        return self.mode

    def write_group(self, group_name, group_path='/'):
        """
        Create group in h5 file by a specified path
        :param group_name: string - end of path group name
        :param group_path: string - path through root -> group -> subgroups -> etc
        :return: None
        """
        if "w" in self.get_mode():
            hf = self.get_hf()
            group_name = group_path + group_name
            hf.create_group(group_name)
        else:
            print("Not in write mode!")

    def write_dataset(self, dataset_name, data, group_path='/', autochunk=False):
        """
        Create dataset in h5 file by specified group path
        :param dataset_name: string - name of dataset. Ex: "raw_images"
        :param data: numpy ndarray - data to be stored in dataset
        :param group_path: string - path to group. Ex: "/images/raw", "/labels/genotype"
        :param autochunk: boolean - whether or not to allow h5 to auto chunk. Helpful for larger dataset access performance
        :return: None
        """
        if "w" in self.get_mode():
            hf = self.get_hf()
            group = hf[group_path]
            group.create_dataset(dataset_name, data=data, chunks=autochunk)
        else:
            print("Not in write mode!")


class HSI_Dataset(Dataset):

    def __init__(self, h5_file, mode='r'):
        super().__init__(h5_file=h5_file, mode=mode)



# pass as list of tuples to make 1 for loop generating file
def build_pent_dataset(file, pentaras):

    hf = h5.File(file, 'w')
    g0 = hf.create_group('images')
    g1 = hf.create_group('labels')
    # g2 = g1.create_group('kernels')

    i = 0
    for pentara in pentaras:
        data_s = 'data_' + str(i)
        g0.create_dataset(data_s, data=pentara.get_img(), compression="gzip", compression_opts=9, dtype='f8')
        k_s = 'kernels_' + str(i)
        g2 = g1.create_group(k_s)
        kernels = pentara.obs
        kernels_k = kernels.keys()
        j = 0
        for kernel in kernels_k:
            kern = 'kernel_' + str(j)
            g2.create_dataset(kern, data=kernel.labels, compression="gzip", compression_opts=9)
            j += 1
        i += 1

    hf.close()


# Don't really need unless one wants to load into separate dicts
def enum_pent_dataset(file):

    hf = h5.File(file, 'r')

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


def open_hsi_bil(file_path):
    """
    Load image array from .bil file
    :param file_path: string - path to .bil file (requires .bil.hdr file in same directory)
    :return: numpy array of image in shape (lines, cols, bands)
    """
    raw = rasterio.open(file_path)
    img = np.array(raw.read())
    img = img.swapaxes(0, 2).swapaxes(0, 1)
    return img


def write_h5(file_path, group):
    hf = h5.File(file_path, 'w')

# def load_hsi_bil_folder(dir_path, packet=None, hormone=None, geno=None):
#     """
#     Load images from folder of .bil files
#     :param dir_path: string - path to directory
#     :param packet: int or list of ints - packet numbers to load
#     :param hormone: string or list of strings - hormone treatments to load
#     :param geno: string - genotype of images in folder
#     :return:
#     """
#
#     if packet is not None and hormone is not None:
#         for file in os.listdir(dir_path):
#             if file.endswith('.bil'):
#                 pac, hor, suf = file.split('.')
#                 if (pac in packet and hor in hormone) or (pac == packet and hor == hormone):
#
#     else:
#         pass
#
#     return None
#
#
# def load_all_hsi(dir_path, geno=None, **kwargs):
#     """
#     Load images from a collection of folders of .bil files
#     :param dir_path: string - path to parent hsi directory
#                             subdirectories are expected to be named after genotypes
#     :param geno: string or list of strings - genotypes to load
#     :param kwargs: arguments for load_hsi_bil_folder
#     :return:
#     """
#     # getting list of immediate child directories (named by genotype)
#     data = pd.DataFrame()
#
#     children = os.listdir(dir_path)
#
#     if geno is not None:
#         for child in children:
#             if child in geno:
#                 c_path = dir_path + child
#                 load_hsi_bil_folder(c_path, geno=child, **kwargs)
#
#     else:
#         pass
#
#     return None
