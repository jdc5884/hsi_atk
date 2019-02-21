import h5py as h5
import os
import numpy as np
import pandas as pd
import rasterio


class Dataset():
    """Class for file I/O of stored datasets"""

    def __init__(self, h5_file, mode='a'):
        self.file_name = h5_file
        self.mode = mode
        self.hf = h5.File(h5_file, mode=mode)

    def get_hf(self):
        return self.hf

    def get_mode(self):
        return self.mode

    #TODO: fix hf manifest list generator
    def get_hf_mfest(self, prefix='', groups=[], dsets=[]):
        """
        Generates and returns list of group and dataset instances in hdf5 file as separate lists
        """
        hf = self.get_hf()
        for key in hf.keys():
            item = hf[key]
            path = '{}/{}'.format(prefix, key)
            if isinstance(item, h5.Dataset):
                dsets.append(path)
            elif isinstance(item, h5.Group):
                groups.append(path)
                g, d = self.get_hf_mfest(prefix=path)
                groups.extend(g)
                dsets.extend(d)
        return groups, dsets


    def get_dset_mfest(self):
        """
        Generates and returns list of dataset instances in hdf5 file
        """
        f = self.get_hf()
        names = f.visititems(list_dataset)
        return names

    def get_group_mfest(self):
        """
        Generates and returns list of group instances in hdf5 file
        """
        f = self.get_hf()
        names = f.visititems(list_group)
        return names

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
            print("Not in write mode! Group not written!")

    def write_dataset(self, dataset_name, data, group_path='/', autochunk=True):
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


class HSIDataset(Dataset):

    def __init__(self, h5_file, mode='a'):
        super().__init__(h5_file=h5_file, mode=mode)


class KernelHSIDset(Dataset):

    def __init__(self, h5_file, genotypes=('B73','CML103'), mode='a'):
        super().__init__(h5_file=h5_file, mode=mode)
        self.genotypes=genotypes
        self.init_group_structure()

    def init_group_structure(self):
        hf = self.get_hf()
        for geno in self.get_genos():
            self.write_group(geno)

    def get_genos(self):
        return self.genotypes


def list_dataset(name, node):
    dsets = []
    if isinstance(node, h5.Dataset):
        dsets.append(name)


def list_group(name, node):
    groups = []
    if isinstance(node, h5.Group):
        groups.append(name)


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


def write_metadata(hf_group, img_path):
    """
    Writes metadata of hsi into h5 file in specified format.

    :param hf_group: h5 file group - expected to be open in h5 file open in a write mode.
    :param img_path: string - path to image file. Expecting the metadata to be stored horizontally.
    :return: None
    """

    pass


#TODO: sort image paths by genotype (should already be done by appending order) and make iteration more greedy
#TODO: add existing dataset conditional handlers
def convert_bil_h5(file_path, img_paths, geno, store_metadata=True):
    """
    Writes set of images into h5 file organized by /group/img
    in the format /geno/packet#.hormone
    :param file_path:
    :param img_paths:
    :param geno:
    :return:
    """
    hf = h5.File(file_path, 'a')

    for gene in geno:
        if gene not in hf.keys():
            group = hf.create_group(gene)
        else:
            group = hf[gene]

        for file in img_paths:
            f = os.path.basename(file)
            packet, hormone, ext = f.split(".")

            if hormone not in group.keys():
                horm = group.create_group(hormone)
                pac = horm.create_group(packet)
            else:
                horm = group[hormone]
                if packet not in horm.keys():
                    pac = horm.create_group(packet)
                else:
                    pac = horm[packet]



            if gene in file:
                img = open_hsi_bil(file)
                name_idx = file.find(gene) + len(gene+'/')
                cut_idx = file.find(".bil")
                dset_name = file[name_idx:cut_idx]
                group.create_dataset(dset_name, data=img, chunks=True)

    hf.close()



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


def enum_hsi_files(dir_path, return_geno=True):
    """
    Return list of file paths for .bil files of HS images.

    Expects images to be stored in "/root_path/geno/packet#.hormone.bil". It may also be of interest to acquire paths
    to the metadata .hdr files. For now, we will assume they are the same path as the reference image except with
    .hdr appended.

    :param dir_path: string - parent directory of images
    :param geno: string - genotype(s) of interest. Ex. 'B73', ['B73', 'CML103'], etc. None selects all
    :param hormone: string - hormone treatment in question. Ex. 'control', 'GA', ['CONTROL', 'GA', 'PAC+GA']
    :param packet: string or int - packet # of seed selection
    :return: list of strings - file paths to HS images
    """
    image_paths = []
    geno_list = []
    genos = os.listdir(dir_path)  # gets subdirectories named by genotype

    for g in genos:
        g_path = dir_path+'/'+g+'/'
        geno_list.append(g)
        gimages = os.listdir(g_path)
        nimages = []
        for img in gimages:
            if img.endswith(".bil"):  # not including .bil.hdr files
                img = img.lower()
                img_p = g_path+img
                nimages.append(img_p)

        image_paths.extend(nimages)

    if return_geno:
        return image_paths, geno_list

    return image_paths



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
