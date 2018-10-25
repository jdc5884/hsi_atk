import h5py
import os
import numpy as np
import pandas as pd
import rasterio


class Dataset:
    """Class for file I/O of stored datasets"""

    def __init__(self, input_folder, file_format, output_folder, lbl_file=None,
                 geno=None, packet=None, treatment=None):

        if not input_folder.endswith('/'):
            input_folder = input_folder + '/'  # ensuring folder path ends with '/' for
                                               # path appending
        if not output_folder.endswith('/'):
            output_folder = output_folder + '/'

        self.input = input_folder   # parent input folder
        self.output = output_folder # parent output folder
        self.format = file_format   # whether loading .bil or .h files/datasets
        self.labels = lbl_file      # .csv file containing experimental labels
        self.genos = geno           # genotypes of interest, also used as folder indexing
        self.packet = packet        # seed packet, used for experimental data and file indexing
        self.treat = treatment      # hormonal treatment used for experimental and file indexing
        self.images = None          # used to hold hsi image arrays temporarily, though usually are just returned
                                    # can be singular 4-D array or list of arrays (referring to numpy array)

    def loadBILFile(self, folder, file, format=1):
        """Load image array from .bil file
        :param file_path: string - path to .bil file (requires .bil.hdr file in same directory)
        :param file: string - packet#.treatment.bil file string
        :param format: 0 or 1 - corresponds to bands first or bands last
                                default is bands last (format=1)
        :return: numpy array of image in shape (lines, cols, bands)
        """
        raw = rasterio.open(folder+file)
        img = np.array(raw.read())

        if format == 0:
            return img  # return bands first format of image

        img = img.swapaxes(0, 2).swapaxes(0, 1)
        return img  # returns bands last format of image

    def composeBILFiles(self, stacked=True):
        """Composes hsi image arrays into single 4-D array."""
        images = []
        if self.genos is not None:
            for geno in self.genos:
                images.append(self.getBILFiles(geno))
            if stacked:
                return np.array(images)
            return images

        # Getting all genotypes if self.genos is None
        all_genos = os.listdir(self.input)
        for geno in all_genos:
            images.append(self.getBILFiles(geno))
        if stacked:
            return np.array(images)
        return images

    def getBILFiles(self, geno):
        """Gets list of files in folders.
        Hyperspectral files organized in folders by genotype.
        Should be passed folder of specific genotype."""
        geno_path = self.input+geno
        children = os.listdir(geno_path)  # list of files in folder
        foi = []  # files of interest
        packet = self.packet
        treatment = self.treat

        for child in children:
            if not child.endswith('.bil'):
                continue                # passing files not ending in .bil

            childl = child.lower()      # enforcing all lower case
            csplit = childl.split('.')  # splitting packet#.treatment.bil format
            if packet is None and treatment is None:
                foi.append(child)  # gets all files of geno
            elif csplit[0] in packet and treatment is None:
                foi.append(child)  # gets all treatments of geno.packet
            elif packet is None and csplit[1] in treatment:
                foi.append(child)  # gets all packets of geno.treatment
            elif csplit[0] in packet and csplit[1] in treatment:
                foi.append(child)  # gets all files of geno.packet.treatment

        if len(foi) == 0:
            raise Exception("No files matching this criteria!")

        return foi



# pass as list of tuples to make 1 for loop generating file
def build_dataset(file, pentaras):

    hf = h5py.File(file, 'w')
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
