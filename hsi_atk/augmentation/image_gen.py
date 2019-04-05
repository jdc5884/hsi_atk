import h5py
import numpy as np
import pandas as pd


def get_input(path):
    """Loads image from h5py file"""
    img = h5py.File("../Data/img_all_4d.h5", "r")
    images = img['datatset'][:]
    img.close()

    return images.swapaxes(0,2).swapaxes(1,2).astype(np.int16)[:, :, 70:570, :]


def get_labels(path, label_idx):
    label_file = pd.read_csv(path, sep='r')
    return np.array(label_file.values[:, label_idx])


def ref_ax1(images, batch):
    ref_image = images[batch,::-1,:,:]
    return ref_image


def ref_ax2(images, batch):
    ref_image = images[batch,:,::-1,:]
    return ref_image


def get_rnd_ndx(batch, total=46):
    indices = np.random.permutation(total)
    training_idx, test_idx = indices[:batch], indices[batch:]
    return training_idx, test_idx


def image_generator(images, label_path, batch_idx, label_idx, h_flip=True, v_flip=True):

    images = images[batch_idx, :, :, :]
    labels = get_labels(label_path, label_idx)
    himages = ref_ax1(images, batch_idx)
    vimages = ref_ax2(images, batch_idx)

    c=0

    while True:
        if c==3:
            c=0

        if c==0:
            c+=1
            yield(images[batch_idx, :, :, :], labels[:, batch_idx])
        elif c==1:
            c+=1
            yield(himages[batch_idx, :, :, :], labels[:, batch_idx])
        elif c==2:
            c+=1
            yield(vimages[batch_idx, :, :, :], labels[:, batch_idx])
