import h5py
import numpy as np
import os
import pandas as pd
from sklearn.metrics import mean_squared_error
from skimage.restoration import denoise_wavelet
from keras import models, optimizers, backend
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import core, convolutional, pooling, SpatialDropout2D
from hsi_atk.weight_logger import WeightsLogger


local_project_path = '/'
local_data_path = os.path.join(local_project_path, 'data')

images_file = h5py.File("../Data/img_all_4d.h5", "r")
images = images_file['datatset'][:]
images_file.close()
images = images.swapaxes(0,2).swapaxes(1,2)
# print(images.shape)
# images_flip_x = np.flip(images, axis=1)
# images_flip_y = np.flip(images, axis=2)
# images_flip_xy = np.flip(images_flip_x, axis=2)
# X_train = np.stack((images, images_flip_x), axis=0).reshape((92, 500, 640, 240))
# del images
# del images_flip_x
# del images_flip_y
# print(X_train.shape)
# X_test = images_flip_xy

df = pd.read_csv("../Data/headers3mgperml.csv", sep=",")
y = np.array(df.values[:, 9])
# y1 = y.copy()
# y2 = y.copy()
# y3 = y.copy()
# y_train = np.stack((y, y1), axis=0).reshape(96)
# print(y_train.shape)
# y_test = y3

indices = np.random.permutation(46)
training_idx, test_idx = indices[:36], indices[36:]
X_train, X_test = images[training_idx, :, :, :], images[test_idx, :, :, :]
y_train, y_test = y[training_idx], y[test_idx]

model = models.Sequential()
model.add(convolutional.Cropping2D(cropping=40, input_shape=(500, 640, 240), data_format='channels_last'))
model.add(convolutional.Convolution2D(filters=240, kernel_size=3, strides=1,
                                      activation='relu', padding='same'))
                                      # input_shape=(500, 640, 240), activation='relu'))
# model.add(SpatialDropout2D(.25))
model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
# model.add(convolutional.Convolution2D(filters=100, kernel_size=3, strides=3,
#                                       activation='relu'))
# model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
# model.add(convolutional.Convolution2D(filters=64, kernel_size=3, strides=3,
#                                       activation='relu'))
# model.add(SpatialDropout2D(.25))
# model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
model.add(convolutional.Convolution2D(filters=32, kernel_size=3, strides=3,
                                      activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
model.add(convolutional.Convolution2D(filters=16, kernel_size=3, strides=3,
                                      activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
model.add(core.Flatten())
model.add(core.Dense(500, activation='relu'))
model.add(core.Dropout(.5))
model.add(core.Dense(240, activation='relu'))
model.add(core.Dropout(.4))
model.add(core.Dense(100, activation='relu'))
model.add(core.Dropout(.25))
model.add(core.Dense(20, activation='relu'))
model.add(core.Dense(1))
model.compile(optimizer=optimizers.Adadelta(lr=1.0), loss='mean_squared_error')

# datagen = ImageDataGenerator(
#     rotation_range=20,
#     horizontal_flip=True,
#     vertical_flip=True,
#     data_format='channels_last'
# )
#
# datagen.fit(X_train)

# history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=46),
#                               sample_per_epoch=92,
#                               epochs=40)

history = model.fit(
    x=X_train,
    y=y_train,
    batch_size=6,
    epochs=40,
    # callbacks=[WeightsLogger(root_path=local_project_path)]
)

predictions = model.predict(
    x=X_test,
    batch_size=2,
)

print(mean_squared_error(y_test, predictions))

backend.clear_session()
