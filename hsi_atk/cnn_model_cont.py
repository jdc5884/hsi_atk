import h5py
import numpy as np
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from keras import backend
from keras.preprocessing.image import ImageDataGenerator


local_project_path = '/'
local_data_path = os.path.join(local_project_path, 'data')

images_file = h5py.File("../Data/img_all_4d.h5", "r")
images = images_file['datatset'][:]
images_file.close()
images = images.swapaxes(0,2).swapaxes(1,2).astype(np.int16)
images = images[:, :, 70:570, :]
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
# label_cols = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12]  # all labels
# label_cols = [5, 6, 7, 9, 10, 11, 12]  # continuous labels
label_cols = [5, 6, 7]
# label_cols = [1]
# label_cols = [9]
# le = LabelEncoder()
y = df.values[:, label_cols]
# y = le.fit_transform(np.array(df.values[:, 1]).ravel())
# y += 1
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


from hsi_atk.model_builder import build_cnn_cont

model = build_cnn_cont(n_out=len(label_cols))

# datagen = ImageDataGenerator(
#     horizontal_flip=True,
#     vertical_flip=True,
#     data_format='channels_last'
# )

# datagen.fit(X_train)
#
# history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=36),
#                               epochs=6)
# from hsi_atk.ImageGen import image_generator
#
# history = model.fit_generator(image_generator(images, "../Data/headers3mgperml.csv", training_idx, label_cols),
#                               epochs=6)

history = model.fit(
    x=X_train,
    y=y_train,
    batch_size=36,
    epochs=10,
    # callbacks=[WeightsLogger(root_path=local_project_path)]
)

predictions = model.predict(
    x=X_test,
    batch_size=10,
)

print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
# print(mean_squared_error(y_test[0], predictions[0]))
# print(mean_squared_error(y_test[1], predictions[1]))
# print(mean_squared_error(y_test[2], predictions[2]))
# print(mean_squared_error(y_test[3], predictions[3]))
# print(mean_squared_error(y_test[4], predictions[4]))
# print(mean_squared_error(y_test[5], predictions[5]))
# print(mean_squared_error(y_test[6], predictions[6]))

backend.clear_session()
