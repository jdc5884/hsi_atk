from keras import models, optimizers
from keras.layers import core, convolutional
from keras.constraints import min_max_norm



def build_cnn_cont(n_out=3):
    model = models.Sequential()

    model.add(convolutional.SeparableConv2D(filters=16, kernel_size=7, strides=5, activation='relu'))
    model.add(convolutional.SeparableConv2D(filters=32, kernel_size=7, strides=3, activation='relu'))
    model.add(convolutional.MaxPooling2D(pool_size=2))
    model.add(convolutional.SeparableConv2D(filters=16, kernel_size=2, strides=2, activation='relu'))
    model.add(convolutional.MaxPooling2D(pool_size=2))
    # model.add(core.Flatten())
    # model.add(core.Dense(256, activation='relu')
    # model.add(core.Dropout(.5)
    # model.add(core.Dense(128, activation='relu')
    # model.add(core.Dropout(.25)
    # model.add(core.Dense(64, activation='relu')
    # model.add(core.Dense(n_out, activation='relu') ** ( activation='sigmoid' if categorical)

    # model.add(convolutional.SeparableConv2D(filters=32, kernel_size=1, strides=1,
    #                                         activation='relu', depth_multiplier=sep_mult))
    # model.add(convolutional.ZeroPadding2D(padding=7))
    # model.add(convolutional.SeparableConv2D(filters=32, kernel_size=9, strides=1,
    #                                         activation='relu', depth_multiplier=2))
    # model.add(convolutional.MaxPooling2D(pool_size=2))
    # model.add(convolutional.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'))
    # model.add(convolutional.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'))
    # model.add(convolutional.MaxPooling2D(pool_size=2))
    # model.add(convolutional.Conv2D(filters=128, kernel_size=3, strides=1, activation='relu'))
    # model.add(convolutional.Conv2D(filters=128, kernel_size=3, strides=1, activation='relu'))
    # model.add(convolutional.MaxPooling2D(pool_size=2))

    model.add(core.Flatten())
    model.add(core.Dense(256, activation='relu'))
    model.add(core.Dropout(.5))
    model.add(core.Dense(128, activation='relu'))
    model.add(core.Dropout(.5))
    model.add(core.Dense(64, activation='relu'))
    model.add(core.Dense(n_out, activation='relu'))
    model.compile(optimizer=optimizers.Adam(lr=1.0), loss='mean_squared_error')
    # if continuous:
    #     model.add(core.Dense(n_out))
    #
    #     model.compile(optimizer=optimizers.Adam(lr=1.0), loss='mean_squared_error')
    #
    # else:
    #     model.add(core.Dense(32, activation='softmax'))
    #     model.add(core.Dense(n_out, activation='softmax', kernel_constraint=min_max_norm(min_value=1.0, max_value=2.0, rate=1.0, axis=0)))
    #
    #     model.compile(optimizer=optimizers.Adam(lr=1.0), loss='binary_crossentropy', metrics=['accuracy'])

    return model
