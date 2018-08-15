import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


df = pd.read_csv("../Data/headers3mgperml.csv", sep=",")
label_cols = df.values[:,1:5]
scaler = MinMaxScaler()


def cnn_model_fn(features, labels, mode, params):

    input_shape = features.shape

    pad_shape = (input_shape[0], 140, input_shape[2], input_shape[3])

    img_tensor = tf.convert_to_tensor(
        value=features,
        dtype=tf.int16,
        name="input_images"
    )

    paddings = tf.constant(np.zeros((pad_shape)))

    img_tensor = tf.pad(
        tensor=img_tensor,
        paddings=paddings,
        mode="CONSTANT",
        name="padded_images",
        constant_values=0,
        dtype=tf.int16
    )

    input_layer = tf.reshape(img_tensor["x"], [-1, 500, 640, 240])

    conv1 = tf.layers.conv3d(
        inputs=input_layer,
        filters=32,
        kernel_size=[8, 8, 240],
        padding="same",
        activation=tf.nn.relu
    )

    pool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=[4, 4, 240], strides=1)

    conv2 = tf.layers.conv3d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 8, 3],
        padding="same",
        activation=tf.nn.relu
    )

    pool2 = tf.layers.max_pooling3d(inputs=conv2, pool_size=2, strides=2)

    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN
    )

    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        "classes": tf.argmax(inputs=logits, axis=1),

        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"]
        )
    }

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops
    )
