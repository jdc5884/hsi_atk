__author__ = "David Ruddell"
# contact: dr1236@uncw.edu, dlruddell@gmail.com

#TODO: Create way to set X-data (wavelength signals) to vars

import urllib
import tempfile
import pandas as pd
import tensorflow as tf

hyper_data = pd.read_csv("headers3mgperml.csv", sep=',')
column_headers = hyper_data.columns.values.tolist()

# Categorical base columns.
genotype = tf.contrib.layers.sparse_column_with_keys(column_name="Genotype", keys=['B73', 'CML103'])
density = tf.contrib.layers.sparse_column_with_keys(column_name="Density", keys=['NORMAL', 'HIGH'])
nitrogen = tf.contrib.layers.sparse_column_with_keys(column_name="Nitrogen", keys=['NORMAL', 'LOW'])
hormone = tf.contrib.layers.sparse_column_with_keys(column_name="Hormone",
                                                    keys=['CONTROL','PAC','PACGA','UCN','PCZ','GA'])

# Continuous base columns
kernel_weight = tf.contrib.layers.sparse_column_with_hash_bucket("Kernelwt", hash_bucket_size=100)
lipid_weight = tf.contrib.layers.sparse_column_with_hash_bucket("Lipidwt", hash_bucket_size=100)
weight_ratio = tf.contrib.layers.sparse_column_with_hash_bucket("wtpercent", hash_bucket_size=100)
palmetic = tf.contrib.layers.sparse_column_with_hash_bucket("PALMETIC", hash_bucket_size=1000)
linoleic = tf.contrib.layers.sparse_column_with_hash_bucket("LINOLEIC", hash_bucket_size=1000)
oleic = tf.contrib.layers.sparse_column_with_hash_bucket("OLEIC", hash_bucket_size=1000)
stearic = tf.contrib.layers.sparse_column_with_hash_bucket("STEARIC", hash_bucket_size=1000)

#wavelengths = tf.contrib.layers.






model_dir = tempfile.mkdtemp()
# m = tf.contrib.learn.DNNLinearCombinedClassifier(
#     model_dir=model_dir,
#     linear_feature_columns=wide_columns,
#     dnn_feature_columns=deep_columns,
#     dnn_hidden_units=[100,50]
# )

COLUMNS = column_headers[1:8]+column_headers[9:14]+column_headers[15:]
LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS = column_headers[1:5]
CONTINUOUS_COLUMNS = column_headers[5:8]+column_headers[9:14]+column_headers[15:]

train_file = tempfile.NamedTemporaryFile()
test_file = tempfile.NamedTemporaryFile()
