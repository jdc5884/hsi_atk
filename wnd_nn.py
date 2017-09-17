# author: David Ruddell
# contact: dr1236@uncw.edu, dlruddell@gmail.com

import sys
import shutil
import tempfile
import argparse
import pandas as pd
import tensorflow as tf

CSV_COLUMNS = [
    "Genotype", "Density", "Nitrogen", "Hormone", "Kernelwt", "Lipidwt", "wtpercent",
    "PALMETIC", "LINOLEIC", "OLEIC", "STEARIC"
]

hyper_data = pd.read_csv("~/PycharmProjects/HyperSpectralImaging/Data/headers3mgperml.csv", sep=',')
column_headers = hyper_data.columns.values.tolist()
wavelengths = column_headers[15:]

# Categorical base columns.
genotype = tf.feature_column.categorical_column_with_vocabulary_list("Genotype", ['B73', 'CML103'])
density = tf.feature_column.categorical_column_with_vocabulary_list("Density", ['NORMAL', 'HIGH'])
nitrogen = tf.feature_column.categorical_column_with_vocabulary_list("Nitrogen", ['NORMAL', 'LOW'])
hormone = tf.feature_column.categorical_column_with_vocabulary_list("Hormone", ['CONTROL','PAC','PACGA','UCN','PCZ','GA'])

# Continuous base columns
kernel_weight = tf.feature_column.categorical_column_with_hash_bucket("Kernelwt", hash_bucket_size=1000)
lipid_weight = tf.feature_column.categorical_column_with_hash_bucket("Lipidwt", hash_bucket_size=1000)
weight_ratio = tf.feature_column.categorical_column_with_hash_bucket("wtpercent", hash_bucket_size=1000)
palmetic = tf.feature_column.categorical_column_with_hash_bucket("PALMETIC", hash_bucket_size=1000)
linoleic = tf.feature_column.categorical_column_with_hash_bucket("LINOLEIC", hash_bucket_size=1000)
oleic = tf.feature_column.categorical_column_with_hash_bucket("OLEIC", hash_bucket_size=1000)
stearic = tf.feature_column.categorical_column_with_hash_bucket("STEARIC", hash_bucket_size=1000)

# Wavelength signal data
ind_121 = tf.contrib.layers.sparse_column_with_hash_bucket(wavelengths[121-14], hash_bucket_size=1000)
ind_122 = tf.contrib.layers.sparse_column_with_hash_bucket(wavelengths[122-14], hash_bucket_size=1000)
ind_144 = tf.contrib.layers.sparse_column_with_hash_bucket(wavelengths[144-14], hash_bucket_size=1000)
ind_145 = tf.contrib.layers.sparse_column_with_hash_bucket(wavelengths[145-14], hash_bucket_size=1000)
ind_185 = tf.contrib.layers.sparse_column_with_hash_bucket(wavelengths[185-14], hash_bucket_size=1000)

# wide_columns = []
#
# for column in wavelengths:
#     wide_columns.append(tf.)
wide_columns = [
    ind_121, ind_122, ind_144, ind_145, ind_185, genotype, density, nitrogen, hormone,
    tf.contrib.layers.crossed_column([genotype, ind_121, ind_122, ind_144, ind_145,
                                      ind_185], hash_bucket_size=int(1e4)),
    tf.contrib.layers.crossed_column([density, ind_121, ind_122, ind_144, ind_145,
                                      ind_185], hash_bucket_size=int(1e4)),
    tf.contrib.layers.crossed_column([nitrogen, ind_121, ind_122, ind_144, ind_145,
                                      ind_185], hash_bucket_size=int(1e4)),
    tf.contrib.layers.crossed_column([hormone, ind_121, ind_122, ind_144, ind_145,
                                      ind_185], hash_bucket_size=int(1e4)),
]

# cross_columns = [
#     tf.feature_column.crossed_column([""])
# ]

deep_columns = [
    tf.contrib.layers.embedding_column(genotype, dimension=8),
    tf.contrib.layers.embedding_column(density, dimension=8),
    tf.contrib.layers.embedding_column(nitrogen, dimension=8),
    tf.contrib.layers.embedding_column(hormone, dimension=8),
    kernel_weight, lipid_weight, weight_ratio, palmetic, linoleic, oleic, stearic,
    ind_121, ind_122, ind_144, ind_145, ind_185

]

model_dir = tempfile.mkdtemp()
m = tf.contrib.learn.DNNLinearCombinedClassifier(
    model_dir=model_dir,
    linear_feature_columns=wide_columns,
    dnn_feature_columns=deep_columns,
    dnn_hidden_units=[100,50]
)

COLUMNS = column_headers[1:8]+column_headers[9:14]+column_headers[15:]
LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS = column_headers[1:5]
CONTINUOUS_COLUMNS = column_headers[5:8]+column_headers[9:14]+column_headers[15:]

train_file = tempfile.NamedTemporaryFile()
test_file = tempfile.NamedTemporaryFile()

df_train = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True)
df_test = pd.read_csv(test_file, names=COLUMNS, skipinitialspace=True, skiprows=1)
df_train[LABEL_COLUMN] = df_train['STEARIC']
df_test[LABEL_COLUMN] = df_test['STEARIC']

def input_fn(df, n_epochs, shuffle):
    continuous_cols = {k: tf.constant(df[k].values)
                       for k in CONTINUOUS_COLUMNS}

    categorical_cols = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(df[k].size)],
        values=df[k].values,
        dense_shape=[df[k].size, 1]
    )                   for k in CATEGORICAL_COLUMNS}

    feature_cols = dict(continuous_cols.item() + categorical_cols.item())
    label = tf.constant(df[LABEL_COLUMN].values)
    return feature_cols, label

def train_input_fn():
    return input_fn(df_train)

def eval_input_fn():
    return input_fn(df_test)

m.fit(input_fn=train_input_fn(), steps=200)
results = m.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
    print("%s: %s" % (key, results[key]))