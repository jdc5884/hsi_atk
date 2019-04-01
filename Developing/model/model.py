__author__ = "David Ruddell"
__credits__ = ["David Ruddell"]
__license__ = "GPL"
__version__ = "0.0.1"
__status__ = "Development"


import tensorflow as tf


class Model(tf.keras.Model):

    def __init__(self):
        super(Model, self).__init__()

        self.build_model()

    def build_model(self):
        pass