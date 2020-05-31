import Tensorflow
import numpy
import matplot.lib.pyplot as plt

import nn_utils

print('TensorFlow Version:', tf.__version__)


class NeuralNetwork:
    """
    Will contains all code necessary for a minimal NN.
    """

    def __init__(self, layers):
        self.layers = layers
        self.L = len(layers)    # Number of layers, "depth" of the network

        # We define the number of features, or input, and number of classes, or output.
        self.nb_features = layers[0]
        self.nb_classes = layers[-1]