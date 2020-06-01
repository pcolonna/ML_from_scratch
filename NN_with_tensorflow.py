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

        # We will store here the weights and biaises
        self.W = {}
        self.b = {}

        # And here the derivate of the weights and biaises
        self.dW = {}
        self.db = {}

        # And finally, run our initialisation function
        self.initialize()


    def initialize(self):
        """ We will iterate for each layer and initialize with a random value from a normal distribution"""
        for i in range(1, self.L):
            self.W[i] = tf.Variable(tf.random.normal(shape=(self.layers[i], self.layers[i-1])))
            self.b[i] = tf.Variable(tf.random.normal(shape=(self.layers[i], 1)))