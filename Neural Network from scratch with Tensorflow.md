# Neural Network from scratch with Tensorflow

I suppose you are somewhat familiar with machine learning.

## What wil we do in this project?

Write the neural network code, some EDA on MNIST, some data visualization. 

## Code structure

## Let's roll!

First let's import some package

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import nn_utils
```

I import Bokeh to later generate pretty interactive graphs. It's not technically necessary to build, train and run your model but it's always useful and nice to see what's going on. Why bokeh? Beautifull, interactive graphs. Great for the Web. Part of an amazing ecosystem for data viz. (Those are the reasons i chose it at my job)

Seaborn is also a great package to generate beautiful statics libs on top of matplotlib

In utils you will find non essential tools for the project, like plotting or loading data.

Now, let's build our NeuralNetworkClass starting with the \__init__ function.

## \__init__

```python
class NeuralNetwork:
    
    def __init__(self, layers):
        
        self.layers = layers
        self.L = len(layers)
```

We will use our class by passing a  list, named layers, where each integer is the number of hidden units in the layers.

For instance,  
$$
NeuralNetwork([8, 16, 16, 2])
$$
will be a neural network with 4 layers, 8 features or inputs, two hidden layers with 16 hidden units and an output layer with 10 classes.

![g884](/home/pcdi/Documents/ML_from_scratch/g884.png)

Then we define the number of features and classes

```python
        self.nb_features = layers[0]
        self.nb_classes = layers[-1]
```

those being equals to the first and last value of your array.

You will also need weights, biases and their derivate for backpropagation later.    
```python
    	self.W = {}
    	self.b = {}

    	self.dW = {}
    	self.db = {}
```


Finally, we initialize W and b with random values from a normal distribution. Turns out there is quite a bit of theory and research behind why it is best to do so, instead of say, start with zeros or ones.  



```python
def initialize(self):
	for i in range(1, self.L):
    	self.W[i] = tf.Variable(tf.random.normal(shape=(self.layers[i], self.layers[i-1])))
        self.b[i] = tf.Variable(tf.random.normal(shape=(self.layers[i], 1)))
```



tf.Variable is the preferred way to declare a ariable in TensorFlow.  

Quick note concerning the shape of tf.random.normal. As you can see, it corresponds to the number of node in the current layer by the number of nodes in the previous layer, but why? 