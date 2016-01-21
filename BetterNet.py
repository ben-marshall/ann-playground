#!/usr/bin/python3

"""
Contains classes for creating a neural network.
"""

import sys
import math
import numpy as np

import Gates

class BetterNet:
    """
    The top level net class which contains the nodes and the connections
    between them.
    """

    def __init__(self, layerSizes = [3,4,2]):
        """
        Initialises the network and allocates memory space for all weights and
        biases. The layerSizes input is a list of integers, with the i'th
        representing the number of nodes in layer i. The zeroth element is
        the input layer, while the N-1'th layer is the output layer.
        Assumes that there are atleast two layers.
        """

        self.numLayers      = len(layerSizes)
        self.weights        = [None] * self.numLayers
        self.biases         = [None] * self.numLayers
        self.activations    = [None] * self.numLayers

        # Input layer
        self.activations[0] = np.matrix([0.0] * layerSizes[0])
        self.biases[0]      = np.matrix([0.0] * layerSizes[0])
        self.weights[0]     = np.matrix([[1.0] * layerSizes[0]]*layerSizes[0])

        # Weighted input to every neuron, prior to being "sigmoided".
        self.Z              = [None] * self.numLayers

        for L in range(1, self.numLayers):
            # Initialise each element of the weights, biases and activations
            # lists with the appropriate values.
            
            self.weights[L]     = np.matrix([[0.0] * layerSizes[L-1]]*layerSizes[L])
            self.biases[L]      = np.matrix([0.0] * layerSizes[L])
            self.activations[L] = np.matrix([0.0] * layerSizes[L])

    def backward(self, desiredOutputs):
        """
        Perform a backward pass on the network and return a tuple of
        gradients with respect to all biases and weights.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases ]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        delta = (self.activations[-1] - desiredOutputs) * self.dSigmoid(self.Z[-1])

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta.transpose(), self.activations[-2])

        for l in range(2, self.numLayers):
            print(">>> %d" % l)
            z   = self.Z[-l]
            sp  = self.dSigmoid(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, self.activations[-l-1].transpose())
        return (nabla_b, nabla_w)


    def forward(self):
        """
        Computes a forward pass on the network.
        """

        for L in range(1, self.numLayers):
            self.Z[L] = np.dot(self.activations[L-1],self.weights[L].transpose())
            self.Z[L] = self.Z[L]  + self.biases[L]
            self.activations[L] = self.sigmoid(self.Z[L])


    def dSigmoid(self, x):
        a = self.sigmoid(x).transpose()
        b = 1- self.sigmoid(x)
        print()
        print(a)
        print(b)
        return a*b

    def sigmoid(self, x):
        return 1.0/(1.0+np.exp(-x))

def main():
    """
    Main function for the script.
    """
    
    toy = BetterNet(layerSizes=[3,4,2])
    print("Network Size: %d Bytes" % sys.getsizeof(toy))

    print("Weights:")
    print(toy.weights)
    print("Biases:")
    print(toy.biases)
    print("Activations:")
    print(toy.activations)

    toy.forward()
    print("-------------------")

    print("Weights:")
    print(toy.weights)
    print("Biases:")
    print(toy.biases)
    print("Activations:")
    print(toy.activations)

    toy.backward(np.array([-1,4]))
    print("-------------------")

    print("Weights:")
    print(toy.weights)
    print("Biases:")
    print(toy.biases)
    print("Activations:")
    print(toy.activations)



if(__name__=="__main__"):
    main()

