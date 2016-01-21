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
        self.activations[0] = np.matrix([0.0] * layerSizes[0])
        self.Z              = [None] * self.numLayers

        for L in range(1, self.numLayers):
            # Initialise each element of the weights, biases and activations
            # lists with the appropriate values.
            
            self.weights[L]     = np.matrix([[0.0] * layerSizes[L-1]]*layerSizes[L])
            self.biases[L]      = np.matrix([0.0] * layerSizes[L])
            self.activations[L] = np.matrix([0.0] * layerSizes[L])

    def forward(self):
        """
        Computes a forward pass on the network.
        """

        for L in range(1, self.numLayers):
            self.Z[L] = np.dot(self.activations[L-1],self.weights[L].transpose())
            self.Z[L] = self.Z[L]  + self.biases[L]
            self.activations[L] = self.sigmoid(self.Z[L])


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



if(__name__=="__main__"):
    main()

