#!/usr/bin/python3

"""
Contains classes for creating a neural network.
"""

import sys
import math
import random

import numpy as np

class Net:
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

        self.L          = len(layerSizes)   # The number of layers in the net
        self.W          = [None] * self.L   # The weights of each layer
        self.B          = [None] * self.L   # The biases of each layer
        self.A          = [None] * self.L   # The activation values of nodes.
        self.Z          = [None] * self.L   # Weighted input to neurons.
        self.dW         = [None] * self.L
        self.dB         = [None] * self.L
        self.dL         = [None] * self.L

        for layer in range(0, self.L):
            rows          = layerSizes[layer]
            cols          = layerSizes[layer-1] if layer > 0 else rows

            # layer size * previous layer size
            self.W[layer] = np.matrix([[0.0]*cols]*rows)
            # layer size * 1
            self.B[layer] = np.matrix([[0.0]]*rows)
            # layer size * 1
            self.A[layer] = np.matrix([[0.0]]*rows)
            self.Z[layer] = np.matrix([[0.0]]*rows)
            self.dB[layer] = np.matrix([[0.0]]*rows)
            self.dW[layer] = np.matrix([[0.0]]*rows)
            self.dL[layer] = np.matrix([[0.0]]*rows)

    def setInputs(self, values):
        """
        Sets all of the inputs to the supplied values. Values should be
        an array like set.
        """
        for i in range(0, len(values)):
            self.A[0][i] = values[i]


    def setInput(self, I, val):
        """
        Sets the value of the Ith input to the net.
        """
        self.A[0][I] = val


    def getOutput(self):
        """
        Returns the output values of the network
        """
        return self.A[self.L-1]


    def forward(self):
        """
        Perform a forward pass on the network.
        """
        for l in range(1, self.L):
            # iterate forwards over the layers computing each activation.
            # Since layer 0 is the input layer, start computing at layer 1
            self.Z[l]   = self.W[l] * self.A[l-1] + self.B[l]
            self.A[l]   = sigmoid(self.Z[l])
       # The output is now in self.A[self.L - 1]


    def backward(self, expected):
        """
        Perform a backward pass on the network.
        """
        # First compute the error in the output at the last layer.
        err = self.A[self.L-1] - expected
        self.dL[self.L-1] = (err) * dSigmoid(self.Z[self.L-1])
        self.dB[self.L-1] = self.dL[self.L-1]
        self.dW[self.L-1] = self.dB[self.L-1] * self.A[self.L-2].transpose()

        for l in range(self.L-2, 0, -1):
            ds = dSigmoid(self.Z[l])
            self.dL[l] = (self.W[l+1].transpose() * self.dL[l+1])
            self.dB[l] = np.multiply(self.dL[l], ds)
            self.dW[l] = self.dB[l] * self.A[l-1].transpose()

    def modWeights(self, stepSize = -0.01):
        """
        Modifies the weights and biases according to the given gradients and
        step sizes.
        """
        for l in range(0, self.L):
            for N in range(0, len(self.W[l])):
                self.W[l][N] = self.W[l][N] + (self.dW[l][N] * stepSize)
                self.B[l][N] = self.B[l][N] + (self.dB[l][N] * stepSize)
        

def dSigmoid( x):
    a = 2.0-sigmoid(x)
    b = sigmoid(x)
    return b.transpose()*a


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))


def main():
    """
    Main function for the script.
    """
 
    print("----------------Start---------------------------")
    layerSizes=[3,3,3,1]
    toy = Net(layerSizes)
    X = 0.5
    Y = 0.8
    Z = -0.3
    toy.setInput(0, X)
    toy.setInput(1, Y)
    toy.setInput(2, Z)

    tgt = sigmoid(X*Y-Z)

    for i in range(0,10000):
        
        toy.forward()
        toy.backward(np.matrix([tgt]))
        toy.modWeights(stepSize=-0.01)

        result = toy.A[toy.L-1][0]
    
    print("%f %f %f" % (result, tgt, abs((result - tgt))))

    print("----------------End-----------------------------")

if(__name__=="__main__"):
    main()
else:
    print("Imported: %s" % __name__)

