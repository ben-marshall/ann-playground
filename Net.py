#!/usr/bin/python3

"""
Contains classes for creating a neural network.
"""

import math
import numpy as np

import Gates

class Net:
  """
  The top level net class which contains the nodes and the connections
  between them.
  """

  def __init__(self, inputCells, hiddenCells, outputCells):
    """
    Initialise the network with the specified number of inputs, hidden
    cells and output cells.
    """

    self.inputValues    = np.array([0.0] * inputCells)
    self.hiddenCells    = [Gates.Gate(inputCells)] * hiddenCells
    self.outputValues   = np.array([0.0] * outputCells)


class CustomNet:
    """
    A parent class for custom networks.
    """

    def __init__(self, inputCells, outputCells):
        """
        Initialise the class with the supplied number of input and output
        cells.
        """
        self.inputValues    = np.array([0.0] * inputCells)
        self.outputValues   = np.array([0.0] * outputCells)

        self.build()

    def build(self):
        """
        Overridable function that should construct the network.
        """
        pass

    def evaluateForward(self):
        """
        Update the output values of the network to refelect changes to the
        weights or inputs.
        """
        pass

    def evaluateBackward(self):
        """
        Computes the gradients for all cells and alters the weights associated
        with them ready for the next pass.
        """
        pass

    def setInput(self, i, val):
        """
        Sets the value of an input to val.
        """
        self.inputCells[i] = val

    def getOutput(self, i):
        """
        Returns the value of output i.
        """
        return self.outputCells[i]
