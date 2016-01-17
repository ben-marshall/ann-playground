#!/usr/bin/python3

"""
File containing the library of simple gates that will make up the NNs.
"""

import math

class Gate:
  """
  Basic gate class that acts as a simple forwarder with N inputs and
  one output. It will only ever forward the first input to the output.
  """

  def __init__(self, inputs):
    """
    Creates a new gate instance with the supplied inputs.
    """
    self.numInputs  = inputs
    self.inputs     = [0] * self.numInputs
    self.weights    = [1] * self.numInputs
    self.gradients  = [0] * self.numInputs
    self.output     =  0

  def setWeight(self, i, val):
    """
    Sets the weighting of input i
    """
    self.weights[i] = val

  def setIn(self, i, val):
    """
    Set the value of an input.
    """
    self.inputs[i] = val

  def computeForward(self):
    """
    Given the current inputs and weights, compute the output of the
    function. Does not return it, simply sets the internal field.
    """
    self.output = self.weights[0] * self.inputs[0]

  def computeBackward(self):
    """
    Computes the gradient of the backward pass. For this gate, that is
    simply the weighting of the first input.
    """
    for i in range(0, self.numInputs):
      self.gradients[i] = self.weights[i]

  def getOutput(self):
    """
    Returns the output value of the gate. For this class, it is
    simply the value of the first input multiplied by its associated
    weight.
    """
    return self.output


