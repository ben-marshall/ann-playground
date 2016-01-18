#!/usr/bin/python3

"""
File containing the library of simple gates that will make up the NNs.
"""

import math
import numpy as np

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
    self.inputs     = np.array([0.0] * self.numInputs)
    self.weights    = np.array([1.0] * self.numInputs)
    self.gradients  = np.array([0.0] * self.numInputs)
    self.output     =  0

  def setWeight(self, i, val):
    """
    Sets the weighting of input i
    """
    self.weights[i] = val

  def updateWeight(self, i, gradient, step = 0.01):
    """
    Updates an input weighting based on the supplied gradient and step size.
    """
    self.weights[i] = self.weights[i] + gradient*step

  def setIn(self, i, val):
    """
    Set the value of an input.
    """
    self.inputs[i] = val

  def getWeightedInput(self, i):
    """
    Returns the weighted value of the input.
    """
    return self.inputs[i] * self.weights[i]

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


class GateAdd (Gate):
  """
  A simple addition gate that sums all of it's inputs.
  """

  def computeForward(self):
    """
    Computes the forward pass of the gate by summing the products
    of every input and it's associated weight. Sets the internal
    field output to the computed value.
    """
    self.output = sum(self.inputs * self.weights)

  def computeBackward(self):
    """
    Computes the gradient of the output with respect to all inputs.
    For the add gate, this is just the weight assigned to each input.
    """
    for i in range(0, self.numInputs):
        self.gradients[i] = self.weights[i]

class GateMul (Gate):
  """
  A simple multiplication gate that returns the weighted dot product
  of it's inputs.
  """

  def computeForward(self):
    """
    Computes the forward pass of the gate by working out the
    dot product of all inputs multiplied by their respective weights.
    """
    self.output = np.prod(self.inputs * self.weights)

  def computeBackward(self):
    """
    Computes the gradient of the output with respect to all inputs.
    For the add gate, this is just the weight assigned to each input.
    """
    for i in range(0, self.numInputs):
      self.gradients[i] = 1
      for j in range(0, self.numInputs):
        self.gradients[i] *= self.inputs[j] * self.weights[i]
