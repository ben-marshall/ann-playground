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
        self.gradients      = np.ones((inputCells, outputCells))

        self.build()

    def build(self):
        """
        Overridable function that should construct the network.
        """
        print("[ERROR] Function not overriden. This network is empty.")
        pass

    def evaluateForward(self):
        """
        Update the output values of the network to refelect changes to the
        weights or inputs.
        """
        print("[ERROR] Function not overriden. This network is empty.")
        pass

    def evaluateBackward(self):
        """
        Computes the gradients for all cells and alters the weights associated
        with them ready for the next pass.
        """
        print("[ERROR] Function not overriden. This network is empty.")
        pass

    def getGradient(self, outputVal, inputVal):
        """
        Returns the gradient of the given output value with respect to the
        given input value. Both are indicies.
        """
        return self.gradients[inputVal][outputVal]

    def setInput(self, i, val):
        """
        Sets the value of an input to val.
        """
        self.inputValues[i] = val

    def setInputs(self,vals):
        """
        Sets the value of all inputs. Note vals must be an np.array type.
        """
        self.inputValues = vals

    def getOutput(self, i):
        """
        Returns the value of output i.
        """
        return self.outputValues[i]


class ToyNet(CustomNet):
    """
    Toy example network from the "Hackers guide to neural networks tutorial"
    http://karpathy.github.io/neuralnets/
    Inputs: X,Y,Z
    Output: F
    Q = X+Y
    F = Q*Z
    """

    def __init__(self):
        """
        Override the default constructor with fixed numbers of inputs
        and outputs. Three inputs, one output.
        """
        super().__init__(3, 1)

    def build(self):
        """
        Construct the same circuit as shown in the tutorial.
        """
        self.addGate = Gates.GateAdd(2)
        self.mulGate = Gates.GateMul(2)

    def evaluateForward(self):
        """
        Run the forward pass on the network.
        """

        self.addGate.setIn(0,self.inputValues[0])
        self.addGate.setIn(1,self.inputValues[1])
        self.addGate.computeForward()

        self.mulGate.setIn(0, self.addGate.getOutput())
        self.mulGate.setIn(1, self.inputValues[2])
        self.mulGate.computeForward()

        self.outputValues[0] = self.mulGate.getOutput()

    def evaluateBackward(self):
        """
        Evaluate the gradient of the output values with respect to
        each input value.
        """

        self.mulGate.computeBackward()
        self.addGate.computeBackward()

        df_wrt_z = self.addGate.getOutput()
        df_wrt_q = self.inputValues[2]

        dq_wrt_x = self.addGate.gradients[0]
        dq_wrt_y = self.addGate.gradients[1]
        
        # df_dx
        self.gradients[0][0] = df_wrt_q * dq_wrt_x
        self.gradients[1][0] = df_wrt_q * dq_wrt_y
        self.gradients[2][0] = df_wrt_z


