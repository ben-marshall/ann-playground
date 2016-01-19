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

    class NetLayer:
        """
        Fully describes a single layer of a neural network.
        """

        def __init__(self, nodeCount, inputCount):
            """
            Instances a layer with the given number of nodes, each with the
            given number of inputs.
            """
            self.nodeCount      = nodeCount
            self.inputCount     = inputCount

            self.nodeWeights    = np.ones((nodeCount,inputCount))
            self.nodeOutputs    = np.zeros(nodeCount)
            self.nodeGradients  = np.ones((nodeCount, inputCount))

        def updateWeights(self, multiplier, stepsize = 0.01):
            """
            Updates all of the weights in the direction specified by multiplier
            with the given stepsize.
            """
            mod = np.array([multiplier*stepsize] * self.nodeCount)
            for i in range(0, self.nodeCount):
                self.nodeWeights[i] = self.nodeWeights[i] + mod*self.nodeGradients[i]

        def outputs(self):
            """
            Return the output values of each node.
            """
            return self.nodeOutputs

        def forward(self, inputs):
            """
            Evaluates the outputs of all nodes in the layer given the supplied
            input values. "inputs" should be a numpy array the same length as
            the number of inputs that the layer expects.
            """
            for i in range(0, self.nodeCount):
                self.nodeOutputs[i] = self.__computeForward__(self.nodeWeights[i], inputs)

        def backward(self):
            """
            Compute the gradients of the output for every node with respect
            to each of their inputs.
            """
            for i in range(0, self.nodeCount):
                self.nodeGradients[i] = self.__computeBackward__(i)

        def __computeForward__(self, weights, inputs):
            """
            Easy to override function that computes the forward value for a
            single node given its weights and inputs. Assume that
            len(weights) == len(inputs)
            By default, computes the weighted product of all inputs.
            """
            return np.prod(weights * inputs)

        def __computeBackward__(self, node):
            """
            Easy to override function that computes the backward gradient for the
            given node i and returns it.
            """
            return 1


    def __init__(self, inputCount, layers = [5]):
        """
        Creates a new network with the supplied number of inputs and outputs.
        The layers argument describes how many nodes are contained within
        each hidden layer. By default, there is one hidden layer with five
        nodes. layers=[10,5] would create two hidden layers, the first with
        ten nodes and the second with five nodes.
        """
        self.inputCount         = inputCount
        self.inputValues        = np.array([0.0] * self.inputCount)

        self.outputCount        = layers[-1]

        self.hiddenLayerCount   = len(layers)
        self.hiddenLayers       = [None] * self.hiddenLayerCount

        for i in range(0,self.hiddenLayerCount):
            nodeCount   = layers[i]
            inputCount  = self.inputCount if i == 0 else layers[i-1]
            self.hiddenLayers[i] = self.NetLayer(nodeCount, inputCount)

    def getOutput(self):
        """
        Returns the output values of the final layer of the network.
        """
        return self.hiddenLayers[-1].outputs()

    def forward(self, inputs):
        """
        Perform a forward pass on the network using the supplied input values.
        """
        self.inputValues = inputs
        
        for l in range(0, self.hiddenLayerCount):
            if(l == 0):
                self.hiddenLayers[l].forward(self.inputValues)
            else:
                self.hiddenLayers[l].forward(self.hiddenLayers[l-1].outputs())
        
    def backward(self):
        """
        Perform a backward pass on the network.
        """
        for l in range(0, self.hiddenLayerCount):
            if(l == 0):
                self.hiddenLayers[l].backward()
            else:
                self.hiddenLayers[l].backward()


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

        self.df_wrt_q = self.inputValues[2]
        self.df_wrt_z = self.addGate.getOutput()

        self.dq_wrt_x = self.addGate.gradients[0]
        self.dq_wrt_y = self.addGate.gradients[1]
        
        self.gradients[0][0] = self.df_wrt_q * self.dq_wrt_x
        self.gradients[1][0] = self.df_wrt_q * self.dq_wrt_y
        self.gradients[2][0] = self.df_wrt_z

    def updateWeights(self, multiplier = 1):
        """
        Toy network function that updates the weights based on the
        computed gradients of each function.
        """
        
        self.mulGate.updateWeight(0, multiplier * self.df_wrt_q)
        self.mulGate.updateWeight(1, multiplier * self.df_wrt_z)
        
        self.addGate.updateWeight(0, multiplier * self.gradients[0][0])
        self.addGate.updateWeight(1, multiplier * self.gradients[1][0])

