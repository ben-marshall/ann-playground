#!/usr/bin/python3

"""
Main script for playing around with the networks I build.
"""

import sys
import random

import numpy as np

import Net
import Gates

def toy():
    """
    Toy network demonstration function.
    """

    toy = Net.ToyNet()

    x = -2
    y =  5
    z = -4
    step_size = 0.01

    for i in range(0, 10):
        
        sys.stdout.write("%d -   Inputs: X=%f,Y=%f,Z=%f \t" % (i,x,y,z))

        toy.setInputs(np.array([x,y,z]))
        toy.evaluateForward()
        
        result = toy.getOutput(0)
        sys.stdout.write("Output: %f \t" % result)

        toy.evaluateBackward()

        dx,dy,dz = [toy.getGradient(0,0),toy.getGradient(0,1),toy.getGradient(0,2)]
        sys.stdout.write("Gradients: dX=%f,dY=%f,dZ=%f\n" % (dx,dy,dz))

        toy.updateWeights()
        
        #x = x + step_size * dx
        #y = y + step_size * dy
        #z = z + step_size * dz

def main():
    """
    Main function for the script.
    """
    
    toy = Net.Net(3, layers=[2,1])
    inputs = [1.0,2.0,3.0]
    toy.forward(inputs)
    result = toy.getOutput()
    sys.stdout.write("Inputs: %s, output = %s " % (str(inputs), str(result)))
    print("")


if(__name__=="__main__"):
    main()
