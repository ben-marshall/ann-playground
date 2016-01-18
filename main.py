#!/usr/bin/python3

"""
Main script for playing around with the networks I build.
"""

import sys
import random

import numpy as np

import Net
import Gates

def main():
    """
    Main function for the script.
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
        sys.stdout.write("Gradients: dX=%f,dY=%f,dZ=%f" % (dx,dy,dz))

        if(result != (x+y)*z):
            print("\t[FAIL]")
        else:
            print("\t[PASS]")
        
        x = x + step_size * dx
        y = y + step_size * dy
        z = z + step_size * dz

if(__name__=="__main__"):
    main()
