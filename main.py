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

    for i in range(0, 10):
        
        x = (random.random()-0.5)*20 
        y = (random.random()-0.5)*20 
        z = (random.random()-0.5)*20 
        
        sys.stdout.write("%d - Inputs: X= %f, Y=%f, Z=%f \t" % (i,x,y,z))

        toy.setInputs(np.array([x,y,z]))
        toy.evaluateForward()
        
        result = toy.getOutput(0)
        sys.stdout.write("Output: %f" % result)

        if(result != (x+y)*z):
            print("\t[FAIL]")
        else:
            print("\t[PASS]")

if(__name__=="__main__"):
    main()
