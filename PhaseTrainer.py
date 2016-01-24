#!/usr/bin/python3

"""
Contains a small program for training a neural net to recognise if one
signal is ahead of another in terms of phase.
"""

import sys
import math
import random

import numpy as np

from tqdm import tqdm

import Net as ann

class PhaseTrainer:
    """
    Top level class for identifying phase difference.
    """

    def __init__(self, maxFreq=100):
        """
        Initialises the class with the supplied arguments.
        - maxFreq - maximum frequency to detect in hertz. Defines the
                    sample rate of training signals.
        """

        self.maxFreq    = maxFreq
        self.sampleLen  = 2 * self.maxFreq

        self.layerSizes = []
        size = self.sampleLen

        while(size > 1):
            self.layerSizes.append(size)
            size = int(size / 2)
        
        self.net = ann.Net(layerSizes = self.layerSizes)


    def train(self, itterations = 100):
        """
        Runs the training program.
        """
        
        for i in tqdm(range(0, itterations)):
            pass


def main():
    """
    Main function for the script.
    """

    program = PhaseTrainer()
    print("Training...")
    program.train()
    print("Training Complete.")

if(__name__=="__main__"):
    main()
else:
    print("Imported: %s" % __name__)

