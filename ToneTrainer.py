#!/usr/bin/python3

"""
Contains classes for training a neural network to differentiate between
two frequencies.
"""

import sys
import math
import random

import numpy as np

from tqdm import tqdm

import Net as ann

class ToneTrainer:
    """
    Top level training class.
    """

    def __init__(self, f1 = 100, f2 = 5):
        """
        Initialises the class with the supplied arguments.
        """

        self.f1         = float(f1)
        self.f2         = float(f2)
        self.L1         = 1.0
        self.L2         = 0.0
        self.sampleRate = 2.0 * max(f1,f2)
        self.numSamples = int(self.sampleRate)

        self.layerSizes = [self.numSamples+5] * 7 + [2]
        self.net        = ann.Net(layerSizes = self.layerSizes)
        
        # A list of tuples containing the input data, and expected outputs.
        self.trainingSet = []

    def genTrainingData(self, numElements = 10):
        """
        Generates training data
        """
        pbar = tqdm(range(0, numElements))
        pbar.set_description("Generating Training Data")
        self.trainingSet = []

        for i in pbar:
            # randomly pick a tone and label.
            tone,target= random.choice([(self.f1,[self.L1, self.L2]),
                                        (self.f2,[self.L2, self.L1])])

            samples = np.linspace(0, 2*np.pi, self.numSamples) * tone
            values  = (np.sin(samples)/2.0) + 0.5

            self.trainingSet.append((values,target,tone))


    def train(self, itterations = 1000):
        """
        Runs the training algorithm
        """

        pbar = tqdm(range(0, itterations))
        pbar.set_description("Running Training")
        score = 0

        for i in pbar:

            error = 0
            score = 0

            for j in range(0, len(self.trainingSet)):
                targetValues    = self.trainingSet[j][1]
                inputValues     = self.trainingSet[j][0]
                self.net.setInputs(inputValues)

                self.net.forward()
                result = self.net.getOutput()
                err = self.net.backward(np.matrix(targetValues).transpose())
                error += err
                self.net.modWeights(stepSize = -0.01)
                
            pbar.set_description("Score: %d " % (score*100.0/len(self.trainingSet)))

        print("Final Score: %d/%d" % (score, len(self.trainingSet)))


def main():
    """
    Main function for the script.
    """

    program = ToneTrainer()
    print("Samples per tone: %d" % program.numSamples)
    program.genTrainingData()
    print("Training...")
    program.train()
    print("Training Complete.")

if(__name__=="__main__"):
    main()
else:
    print("Imported: %s" % __name__)


