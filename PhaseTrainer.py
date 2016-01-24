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

    def __init__(self, maxFreq=10):
        """
        Initialises the class with the supplied arguments.
        - maxFreq - maximum frequency to detect in hertz. Defines the
                    sample rate of training signals.
        """

        self.maxFreq    = maxFreq
        self.sampleLen  = int(2*np.pi / 2*self.maxFreq)

        size = self.sampleLen*1
        self.layerSizes = [size] * 3

        while(size > 1):
            self.layerSizes.append(size)
            size = int(size * 0.9)
        self.layerSizes.append(1)
        
        self.net = ann.Net(layerSizes = self.layerSizes)

    def genSample(self, phase=0.0):
        """
        Generates a signal sample with the
        supplied phase modification. The amplitude of the signal will be
        between 0 and 1.
        """
        x = np.linspace(0, 2*np.pi, num = self.sampleLen)
        x = x + phase
        return (np.sin(x)/2.0) + 0.5

    def train(self, itterations = 1000):
        """
        Runs the training program.
        """

        baseSignal = self.genSample(phase = 0)
        
        print("Base Signal Length: %d" % len(baseSignal))
        print("Layer Sizes       : %s" % str(self.layerSizes))

        correct    = 0
        score      = 0
        
        pbar = tqdm(range(0, itterations))
        for i in pbar:
            
            # Generate a new signal with a random phase difference between
            # 0 and 2*pi
            phaseDiff       = -np.pi/4
            if(random.choice([True,False])):
                phaseDiff = - phaseDiff
            sampledSignal   = self.genSample(phase = phaseDiff) - baseSignal
            
            self.net.setInputs(sampledSignal)

            self.net.forward()
            result = self.net.getOutput()[0]

            target = 1.0
            if(phaseDiff < 0.0):
                target = -1.0

            self.net.backward(np.matrix([target]))
            self.net.modWeights(stepSize = -0.1)

            if(phaseDiff > 0.0 and result < 0.0):
                correct += 1
                score +=1
            elif(phaseDiff < 0.0 and result > 0.0):
                correct += 1
                score +=1
            else:
                score -=1

                pbar.set_description(str(score))

        print("%d / %d Correct" % (correct, itterations))


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

