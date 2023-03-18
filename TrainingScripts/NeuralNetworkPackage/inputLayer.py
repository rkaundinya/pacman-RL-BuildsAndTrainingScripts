from . layers import Layer
import numpy as np

class InputLayer(Layer):
    #Input: dataIn, an NxD matrix
    #Output: None
    def __init__(self, dataIn, bShouldZScore=True):
        self.meanX = np.mean(dataIn, axis=0)
        self.stdX = np.std(dataIn, axis=0, ddof=1)
        self.stdX[self.stdX == 0] = 1
        self.bShouldZScore = bShouldZScore

    def Print(self):
        print("Mean: " + str(self.meanX) + "\nStd: " + str(self.stdX))

    #Input: dataIn, an NxD matrix
    #Output: a NxD matrix
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        if (self.bShouldZScore):
            zScoredData = np.divide(dataIn - self.meanX, self.stdX)
            self.setPrevOut(zScoredData)
            return zScoredData
        else:
            dataIn = dataIn / np.max(dataIn)
            self.setPrevOut(dataIn)
            return dataIn

    #TODO - functions below will be implemented later
    def gradient(self):
        pass

    def backward(self, gradIn):
        pass