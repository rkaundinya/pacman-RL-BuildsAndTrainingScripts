from . layers import Layer
import numpy as np

class ReLuLayer(Layer):
    #Input: None
    #Output: None
    def __init__(self):
        pass

    #Input: dataIn an NxK matrix
    #Output: An NxK matrix
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        output = np.clip(dataIn, 0, None)
        self.setPrevOut(output)
        return output

    #Input: None
    #Output: NxD identity matrix (after going thru relu, all prev output will have min value 0)
    def gradient(self):
        prevIn = self.getPrevIn()
        condensedOut = np.ones((prevIn.shape[0], prevIn.shape[1]))
        
        for rowCount, row in enumerate(condensedOut):
            for elCount, element in enumerate(row):
                condensedOut[rowCount][elCount] = 1 if prevIn[rowCount][elCount]  >= 0 else 0

        return condensedOut

    #Input: NxK matrix
    #Output: NxK matrix
    def backward(self, gradient):
        return np.multiply(gradient, self.gradient())