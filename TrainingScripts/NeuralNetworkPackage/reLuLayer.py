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
        output = np.zeros_like(dataIn)
        for i, j  in enumerate(dataIn):
            for l,m in enumerate(dataIn[i]):
                output[i][l] = np.clip(dataIn[i][l], 0, None)
        self.setPrevOut(output)
        return output

    #Input: None
    #Output: NxD identity matrix (after going thru relu, all prev output will have min value 0)
    #This assumes a PrevIn input with a min value of 0
    def gradient(self):
        prevIn = self.getPrevIn()
        return prevIn

    #Input: NxK matrix
    #Output: NxK matrix
    def backward(self, gradient):
        return np.multiply(gradient, self.gradient())