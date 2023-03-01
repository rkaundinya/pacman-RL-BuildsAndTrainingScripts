from . layers import Layer
import numpy as np

class LinearLayer(Layer):
    #Input: None
    #Output: None
    def __init__(self):
        pass

    #Input: dataIn an NxK matrix
    #Output: An NxK matrix
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        self.setPrevOut(dataIn)
        return dataIn

    #Input: None
    #Output: NxD identity matrix
    def gradient(self):
        prevIn = self.getPrevIn()
        return np.ones((prevIn.shape[0], prevIn.shape[1]))

    def backward(self, gradient):
        return gradient