from . objectiveLayer import ObjectiveLayer
import numpy as np

class SquaredErrorLayer(ObjectiveLayer):
    def __init__(self):
        self.__prevIn = []
        self.__prevOut = []
        pass
    
    #Input: Y is an N by K matrix of target values
    #Input: Yhat is a N by K matrix of estimated values
    #Output: A single floating point number
    def eval(self, Y, Yhat):
        diff = Y - Yhat
        out = 1/Y.shape[0] * np.sum(np.multiply(diff, diff))
        return out

    #Input: Y is an N by K matrix of target values
    #Input: Yhat is an N by K matrix of estimated values
    #Output: an N by K matrix
    def gradient(self, Y, Yhat):
        diff = Y - Yhat
        return -2 * diff