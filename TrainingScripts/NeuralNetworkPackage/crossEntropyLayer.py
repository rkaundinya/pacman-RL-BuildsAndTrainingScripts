from . objectiveLayer import ObjectiveLayer
import numpy as np

class CrossEntropyLayer(ObjectiveLayer):
    #Input: Y is an N by K matrix of target values
    #Input: Yhat is a N by K matrix of estimated values
    #Output: A single floating point number
    def eval(self, Y, Yhat):
        eps = pow(10,-7)
        meanDivisor = Y.shape[0] * Y.shape[1]
        return -1/meanDivisor * np.sum(Y * np.log(Yhat + eps))

    #Input: Y is an N by K matrix of target values
    #Input: Yhat is an N by K matrix of estimated values
    #Output: an N by K matrix
    def gradient(self, Y, Yhat):
        eps = pow(10,-7)
        return -Y / (Yhat + eps)