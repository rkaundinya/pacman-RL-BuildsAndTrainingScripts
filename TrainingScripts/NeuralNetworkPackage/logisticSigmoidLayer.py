from . layers import Layer
import numpy as np

class LogisticSigmoidLayer(Layer):
    #Input: None
    #Output: None
    def __init__(self):
        pass

    #Input: Takes in single element from matrix
    #Output: applies logistic sigmoid to input element and returns
    def calc(self, num):
        return 1 / (1 + np.exp(-num))

    #Input: dataIn an NxK matrix
    #Output: An NxK matrix
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        sigmoid = np.vectorize(self.calc)
        dataOut = sigmoid(dataIn)
        self.setPrevOut(dataOut)
        return dataOut

    #Input: None
    #Output: NxK matrix storing diagonal values of Jacobian Nx(KxK) in each row 
    #        Each row corresponding to diagonal of Jacobian per observation
    def gradient(self):
        prevIn = self.getPrevIn()
        prevOut = self.getPrevOut()
        grad = np.zeros((prevIn.shape[0], prevIn.shape[1]))

        rowIdx = 0
        colIdx = 0
        prevOutElVal = 0
        for row in prevIn:
            for element in row:
                prevOutElVal = prevOut[rowIdx][colIdx]
                grad[rowIdx][colIdx] = prevOutElVal * (1 - prevOutElVal)
                colIdx += 1    
            rowIdx += 1
            colIdx = 0
        
        return grad

    #Input: NxK matrix
    #Output: NxK matrix
    def backward(self, gradient):
        return np.multiply(gradient, self.gradient())
