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
        elementsAreNPArrays = False
        if (type(prevIn[0][0]) == np.ndarray):
            elementsAreNPArrays = True

        if not elementsAreNPArrays:
            for idx1,j in enumerate(prevIn):
                for idx2,m in enumerate(prevIn[idx1]):
                    prevIn[idx1][idx2] = 1 if prevIn[idx1][idx2] >= 0 else 0
        #Assuming 4 dimensional prevIn
        else:
            for tensor in prevIn:
                for matrix in tensor:
                    for row in matrix:
                        for element in row:
                            element = 1 if element >= 0 else 0

        return prevIn

    #Input: NxK matrix
    #Output: NxK matrix
    def backward(self, gradient):
        return np.multiply(gradient, self.gradient())
    
test = np.array([1,2])
print(type(test) == np.ndarray)