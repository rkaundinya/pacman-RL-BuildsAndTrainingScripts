from . layers import Layer
import numpy as np

class SoftmaxActivationLayer(Layer):
    #Input: None
    #Output: None
    def __init__(self):
        pass

    def exp(self, num, max):
        return np.exp(num - max)

    #Input: Takes in single element from matrix
    #Output: applies softmax activation to input element and returns
    def calc(self, rowNum, sum):
        return rowNum / sum

    #Input: dataIn an NxK matrix
    #Output: An NxK matrix
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        expFunc = np.vectorize(self.exp)
        calcFunc = np.vectorize(self.calc)
        
        dataOut = np.ones(dataIn.shape)

        rowIdx = 0
        for row in dataIn:
            max = np.max(row)
            expRaisedRow = expFunc(row, max)
            sum = np.sum(expRaisedRow)
            dataOut[rowIdx, :] = calcFunc(expRaisedRow, sum)
            rowIdx += 1

        self.setPrevOut(dataOut)
        return dataOut

    #Input: None
    #Output: Nx(KxK) tensor of Jacobian Matrices
    def gradient(self):
        prevIn = self.getPrevIn()
        prevOut = self.getPrevOut()

        grad = np.zeros((prevIn.shape[0], prevIn.shape[1], prevIn.shape[1]))

        rowIdx = 0
        colIdx = 0
        obsNum = 0

        for matrix in grad:
            for row in matrix:
                for el in row:
                    # If we're on a diagonal use diagonal gradient calc
                    if (rowIdx == colIdx):
                        row[colIdx] = prevOut[obsNum][colIdx] * (1 - prevOut[obsNum][colIdx])
                    else:
                        # Otherwise use off diagonal gradient calc
                        row[colIdx] = -prevOut[obsNum][rowIdx] * prevOut[obsNum][colIdx]
                    colIdx += 1
                rowIdx += 1
                colIdx = 0
            obsNum += 1
            rowIdx = 0

        return grad

    #Input: Nx(KxK) Jacobian
    #Output: NxK matrix
    def backward(self, gradient):
        prevIn = self.getPrevIn()
        result = np.zeros((prevIn.shape[0], prevIn.shape[1]))

        selfGradient = self.gradient()
        for rowIdx, matrix in enumerate(selfGradient):
            result[rowIdx] = np.matmul(gradient[rowIdx], matrix)
        
        return result