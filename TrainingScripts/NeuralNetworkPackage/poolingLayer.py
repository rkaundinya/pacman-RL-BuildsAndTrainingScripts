from layers import Layer
from maxPoolingCalc import MaxPoolingCalc
import numpy as np
import math

class PoolingLayer(Layer):
    #Input: qRows, rows in window Q
    #Input: qCols, cols window Q
    #Input: stride, the stride length of pooling window
    #Input: poolingCalc, set the pooling calculation type for this layer - default to max pooling
    #Output: None
    def __init__(self, qRows, qCols, stride, poolingCalc=MaxPoolingCalc()):
        self.qRows = qRows
        self.qCols = qCols
        self.stride = stride
        self.poolingCalc = poolingCalc

    #Input: poolingCalcIn, set the pooling calculation type for this layer
    #Output: None
    def setPoolingCalc(self, poolingCalcIn):
        self.poolingCalc = poolingCalcIn

    #Input: newStride, the new stride length for this layer
    #Output: None
    def setStride(self, newStride):
        self.stride = newStride

    #Input: dataIn, a N x (H - M + 1) x (W - M + 1) matrix where M is kernelDim from convolution layer
    #Output: a N x (floor((D - Q) / S) + 1) x (floor((E - Q) / S)) + 1) tensor
    #        where each feature map in dataIn is DxE and pooling window is QxQ
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        #Make result tensor
        windowDim = self.qRows
        n = dataIn.shape[0]
        windowOutputNumRows = math.floor((dataIn.shape[1] - windowDim) / self.stride) + 1
        windowOutputNumCols = math.floor((dataIn.shape[2] - windowDim) / self.stride) + 1

        result = np.zeros((n, windowOutputNumRows, windowOutputNumCols))
        for resultIdx, featureMap in enumerate(dataIn):
            result[resultIdx] = self.poolCalc(featureMap, self.qRows, self.stride)
        
        print(result)
        self.setPrevOut(result)
        

    def gradient(self):
        pass

    def backward(self, gradIn):
        pass

    def poolCalc(self, dataIn, windowDim, stride=1):
        dataInNumRows = dataIn.shape[0]
        dataInNumCols = dataIn.shape[1]

        numRowsToIterate = math.floor((dataInNumRows - windowDim) / stride) + 1
        numColsToIterate = math.floor((dataInNumCols - windowDim) / stride) + 1

        result = np.zeros((numRowsToIterate, numColsToIterate))

        #Go through each row of pooling output
        for featureMapRowIdx, maxRow in enumerate(range(windowDim - 1, dataInNumRows, stride)):
            #Go through each column of pooling output
            for featureMapColIdx, maxCol in enumerate(range(windowDim - 1, dataInNumCols, stride)):
                featureMapSnippet = dataIn[featureMapRowIdx*stride:maxRow+1, featureMapColIdx*stride:maxCol+1]
                calcResult = self.poolingCalc.calculate(featureMapSnippet)
                result[featureMapRowIdx, featureMapColIdx] = calcResult

        return result
    
'''
Testing code
poolingLayer = PoolingLayer(2,2,2)
featureMap = np.array([[1,1,2,4],[5,6,7,8],[3,2,1,0],[1,2,3,4]])
dataIn = np.array([featureMap])
poolingLayer.forward(dataIn)'''