from . layers import Layer
from . maxPoolingCalc import MaxPoolingCalc
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
        #Keep track of the indices we choose from during forward pass
        self.modifiedIndices = []

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
        windowOutputNumRows = math.floor((dataIn.shape[2] - windowDim) / self.stride) + 1
        windowOutputNumCols = math.floor((dataIn.shape[3] - windowDim) / self.stride) + 1
        numFeatureMapsPerObs = dataIn.shape[1]

        #Keep track of the indices we choose from during forward pass
        #For later use in gradient pass; could be multiple indices depending on pooling 
        #implementation so need to have an array of modified indices per window output element
        self.modifiedIndices = np.empty((n, numFeatureMapsPerObs, windowOutputNumRows, windowOutputNumCols), dtype=np.ndarray)

        result = np.zeros((n, numFeatureMapsPerObs, windowOutputNumRows, windowOutputNumCols))
        for obsIdx, obsFeatureMaps in enumerate(dataIn):
            for featureMapIdx, featureMap in enumerate(obsFeatureMaps):
                result[obsIdx][featureMapIdx] = self.poolCalc(featureMap, self.qRows, obsIdx, featureMapIdx, self.stride)
            
        
        self.setPrevOut(result)
        return result
        

    def gradient(self):
        pass

    #Input: gradIn, a N x (floor((D - Q) / S) + 1) x (floor((E - Q) / S)) + 1) tensor
    #       where each feature map in dataIn is DxE and pooling window is QxQ
    #Output: a N x (H - M + 1) x (W - M + 1) tensor where HxW are dimensions of input image
    #        and MxM are dimensions of convolutional kernel
    def backward(self, gradIn):
        prevIn = self.getPrevIn()
        result = np.zeros(prevIn.shape)

        for obsIdx, observationGrad in enumerate(prevIn):
            for gradMatrixIdx, featureMap in enumerate(observationGrad):
                featureMap = np.zeros((featureMap.shape[0], featureMap.shape[1]))
                gradMatrix = gradIn[obsIdx][gradMatrixIdx]
                modifiedIndicesTensor = self.modifiedIndices[obsIdx]

                for modifiedIndicesMatrix in modifiedIndicesTensor:
                    rowsToIterate = modifiedIndicesMatrix.shape[0]
                    colsToIterate = modifiedIndicesMatrix.shape[1]

                    for row in range(rowsToIterate):
                        for col in range(colsToIterate):
                            for tuple in modifiedIndicesMatrix[row][col]:
                                result[obsIdx][gradMatrixIdx][tuple] = gradMatrix[row][col]
        
                    
        return result

    def poolCalc(self, dataIn, windowDim, observationNum, obsFeatureMapNum, stride=1):
        dataInNumRows = dataIn.shape[0]
        dataInNumCols = dataIn.shape[1]

        numRowsToIterate = math.floor((dataInNumRows - windowDim) / stride) + 1
        numColsToIterate = math.floor((dataInNumCols - windowDim) / stride) + 1

        result = np.zeros((numRowsToIterate, numColsToIterate))

        modifiedIndicesMatrix = self.modifiedIndices[observationNum][obsFeatureMapNum]

        #Declare vars outside of loop for memory and compiler optimization
        featureMapSnippetRowStartIdx = 0
        featureMapSnippetColStartIdx = 0

        #Go through each row of pooling output
        for featureMapRowIdx, maxRow in enumerate(range(windowDim - 1, dataInNumRows, stride)):
            #Go through each column of pooling output
            for featureMapColIdx, maxCol in enumerate(range(windowDim - 1, dataInNumCols, stride)):
                #Keep track of starting row and col indicies of feature map
                featureMapSnippetRowStartIdx = featureMapRowIdx*stride
                featureMapSnippetColStartIdx = featureMapColIdx*stride

                #Get the snippet of feature map we want to operate on
                featureMapSnippet = dataIn[featureMapSnippetRowStartIdx:maxRow+1, featureMapSnippetColStartIdx:maxCol+1]
                #Store pooling calc result and the indices we modified (for gradient later)
                calcResult, modifiedIndices = self.poolingCalc.calculate(featureMapSnippet)

                modifiedIndicesEntries = np.zeros((len(modifiedIndices)), dtype=tuple)

                #Adjust modified indices from feature map snippet indices to indices of whole feature map
                result[featureMapRowIdx, featureMapColIdx] = calcResult
                for modifiedIndexEntryIdx, modifiedIndex in enumerate(modifiedIndices):
                    modifiedIndex = (featureMapSnippetRowStartIdx + modifiedIndex[0], featureMapSnippetColStartIdx + modifiedIndex[1])
                    modifiedIndicesEntries[modifiedIndexEntryIdx] = modifiedIndex

                modifiedIndicesMatrix[featureMapRowIdx][featureMapColIdx] = modifiedIndicesEntries

        return result
    
'''
Test code for forward and gradient in pooling layer
gradBackTest = np.array([[[-2,0],[6,-2]]])
poolingLayer = PoolingLayer(2,2,2)
featureMap = np.array([[1,1,2,4],[5,6,7,8],[3,2,1,0],[1,2,3,4]])
print(featureMap)
dataIn = np.array([featureMap]) 
poolingLayer.forward(dataIn)
poolingLayer.backward(gradBackTest)

#Testing slide 27 feature map gradient example
row1 = np.array([[4,7,1,7,2,3]])
row2 = np.array([[6,3,5,6,4,2]])
row3 = np.array([[6,5,6,4,3,7]])
row4 = np.array([[4,2,5,2,5,0]])
row5 = np.array([[5,6,6,2,5,3]])
row6 = np.array([[2,1,2,3,2,3]])

slide27FeatureMap = np.array([np.concatenate((row1, row2, row3, row4, row5, row6), axis=0)])
print(slide27FeatureMap)

poolingLayer = PoolingLayer(3,3,3)
poolingLayer.forward(slide27FeatureMap)
print(poolingLayer.backward(gradBackTest))'''