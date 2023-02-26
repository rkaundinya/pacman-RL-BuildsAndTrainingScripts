import numpy as np
import math
from layers import Layer

class ConvolutionalLayer(Layer):
    #Input: kernelDimension, determines size of kernel K 
    # (kernelDimension x kernelDimension square matrix)
    #Input: numKernels, the number of kernel matrices of kernel tensor
    #Output: None
    def __init__(self, kernelDimension, numKernels=1):
        self.kernelDim = kernelDimension
        assert numKernels >= 1, "Trying to make kernel tensor with less than one entry"
        self.kernel = np.random.uniform(low=-pow(10,-4), high=pow(10,-4), size=(numKernels, kernelDimension, kernelDimension))

    def setKernel(self, newKernel):
        assert newKernel.shape[1] == self.kernel.shape[1] and newKernel.shape[2] == self.kernel.shape[2], "Need to re-initialize layer with new dimensions"
        self.kernel = newKernel

    #Input: dataIn, N x H x W matrix
    #Output: N x (H - M + 1) x (W - M + 1) matrix where M is kernelDim
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        n = dataIn.shape[0]
        h = dataIn.shape[1]
        w = dataIn.shape[2]
        numRowsToIterate = h - self.kernelDim + 1
        numColsToIterate = w - self.kernelDim + 1

        result = np.zeros((n, self.kernelDim, self.kernelDim))

        #Go through each observation 1-N
        for obsIdx, observation in enumerate(dataIn):
            result[obsIdx] = self.crossCorrelate(observation, self.kernel[obsIdx], numRowsToIterate, numColsToIterate)

    #Output: result, returns a tensor with each kernel matrix transposed
    def gradient(self):
        result = np.zeros(self.kernel.shape)
        for resultIdx, kernelMatrix in enumerate(self.kernel):
            result[resultIdx] = np.transpose(kernelMatrix)

        return result

    #Input: gradIn, a N x (H - M + 1) x (W - M + 1) tensor
    def backward(self, gradIn):
        padAmnt = math.ceil(self.kernelDim / 2)

        oldXDimension = gradIn[0].shape[0]
        oldYDimension = gradIn[0].shape[1]
        
        newXDimension = oldXDimension + 2 * padAmnt
        newYDimension = oldYDimension + 2 * padAmnt

        numRowsToIterate = newXDimension - self.kernelDim + 1
        numColsToIterate = newYDimension - self.kernelDim + 1

        result = np.zeros((gradIn.shape[0], numRowsToIterate, numColsToIterate))
        kernelsTransposed = self.gradient()

        for matrixIdx, gradInMatrix in enumerate(gradIn):
            #Pad gradient in with 0's to match dimensions of previous X input
            zPaddedGradIn = np.zeros((newXDimension, newYDimension))

            startingRowIdx = padAmnt
            endingRowIdx = startingRowIdx + oldXDimension
            startingColIdx = padAmnt
            endingColIdx = oldYDimension + padAmnt
            
            zPaddedGradIn[startingRowIdx:endingRowIdx, startingColIdx:endingColIdx] = gradInMatrix
            
            #Cross-correlate by kernel (K) transposed to get backcoming gradient
            kernel = kernelsTransposed[matrixIdx]
            result[matrixIdx] = self.crossCorrelate(zPaddedGradIn, kernel, numRowsToIterate, numColsToIterate)

        return result
    
    #Input: kernel, assumes square kernel - kernel/filter we are cross-correlating with
    #Output: cross-correlated matrix
    def crossCorrelate(self, matrix1, kernel, numRowsToIterate, numColsToIterate):
        resultXDim = matrix1.shape[0] - kernel.shape[0] + 1
        resultyDim = matrix1.shape[1] - kernel.shape[1] + 1

        result = np.zeros((resultXDim, resultyDim))
        kernelDim = kernel.shape[0]
        
        #Go through each row of feature map
        for featureMapRowIdx, maxRow in enumerate(range(kernelDim - 1, numRowsToIterate + 1)):
            #Go through each column of feature map
            for featureMapColIdx, maxCol in enumerate(range(kernelDim - 1, numColsToIterate + 1)):
                #Cache sum aggregator for compiler optimization
                currentFeatureMapSum = 0

                #Go through each row and column of input according to current featuremap index
                for kernelRowIdx, inputRowIdx in enumerate(range(featureMapRowIdx, maxRow + 1)):
                    for kernelColIdx, inputColIdx in enumerate(range(featureMapColIdx, maxCol + 1)):
                        currentFeatureMapSum += matrix1[inputRowIdx,inputColIdx] * kernel[kernelRowIdx,kernelColIdx]

                #Assign obs feature map index result to aggregated sum  
                result[featureMapRowIdx,featureMapColIdx] = currentFeatureMapSum

        return result
    
'''
Updated Toy Code for testing layer
test = np.array([[[1,2,3],[2,2,3],[1,3,3]]])
kernelTest1 = np.ones((1,2,2))
convLayerTest1 = ConvolutionalLayer(2)
convLayerTest1.setKernel(kernelTest1)
convLayerTest1.forward(test)

kernel = np.array([[[2,2,1],[-1,-1,0],[2,0,2]]])

for kernelIdx, kernelMatrix in enumerate(kernel):
    kernel[kernelIdx] = np.transpose(kernelMatrix)

gradIn = np.zeros((1,6,6))
gradIn[0][0][1] = -2
gradIn[0][2][2] = 1
gradIn[0][3][4] = -2
gradIn[0][4][1] = 6
convLayer = ConvolutionalLayer(3)
convLayer.setKernel(kernel)
print(convLayer.backward(gradIn))
'''