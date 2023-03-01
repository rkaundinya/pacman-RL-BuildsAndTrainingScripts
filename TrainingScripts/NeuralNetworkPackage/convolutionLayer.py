from . layers import Layer
import numpy as np
import math
import NeuralNetworkPackage.mathHelperLibrary as mhl

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

        result = np.zeros((n, numRowsToIterate, numColsToIterate))

        #Go through each observation 1-N
        for obsIdx, observation in enumerate(dataIn):
            result[obsIdx] = mhl.crossCorrelate(observation, self.kernel[obsIdx], numRowsToIterate, numColsToIterate)

        self.setPrevOut(result)
        return result

    def updateWeights(self, gradIn, eta=0.0001):
        prevIn = self.getPrevIn()

        #Declare variables outside of loop so we don't have to reallocate memory
        #for vars each loop cycle (compiler optimization)
        numRowsToIterate = 0
        numColsToIterate = 0

        for matrix in prevIn:
            for kernelIdx, kernelMatrix in enumerate(self.kernel):
                gradMatrix = gradIn[kernelIdx]
                numRowsToIterate = matrix.shape[0] - gradMatrix.shape[0] + 1
                numColsToIterate = matrix.shape[1] - gradMatrix.shape[1] + 1
                dJdK = mhl.crossCorrelate(matrix, gradMatrix, numRowsToIterate, numColsToIterate)
                kernelMatrix -= eta * dJdK
        
    #Output: result, returns a tensor with each kernel matrix transposed
    def gradient(self):
        result = np.zeros(self.kernel.shape)
        for resultIdx, kernelMatrix in enumerate(self.kernel):
            result[resultIdx] = np.transpose(kernelMatrix)

        return result

    #Input: gradIn, a N x (H - M + 1) x (W - M + 1) tensor
    #Output: a N x H x W tensor
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
            result[matrixIdx] = mhl.crossCorrelate(zPaddedGradIn, kernel, numRowsToIterate, numColsToIterate)

        return result
    
'''
Still more test code --- weight update, forward, backprop tests
row1 = np.array([[1,1,0,1,0,0,1,1]])
row2 = np.array([[1,1,1,1,0,0,1,0]])
row3 = np.array([[0,0,1,1,0,1,0,1]])
row4 = np.array([[1,1,1,0,1,1,1,0]])
row5 = np.array([[1,1,1,1,1,0,1,1]])
row6 = np.zeros((1,8))
row7 = np.array([[0,1,1,1,1,0,0,1]])
row8 = np.array([[1,0,1,0,0,1,0,1]])

test = np.array([np.concatenate((row1, row2, row3, row4, row5, row6, row7, row8), axis=0)])

dJdF = np.zeros((1,6,6))

dJdF[0][0][1] = -2
dJdF[0][3][4] = -2
dJdF[0][4][1] = 6

print(dJdF)

convLayer = ConvolutionalLayer(3)
convLayer.forward(test)
convLayer.updateWeights(dJdF)

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
gradIn[0][3][4] = -2
gradIn[0][4][1] = 6

print("dJdF:")
print(gradIn)

convLayer = ConvolutionalLayer(3)
convLayer.setKernel(kernel)
print("Updated Kernel:")
print(convLayer.backward(gradIn))'''