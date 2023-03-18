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
        self.kernelDims = [kernelDimension]
        assert numKernels >= 1, "Trying to make kernel tensor with less than one entry"
        self.kernels = []
        for i in range(numKernels):
            self.kernels.append(np.random.uniform(low=-pow(10,-4), high=pow(10,-4), size=(kernelDimension, kernelDimension)))

    def addKernel(self, newKernel):
        self.kernels.append(newKernel)
        self.kernelDims.append(newKernel.shape[0])

    def addKernels(self, newKernels):
        for newKernel in newKernels:
            self.kernels.append(newKernel)
            self.kernelDims.append(newKernel.shape[0])

    #Input: list of kernels
    def setKernels(self, newKernels):
        self.kernelDims = np.zeros((len(newKernels)), dtype=int)
        for newKernelIdx, newKernel in enumerate(newKernels):
            self.kernelDims[newKernelIdx] = newKernel.shape[0]
        
        self.kernels = newKernels

    def getKernels(self):
        return self.kernels.copy()
    

    #Input: dataIn, N x H x W matrix
    #Output: N x (H - M + 1) x (W - M + 1) matrix where M is kernelDim
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        n = dataIn.shape[0]
        h = dataIn.shape[1]
        w = dataIn.shape[2]

        numKernels = len(self.kernels)

        result = np.zeros((n, numKernels), dtype=np.ndarray)

        #Go through each observation 1-N
        for obsIdx, observation in enumerate(dataIn):
            for kernelIdx, kernel in enumerate(self.kernels):
                numRowsToIterate = h - kernel.shape[0] + 1
                numColsToIterate = w - kernel.shape[0] + 1
                result[obsIdx][kernelIdx] = mhl.crossCorrelate(observation, kernel, numRowsToIterate, numColsToIterate)

        self.setPrevOut(result)
        return result

    def updateWeights(self, gradIn, eta=0.0001):
        prevIn = self.getPrevIn()

        #Declare variables outside of loop so we don't have to reallocate memory
        #for vars each loop cycle (compiler optimization)
        numRowsToIterate = 0
        numColsToIterate = 0

        for obsNum, matrix in enumerate(prevIn):
            for kernelIdx, kernelMatrix in enumerate(self.kernels):
                gradMatrix = gradIn[obsNum][kernelIdx]
                numRowsToIterate = matrix.shape[0] - gradMatrix.shape[0] + 1
                numColsToIterate = matrix.shape[1] - gradMatrix.shape[1] + 1
                dJdK = mhl.crossCorrelate(matrix, gradMatrix, numRowsToIterate, numColsToIterate)
                kernelMatrix -= eta * dJdK

    #Output: result, returns a tensor with each kernel matrix transposed
    def gradient(self):
        result = np.zeros((len(self.kernels)), dtype=np.ndarray)
        for resultIdx, kernelMatrix in enumerate(self.kernels):
            result[resultIdx] = np.transpose(kernelMatrix)

        return result

    #Input: gradIn, a N x (H - M + 1) x (W - M + 1) tensor
    #Output: a N x H x W tensor
    def backward(self, gradIn):
        prevIn = self.getPrevIn()
        result = np.zeros(prevIn.shape)
        kernelsTransposed = self.gradient()

        #TODO - average max pool backcoming gradients for two gradients assigned to 
        #same zPad location; update loop so it works with multiple sized multiple kernels
        for gradInIdx, gradInTensor in enumerate(gradIn):
            for matrixIdx, gradInMatrix in enumerate(gradInTensor):
                #Get pad amount for kernel
                padAmnt = self.kernelDims[matrixIdx] - 1
                #Offset starting row and col idx by pad amount
                startingRowIdx = padAmnt
                startingColIdx = padAmnt
                #Cache old gradient matrix size
                oldXDimension = gradInMatrix.shape[0]
                oldYDimension = gradInMatrix.shape[1]
                #Offset ending row and col idx by old dimensions
                endingRowIdx = startingRowIdx + oldXDimension
                endingColIdx = oldYDimension + padAmnt

                #Get new x and y needed to pad and get proper cross-correlate dimensions for result
                newXDimension = oldXDimension + 2 * padAmnt
                newYDimension = oldYDimension + 2 * padAmnt

                #
                numRowsToIterate = newXDimension - self.kernelDims[matrixIdx] + 1
                numColsToIterate = newYDimension - self.kernelDims[matrixIdx] + 1

                #Pad gradient in with 0's to match dimensions of previous X input
                zPaddedGradIn = np.zeros((newXDimension, newYDimension))
                
                zPaddedGradIn[startingRowIdx:endingRowIdx, startingColIdx:endingColIdx] = gradInMatrix
                
                #Cross-correlate by kernel (K) transposed to get backcoming gradient
                crossCorrResult = mhl.crossCorrelate(zPaddedGradIn, kernelsTransposed[matrixIdx], numRowsToIterate, numColsToIterate)
                result[gradInIdx] += crossCorrResult
            result[gradInIdx] /= kernelsTransposed.shape[0]

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