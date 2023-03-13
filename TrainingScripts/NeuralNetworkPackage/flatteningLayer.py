from . layers import Layer
import numpy as np

class FlatteningLayer(Layer):
    def __init__(self):
        self.prevInRowDim = 0
        self.prevInColDim = 0
        self.prevInNumKernels = 0
        pass

    #Input: a N x (floor((D - Q) / S) + 1) x (floor((E - Q) / S)) + 1) tensor
    #       where each feature map in dataIn is DxE and pooling window is QxQ
    #Output: a N x D array
    #Note: this reshapes in row-major order since numpy stores values in row-major
    def forward(self, dataIn):
        #Cache row and col dimension of each matrix in dataIn
        prevInNumKernels = dataIn.shape[1]
        prevInRowDim = dataIn.shape[2]
        prevInColDim = dataIn.shape[3]

        #Set vars caching prev Row and Col Dimensions for backwards pass
        self.prevInNumKernels = prevInNumKernels
        self.prevInRowDim = prevInRowDim
        self.prevInColDim = prevInColDim

        #Length for a single observation and kernel pooled result
        observationReshapedRowLength = prevInColDim * prevInRowDim
        #Length for all pooled kernel results for single obs
        finalReshapeRowLength = prevInRowDim * prevInColDim * prevInNumKernels

        result = np.zeros((dataIn.shape[0], finalReshapeRowLength))

        #Reshape input matrices and add to result
        obsPooledResult = np.zeros((dataIn.shape[0], prevInNumKernels, observationReshapedRowLength))
        for tensorIdx, tensor in enumerate(dataIn):
            for matrixIdx, matrix in enumerate(tensor):
                obsPooledResult[tensorIdx][matrixIdx] = np.reshape(matrix, (1, observationReshapedRowLength))
            result[tensorIdx] = np.reshape(obsPooledResult[tensorIdx], (1, finalReshapeRowLength))    
        

        return result

    def gradient(self):
        pass

    #Input: gradIn, a N x D array
    #Output, a N x (floor((D - Q) / S) + 1) x (floor((E - Q) / S)) + 1) tensor
    #       where each feature map in dataIn is DxE and pooling window is QxQ
    #Note - this reshapes in row-major order since numpy stores values in row-major
    def backward(self, gradIn):
        result = np.zeros((gradIn.shape[0], self.prevInNumKernels, self.prevInRowDim, self.prevInColDim))
        kernelSize = self.prevInRowDim * self.prevInColDim
        for matrixIdx, row in enumerate(gradIn):
            for kernelIdx in range(self.prevInNumKernels):
                startIdx = kernelIdx * kernelSize
                endIdx = startIdx + kernelSize
                result[matrixIdx][kernelIdx] = np.reshape(row[startIdx:endIdx], (self.prevInRowDim, self.prevInColDim))
            
        
        return result

'''
Testing code
test = np.array([[[2, 1], [3, 4]]])
flatteningLayer = FlatteningLayer()
flattenedOutput = flatteningLayer.forward(test)
print(flatteningLayer.backward(flattenedOutput))

#Ordering modified to fit row-major reshaping
slide25Test = np.array([[-2, 0, 6, -2]])
flatteningLayer.forward(np.array([[[-2, 0], [6, -2]]]))
print(flatteningLayer.backward(slide25Test))'''