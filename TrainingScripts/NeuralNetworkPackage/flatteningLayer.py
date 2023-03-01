from . layers import Layer
import numpy as np

class FlatteningLayer(Layer):
    def __init__(self):
        self.prevInRowDim = 0
        self.prevInColDim = 0
        pass

    #Input: a N x (floor((D - Q) / S) + 1) x (floor((E - Q) / S)) + 1) tensor
    #       where each feature map in dataIn is DxE and pooling window is QxQ
    #Output: a N x D array
    #Note: this reshapes in row-major order since numpy stores values in row-major
    def forward(self, dataIn):
        #Cache row and col dimension of each matrix in dataIn
        prevInRowDim = dataIn.shape[1]
        prevInColDim = dataIn.shape[2]

        #Set vars caching prev Row and Col Dimensions for backwards pass
        self.prevInRowDim = prevInRowDim
        self.prevInColDim = prevInColDim

        reshapeRowLength = prevInRowDim * prevInColDim

        result = np.zeros((dataIn.shape[0], reshapeRowLength))

        #Reshape input matrices and add to result
        for matrixIdx, matrix in enumerate(dataIn):
            result[matrixIdx] = np.reshape(matrix, (1, reshapeRowLength))

        return result

    def gradient(self):
        pass

    #Input: gradIn, a N x D array
    #Output, a N x (floor((D - Q) / S) + 1) x (floor((E - Q) / S)) + 1) tensor
    #       where each feature map in dataIn is DxE and pooling window is QxQ
    #Note - this reshapes in row-major order since numpy stores values in row-major
    def backward(self, gradIn):
        result = np.zeros((gradIn.shape[0], self.prevInRowDim, self.prevInColDim))
        for matrixIdx, row in enumerate(gradIn):
            result[matrixIdx] = np.reshape(row, (self.prevInRowDim, self.prevInColDim))
        
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