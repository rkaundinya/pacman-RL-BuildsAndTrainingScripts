from . layers import Layer
import numpy as np

class FlatteningLayer(Layer):
    def __init__(self):
        self.prevInRowDims = []
        self.prevInColDims = []
        self.prevInNumKernels = 0
        pass

    #Input: a N x (floor((D - Q) / S) + 1) x (floor((E - Q) / S)) + 1) tensor
    #       where each feature map in dataIn is DxE and pooling window is QxQ
    #Output: a N x D array
    #Note: this reshapes in row-major order since numpy stores values in row-major
    def forward(self, dataIn):
        #Cache row and col dimension of each matrix in dataIn
        numObs = len(dataIn)
        prevInNumKernels = len(dataIn[0])
        prevInRowDim = np.zeros((prevInNumKernels * numObs), dtype=int)
        prevInColDim = np.zeros((prevInNumKernels * numObs), dtype=int)

        #Length of reshaped single observation pooled results
        obsRowLength = 0

        for obsIdx, obsPooledKernel in enumerate(dataIn):
            for idx, pooledKernel in enumerate(obsPooledKernel):
                prevInRowDim[2*obsIdx + idx] = pooledKernel.shape[0]
                prevInColDim[2*obsIdx + idx] = pooledKernel.shape[1]
                #Keep track of reshape obs row length, but only count for first obs pooled results 
                #(otherwise we're adding row length for every observation which gives wrong result)
                if (obsIdx == 0):
                    obsRowLength += prevInRowDim[idx] * prevInColDim[idx]

        #Set vars caching prev Row and Col Dimensions for backwards pass
        self.prevInNumKernels = prevInNumKernels
        self.prevInRowDims = prevInRowDim
        self.prevInColDims = prevInColDim

        result = np.zeros((numObs, obsRowLength))

        #Reshape input matrices and add to result
        obsPooledResult = np.zeros((numObs, obsRowLength))
        for tensorIdx, tensor in enumerate(dataIn):
            reshapeEndIdx = 0
            reshapeStartIdx = 0
            for matrix in tensor:
                reshapeSize = matrix.shape[0] * matrix.shape[1]
                reshapeStartIdx = reshapeEndIdx
                reshapeEndIdx += reshapeSize
                obsPooledResult[tensorIdx][reshapeStartIdx:reshapeEndIdx] = np.reshape(matrix, (1, reshapeSize))
            result[tensorIdx] = np.reshape(obsPooledResult[tensorIdx], (1, obsRowLength))    
        

        return result

    def gradient(self):
        pass

    #Input: gradIn, a N x D array
    #Output, a N x (floor((D - Q) / S) + 1) x (floor((E - Q) / S)) + 1) tensor
    #       where each feature map in dataIn is DxE and pooling window is QxQ
    #Note - this reshapes in row-major order since numpy stores values in row-major
    def backward(self, gradIn):
        result = np.zeros((gradIn.shape[0], self.prevInNumKernels), dtype=np.ndarray)
        for matrixIdx, row in enumerate(gradIn):
            startIdx = 0
            endIdx = 0
            for kernelIdx in range(self.prevInNumKernels):
                currentKernelRowSize = self.prevInRowDims[2*matrixIdx + kernelIdx]
                currentKernelColSize = self.prevInColDims[2*matrixIdx + kernelIdx]
                kernelSize = currentKernelRowSize * currentKernelColSize
                startIdx = endIdx
                endIdx = startIdx + kernelSize
                result[matrixIdx][kernelIdx] = np.reshape(row[startIdx:endIdx], (currentKernelRowSize, currentKernelColSize))
            
        
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