import numpy as np
from layers import Layer

class ConvolutionalLayer(Layer):
    #Input: kernelDimension, determines size of kernel K 
    # (kernelDimension x kernelDimension square matrix)
    #Output: None
    def __init__(self, kernelDimension):
        self.kernelDim = kernelDimension
        self.kernel = np.random.uniform(low=-pow(10,-4), high=pow(10,-4), size=(kernelDimension, kernelDimension))

    #Input: dataIn, N x H x W matrix
    #Output: N x H x W matrix
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        n = dataIn.shape[0]
        h = dataIn.shape[1]
        w = dataIn.shape[2]
        numRowsToIterate = h - self.kernelDim + 2
        numColsToIterate = w - self.kernelDim + 2

        result = np.zeros((n, self.kernelDim, self.kernelDim))

        #Go through each observation 1-N
        for obsIdx, observation in enumerate(dataIn):
            currentObsResult = result[obsIdx]

            inputRowIdx = 0
            inputColIdx = 0
            #Go through each row of feature map
            for featureMapRowIdx, maxRow in enumerate(range(self.kernelDim - 1, numRowsToIterate)):
                #Go through each column of feature map
                for featureMapColIdx, maxCol in enumerate(range(self.kernelDim - 1, numColsToIterate)):
                    #Cache sum aggregator for compiler optimization
                    currentFeatureMapSum = 0

                    #Go through each row and column of input according to current featuremap index
                    for kernelRowIdx, inputRowIdx in enumerate(range(featureMapRowIdx, maxRow + 1)):
                        for kernelColIdx, inputColIdx in enumerate(range(featureMapColIdx, maxCol + 1)):
                            currentFeatureMapSum += observation[inputRowIdx,inputColIdx] * self.kernel[kernelRowIdx,kernelColIdx]

                    #Assign obs feature map index result to aggregated sum  
                    currentObsResult[featureMapRowIdx,featureMapColIdx] = currentFeatureMapSum
            print(currentObsResult)
                    
    def gradient(self):
        pass

    def backward(self, gradient):
        pass