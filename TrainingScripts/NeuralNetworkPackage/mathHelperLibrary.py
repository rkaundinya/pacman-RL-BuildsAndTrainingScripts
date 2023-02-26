import numpy as np

#Input: kernel, assumes square kernel - kernel/filter we are cross-correlating with
#Output: cross-correlated matrix
#Note: Assumes kernel is smaller dimension than matrix1 - always make sure matrix1 has higher 
#dimennsionality than kernel
def crossCorrelate(matrix1, kernel, numRowsToIterate, numColsToIterate):
    finalRowIdx = matrix1.shape[0]
    finalColIdx = matrix1.shape[1]

    result = np.zeros((numRowsToIterate, numColsToIterate))
    kernelDim = kernel.shape[0]
    
    #Go through each row of feature map
    for featureMapRowIdx, maxRow in enumerate(range(kernelDim - 1, finalRowIdx)):
        #Go through each column of feature map
        for featureMapColIdx, maxCol in enumerate(range(kernelDim - 1, finalColIdx)):
            #Cache sum aggregator for compiler optimization
            currentFeatureMapSum = 0

            #Go through each row and column of input according to current featuremap index
            for kernelRowIdx, inputRowIdx in enumerate(range(featureMapRowIdx, maxRow + 1)):
                for kernelColIdx, inputColIdx in enumerate(range(featureMapColIdx, maxCol + 1)):
                    currentFeatureMapSum += matrix1[inputRowIdx,inputColIdx] * kernel[kernelRowIdx,kernelColIdx]

            #Assign obs feature map index result to aggregated sum  
            result[featureMapRowIdx,featureMapColIdx] = currentFeatureMapSum

    return result