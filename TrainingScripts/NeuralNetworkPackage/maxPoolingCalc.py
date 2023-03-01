from poolingCalcBase import PoolingCalcBase
import numpy as np

class MaxPoolingCalc(PoolingCalcBase):
    def __init__(self):
        pass

    #Input: featureMapSnippet, the portion of the feature map you want to calculate on
    #Output: result of custom pooling calculation, array of pooling layer modified indices
    def calculate(self, featureMapSnippet):
        #Get index of max value (only chooses one even if there are multiple)
        maxValIdx = np.argmax(featureMapSnippet)
        
        #Return an array of modified indices
        modifiedIndices = [np.unravel_index(maxValIdx, featureMapSnippet.shape)]
        return np.amax(featureMapSnippet), modifiedIndices