from poolingCalcBase import PoolingCalcBase
import numpy as np

class MaxPoolingCalc(PoolingCalcBase):
    def __init__(self):
        pass

    def calculate(self, featureMapSnippet):
        return np.amax(featureMapSnippet)