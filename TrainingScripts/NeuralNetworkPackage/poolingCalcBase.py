from abc import ABC, abstractmethod
import numpy as np

#Abstract base class for user defined pooling calculations
class PoolingCalcBase(ABC):
    def __init__(self):
        pass

    #Input: featureMapSnippet, the portion of the feature map you want to calculate on
    #Output: result of custom pooling calculation
    @abstractmethod
    def calculate(self, featureMapSnippet):
        pass