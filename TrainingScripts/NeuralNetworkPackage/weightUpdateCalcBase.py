from abc import ABC, abstractmethod

class CalcData:
    def __init__(self):
        self.prevIn = None
        self.gradIn = None
        self.eta = None

    def __init__(self, prevIn=None, gradIn=None, eta=None):
        self.prevIn = prevIn
        self.gradIn = gradIn
        self.eta = eta

    def setPrevIn(self, prevIn):
        self.prevIn = prevIn

    def getPrevIn(self):
        return self.prevIn

    def setGradIn(self, gradIn):
        self.gradIn = gradIn

    def getGradIn(self):
        return self.gradIn

class WeightUpdateCalc(ABC):
    def __init__(self):
        self.calcData = CalcData()

    def UpdateCalcDataPrevIn(self, prevIn):
        self.calcData.setPrevIn(prevIn)

    def UpdateCalcDataGradIn(self, gradIn):
        self.calcData.setGradIn(gradIn)

    @abstractmethod
    def CalculateUpdate(self, eta, layer):
        pass