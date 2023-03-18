from . weightUpdateCalcBase import WeightUpdateCalc, CalcData
from . fullyConnectedLayer import FullyConnectedLayer
import numpy as np
import math

class AdamCalcData(CalcData):
    def __init__(self, p1, p2, sW, rW, sB, rB, delta, t=0):
        self.p1 = p1
        self.p2 = p2
        self.sWeights = sW
        self.rWeights = rW
        self.sBiases = sB
        self.rBiases = rB
        self.delta = delta
        self.t = t
        super().__init__()

class AdamWeightUpdateCalc(WeightUpdateCalc):
    def __init__(self, p1=0.9, p2=0.999, delta=math.pow(10,-8), sW=0, rW=0, sB=0, rB=0, t=1):
        self.calcData = AdamCalcData(p1, p2, sW, rW, sB, rB, delta, t)

    def momentumUpdate(self, dJdW, dJdB):
        self.calcData.sWeights = self.calcData.p1 * self.calcData.sWeights + (1 - self.calcData.p1) * dJdW
        self.calcData.sBiases = self.calcData.p1 * self.calcData.sBiases + (1 - self.calcData.p1) * dJdB

    def RMSPropUpdate(self, dJdW, dJdB):
        self.calcData.rWeights = self.calcData.p2*self.calcData.rWeights + (1-self.calcData.p2) * (np.multiply(dJdW, dJdW))
        self.calcData.rBiases = self.calcData.p2*self.calcData.rBiases + (1-self.calcData.p2) * (np.multiply(dJdB, dJdB))

    def AdamUpdate(self):
        #Calc numerator and denominator
        numeratorWeights = (self.calcData.sWeights/(1-self.calcData.p1**self.calcData.t))
        denominatorWeights = np.sqrt(self.calcData.rWeights/(1-self.calcData.p2**self.calcData.t)) + self.calcData.delta
        
        numeratorBiases = (self.calcData.sBiases/(1-self.calcData.p1**self.calcData.t))
        denominatorBiases = np.sqrt(self.calcData.rBiases/(1-self.calcData.p2**self.calcData.t)) + self.calcData.delta

        #Increment time var
        self.calcData.t += 1
        return numeratorWeights / denominatorWeights, numeratorBiases / denominatorBiases
    
    def CalculateUpdate(self, eta, layer):
        #Throw an error if input layer is not of class FullyConnectedLayer
        if (not isinstance(layer, FullyConnectedLayer)):
            TypeError(layer)
        
        prevIn = self.calcData.getPrevIn()
        gradIn = self.calcData.getGradIn()

        #Check for uninitialized calculation data
        if (prevIn.any() == None or gradIn.any() == None):
            print("Warning - uninitialized calculation data")
            ValueError()

        dJdW = np.matmul(np.transpose(prevIn), gradIn) / gradIn.shape[0]
        dJdb = np.mean(gradIn, axis=0)

        self.momentumUpdate(dJdW, dJdb)
        self.RMSPropUpdate(dJdW, dJdb)
        adamValWeights, adamValBiases = self.AdamUpdate()
        
        currentWeights = layer.getWeights()
        layer.setWeights(currentWeights - eta * adamValWeights)

        #Update biases
        layer.setBiases(layer.biases - eta * adamValBiases) 