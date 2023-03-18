from . layers import Layer
from . weightUpdateCalcBase import WeightUpdateCalc
import numpy as np

class BasicWeightUpdateCalc(WeightUpdateCalc):
    def __init__(self):
        super().__init__()

    def CalculateUpdate(self, eta, layer):
        #Throw an error if input layer is not of class FullyConnectedLayer
        if (not isinstance(layer, FullyConnectedLayer)):
            TypeError(layer)
        
        prevIn = self.calcData.getPrevIn()
        gradIn = self.calcData.getGradIn()

        #Update weights
        dJdW = -np.matmul(np.transpose(prevIn), gradIn) / gradIn.shape[0]
        layer.setWeights(layer.weights + eta * dJdW)

        #Update biases
        dJdb = np.mean(gradIn, axis=0)
        layer.setBiases(layer.biases - eta * dJdb) 

class FullyConnectedLayer(Layer):
    #Input: sizeIn, the number of features of data coming in
    #Input: sizeOut, the number of features for the data coming out
    #Input: calcClass, the type of weight update calc we want - default to basic
    #Output: None
    def __init__(self, sizeIn, sizeOut, calcClass=BasicWeightUpdateCalc()):
        # Create weights matrix of random ints in range -10^-4, 10^-4
        # Create biases matrix of random ints in same range of size 1xk
        self.weights = np.random.uniform(low=-pow(10,-4), high=pow(10,-4), size=(sizeIn, sizeOut))
        self.biases = np.random.uniform(low=-pow(10,-4), high=pow(10,-4), size=(1, sizeOut))
        self.calcClass = calcClass
        
    #Input: None
    #Output: The sizeIn x sizeOut weight matrix
    def getWeights(self):
        return self.weights.copy()

    #Input: The sizeIn x sizeOut weight matrix
    #Output: None
    def setWeights(self, weights):
        self.weights = weights

    #Input: None
    #Output: The 1 x sizeOut bias vector
    def getBiases(self):
        return self.biases.copy()

    #Input: The 1 x sizeOut bias vector
    #Output: None
    def setBiases(self, biases):
        self.biases = biases

    #Input: dataIn, an NxD data matrix
    #Output: An NxK data matrix
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        dataOut = np.matmul(dataIn, self.weights) + self.biases
        self.setPrevOut(dataOut)

        #Update calculation class data
        self.calcClass.UpdateCalcDataPrevIn(dataIn)

        return dataOut

    #Input: None
    #Output: Technically a N x (K x D), but we'll just use a K x D to optimize
    def gradient(self):
        return np.transpose(self.weights)

    #Input: NxK matrix
    #Output: NxD matrix
    def backward(self, gradIn):
        return np.matmul(gradIn, self.gradient())
    
    #Input: gradIn, backcoming gradient
    #Input: eta, learning rate
    #Output: None
    def updateWeights(self, gradIn, eta = 0.0001):
        self.calcClass.UpdateCalcDataGradIn(gradIn)
        self.calcClass.CalculateUpdate(eta, self)