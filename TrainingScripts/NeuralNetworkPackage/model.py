from . layers import Layer
from . convolutionLayer import ConvolutionalLayer
from . fullyConnectedLayer import FullyConnectedLayer
from . objectiveLayer import ObjectiveLayer
import numpy as np

class Model:
    def __init__(self, layers=[], eta=0.01):
        self.layers = layers
        self.eta = eta

        self.fcLayers = []
        self.convLayers = []
        for layer in layers:
            if (isinstance(layer, FullyConnectedLayer)):
                self.fcLayers.append(layer)
            if (isinstance(layer, ConvolutionalLayer)):
                self.convLayers.append(layer)
        return
    
    def add(self, layer):
        if (isinstance(layer, ObjectiveLayer) or isinstance(layer, Layer)):
            self.layers.append(layer)
        else:
            TypeError(layer)

        if (isinstance(layer, FullyConnectedLayer)):
            self.fcLayers.append(layer)
        if (isinstance(layer, ConvolutionalLayer)):
            self.convLayers.append(layer)
    
    #Inputs: x, input to network
    #Inputs: y, expected output for input
    #Executes a single forward/backward prop training of model
    def train(self, x, y):
        prediction = self.predict(x)
        loss = 0

        if (isinstance(self.layers[-1], ObjectiveLayer)):
            loss = self.layers[-1].eval(y, prediction)

        gradient = self.layers[-1].gradient(y, prediction)

        for i in range(len(self.layers) - 2, 0, -1):
            newGradient = self.layers[i].backward(gradient)

            if (isinstance(self.layers[i], FullyConnectedLayer) or isinstance(self.layers[i], ConvolutionalLayer)):
                self.layers[i].updateWeights(gradient, self.eta)

            gradient = newGradient
    
    def predict(self, input):
        h = input

        for layer in self.layers:
            if (isinstance(layer, ObjectiveLayer)):
                return h

            h = layer.forward(h)

    def getWeights(self):
        numConvLayers = len(self.convLayers)
        numKernelsPerObs = 0
        if numConvLayers > 0:
            numKernelsPerObs = len(self.convLayers[0].getKernels())
            convLayersWeights = [None] * numConvLayers * numKernelsPerObs
        
        fcLayersWeights = [None] * (len(self.fcLayers) * 2)

        for convLayerIdx, convLayer in enumerate(self.convLayers):
            for kernelIdx, kernel in enumerate(convLayer.getKernels()):
                convLayersWeights[convLayerIdx*numKernelsPerObs + kernelIdx] = kernel

        for fcLayerIdx in range(0, len(self.fcLayers), 2):
            fcLayersWeights[fcLayerIdx] = self.fcLayers[fcLayerIdx].getWeights()
            fcLayersWeights[fcLayerIdx+1] = self.fcLayers[fcLayerIdx].getBiases()

        return convLayersWeights, numKernelsPerObs, fcLayersWeights
            
    #TODO - add an assert here that makes sure same num input weights and model weights
    def setWeights(self, convLayersWeights, numKernelsPerObs, fcLayersWeights):
        if (len(self.convLayers) != 0):
            for convLayerIdx in range(0, len(convLayersWeights), numKernelsPerObs):
                kernelsStartIdx = convLayerIdx * numKernelsPerObs
                kernelsEndIdx = kernelsStartIdx + numKernelsPerObs
                obsKernels = convLayersWeights[kernelsStartIdx:kernelsEndIdx]
                self.convLayers[convLayerIdx].setKernels(obsKernels)
        if (len(self.fcLayers) != 0):
            for idx in range(0, len(self.fcLayers), 2):
                self.fcLayers[idx].setWeights(fcLayersWeights[idx])
                self.fcLayers[idx].setBiases(fcLayersWeights[idx+1])