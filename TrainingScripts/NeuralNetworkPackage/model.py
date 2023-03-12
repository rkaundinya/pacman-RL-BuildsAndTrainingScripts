from . layers import Layer
from . convolutionLayer import ConvolutionalLayer
from . fullyConnectedLayer import FullyConnectedLayer
from . objectiveLayer import ObjectiveLayer
import numpy as np

class Model:
    def __init__(self, layers=[], eta=0.01):
        self.layers = layers
        self.eta = eta
        return
    
    def add(self, layer):
        if (isinstance(layer, ObjectiveLayer) or isinstance(layer, Layer)):
            self.layers.append(layer)
        else:
            TypeError(layer)
    
    #Inputs: x, input to network
    #Inputs: y, expected output for input
    #Executes a single forward/backward prop training of model
    def train(self, x, y):
        prediction = self.predict(x)
        loss = 0

        if (isinstance(self.layers[-1], ObjectiveLayer)):
            loss = self.layers[-1].eval(y, prediction)

        print(loss)

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