from NeuralNetworkPackage.layers import Layer
from NeuralNetworkPackage.inputLayer import InputLayer
from NeuralNetworkPackage.convolutionLayer import ConvolutionalLayer
from NeuralNetworkPackage.flatteningLayer import FlatteningLayer
from NeuralNetworkPackage.poolingLayer import PoolingLayer
from NeuralNetworkPackage.maxPoolingCalc import MaxPoolingCalc
from NeuralNetworkPackage.fullyConnectedLayer import FullyConnectedLayer
from NeuralNetworkPackage.softmaxActivationLayer import SoftmaxActivationLayer
from NeuralNetworkPackage.crossEntropyLayer import CrossEntropyLayer
from NeuralNetworkPackage.objectiveLayer import ObjectiveLayer
import numpy as np

X = np.array([np.random.randint(8, size=168).reshape((12,14))])

#Set up layers
il = InputLayer(X, False)
cl = ConvolutionalLayer(3)
mpc = MaxPoolingCalc()
pl = PoolingLayer(3, 3, 2, mpc)
fl = FlatteningLayer()
fcl = FullyConnectedLayer(20,4)
sal = SoftmaxActivationLayer()
cel = CrossEntropyLayer()

layers = [il, cl, pl, fl, fcl, sal, cel]

h = X
crossEntropyLoss = 0

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
    def train(self, x, y):
        prediction = self.predict(x)
        loss = 0

        if (isinstance(self.layers[-1], ObjectiveLayer)):
            loss = self.layers[-1].eval(y, prediction)

        print(loss)

        gradient = layers[-1].gradient(y, prediction)

        for i in range(len(layers) - 2, 0, -1):
            newGradient = layers[i].backward(gradient)

            if (isinstance(layers[i], FullyConnectedLayer) or isinstance(layers[i], ConvolutionalLayer)):
                layers[i].updateWeights(gradient, self.eta)

            gradient = newGradient
    
    def predict(self, input):
        h = input

        for layer in self.layers:
            if (isinstance(layer, ObjectiveLayer)):
                return h

            h = layer.forward(h)

'''
Mocking up a model class and test code
modelTest = Model()
for layer in layers:
    modelTest.add(layer)

prediction = modelTest.predict(X)
Y = np.array([[0,0,1,0]])
modelTest.train(X, Y)
'''