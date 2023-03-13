from NeuralNetworkPackage.inputLayer import InputLayer
from NeuralNetworkPackage.convolutionLayer import ConvolutionalLayer
from NeuralNetworkPackage.flatteningLayer import FlatteningLayer
from NeuralNetworkPackage.poolingLayer import PoolingLayer
from NeuralNetworkPackage.maxPoolingCalc import MaxPoolingCalc
from NeuralNetworkPackage.fullyConnectedLayer import FullyConnectedLayer
from NeuralNetworkPackage.linearLayer import LinearLayer
from NeuralNetworkPackage.squaredErrorLayer import SquaredErrorLayer
import numpy as np

#X Input Image
row1 = np.array([[1,1,0,1,0,0,1,1]])
row2 = np.array([[1,1,1,1,0,0,1,0]])
row3 = np.array([[0,0,1,1,0,1,0,1]])
row4 = np.array([[1,1,1,0,1,1,1,0]])
row5 = np.array([[1,1,1,1,1,0,1,1]])
row6 = np.zeros((1,8))
row7 = np.array([[0,1,1,1,1,0,0,1]])
row8 = np.array([[1,0,1,0,0,1,0,1]])

X = np.array([np.concatenate((row1, row2, row3, row4, row5, row6, row7, row8), axis=0)])
Y = np.array([[5]], dtype=float)

FCLAYER_INPUTNUM = 8

#Set up layers
il = InputLayer(X, False)
cl = ConvolutionalLayer(3)
mpc = MaxPoolingCalc()
pl = PoolingLayer(3, 3, 3, mpc)
fl = FlatteningLayer()
fcl = FullyConnectedLayer(FCLAYER_INPUTNUM,1)
lal = LinearLayer()
sel = SquaredErrorLayer()

#Setting weights
clKernel = np.array([[[2, -1, 2], [2, -1, 0], [1, 0, 2]], [[2, -1, 2], [2, -1, 0], [1, 0, 2]]], dtype=float)
fclWeights = np.array([[-1],[0],[3],[-1],[-1],[0],[3],[-1]])

#clKernel = np.array([[[2, -1, 2], [2, -1, 0], [1, 0, 2]]], dtype=float)
#fclWeights = np.array([[-1],[0],[3],[-1]])
fclBiases = np.array([[0]])

cl.setKernel(clKernel)
fcl.setWeights(fclWeights)
fcl.setBiases(fclBiases)

layers = [il, cl, pl, fl, fcl, lal, sel]

h = X
squaredErrorLoss = 0

eta = 1

for layer in layers:
    if (isinstance(layer, SquaredErrorLayer)):
        squaredErrorLoss = layer.eval(Y, h)
        continue

    h = layer.forward(h)

print(squaredErrorLoss)

gradient = layers[-1].gradient(Y, h)

for i in range(len(layers) - 2, 0, -1):
    newGradient = layers[i].backward(gradient)

    if (isinstance(layers[i], FullyConnectedLayer) or isinstance(layers[i], ConvolutionalLayer)):
        layers[i].updateWeights(gradient, eta)

    gradient = newGradient