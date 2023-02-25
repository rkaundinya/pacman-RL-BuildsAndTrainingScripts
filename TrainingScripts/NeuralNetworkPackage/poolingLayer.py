from layers import Layer
import numpy as np

class PoolingLayer(Layer):
    #Input: qRows, rows in window Q
    #Input: qCols, cols window Q
    #Input: stride, the stride length of pooling window
    #Output: None
    def __init__(self, qRows, qCols, stride):
        self.qRows = qRows
        self.qCols = qCols
        self.stride = stride

    def forward(self, dataIn):
        pass

    def gradient(self):
        pass

    def backward(self, gradIn):
        pass