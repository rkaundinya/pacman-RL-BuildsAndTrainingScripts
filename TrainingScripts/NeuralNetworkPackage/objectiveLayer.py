from abc import ABC, abstractmethod

class ObjectiveLayer(ABC):
    def __init__(self):
        return

    @abstractmethod
    def eval(self, Y, Yhat):
        pass

    @abstractmethod
    def gradient(self, Y, Yhat):
        pass