import numpy as np

from nn.activation_functions.activation_function import ActivationFunction


class ELU(ActivationFunction):
    def __init__(self, alpha: float):
        self.__alpha = alpha

    def activate(self, z: float):
        return (self.__alpha * (np.exp(z) - 1)) if z < 0 else z

    def derivative(self, z: float):
        return (self.activate(z) + self.__alpha) if z < 0 else 1
