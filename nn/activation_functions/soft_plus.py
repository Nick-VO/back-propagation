import numpy as np

from nn.activation_functions.activation_function import ActivationFunction


class SoftPlus(ActivationFunction):
    def activate(self, z: float):
        return np.log(1 + np.exp(z))

    def derivative(self, z: float):
        return 1 / (1 + np.exp(-z))
