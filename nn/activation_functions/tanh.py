import numpy as np

from nn.activation_functions.activation_function import ActivationFunction


class TanH(ActivationFunction):
    def activate(self, z: float):
        return np.tanh(z)

    def derivative(self, z: float):
        return 1 - self.activate(z) ** 2
