import numpy as np

from nn.activation_functions.activation_function import ActivationFunction


class ArcTan(ActivationFunction):
    def activate(self, z: float):
        return np.arctan(z)

    def derivative(self, z: float):
        return 1 / (z ** 2 + 1)
