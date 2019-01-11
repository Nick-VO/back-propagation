import numpy as np

from nn.activation_functions.activation_function import ActivationFunction


class Logistic(ActivationFunction):
    def activate(self, z: float):
        return 1 / (1 + np.exp(-z))

    def derivative(self, z: float):
        return self.activate(z) * (1 - self.activate(z))
