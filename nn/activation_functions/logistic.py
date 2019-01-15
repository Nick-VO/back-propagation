import numpy as np

from nn.activation_functions.activation_function import ActivationFunction


class Logistic(ActivationFunction):
    def activate(self, z):
        x = 1 / (1 + np.exp(-z))
        return x

    def derivative(self, z):
        return self.activate(z) * (1 - self.activate(z))
