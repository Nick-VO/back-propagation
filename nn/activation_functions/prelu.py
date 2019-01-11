from nn.activation_functions.activation_function import ActivationFunction


class PReLU(ActivationFunction):
    def __init__(self, alpha: float):
        self.__alpha = alpha

    def activate(self, z: float):
        return z * (self.__alpha if z < 0 else 1)

    def derivative(self, z: float):
        return self.__alpha if z < 0 else 1
