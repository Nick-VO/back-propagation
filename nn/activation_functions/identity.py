from nn.activation_functions.activation_function import ActivationFunction


class Identity(ActivationFunction):
    def activate(self, z: float):
        return z

    def derivative(self, z: float):
        return 1