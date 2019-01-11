from nn.activation_functions.activation_function import ActivationFunction


class BinaryStep(ActivationFunction):
    def activate(self, z: float):
        return 0 if z < 0 else 1

    def derivative(self, z: float):
        return 0 if z != 0 else float('Inf')
