import numpy as np

from nn.cost_functions.cost_function import CostFunction


class HellingerDistance(CostFunction):
    def calculate(self, actual_output: float, desired_output: float):
        sqrt_a = np.sqrt(actual_output)
        return (1 / np.sqrt(2)) * (sqrt_a - np.sqrt(desired_output)) ** 2
