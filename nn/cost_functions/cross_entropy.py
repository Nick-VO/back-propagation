import numpy as np

from nn.cost_functions.cost_function import CostFunction


class CrossEntropyCost(CostFunction):
    def calculate(self, actual_output: float, desired_output: float):
        return desired_output * np.log(actual_output) + (1 - desired_output) * np.log(1 - actual_output)
