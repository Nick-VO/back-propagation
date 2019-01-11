import numpy as np

from nn.cost_functions.cost_function import CostFunction


class GeneralizedKullbackLeiblerDivergence(CostFunction):
    def calculate(self, actual_output: float, desired_output: float):
        return desired_output * np.log10(desired_output / actual_output) - desired_output + actual_output
