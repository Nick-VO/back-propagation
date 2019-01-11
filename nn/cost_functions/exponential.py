import numpy as np

from nn.cells.cell import Cell
from nn.cost_functions.cost_function import CostFunction


class ExponentialCost(CostFunction):
    def __init__(self, cell: Cell, tau: float):
        super().__init__(cell)
        self.__tau = tau

    def calculate(self, actual_output: float, desired_output: float):
        return self.__tau * np.log((actual_output - desired_output) ** 2)
