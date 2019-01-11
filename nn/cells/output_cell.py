from random import random

from nn.activation_functions.activation_function import ActivationFunction
from nn.activation_functions.logistic import Logistic
from nn.cells.cell import Cell
from nn.cost_functions.cost_function import CostFunction
from nn.cost_functions.quadratic import QuadraticCost


class OutputCell(Cell):
    def __init__(self, bias: float = random(), activation_function: ActivationFunction = Logistic(),
                 cost_function: CostFunction = QuadraticCost()):
        super().__init__(bias, activation_function, cost_function)
        self.__desired_output = .0

    @property
    def desired_output(self):
        return self.__desired_output

    @desired_output.setter
    def desired_output(self, value):
        self.__desired_output = value

    @property
    def outputs(self):
        return []

    def calc_error(self):
        self.cost = self.cost_function.calculate(self.activation, self.desired_output)
        self.error = ((self.cost - self.prev_cost) / (self.activation - self.prev_activation)) * \
                     self.activation_function.derivative(self.weighted_input)
        return self.error
