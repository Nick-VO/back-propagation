from random import random

from nn.activation_functions.activation_function import ActivationFunction
from nn.activation_functions.logistic import Logistic
from nn.cost_functions.cost_function import CostFunction
from nn.cost_functions.quadratic import QuadraticCost


class Cell(object):
    def __init__(self, bias: float = random(), activation_function: ActivationFunction = Logistic(),
                 cost_function: CostFunction = QuadraticCost()):
        self.__inputs = []
        self.__outputs = []
        self.__bias = bias
        self.__weighted_input = .0
        self.__prev_weighted_input = .0
        self.__activation = .0
        self.__prev_activation = .0
        self.__cost = .0
        self.__prev_cost = .0
        self.__error = .0
        self.__activation_function = activation_function
        self.__cost_function = cost_function

    @property
    def inputs(self):
        return self.__inputs

    @property
    def outputs(self):
        return self.__outputs

    @property
    def bias(self):
        return self.__bias

    @property
    def weighted_input(self):
        return self.__weighted_input

    @property
    def activation(self):
        return self.__activation

    @activation.setter
    def activation(self, value):
        self.__prev_activation = self.activation
        self.__activation = value

    @property
    def prev_activation(self):
        return self.__prev_activation

    @property
    def cost(self):
        return self.__cost

    @cost.setter
    def cost(self, value):
        self.__prev_cost = self.cost
        self.__cost = value

    @property
    def prev_cost(self):
        return self.__prev_cost

    @property
    def error(self):
        return self.__error

    @error.setter
    def error(self, value):
        self.__error = value

    @property
    def activation_function(self):
        return self.__activation_function

    @property
    def cost_function(self):
        return self.__cost_function

    def calc_activation(self):
        # get weighted input and activation
        self.__prev_weighted_input = self.__weighted_input
        self.__weighted_input = sum(i.weight * i.input_cell.activation for i in self.inputs) + self.bias
        self.__activation = self.activation_function.activate(self.weighted_input)

    def fire(self):
        # check if all inputs have been fired
        if not all(i.fired for i in self.inputs):
            return

        # get weighted input and activation
        self.calc_activation()

        # set all inputs to unfired
        for i in self.inputs:
            i.fired = False

        # fire outputs
        for o in self.outputs:
            o.fire()

    def calc_error(self):
        s = self.activation_function.derivative(self.weighted_input)
        self.__error = sum(o.weight * o.output_cell.error for o in self.outputs) * s
        return self.error

    def improve(self, optimization):
        # calculate error, bias change = error
        bias_change = self.calc_error()
        self.__bias -= optimization.optimize(self.bias, bias_change)

        # optimize input connections
        for i in self.inputs:
            i.improve(optimization)
