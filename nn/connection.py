from random import random


class Connection(object):
    def __init__(self, input_cell, output_cell, weight: float = random()):
        self.__input_cell = input_cell
        self.__output_cell = output_cell
        self.__weight = weight
        self.__fired = False

    @property
    def input_cell(self):
        return self.__input_cell

    @property
    def output_cell(self):
        return self.__output_cell

    @property
    def weight(self):
        return self.__weight

    @property
    def fired(self):
        return self.__fired

    @fired.setter
    def fired(self, value):
        self.__fired = value

    def fire(self):
        self.fired = True
        self.output_cell.fire()

    def improve(self, optimization):
        # improve weight
        weight_change = self.input_cell.activation * self.output_cell.error
        self.__weight -= optimization.optimize(self.__weight, weight_change)

        # improve input cell
        self.input_cell.improve(optimization)
