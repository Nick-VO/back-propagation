from random import random

import numpy as np

from nn.activation_functions.activation_function import ActivationFunction
from nn.cost_functions.cost_function import CostFunction
from nn.optimization_strategies.optimization_strategy import OptimizationStrategy


class Network(object):
    def __init__(self, topology: [[int]],
                 activation_function: ActivationFunction,
                 cost_function: CostFunction,
                 optimization_strategy: OptimizationStrategy):
        self.__num_layers = len(topology)
        self.__topology = topology
        self.__a = [np.zeros(l) for l in topology]
        self.__z = [np.zeros(l) for l in topology]
        self.__b = [np.ones(l) for l in topology]
        self.__w = [np.array([[random() for k in range(topology[l - 1])] for j in range(layer)]) for l, layer in
                    enumerate(topology[1:], 1)]
        self.__delta = [np.zeros(layer) for l, layer in enumerate(self.__topology)]
        self.__activation_function = activation_function
        self.__cost_function = cost_function
        self.__optimization_strategy = optimization_strategy

    @property
    def a(self):
        return self.__a

    @property
    def z(self):
        return self.__z

    @property
    def b(self):
        return self.__b

    @property
    def w(self):
        return self.__w

    @property
    def delta(self):
        return self.__delta

    @property
    def sigma(self):
        return self.__activation_function.activate

    @property
    def delta_sigma(self):
        return self.__activation_function.derivative

    @property
    def C(self):
        return self.__cost_function.calculate

    @property
    def delta_C(self):
        return self.__cost_function.gradient

    @property
    def O(self):
        return self.__optimization_strategy.optimize

    def run(self, episodes: int, inputs: [[float]], outputs: [[float]]):
        for n in range(episodes):
            print('\n\nEpisode ', n + 1)

            for i, o in zip(inputs, outputs):
                self.feedforward(i)
                self.backpropagate(o)

                print('\nInputs:  ', i)
                print('Outputs: ', self.a[-1])
                print('Desired: ', o)

    def feedforward(self, inputs: [float]):
        for i, ic in enumerate(self.a[0]):
            self.a[0][i] = inputs[i]

        for l in range(1, self.__num_layers):
            self.z[l] = (self.w[l - 1] @ self.a[l - 1]) + self.b[l]
            self.a[l] = self.sigma(self.z[l])

    def backpropagate(self, outputs: [float]):
        # output layer cost
        L = -1
        self.delta[L] = self.delta_C(self.a[L], np.array(outputs).T) * self.delta_sigma(self.z[L])

        # hidden layers
        for l in range(2, self.__num_layers):
            self.delta[-l] = (self.w[-l + 1].T @ self.delta[-l + 1]) * self.delta_sigma(self.z[-l])

        # update weights and biases
        for l in range(self.__num_layers - 1):
            delta = self.delta[l + 1][np.newaxis]
            self.w[l] -= self.O(self.w[l], self.a[l] * delta.T)
            self.b[l] -= self.O(self.b[l], self.delta[l])

        """
        for l, wl in enumerate(self.w):
            for j, wj in enumerate(wl):
                for k, w in enumerate(wj):
                    delta_w = self.a[l][k] * self.delta[l + 1][j]
                    self.w[l][j][k] -= self.O(self.w[l][j][k], delta_w)
                delta_b = self.delta[l + 1][j]
                self.b[l] -= self.O(self.b[l], delta_b)
        """
