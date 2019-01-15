from nn.cost_functions.cost_function import CostFunction


class QuadraticCost(CostFunction):
    def calculate(self, actual_output, desired_output):
        return .5 * (desired_output - actual_output) ** 2

    def gradient(self, actual_output, desired_output):
        return actual_output - desired_output
