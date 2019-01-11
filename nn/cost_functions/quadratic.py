from nn.cost_functions.cost_function import CostFunction


class QuadraticCost(CostFunction):
    def calculate(self, actual_output: float, desired_output: float):
        return .5 * (desired_output - actual_output) ** 2
