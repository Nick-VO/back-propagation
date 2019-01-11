from nn.optimization_strategies.optimization_strategy import OptimizationStrategy


class GradientDescent(OptimizationStrategy):
    def optimize(self, theta: float, delta: float):
        return self.learning_rate * delta
