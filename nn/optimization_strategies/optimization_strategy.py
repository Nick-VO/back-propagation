class OptimizationStrategy(object):
    def __init__(self, learning_rate: float):
        self.__learning_rate = learning_rate

    @property
    def learning_rate(self):
        return self.__learning_rate

    def optimize(self, theta: float, delta: float):
        raise NotImplementedError()
