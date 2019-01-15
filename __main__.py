from nn.activation_functions.logistic import Logistic
from nn.cost_functions.quadratic import QuadraticCost
from nn.network import Network
from nn.optimization_strategies.gradient_descent import GradientDescent

if __name__ == '__main__':
    learning_rate = 5
    activation = Logistic()
    cost = QuadraticCost()
    optimization = GradientDescent(learning_rate)
    nn = Network([2, 6, 1], activation, cost, optimization)

    episodes = 10000
    input_sample = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]

    output_sample = [
        [0],
        [1],
        [1],
        [0]
    ]

    nn.run(episodes, input_sample, output_sample)
