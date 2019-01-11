from nn.cells.hidden_cell import HiddenCell
from nn.cells.input_cell import InputCell
from nn.cells.output_cell import OutputCell
from nn.network import Network
from nn.optimization_strategies.gradient_descent import GradientDescent

if __name__ == '__main__':
    inputs = [InputCell(), InputCell()]
    outputs = [OutputCell()]

    learning_rate = 1
    optimization = GradientDescent(learning_rate)
    nn = Network(inputs, outputs, optimization)

    hidden1 = HiddenCell()
    hidden2 = HiddenCell()

    nn.add_connected_hidden(hidden1, inputs[0], outputs[0])
    nn.add_connected_hidden(hidden2, inputs[0], outputs[0])
    nn.add_connection(inputs[1], hidden1)
    nn.add_connection(inputs[1], hidden2)

    episodes = 100000
    input_samples = [
        [0.001, 0.001],
        [0.001, 1],
        [1, 0.001],
        [1, 1]
    ]
    desired_outputs = [
        [0.001],
        [.5],
        [.5],
        [0.001],
    ]

    nn.run(episodes, input_samples, desired_outputs)
