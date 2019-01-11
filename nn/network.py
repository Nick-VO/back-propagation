from nn.cells.cell import Cell
from nn.cells.input_cell import InputCell
from nn.cells.output_cell import OutputCell
from nn.connection import Connection
from nn.optimization_strategies.optimization_strategy import OptimizationStrategy


class Network(object):
    def __init__(self, inputs: [InputCell], outputs: [OutputCell],
                 optimization: OptimizationStrategy):
        self.__inputs = inputs
        self.__outputs = outputs
        self.__hidden = []
        self.__optimization = optimization

    @property
    def inputs(self):
        return self.__inputs

    @property
    def outputs(self):
        return self.__outputs

    @property
    def hidden(self):
        return self.__hidden

    def add_connection(self, input_cell: Cell, output_cell: Cell):
        if not (input_cell in self.inputs or input_cell in self.hidden):
            raise ValueError('input cell not present in network')
        if not (output_cell in self.outputs or output_cell in self.hidden):
            raise ValueError('output cell not present in network')

        conn = Connection(input_cell, output_cell)
        input_cell.outputs.append(conn)
        output_cell.inputs.append(conn)

    def add_hidden(self, new_cell: Cell):
        self.__hidden.append(new_cell)

    def add_connected_hidden(self, new_cell: Cell, input_cell: Cell, output_cell: Cell):
        self.add_hidden(new_cell)
        self.add_connection(input_cell, new_cell)
        self.add_connection(new_cell, output_cell)

    def feedforward(self, inputs: [float]):
        # set inputs and fire them
        for ic, i in zip(self.inputs, inputs):
            ic.activation = i
            ic.fire()

        return [o.activation for o in self.outputs]

    def backpropagate(self, desired_outputs):
        # set desired outputs and improve the weights
        for oc, o in zip(self.outputs, desired_outputs):
            oc.desired_output = o
            oc.improve(self.__optimization)

    def run(self, episodes: int, input_samples: [[float]], desired_outputs: [[float]]):
        for n in range(episodes):
            print('\n\nEpisode ', n + 1)
            for input_sample, output_sample in zip(input_samples, desired_outputs):
                if len(self.inputs) != len(input_sample):
                    raise ValueError('wrong amount of input values, expected ', len(self.inputs),
                                     ' but got ', len(input_sample))
                if len(self.outputs) != len(output_sample):
                    raise ValueError('wrong amount of output values, expected ', len(self.outputs),
                                     ' but got ', len(output_sample))

                ff_output = self.feedforward(input_sample)
                self.backpropagate(output_sample)

                print('\nInputs:  ', input_sample)
                print('Outputs: ', ff_output)
                print('Desired: ', output_sample)
