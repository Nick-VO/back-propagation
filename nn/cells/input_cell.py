from nn.cells.cell import Cell


class InputCell(Cell):
    def __init__(self):
        super().__init__()
        self.__input_value = .0

    @property
    def inputs(self):
        return []

    @property
    def activation(self):
        return self.__input_value

    @activation.setter
    def activation(self, value):
        self.__input_value = value

    def calc_activation(self):
        pass
