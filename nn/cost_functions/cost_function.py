class CostFunction(object):
    def calculate(self, actual_output, desired_output):
        raise NotImplementedError()

    def gradient(self, actual_output, desired_output):
        raise NotImplementedError()
