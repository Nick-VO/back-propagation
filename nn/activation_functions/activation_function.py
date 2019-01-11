class ActivationFunction(object):
    def activate(self, z: float):
        raise NotImplementedError()

    def derivative(self, z: float):
        raise NotImplementedError()
