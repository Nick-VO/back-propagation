class ActivationFunction(object):
    def activate(self, z):
        raise NotImplementedError()

    def derivative(self, z):
        raise NotImplementedError()
