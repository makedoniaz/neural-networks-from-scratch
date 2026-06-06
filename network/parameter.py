class Parameter:
    def __init__(self, data, weight_decay = True):
        self.data = data
        self.grad = None
        self.weight_decay = weight_decay