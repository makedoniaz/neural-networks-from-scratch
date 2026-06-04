from .base_layer import Layer

class Flatten(Layer):
    def forward(self, X):
        self.cache = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, dout):
        original_shape = self.cache
        return dout.reshape(original_shape)