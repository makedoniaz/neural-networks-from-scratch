
from layers import Layer
import pickle
import os

class Network():
    def __init__(self, layers: list[Layer], model_name="model_name"):
        self.layers = layers
        self.model_name = model_name
    
    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def get_parameter_values(self):
        return [
            param.data.copy()
            for param in self.parameters()
        ]
    
    def parameters(self):
        params = []

        for layer in self.layers:
            params.extend(layer.parameters())

        return params
    
    def set_parameters(self, parameters):
        for param, data in zip(self.parameters(), parameters):
            param.data[:] = data
    
    def save_model(self, directory="models"):
        path = os.path.join(directory, f"{self.model_name}.p")

        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(path, 'wb') as file:
            pickle.dump(self, file)

    def __call__(self, X):
        return self.forward(X)
    
    def __repr__(self):
        layers = "\n".join(
            f"  ({i}): {layer}"
            for i, layer in enumerate(self.layers)
        )

        return f"Network(\n{layers}\n)"