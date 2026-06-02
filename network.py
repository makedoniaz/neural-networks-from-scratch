import pickle
import os

class Network():
    def __init__(self, layers, model_name="model_name"):
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
    
    def save_model(self, directory="models"):
        path = os.path.join(directory, f"{self.model_name}.p")

        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(path, 'wb') as file:
            pickle.dump(self, file)