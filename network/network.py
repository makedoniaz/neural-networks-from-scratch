from layers import Layer
import numpy as np
import pickle
import os

class Network:
    """Neural network container that manages layers, forward/backward propagation, and model persistence.

    Args:
        layers: List of Layer objects forming the network.
        model_name: Name for the model (used when saving). Defaults to 'model_name'.
    """

    def __init__(self, layers: list[Layer], model_name: str = "model_name") -> None:
        self.layers = layers
        self.model_name = model_name
    
    def train(self) -> None:
        """Set all layers to training mode."""
        for layer in self.layers:
            layer.training = True

    def eval(self) -> None:
        """Set all layers to evaluation mode."""
        for layer in self.layers:
            layer.training = False

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through all layers sequentially.

        Args:
            X: Input array.

        Returns:
            Output from the final layer.
        """
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Backward pass through all layers in reverse order.

        Args:
            dout: Gradient of loss with respect to network output.

        Returns:
            Gradient with respect to network input.
        """
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def get_parameter_values(self) -> list:
        """Get a copy of all trainable parameter values.

        Returns:
            List of parameter data arrays.
        """
        return [
            param.data.copy()
            for param in self.parameters()
        ]
    
    def parameters(self) -> list:
        """Collect all trainable parameters from all layers.

        Returns:
            List of Parameter objects.
        """
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    
    def set_parameters(self, parameters: list) -> None:
        """Set parameter values for the entire network.

        Args:
            parameters: List of parameter data arrays to assign.
        """
        for param, data in zip(self.parameters(), parameters):
            param.data[:] = data
    
    def save_model(self, directory: str = "models") -> None:
        """Save model parameters and metadata to disk.
        
        Args:
            directory: Directory path where model will be saved. Defaults to 'models'.
        """

        if not os.path.exists(directory):
            os.makedirs(directory)

        path = os.path.join(directory, f"{self.model_name}.p")

        state = {
            "model_name": self.model_name,
            "parameters": self.get_parameter_values(),
        }

        with open(path, "wb") as file:
            pickle.dump(state, file)

    @classmethod
    def load_model(
        cls,
        model: "Network",
        model_name: str,
        directory: str = "models",
    ) -> "Network":
        """Load parameters into an existing model.
        
        Args:
            model_name: Name of the model to load (without .p extension).
            directory: Directory path where model is saved. Defaults to 'models'.

        Returns:
            Loaded Network instance.

        Raises:
            FileNotFoundError: If model file does not exist.
        """

        path = os.path.join(directory, f"{model_name}.p")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, "rb") as file:
            state = pickle.load(file)

        model.set_parameters(state["parameters"])
        model.model_name = state["model_name"]

        return model

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)
    
    def __repr__(self) -> str:
        layers = "\n".join(
            f"  ({i}): {layer}"
            for i, layer in enumerate(self.layers)
        )
        return f"Network(\n{layers}\n)"