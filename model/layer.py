import numpy as np


class Layer:
    """
    Layer class valid for input, hidden and output layers
    """

    def __init__(self, n_input_nodes: int = None, n_current_layer_nodes: int = None,  activation: str = None, name: str = None):
        self.activation_function = activation
        self.name = name

        self.n_input_nodes = n_input_nodes
        self.n_output_nodes = n_current_layer_nodes

        # Initialize weights only if the layer is not the input layer
        if n_input_nodes is not None:
            self.weights = np.random.randn(n_input_nodes, n_current_layer_nodes) * np.sqrt(2 / n_input_nodes)
            self.bias = np.random.uniform(-1, 1, (1, n_current_layer_nodes))
        else:
            self.weights = None
            self.bias = None

        self.output = None

    def activation(self, x):
        """
        Activation function
        """
        if self.activation_function == 'relu':
            return np.maximum(0, x)
        elif self.activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation_function == 'softmax':
            # Subtract the maximum value to avoid overflow and underflow
            x = x - np.max(x)

            # Clip the array to avoid very large negative numbers
            x = np.clip(x, -500, 500)

            # Compute the softmax activation
            exps = np.exp(x)
            if exps.ndim == 1:
                return exps / np.sum(exps, axis=1)
            else:
                return exps / np.sum(exps, axis=1, keepdims=True)
        else:
            raise ValueError('Activation function not supported')

    def activation_derivative(self, x):
        """
        Derivative of the activation function
        """
        if self.activation_function == 'relu':
            return np.where(x <= 0, 0, 1)
        elif self.activation_function == 'sigmoid':
            return x * (1 - x)
        elif self.activation_function == 'softmax':
            return x * (1 - x)
        else:
            raise ValueError('Activation function not supported')
