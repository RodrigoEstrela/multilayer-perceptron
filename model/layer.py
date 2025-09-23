import numpy as np


class Layer:
    """
    Layer class valid for input, hidden and output layers
    """

    def __init__(self, n_input_nodes: int = None, n_nodes: int = None,
                 activation: str = None, name: str = None,
                 weights: np.ndarray = None):
        # Check if the number of nodes in each layer is greater than 0 and less than 300
        if n_nodes is not None and (n_nodes <= 0 or n_nodes > 300):
            raise ValueError("Number of nodes in each layer must be greater than 0 and less than 300.")
        # Check if the number of input nodes is greater than 0 and less than 300
        if n_input_nodes is not None and (n_input_nodes <= 0 or n_input_nodes > 300):
            raise ValueError("Number of input nodes must be greater than 0 and less than 300.")
        
        self.activation_function = activation
        self.name = name

        self.n_input_nodes = n_input_nodes
        self.n_nodes = n_nodes

        # Initialize weights and bias
        if weights is not None:
            self.weights = weights
            self.bias = np.ones((1, weights.shape[1]))
        elif n_input_nodes is not None:
            self.weights = np.random.randn(n_input_nodes, n_nodes) * np.sqrt(2 / n_input_nodes)
            self.bias = np.ones((1, n_nodes))
        else:
            self.weights = np.array([])
            self.bias = np.array([])

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

    @staticmethod
    def add_layer(type: str = None, n_input=None, n_nodes=None):
        if type == 'input':
            return Layer(n_nodes=30)
        elif type == 'output':
            return Layer(n_input_nodes=n_input, n_nodes=2, activation='softmax')
        else:
            return Layer(n_input_nodes=n_input, n_nodes=n_nodes, activation='relu')

    def __str__(self) -> str:
        return f"Layer: {self.name}\nActivation: {self.activation_function}\n" \
               f"Input nodes: {self.n_input_nodes}\nLayer n_nodes: {self.n_nodes}\n" \
               f"Weights shape: {self.weights.shape}\nBias shape: {self.bias.shape}\n"
