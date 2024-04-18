import numpy as np


def he_initialization(input_size):
    """Initialize weights using He initialization."""
    # Scale factor for He initialization
    scale = np.sqrt(2.0 / input_size)
    # Draw weights from a normal distribution with mean 0 and standard deviation scale
    return np.random.normal(0, scale, input_size)


class Node:
    """
    Node class valid for input, hidden and output nodes
    """

    def __init__(self, shape: int | tuple, activation, input_size):
        self.output = None
        if isinstance(shape, tuple):
            self.weights = np.ones(1)
            self.delta = np.zeros(1)
        else:
            self.weights = he_initialization(input_size)
            self.delta = np.zeros(shape)
        self.activation_function = activation

    def activation(self, x):
        """
        Activation function
        """
        if self.activation_function == 'relu':
            return np.maximum(0, x)
        elif self.activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation_function == 'softmax':
            exps = np.exp(x - np.max(x))
            return exps / np.sum(exps, axis=0)
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

    def __str__(self):
        return (f'--------------------Node--------------------'
                f'\nWeights: {self.weights}\nActivation function: {self.activation_function}')

