import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys


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

    def __init__(self, inputs: int, activation):
        self.weights = he_initialization(inputs)
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

    def __str__(self):
        return (f'--------------------Node--------------------'
                f'\nWeights: {self.weights}\nActivation function: {self.activation_function}')


class Layer:
    """
    Layer class valid for input, hidden and output layers
    """

    def __init__(self, shape: str | int, activation: str):
        self.nodes = [Node(shape, activation) for _ in range(shape)]

    def __str__(self):
        # Print each one of the nodes of the layer
        return ('\n====================================LAYER====================================\n'
                + '\n'.join([str(node) for node in self.nodes]))


class Network:
    """
    multilayer-perceptron model
    """

    def __init__(self, layers):
        self.layers = layers

    def feedforward(self, X):
        """
        Forward pass through the network
        """
        for layer in self.layers:
            # Run activation function for each node in the layer
            layer_output = np.array([node.activation(node.weights @ X) for node in layer.nodes])
            # Set the output of the layer as the input of the next layer
            X = layer_output
        return X

    def __str__(self):
        # Print each one of the layers of the network
        return '\n'.join([str(layer) for layer in self.layers])


def main():
    df = pd.read_csv(sys.argv[1])

    y = df.iloc[:, 1].values
    y = np.array([0 if label == 'M' else 1 for label in y])
    X = df.iloc[:, 2:31].values
    X = np.nan_to_num(X)
    scaler = MinMaxScaler(copy=False)
    X = scaler.fit_transform(X)

    # Define the network
    network = Network([
        # Input layer
        Layer(shape=X.shape[1], activation='relu'),
        # Hidden layer 1
        Layer(shape=20, activation='relu'),
        # Output layer
        Layer(shape=2, activation='softmax')
    ])

    print(network)


if __name__ == '__main__':
    main()
