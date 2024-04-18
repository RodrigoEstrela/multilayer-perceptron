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

    def __init__(self, shape: int | str, activation, features_size):
        if isinstance(shape, str):
            self.weights = np.ones(features_size)
        else:
            self.weights = he_initialization(shape)
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

    def __init__(self, shape: str | int, activation: str, size=0):
        if isinstance(shape, str):
            self.nodes = [Node(shape, activation, size) for _ in range(size)]
        else:
            self.nodes = [Node(shape, activation, size) for _ in range(shape)]

    def __str__(self):
        # Print each one of the nodes of the layer
        return ('\n====================================LAYER====================================\n'
                + '\n'.join([str(node) for node in self.nodes]))


class Network:
    """
    multilayer-perceptron model
    """

    def __init__(self, layers, features, labels):
        self.layers = layers
        self.features = features
        self.labels = labels

    def feedforward(self):
        """
        Forward pass through the network
        """
        X = self.features
        for layer in self.layers:
            # Initialize list to store layer outputs
            layer_outputs = []
            # Iterate over nodes in the layer
            for node in layer.nodes:
                # Transpose weights matrix before multiplying with inputs
                # print(node.weights.shape, X.shape)
                weighted_sum = node.weights @ X.T
                # Apply activation function
                activation_output = node.activation(weighted_sum)
                # Append output to layer_outputs
                print(activation_output.shape)
                layer_outputs.append(activation_output)
            # Set the output of the layer as the input of the next layer
            print(X.shape)
            X = np.array(layer_outputs)
            print(X.shape)
        return X

    def predict(self):
        """
        Predict the class of each sample
        """
        # Get the output of the last layer
        output = self.feedforward()
        # Return the index of the node with the highest value
        return np.argmax(output, axis=0)

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
        Layer(shape='input_shape', activation='relu', size=X.shape[1]),
        # Hidden layer 1
        Layer(shape=20, activation='relu'),
        # Output layer
        Layer(shape=2, activation='softmax')],
        # Features and labels
        features=X, labels=y
    )

    print(network.feedforward())


if __name__ == '__main__':
    main()
