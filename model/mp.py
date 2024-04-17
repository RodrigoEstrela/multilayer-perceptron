import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def he_initialization(input_size):
    """Initialize weights using He initialization."""
    # Scale factor for He initialization
    scale = np.sqrt(2.0 / input_size)
    # Draw weights from a normal distribution with mean 0 and standard deviation scale
    return np.random.normal(0, scale, input_size)


class node:
    """
    node class with things of nodes
    """
    weights = []
    activation_lvl : int
    

class layer:
    """
    layers with things
    """

    def create_layer(self, shape, activiation, weights_initializer=0):
        self.shape = shape
        self.activation = activiation
        self.weights_initializer = weights_initializer


class model:
    """
    multilayer-perceptron model
    """
    layers = []
    input_size = 0
    output_size = 0

    def __init__():
        layers =



if __name__ == '__main__':
    df = pd.read_csv('../data/train.csv')

    y = df.iloc[:, 1].values
    y = np.array([0 if label == 'M' else 1 for label in y])
    X = df.iloc[:, 2:31].values
    X = np.nan_to_num(X)
    scaler = MinMaxScaler(copy=False)
    X = scaler.fit_transform(X)
