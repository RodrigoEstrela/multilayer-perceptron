import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys

from layer import Layer
from network import Network


def main():
    df = pd.read_csv(sys.argv[1])

    y = df.iloc[:, 1].values
    y = np.array([0 if label == 'M' else 1 for label in y])
    X = df.iloc[:, 2:32].values
    X = np.nan_to_num(X)
    scaler = MinMaxScaler(copy=False)
    X = scaler.fit_transform(X)

    # Define the network
    network = Network([
        # Input layer
        Layer(shape=('input_shape', X.shape[1]), activation='relu', input_size=1),
        # Hidden layer 1
        Layer(shape=20, activation='relu', input_size=30),
        # Hidden layer 2
        Layer(shape=20, activation='relu', input_size=20),
        # Output layer
        Layer(shape=2, activation='softmax', input_size=20)],
        # Features and labels
        features=X, labels=y
    )

    print(network.evaluate())


if __name__ == '__main__':
    main()
