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

    network = Network([
        # Input layer
        Layer(shape=('input_shape', X.shape[1]), activation='relu', input_size=1),
        # Hidden layer 1
        Layer(shape=24, activation='relu', input_size=30),
        # Hidden layer 2
        Layer(shape=24, activation='relu', input_size=24),
        # # Hidden layer 3
        # Layer(shape=24, activation='relu', input_size=24),
        # Output layer
        Layer(shape=2, activation='softmax', input_size=24)],
        # Features and labels
        features=X, labels=y
    )

    network.fit(epochs=100, learning_rate=0.014)
    print("Training Finished")
    print(network.evaluate())

    network.plot_cost()

    """
    # Define the networks
    networks = [Network([
        # Input layer
        Layer(shape=('input_shape', X.shape[1]), activation='relu', input_size=1),
        # Hidden layer 1
        Layer(shape=20, activation='relu', input_size=30),
        # Hidden layer 2
        Layer(shape=20, activation='relu', input_size=20),
        # Hidden layer 3
        Layer(shape=20, activation='relu', input_size=20),
        # Output layer
        Layer(shape=2, activation='softmax', input_size=20)],
        # Features and labels
        features=X, labels=y
    ) for _ in range(5)]

    for i in range(5):
        networks[i].train_loop(epochs=1200)
        print("Training Finished")
        print(networks[i].evaluate())
    """


if __name__ == '__main__':
    main()
