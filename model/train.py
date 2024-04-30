import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

from layer import Layer
from network import Network


def main():
    # make directory model_save
    if not os.path.exists('model_save'):
        os.makedirs('model_save')
    try:
        # Training data preprocessing
        df = pd.read_csv(sys.argv[1])
        y = df.iloc[:, 1]
        y = np.array([0 if label == 'M' else 1 for label in y])
        X = df.iloc[:, 2:32]
        X = np.nan_to_num(X)
        scaler = MinMaxScaler(copy=False)
        X = scaler.fit_transform(X)
        # Save the scaler
        joblib.dump(scaler, 'model_save/scaler.pkl')
        # Create the network
        network = Network([
            # Input layer
            Layer(name="input"),
            # Hidden layer 1
            Layer(n_input_nodes=30, n_current_layer_nodes=100, activation='relu', name="hidden1"),
            # Hidden layer 2
            Layer(n_input_nodes=100, n_current_layer_nodes=200, activation='relu', name="hidden2"),
            # Hidden layer 3
            Layer(n_input_nodes=200, n_current_layer_nodes=100, activation='relu', name="hidden3"),
            # Output layer
            Layer(n_input_nodes=100, n_current_layer_nodes=2, activation='softmax', name="output")],
            # Features and labels
            features=X, labels=y
        )

        # Train the network
        network.fit(epochs=1000, learning_rate=0.01)
        print("Training Phase - Completed.")
        # Save the model
        network.save_model()
        print("Model saved to [model_save] directory.")
        # Plot the cost evolution
        network.plot_cost()

    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
