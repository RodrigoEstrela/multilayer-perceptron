import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

from layer import Layer
from network import Network
from network_from_cli import network_from_cli
from args_parser import parser_function


def main():
    # make directory model_save
    this_file_path = os.path.dirname(os.path.realpath(__file__))
    save_model_path = os.path.join(this_file_path, '..', 'model_save')
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    try:
        # Get path for training dataset
        this_file_path = os.path.dirname(os.path.realpath(__file__))
        dataset_path = os.path.join(this_file_path, '..', 'data', 'train.csv')
        # Training data preprocessing
        df = pd.read_csv(dataset_path)
        y = df.iloc[:, 1]
        y = np.array([0 if label == 'M' else 1 for label in y])
        X = df.iloc[:, 2:32]
        X = np.nan_to_num(X)
        scaler = MinMaxScaler(copy=False)
        X = scaler.fit_transform(X)
        # Save the scaler
        this_file_path = os.path.dirname(os.path.realpath(__file__))
        save_model_path = os.path.join(this_file_path, '..', 'model_save')
        joblib.dump(scaler, save_model_path + '/scaler.pkl')
        # Decide whether to build the model from CLI arguments or not
        if network_from_cli(features=X, labels=y):
            return
        # Create the network
        network = Network([
            # Input layer
            Layer(),
            # Hidden layers
            Layer(n_input_nodes=30, n_nodes=24, activation='relu'),
            Layer(n_input_nodes=24, n_nodes=24, activation='relu'),
            Layer(n_input_nodes=24, n_nodes=24, activation='relu'),
            # Output layer
            Layer(n_input_nodes=24, n_nodes=2, activation='softmax')],
            # Features and labels
            features=X, labels=y
        )
        # Check if there are at least 2 hidden layers
        if len(network.layers) < 4:
            raise ValueError("At least 2 hidden layers are required.")
        # Train the network
        network.fit(epochs=84, learning_rate=0.0314)
        # Save the model
        network.save_model()
        # Plot the cost evolution
        if parser_function().plot_cost:
            network.plot_cost()

    except Exception as e:
        print(e)


if __name__ == '__main__': 
    main()
