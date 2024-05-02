import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

from layer import Layer
from network import Network
from mlp_parser import network_from_cli


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
            # Hidden layer 1
            Layer(n_input_nodes=30, n_current_layer_nodes=100, activation='relu'),
            # Hidden layer 2
            Layer(n_input_nodes=100, n_current_layer_nodes=200, activation='relu'),
            # Hidden layer 3
            Layer(n_input_nodes=200, n_current_layer_nodes=100, activation='relu'),
            # Output layer
            Layer(n_input_nodes=100, n_current_layer_nodes=2, activation='softmax')],
            # Features and labels
            features=X, labels=y
        )
        # Train the network
        network.fit(epochs=1000, learning_rate=0.01)
        # Save the model
        network.save_model()
        # Plot the cost evolution
        network.plot_cost()

    except Exception as e:
        print(e)


if __name__ == '__main__': 
    main()
