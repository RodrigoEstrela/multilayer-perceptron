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
    else:
        for filename in os.listdir(save_model_path):
            file_path = os.path.join(save_model_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(e)
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

        if network_from_cli(features=X, labels=y): # this checks if there are sufficient arguments to create the network from CLI
            return
        # Create the network
        network = Network([
            # Input layer
            Layer(),
            # Hidden layers -> Edit here to change the architecture of the network ---------------
            Layer(n_input_nodes=30, n_nodes=24, activation='relu'),
            Layer(n_input_nodes=24, n_nodes=24, activation='relu'),
            Layer(n_input_nodes=24, n_nodes=24, activation='relu'),
            # ------------------------------------------------------------------------------------
            # Output layer
            Layer(n_input_nodes=24, n_nodes=2, activation='softmax')],
            features=X, labels=y
        )

        if len(network.layers) < 4:
            raise ValueError("At least 2 hidden layers are required.")

        network.fit(epochs=84, learning_rate=0.0314)
        network.save_model()
        if parser_function().plot_cost:
            network.plot_cost()

    except Exception as e:
        print(e)


if __name__ == '__main__': 
    main()
