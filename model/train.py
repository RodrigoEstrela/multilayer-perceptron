import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
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

        # Divide the dataset into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)

        if network_from_cli(checking=True): # this checks if there are sufficient arguments to create the network from CLI
            layers, epochs, learning_rate = network_from_cli()
        else:
            layers = [
                # Input layer
                Layer(),
                # Hidden layers -> Edit here to change the architecture of the network ---------------
                Layer(n_input_nodes=30, n_nodes=24, activation='relu'),
                Layer(n_input_nodes=24, n_nodes=24, activation='relu'),
                Layer(n_input_nodes=24, n_nodes=24, activation='relu'),
                # ------------------------------------------------------------------------------------
                # Output layer
                Layer(n_input_nodes=24, n_nodes=2, activation='softmax')]
            epochs = 70  # Edit here to change the number of epochs
            learning_rate = 0.0314  # Edit here to change the learning rate
        
        if len(layers) < 4:
            raise ValueError("At least 2 hidden layers are required.")
        
        network = Network(
            layers,
            features_train=X_train,
            labels_train=y_train,
            features_val=X_val, 
            labels_val=y_val
        )
        # Train the model 
        network.fit(epochs=epochs, learning_rate=learning_rate)
        
        network.save_model()
        if parser_function().plot_loss:
            network.plot_metrics(name='loss', metrics_indexes=[1, 2])
        if parser_function().plot_accuracy:
            network.plot_metrics(name='accuracy', metrics_indexes=[3, 4])

    except Exception as e:
        print(e)


if __name__ == '__main__': 
    main()
