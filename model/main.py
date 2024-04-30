import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys

from layer import Layer
from network import Network


def main():
    # Training data preprocessing
    df = pd.read_csv(sys.argv[1])
    y = df.iloc[:, 1]
    y = np.array([0 if label == 'M' else 1 for label in y])
    X = df.iloc[:, 2:32]
    X = np.nan_to_num(X)
    scaler = MinMaxScaler(copy=False)
    X = scaler.fit_transform(X)
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

    # =============================================================================
    # TRAINING PHASE
    # =============================================================================

    # Train the network
    network.fit(epochs=1000, learning_rate=0.01)
    print("Training Finished")
    network.plot_cost()

    # =============================================================================
    # VALIDATION PHASE
    # =============================================================================

    # Validation data preprocessing
    validate_data = pd.read_csv(sys.argv[2])
    y_validate = validate_data.iloc[:, 1]
    y_validate = np.array([0 if label == 'M' else 1 for label in y_validate])
    X_validate = validate_data.iloc[:, 2:32]
    X_validate = np.nan_to_num(X_validate)
    X_validate = scaler.transform(X_validate)
    # Validate the network
    validate_predictions = network.feedforward(X_validate)
    predicted_classes = np.argmax(validate_predictions, axis=1)
    accuracy = np.mean(predicted_classes == y_validate)
    print("accuracy: ", accuracy)


if __name__ == '__main__':
    main()
