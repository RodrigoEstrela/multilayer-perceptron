import pandas as pd
import numpy as np
import sys
import joblib

from network import Network


def main():
    try:
        # Validation data preprocessing
        validate_data = pd.read_csv(sys.argv[1])
        y_validate = validate_data.iloc[:, 1]
        y_validate = np.array([0 if label == 'M' else 1 for label in y_validate])
        X_validate = validate_data.iloc[:, 2:32]
        X_validate = np.nan_to_num(X_validate)
        # Load the scaler
        scaler = joblib.load('model_save/scaler.pkl')
        X_validate = scaler.transform(X_validate)
        # Load the model
        trained_model = Network.load_model(num_layers=5)
        print("Model loaded.")
        # Validate the network
        validate_predictions = trained_model.feedforward(X_validate)
        predicted_classes = np.argmax(validate_predictions, axis=1)
        accuracy = np.mean(predicted_classes == y_validate)
        # Print the accuracy
        print(f"Accuracy: {accuracy:.2f}.")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
