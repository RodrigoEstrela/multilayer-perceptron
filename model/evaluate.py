import pandas as pd
import numpy as np
import sys
import joblib

from network import Network


def main():
    try:
        # Validation data preprocessing
        evaluate_data = pd.read_csv(sys.argv[1])
        y_evaluate = evaluate_data.iloc[:, 1]
        y_evaluate = np.array([0 if label == 'M' else 1 for label in y_evaluate])
        X_evaluate = evaluate_data.iloc[:, 2:32]
        X_evaluate = np.nan_to_num(X_evaluate)
        # Load the scaler
        scaler = joblib.load('model_save/scaler.pkl')
        X_evaluate = scaler.transform(X_evaluate)
        # Load the model
        trained_model = Network.load_model(num_layers=5)
        # Evaluate the network
        trained_model.evaluate(input_data=X_evaluate, labels=y_evaluate)

    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
