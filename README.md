# Multilayer Perceptron (MLP)

This project implements a customizable Multilayer Perceptron (MLP) neural network for binary classification tasks, using Python and NumPy. It includes data preprocessing, model training, evaluation, and model saving/loading functionalities.

## Project Structure

```
multilayer-perceptron/
├── README.md
├── data/
│   ├── csv_spliter.py      # Utility to split CSV into train/test
│   ├── data.csv           # Raw dataset
│   ├── train.csv          # Training data
│   └── test.csv           # Test data
├── model/
│   ├── args_parser.py     # CLI argument parser
│   ├── evaluate.py        # Model evaluation script
│   ├── layer.py           # Layer class (input, hidden, output)
│   ├── network.py         # MLP network implementation
│   ├── network_from_cli.py# Build/train network from CLI
│   ├── train.py           # Model training script
├── model_save/
│   ├── *.npy              # Saved weights and biases
│   └── scaler.pkl         # Saved data scaler
```

## Features

- Customizable number of hidden layers and nodes
- CLI-based configuration for layers, epochs, and learning rate
- Data normalization using MinMaxScaler
- Model saving/loading for reproducibility
- Loss evolution plotting

## Setup

1. **Install dependencies:**
	```zsh
	pip install numpy pandas scikit-learn matplotlib joblib
	```

2. **Prepare data:**
	Place your dataset as `data/data.csv` (first column: ID, second: label ['M', 'B'], rest: features).
	Split into train/test:
	```zsh
	python data/csv_spliter.py data/data.csv
	```

## Training

You can train the model using two methods.

### From what is defined in `model/train.py`:
```zsh
python3 model/train.py --plot_loss --show_metrics
```

- `--plot_loss`: Plot loss evolution after training
- `--show_metrics`: Show loss per epoch

### From CLI arguments:
```zsh
python3 model/train.py --layer 24 24 24 --epochs 100 --learning_rate 0.01
```

- `--layer`: List of hidden layer sizes (e.g., 24 24 24 for three hidden layers)
- `--epochs`: Number of training epochs
- `--learning_rate`: Learning rate
- plot_loss and show_metrics can also be used here.

Model weights, biases, and scaler are saved in `model_save/`.

## Evaluation

Evaluate the trained model on test data:

```zsh
python3 model/evaluate.py
```

## Model Architecture

- **Layer**: Implements input, hidden, and output layers with activation functions (ReLU, Softmax)
- **Network**: Handles feedforward, backpropagation, training, evaluation, and model persistence

## Customization

- Modify `model/train.py` to experiment with different architectures or activation functions.
- Use CLI arguments to change layer sizes, epochs, and learning rate.

## 42 Advanced - AI Track

- This project is part of the 42 Advanced - AI Track Curriculum and as such, it's corresponding subject is propriety of 42 and will not be shared here.
- The code was developed by myself and it's intended for my personal and student use only.

## Useful Learning Resources

- https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
