import numpy as np
import matplotlib.pyplot as plt
import os

from layer import Layer
from args_parser import parser_function


class Network:
    """
    multilayer-perceptron model
    """

    def __init__(self, layers, features_train=None, labels_train=None, features_val=None, labels_val=None):
        """
        Initialize the network with the given layers, features, and labels
        """
        self.layers = layers
        self.features_train = features_train
        self.labels_train = labels_train
        self.features_val = features_val
        self.labels_val = labels_val
        self.predictions = None
        self.learning_rate = 0.01
        self.iterations = []
        self.gradients = []
        self.save_model_path = None
        self.epochs = None
        self.track_metrics = parser_function().show_metrics or parser_function().plot_accuracy or parser_function().plot_loss
        self.track_accuracy = []
        self.show_metrics = parser_function().show_metrics


    def feedforward(self, input_data=None, in_training=True):
        """
        Feedforward the input data through the network
        """ 
        if in_training:
            self.layers[0].output = self.features_train
            # Iterate over each layer (starting from the second layer)
            for i in range(1, len(self.layers)):
                # Compute the weighted sum of inputs for each node in the current layer
                weighted_sums = np.dot(self.layers[i - 1].output, self.layers[i].weights) + self.layers[i].bias
                # Apply the activation function to the weighted sum for each node
                outputs = self.layers[i].activation(weighted_sums)
                # Store the outputs of the current layer
                self.layers[i].output = outputs
            # Save the output of the last layer as the predictions
            self.predictions = self.layers[-1].output
        else:
            outputs_list = []
            outputs_list.append(input_data)
            for i in range(1, len(self.layers)):
                weighted_sums = np.dot(outputs_list[i -1], self.layers[i].weights) + self.layers[i].bias
                outputs_list.append(self.layers[i].activation(weighted_sums))
            return outputs_list[-1]


    def training_validation_metrics(self, iteration: int):
        """
        Compute and print the training and validation metrics for the current iteration
        """
        if self.track_metrics:
            # Compute the cross-entropy loss on the training data
            train_loss = -np.mean(np.log(self.predictions[np.arange(len(self.predictions)), self.labels_train]))
            
            # Perform forward pass to get predictions on the validation data
            val_predictions = self.feedforward(input_data=self.features_val, in_training=False)
            # Compute the cross-entropy loss on the validation data
            val_loss = -np.mean(np.log(val_predictions[np.arange(len(val_predictions)), self.labels_val]))
            
            # Compute the accuracy on the training data
            train_accuracy = np.mean(np.argmax(self.predictions, axis=1) == self.labels_train)
            # Compute the accuracy on the validation data
            val_accuracy = np.mean(np.argmax(val_predictions, axis=1) == self.labels_val)

            # Store the training and validation metrics for the current iteration
            self.iterations.append((iteration, train_loss, val_loss, train_accuracy, val_accuracy))

            if self.show_metrics:
                if iteration < 9:
                    print(f"epoch 0{iteration + 1}/{self.epochs} - loss: {train_loss:.2f} - val_loss: {val_loss:.2f} - acc: {train_accuracy:.2f} - val_acc: {val_accuracy:.2f}")
                else:
                    print(f"epoch {iteration + 1}/{self.epochs} - loss: {train_loss:.2f} - val_loss: {val_loss:.2f} - acc: {train_accuracy:.2f} - val_acc: {val_accuracy:.2f}")
        else:
            return

    def backpropagation(self):
        """
        Backpropagation algorithm to compute the gradients of the loss with respect to the weights and biases
        """
        # Compute the error of the output layer
        error = self.predictions
        error[np.arange(len(self.predictions)), self.labels_train] -= 1
        error /= len(self.predictions)

        # Initialize the list of gradients
        gradients = []

        # Iterate over each layer in reverse order
        for i in range(len(self.layers) - 1, 0, -1):
            # Compute the gradients of the loss with respect to the weights and biases
            gradients.append([np.dot(self.layers[i - 1].output.T, error)])
            gradients[-1].append(np.mean(error, axis=0))

            # Compute the error of the current layer
            if i > 1:  # avoid index error
                error = np.dot(error, self.layers[i].weights.T)
                error *= self.layers[i - 1].activation_derivative(self.layers[i - 1].output)

        # Reverse the list of gradients to match the order of the layers
        gradients.reverse()

        self.gradients = gradients


    def update_weights(self):
        """
        Update the weights of the network using the computed gradients
        """
        for i in range(1, len(self.layers)):
            # Update the weights of the current layer
            self.layers[i].weights -= self.learning_rate * self.gradients[i - 1][0]


    def fit(self, epochs: int = 1000, learning_rate: float = 0.01):
        """
        Train the network using the given features and labels
        """
        args = parser_function()
        # Check if number of epochs and learning rate are greater than 0
        if epochs <= 0:
            raise ValueError("Number of epochs must be greater than 0.")
        if learning_rate <= 0:
            raise ValueError("Learning rate must be greater than 0.")

        self.epochs = epochs
        self.learning_rate = learning_rate
        # print the shape of the features
        print(f"Training on {self.features_train.shape[0]} samples with {self.features_train.shape[1]} features.")
        for i in range(epochs):
            self.feedforward()
            self.training_validation_metrics(i)
            self.backpropagation()
            self.update_weights()

        print(f"Training completed.")


    def evaluate(self, input_data, labels):
        """
        Evaluate the accuracy and binary cross-entropy loss of the model
        on the given input data and labels.
        """
        # Perform prediction on the input data (get probabilities)
        predictions = self.feedforward(input_data, in_training=False)  
        
        # If using softmax output with 2 classes, take probability of class 1
        if predictions.shape[1] == 2:
            probs = predictions[:, 1]
        else:
            # If using sigmoid with single output neuron
            probs = predictions.flatten()

        # Compute predicted class labels
        predicted_classes = np.argmax(predictions, axis=1) if predictions.shape[1] > 1 else (probs >= 0.5).astype(int)

        # Compute accuracy
        accuracy = np.mean(predicted_classes == labels)

        # Compute binary cross entropy
        eps = 1e-15  # to avoid log(0)
        probs = np.clip(probs, eps, 1 - eps)
        bce = -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))

        print(f"Accuracy: {accuracy:.2f}, Binary Cross-Entropy: {bce:.4f}")
        return accuracy, bce


    def plot_metrics(self, name=None, metrics_indexes=None):
        """
        Plot the metrics for each epoch with two lines: one for training and one for validation
        """
        x = [item[0] for item in self.iterations]
        y_train = [item[metrics_indexes[0]] for item in self.iterations]
        y_val = [item[metrics_indexes[1]] for item in self.iterations]

        # Plot the data as a line graph
        plt.plot(x, y_train, label='Training')
        plt.plot(x, y_val, label='Validation')
        plt.xlabel('Epochs')
        plt.ylabel(f'{name.capitalize()}')
        plt.title(f'{name.capitalize()} evolution')
        plt.grid(True)
        plt.legend()
        plt.show()

    def save_model(self):
        """
        Save the model weights and biases to files
        """
        this_file_path = os.path.dirname(os.path.realpath(__file__))
        self.save_model_path = os.path.join(this_file_path, '..', 'model_save')
        for i, layer in enumerate(self.layers[1:], start=1):
            if i == len(self.layers) - 1:  # output layer
                weights_filename = self.save_model_path + f"/mlp_output_layer_weights.npy"
            else:  # hidden layers
                weights_filename = self.save_model_path + f"/mlp_hidden_layer_{i}_weights.npy"
            np.save(weights_filename, layer.weights)

        print("Model saved.")


    @staticmethod
    def load_model():
        """
        Load the model weights and biases from files
        """
        try: 
            this_file_path = os.path.dirname(os.path.realpath(__file__))
            save_model_path = os.path.join(this_file_path, '..', 'model_save')
            layers = [Layer(name="input")]
            if not os.path.exists(save_model_path + "/mlp_output_layer_weights.npy"):
                raise FileNotFoundError("Model files not found. Please train the model first.")
            num_layers = 1  # start with input layer
            for filename in os.listdir(save_model_path):
                if filename.startswith("mlp_hidden_layer_") and filename.endswith("_weights.npy"):
                    try:
                        layer_index = int(filename.split("_")[3])
                        if layer_index + 1 > num_layers:
                            num_layers = layer_index + 1
                    except ValueError:
                        continue
            num_layers += 1  # add 1 for output layer

            for i in range(1, num_layers):
                if i == num_layers - 1:  # output layer
                    weights_filename = save_model_path + f"/mlp_output_layer_weights.npy"
                    weights = np.load(weights_filename)
                    layers.append(Layer(weights=weights, activation='softmax'))
                else:  # hidden layers
                    weights_filename = save_model_path + f"/mlp_hidden_layer_{i}_weights.npy"
                    weights = np.load(weights_filename)
                    layers.append(Layer(weights=weights, activation='relu'))

            print("Model loaded.")
            return Network(layers=layers)

        except Exception as e:
            print(e)
            return None
