import numpy as np
import matplotlib.pyplot as plt
import os

from layer import Layer
from args_parser import parser_function


class Network:
    """
    multilayer-perceptron model
    """

    def __init__(self, layers, features=None, labels=None):
        """
        Initialize the network with the given layers, features, and labels
        """
        self.layers = layers
        self.features_train = features
        self.labels_train = labels
        self.predictions = None
        self.learning_rate = 0.01
        self.iterations = []
        self.gradients = []
        self.save_model_path = None
        self.epochs = None
        self.show_epochs = parser_function().show_epochs


    def feedforward(self, input_data=None):
        """
        Feedforward the input data through the network
        """
        # Set the output of the input layer to the input features
        if input_data is not None:
            self.layers[0].output = input_data
        else:
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

    def calculate_loss(self, iteration: int):
        """
        Calculate the loss (error) of the current predictions
        """
        # Perform forward pass to get predictions
        self.feedforward()

        # Compute the cross-entropy loss
        loss = -np.mean(np.log(self.predictions[np.arange(len(self.predictions)), self.labels_train]))

        self.iterations.append((iteration, loss))
        # print the loss for each epoch, if epoch x < 10, print 0x instead of x, otherwise print x
        if self.show_epochs:
            if iteration < 9:
                print(f"epoch 0{iteration + 1}/{self.epochs} - loss: {loss:.2f}")
            else:
                print(f"epoch {iteration + 1}/{self.epochs} - loss: {loss:.2f}")

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
            self.calculate_loss(i)
            self.backpropagation()
            self.update_weights()

        print(f"Training completed. Loss: {self.iterations[-1][1]:.2f}")

    def evaluate(self, input_data, labels):
        """
        Evaluate the accuracy of the model on the given input data and labels
        """

        # Perform prediction on the input data
        self.feedforward(input_data)
        predictions = np.argmax(self.predictions, axis=1)
        # Compute the accuracy of the model
        accuracy = np.mean(predictions == labels)
        # Print the accuracy
        print(f"Accuracy: {accuracy:.2f}.")

    def plot_cost(self):
        """
        Plot the cost evaluation
        """
        x = [item[0] for item in self.iterations]
        y = [item[1] for item in self.iterations]

        # Plot the data as a scatter graph
        plt.scatter(x, y)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Cost evolution')
        plt.grid(True)
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
