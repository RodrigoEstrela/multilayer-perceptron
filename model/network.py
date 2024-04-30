import numpy as np
import matplotlib.pyplot as plt

from layer import Layer


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

    def feedforward(self, input_data):
        """
        Feedforward the input data through the network
        """
        # Set the output of the input layer to the input features
        self.layers[0].output = input_data
        # Iterate over each layer (starting from the second layer)
        for i in range(1, len(self.layers)):
            # Compute the weighted sum of inputs for each node in the current layer
            weighted_sums = np.dot(self.layers[i - 1].output, self.layers[i].weights) + self.layers[i].bias

            # Apply the activation function to the weighted sum for each node
            outputs = self.layers[i].activation(weighted_sums)

            # Store the outputs of the current layer
            self.layers[i].output = outputs

        # Return the output of the last layer
        return self.layers[-1].output

    def calculate_loss(self):
        """
        Calculate the loss (error) of the current predictions
        """
        # Perform forward pass to get predictions
        predictions = self.feedforward(self.features_train)

        # Compute the cross-entropy loss
        loss = -np.mean(np.log(predictions[np.arange(len(predictions)), self.labels_train]))

        return loss

    def backpropagation(self, predictions, labels):
        """
        Backpropagation algorithm to compute the gradients of the loss with respect to the weights and biases
        """
        # Compute the error of the output layer
        error = predictions
        error[np.arange(len(predictions)), labels] -= 1
        error /= len(predictions)

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

        return gradients

    def update_weights(self, gradients):
        """
        Update the weights and biases of the network using the computed gradients
        """
        for i in range(1, len(self.layers)):
            # Update the weights and biases of the current layer
            self.layers[i].weights -= self.learning_rate * gradients[i - 1][0]
            self.layers[i].bias -= self.learning_rate * gradients[i - 1][1]

    def fit(self, epochs: int, learning_rate: float):
        """
        Train the network using the given features and labels
        """
        self.learning_rate = learning_rate
        for i in range(epochs):
            # feedforward
            self.predictions = self.feedforward(self.features_train)
            # calculate loss
            loss = self.calculate_loss()
            self.iterations.append((i, loss))
            # backpropagation
            gradients = self.backpropagation(self.predictions, self.labels_train)
            # update weights
            self.update_weights(gradients)

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
        for i, layer in enumerate(self.layers[1:], start=1):
            if i == len(self.layers) - 1:  # output layer
                weights_filename = f"model_save/mlp_output_layer_weights.npy"
                bias_filename = f"model_save/mlp_output_layer_bias.npy"
            else:  # hidden layers
                weights_filename = f"model_save/mlp_hidden_layer_{i}_weights.npy"
                bias_filename = f"model_save/mlp_hidden_layer_{i}_bias.npy"
            np.save(weights_filename, layer.weights)
            np.save(bias_filename, layer.bias)

    @staticmethod
    def load_model(num_layers):
        """
        Load the model weights and biases from files
        """
        layers = [Layer(name="input")]
        for i in range(1, num_layers):
            if i == num_layers - 1:  # output layer
                weights_filename = f"model_save/mlp_output_layer_weights.npy"
                bias_filename = f"model_save/mlp_output_layer_bias.npy"
                weights = np.load(weights_filename)
                bias = np.load(bias_filename)
                layers.append(Layer(weights=weights, bias=bias, activation='softmax'))
            else:  # hidden layers
                weights_filename = f"model_save/mlp_hidden_layer_{i}_weights.npy"
                bias_filename = f"model_save/mlp_hidden_layer_{i}_bias.npy"
                weights = np.load(weights_filename)
                bias = np.load(bias_filename)
                layers.append(Layer(weights=weights, bias=bias, activation='relu'))
        return Network(layers=layers)
