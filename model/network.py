import numpy as np


class Network:
    """
    multilayer-perceptron model
    """

    def __init__(self, layers, features, labels):
        self.layers = layers
        self.features = features
        self.labels = labels
        self.predictions = None

    def feed_input(self):
        """
        Gets input features to the network's input layer
        """
        input_layer = self.layers[0]
        layer_outputs = []
        for i in range(input_layer.shape):
            layer_outputs.append(input_layer.nodes[i].activation(self.features.T[i]))
            input_layer.nodes[i].output = layer_outputs[i]

        return np.array(layer_outputs)

    def feedforward(self):
        """
        Forward pass through the network
        """
        X = self.feed_input()
        for i in range(1, len(self.layers)):
            # Initialize list to store layer outputs
            layer_outputs = []
            # Iterate over nodes in the layer
            for node in self.layers[i].nodes:
                # Transpose weights matrix before multiplying with inputs
                # print(node.weights.shape, X.shape)
                weighted_sum = node.weights @ X
                # Apply activation function
                activation_output = node.activation(weighted_sum)
                # Append output to layer_outputs
                layer_outputs.append(activation_output)
                node.output = activation_output
            # Set the output of the layer as the input of the next layer
            X = np.array(layer_outputs)
        self.predictions = np.argmax(X, axis=0)

        return self.predictions

    def compute_cost(self):
        """
        Function to compute the cost of the network using the following formula:
        (y - ŷ)^2 / 2
        """
        cost = ((self.labels - self.predictions) ** 2) / 2
        # print(cost)

    def compute_output_layer_delta(self):
        """
        Compute the delta of the output layer
        """
        output_layer = self.layers[-1]
        for i in range(output_layer.shape):
            node = output_layer.nodes[i]
            # Compute the delta for each node in the output layer
            node.delta = (self.labels - self.predictions) * node.activation_derivative(node.output)

    def compute_hidden_layer_delta(self):
        """
        Compute the delta of the hidden layers
        """
        current_layer = self.layers[1]
        next_layer = self.layers[2]
        for j in range(current_layer.shape):
            node = current_layer.nodes[j]
            # Compute the delta for each node in the hidden layer
            node.delta = np.dot(next_layer.nodes[j].weights, next_layer.nodes[j].delta) * node.activation_derivative(
                node.output)

    def update_output_layer_weights(self):
        """
        Update the weights of the output layer
        """
        learning_rate = 0.0314
        output_layer = self.layers[2]
        hidden_layer = self.layers[1]
        for i in range(output_layer.shape):
            node = output_layer.nodes[i]
            # Update the weights for each node in the output layer
            node.weights -= learning_rate * node.delta @ hidden_layer.nodes[i].output

    def train_loop(self, epochs: int):
        for i in range(epochs):
            if i == 1:
                print(self.evaluate())
            self.feedforward()
            self.compute_cost()
            self.compute_output_layer_delta()
            self.compute_hidden_layers_delta()
            self.update_output_layer_weights()

    def evaluate(self):
        """
        Evaluate the accuracy of the model
        """
        # Compare the predicted classes with the true classes
        accuracy = np.mean(self.predictions == self.labels)
        return accuracy

    def __str__(self):
        # Print each one of the layers of the network
        return '\n'.join([str(layer) for layer in self.layers])
