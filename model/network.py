import numpy as np
import matplotlib.pyplot as plt


class Network:
    """
    multilayer-perceptron model
    """

    def __init__(self, layers, features, labels):
        self.layers = layers
        self.features = features
        self.labels = labels
        self.predictions = None
        self.learning_rate = 0.01
        self.iterations = []

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
                weighted_sum = node.weights @ X + node.bias
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
        n = len(self.labels)
        cost = np.sum((self.labels - self.predictions) ** 2) / n

        return cost

    def compute_output_layer_delta(self):
        """
        Compute the delta of the output layer
        """
        output_layer = self.layers[-1]
        for i in range(output_layer.shape):
            node = output_layer.nodes[i]
            # Compute the delta for each node in the output layer
            node.delta = (self.predictions - self.labels) * node.activation_derivative(node.output)

    def update_output_layer_weights(self):
        """
        Update the weights of the output layer
        """
        output_layer = self.layers[-1]
        hidden_layer = self.layers[-2]
        for i in range(output_layer.shape):
            node = output_layer.nodes[i]
            # Update the weights for each node in the output layer
            transposed_hidden_outputs = np.array([hidden_node.output for hidden_node in hidden_layer.nodes]).T
            node.weights -= self.learning_rate * node.delta @ transposed_hidden_outputs
            node.weights = [0 if weight < 0 else weight for weight in node.weights]
            node.bias -= self.learning_rate * node.delta

    def compute_hidden_layers_delta(self):
        """
        Compute the delta of the hidden layers
        """
        for i in range(len(self.layers) - 2, 0, -1):
            hidden_layer = self.layers[i]
            next_layer = self.layers[i + 1]
            for j in range(hidden_layer.shape):
                node = hidden_layer.nodes[j]
                # Compute the delta for each node in the hidden layer
                node.delta = sum([next_node.weights[j] * next_node.delta for next_node in
                                  next_layer.nodes]) * node.activation_derivative(node.output)

    def update_hidden_layers_weights(self):
        """
        Update the weights of the hidden layers
        """
        for i in range(len(self.layers) - 2, 0, -1):
            hidden_layer = self.layers[i]
            previous_layer = self.layers[i - 1]
            for j in range(hidden_layer.shape):
                node = hidden_layer.nodes[j]
                # Update the weights for each node in the hidden layer
                transposed_previous_outputs = np.array([previous_node.output for previous_node in previous_layer.nodes]).T
                node.weights -= self.learning_rate * node.delta @ transposed_previous_outputs
                node.weights = [0 if weight < 0 else weight for weight in node.weights]
                node.bias -= self.learning_rate * node.delta

    def fit(self, epochs: int, learning_rate: float):
        self.learning_rate = learning_rate
        self.iterations : np.array
        print("Begin Training")
        for i in range(epochs):
            if i  == 1:
                print(self.evaluate())
            self.feedforward()
            self.iterations.append((i, self.compute_cost()))
            self.compute_output_layer_delta()
            self.compute_hidden_layers_delta()
            self.update_output_layer_weights()
            self.update_hidden_layers_weights()

    def evaluate(self):
        """
        Evaluate the accuracy of the model
        """
        # Compare the predicted classes with the true classes
        accuracy = np.mean(self.predictions == self.labels)
        return accuracy
    
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
        plt.title('Cost evaluation')
        plt.grid(True)
        plt.show()

    def __str__(self):
        # Print each one of the layers of the network
        return '\n'.join([str(layer) for layer in self.layers])
