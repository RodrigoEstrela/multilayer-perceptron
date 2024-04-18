import numpy as np


class Network:
    """
    multilayer-perceptron model
    """

    def __init__(self, layers, features, labels):
        self.layers = layers
        self.features = features
        self.labels = labels

    def feed_input(self):
        """
        Gets input features to the network's input layer
        """
        input_layer = self.layers[0]
        layer_outputs = []
        for i in range(input_layer.shape):
            layer_outputs.append(input_layer.nodes[i].activation(self.features.T[i]))

        return layer_outputs

    def feedforward(self):
        """
        Forward pass through the network
        """
        X = np.array(self.feed_input())
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
            # Set the output of the layer as the input of the next layer
            X = np.array(layer_outputs)
        return X

    def predict(self):
        """
        Predict the class of each sample
        """
        # Get the output of the last layer
        output = self.feedforward()
        # Return the index of the node with the highest value
        return np.argmax(output, axis=0)

    def evaluate(self):
        """
        Evaluate the accuracy of the model
        """
        # Get the predicted classes
        predictions = self.predict()
        # Compare the predicted classes with the true classes
        accuracy = np.mean(predictions == self.labels)
        return accuracy

    def __str__(self):
        # Print each one of the layers of the network
        return '\n'.join([str(layer) for layer in self.layers])
