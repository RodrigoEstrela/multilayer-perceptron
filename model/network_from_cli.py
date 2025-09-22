from network import Network
from layer import Layer
from args_parser import parser_function


def network_from_cli(features=None, labels=None):
    args = parser_function()
    if args.layer and args.epochs and args.learning_rate:
        # Check if there are at least 2 hidden layers
        if len(args.layer) < 2:
            raise ValueError("At least 2 hidden layers are required.")
        # Check if the number of nodes in each layer is greater than 0 and less than 300
        for layer in args.layer:
            if layer <= 0 or layer > 300:
                raise ValueError("Number of nodes in each layer must be greater than 0 and less than 300.")
        # Check if the number of epochs is greater than 0
        if args.epochs[0] <= 0:
            raise ValueError("Number of epochs must be greater than 0.")
        # Check if the learning rate is greater than 0
        if args.learning_rate[0] <= 0:
            raise ValueError("Learning rate must be greater than 0.")
        # All checks passed, building network
        print("Building network from cli arguments.")

        # Input layer
        layers = [Layer.add_layer(type='input', n_nodes=30)]
        # Hidden layers
        for i in range(0, len(args.layer)):
            layers.append(Layer.add_layer(n_input=layers[i].n_nodes, n_nodes=args.layer[i]))
        # Output layer
        layers.append(Layer.add_layer(type='output', n_input=layers[-1].n_nodes, n_nodes=2))

        # Create the network and train it
        network = Network(layers, features=features, labels=labels)
        network.fit(epochs=args.epochs[0], learning_rate=args.learning_rate[0])

        # Save the model
        network.save_model()
        # Plot the cost evolution
        if args.plot_cost:
            network.plot_cost()

        return True
    
    else:
        print("Network from CLI arguments not found. Building network from file.")
        return False
