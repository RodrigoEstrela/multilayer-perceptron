import argparse

from network import Network
from layer import Layer


def parser_function():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', nargs='+', type=int, help='Layer(s) size(s)')
    parser.add_argument('--epochs', nargs=1, type=int, help='Number of Epochs')
    parser.add_argument('--learning_rate', nargs=1, type=float, help='Learning Rate')

    args = parser.parse_args()
    return args


def network_from_cli(features=None, labels=None):
    args = parser_function()
    if args.layer and args.epochs and args.learning_rate:
        print("Building network from cli arguments.")
        layers = [Layer.add_layer(type='input',
                                  n_nodes=30)]
        for i in range(0, len(args.layer)):
            layers.append(Layer.add_layer(type='hidden', 
                                          n_input=layers[i].n_output_nodes,
                                          n_nodes=args.layer[i]))
        layers.append(Layer.add_layer(type='output',
                                      n_input=layers[-1].n_output_nodes,
                                      n_nodes=2))
        
        network = Network(layers, features=features, labels=labels)
        epochs, lr = 1000, 0.01
        if args.epochs:
            epochs = args.epochs[0]
        if args.learning_rate:
            lr = args.learning_rate[0]
        network.fit(epochs=epochs, learning_rate=lr)

        network.save_model()
        # network.plot_cost()
        return True
    else:
        print("Insufficient arguments passed. Building default network.")
        return False
