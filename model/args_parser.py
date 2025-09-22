import argparse

def parser_function():
    parser = argparse.ArgumentParser()
    
    # building network arguments
    parser.add_argument('--layer', nargs='+', type=int, help='Layer(s) size(s)')
    parser.add_argument('--epochs', nargs=1, type=int, help='Number of Epochs')
    parser.add_argument('--learning_rate', nargs=1, type=float, help='Learning Rate')
    # training arguments
    parser.add_argument('--show_epochs', action='store_true', help='Show epoch details during training')
    parser.add_argument('--plot_cost', action='store_true', help='Plot cost evolution after training')

    args = parser.parse_args()
    return args
