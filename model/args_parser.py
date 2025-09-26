import argparse

def parser_function():
    parser = argparse.ArgumentParser()
    
    # building network arguments
    parser.add_argument('--layer', nargs='+', type=int, help='Layer(s) size(s)')
    parser.add_argument('--epochs', nargs=1, type=int, help='Number of Epochs')
    parser.add_argument('--learning_rate', nargs=1, type=float, help='Learning Rate')
    # training arguments
    parser.add_argument('--show_metrics', action='store_true', help='Show metrics for each epoch during training')
    parser.add_argument('--plot_loss', action='store_true', help='Plot loss evolution after training')
    parser.add_argument('--plot_accuracy', action='store_true', help='Plot accuracy evolution after training')

    args = parser.parse_args()
    return args
