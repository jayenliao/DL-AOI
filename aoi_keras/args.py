import argparse

def init_arguments():
    parser = argparse.ArgumentParser(prog='Deep Learning - HW5: AOI (keras version)')

    # General
    parser.add_argument('--seed', type=int, default=4028)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--label', type=str, default='Label')
    parser.add_argument('--dataPATH', type=str, default='./train_images/', help='The path where should the data be loaded in.')
    parser.add_argument('--trainDATA', type=str, default='./train.csv')
    parser.add_argument('--savePATH', type=str, default='./output_keras/', help='The path to store the outputs, including models, plots, and training and evalution results.')
    parser.add_argument('--save_model', action='store_true', help='Whether to save the trained models')
    parser.add_argument('--model_name', type=str, default='CNN', choices=['CNN', 'VGG'], help='Which kind of model is going to be trained?')
    
    # Preprocessing
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--rescale', type=int, default=256)
    parser.add_argument('--width_shift_range', type=float, default=.2)
    parser.add_argument('--height_shift_range', type=float, default=.2)
    parser.add_argument('--horizontal_flip', action='store_false', default=True)
    parser.add_argument('--vertical_flip', action='store_false', default=True)
    parser.add_argument('--brightness_range', nargs='+', type=float, default=[0.8, 1.2])
    parser.add_argument('--zoom_range', nargs='+', type=float, default=[0.8, 1.1])

    # Model structure
    parser.add_argument('--l2', type=float, default=.01, help='L2 regularization')
    parser.add_argument('--dropout', type=float, default=.2, help='Dropout ratio')
    parser.add_argument('--activation', type=str, default='relu', choices=['sigmoid', 'relu', 'tanh'], help='Activation function of all hidden layers (except for the last layer)')
    
    # Training
    parser.add_argument('--val_size', type=float, default=0.15)
    parser.add_argument('--test_size', type=float, default=0.15)
    parser.add_argument('--device', type=str, default='cuda:1', choices=['cpu', 'cuda', 'cuda:1'], help='Device name')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'], help='Optimizer')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--decay', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='No. of epochs')
    parser.add_argument('--early_stop', action='store_true', default=False)
    parser.add_argument('--pretrained_model', type=str, default='', help='File name of the pretained model that is going to keep being trained or to be evaluated. Set an empty string if not using a pretrained model.')
    parser.add_argument('--plot_figsize', nargs='+', type=int, default=[8, 6], help='Figure size of model performance plot. Its length should be 2.')
    
    return parser