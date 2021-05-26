import argparse

def init_arguments():
    parser = argparse.ArgumentParser(prog='Deep Learning - HW5: AOI')

    # General
    parser.add_argument('--NonCNN', action='store_true')
    parser.add_argument('--seed', type=int, default=4028)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--label', type=str, default='Label')
    parser.add_argument('--dataPATH', type=str, default='./train_images/', help='The path where should the data be loaded in.')
    parser.add_argument('--trainDATA', type=str, default='./train.csv')
    parser.add_argument('--savePATH', type=str, default='./output/', help='The path to store the outputs, including models, plots, and training and evalution results.')
    parser.add_argument('--save_model', action='store_true', help='Whether to save the trained models')
    parser.add_argument('--pretrained_model', type=str, default='resnet50', choices=['resnet18', 'resnet50', 'resnet101', 'vgg16'])


    parser.add_argument('--model_name', type=str, default='lenet', choices=['lenet', 'VGG'], help='Which kind of model is going to be trained?')

    # Model structure
    parser.add_argument('--channels', type=int, default=3, help='Dimension of the input image array')
    parser.add_argument('--num_conv_layers', type=int, default=2, help='No. of convolution layers')
    parser.add_argument('--filter_size', type=int, default=5, help='List of filter size of each conv. layer. Its length should be same as num_conv_layers.')
    parser.add_argument('--pooling_size', type=int, default=2, help='Size of max pooling layer')
    parser.add_argument('--hidden_sizes', nargs='+', type=int, default=[32, 16], help='Dimension of the hidden state of the linear layers')
    parser.add_argument('--hidden_act', type=str, default='ReLU', choices=['Sigmoid', 'ReLU', 'tanh'], help='Activation function of all hidden layers (except for the last layer)')
    
    # Training
    parser.add_argument('--val_size', type=float, default=0.15)
    parser.add_argument('--test_size', type=float, default=0.15)
    parser.add_argument('--device', type=str, default='cuda:1', choices=['cpu', 'cuda', 'cuda:1'], help='Device name')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'], help='Optimizer')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--decay', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=48, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='No. of epochs')
    parser.add_argument('--early_stop', action='store_true', default=False)
    parser.add_argument('--plot_figsize', nargs='+', type=int, default=[8, 6], help='Figure size of model performance plot. Its length should be 2.')
    
    return parser