'''
Deep Learning - HW5: AOI (torch versiob)
Jay Liao (re6094028@gs.ncku.edu.tw)
'''

from aoi_torch.args import init_arguments
from aoi_torch.load_data import files_preprocess
from tqdm import tqdm
import os, shutil, time, pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import torch
from torch._C import device
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import models

def main(args):
    # load the data
    if not os.path.exists('images_va'):
        files_preprocess('train', args.label, args.val_size, args.test_size, args.random_state)
    if not os.path.exists('images_test'):
        files_preprocess('test', args.label, args.val_size, args.test_size, args.random_state)
    data_tr = datasets.ImageFolder(root='images_tr', transform=transforms.ToTensor())
    data_va = datasets.ImageFolder(root='images_va', transform=transforms.ToTensor())
    data_te = datasets.ImageFolder(root='images_te', transform=transforms.ToTensor())
    data_test = datasets.ImageFolder(root='images_test', transform=transforms.ToTensor())

    # preprocess the data
    # put the data into the trainer
    # train the model
    # evaluate the model
    # predict
    ''

if __name__ == '__main__':
    args = init_arguments().parse_args()
    main(args)