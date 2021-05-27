import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from collections import namedtuple
from aoi_torch.models import *
from efficientnet_pytorch import EfficientNet

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Trainer:
    def __init__(self, data_tr, data_va, data_te, device:str, pretrained_model:str, optimizer:str, lr:float, epochs:int, batch_size:int, savePATH:str, save_model:bool, verbose:bool, random_state=4028):
        self.savePATH = savePATH if savePATH[-1] == '/' else savePATH + '/'
        self.random_state = random_state
        self.verbose = verbose
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss_tr = []
        self.criterion = nn.CrossEntropyLoss()
        
        # data
        OUTPUT_DIM = len(data_tr.classes)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.dataloader_tr = DataLoader(data_tr, shuffle=True, batch_size=batch_size)
        self.dataloader_va = DataLoader(data_va, shuffle=False, batch_size=len(data_va))
        self.dataloader_te = DataLoader(data_te, shuffle=False, batch_size=len(data_te))

        # model
        ModelConfig = namedtuple('ModelConfig', ['block', 'n_blocks', 'channels'])
        self.model_name = pretrained_model.lower()
        if self.model_name == 'resnet18':
            self.pretrained_model = models.resnet18(pretrained=True)
            self.model_config = ModelConfig(block=Bottleneck, n_blocks=[3, 4, 6, 3], channels=[64, 128, 256, 512])
        elif self.model_name == 'resnet50':
            self.pretrained_model = models.resnet50(pretrained=True)
            self.model_config = ModelConfig(block=Bottleneck, n_blocks=[3, 4, 6, 3], channels=[64, 128, 256, 512])
        elif self.model_name == 'resnet101':
            self.pretrained_model = models.resnet101(pretrained=True)
            self.model_config = ModelConfig(block=Bottleneck, n_blocks=[3, 4, 6, 3], channels=[64, 128, 256, 512])
        elif self.model_name == 'vgg16':
            self.pretrained_model = models.vgg16(pretrained=True)
        elif self.model_name == 'efficientnet-b7':
            self.model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=OUTPUT_DIM)

        if 'resnet' in pretrained_model.lower():
            IN_FEATURES = self.pretrained_model.fc.in_features
            self.pretrained_model.fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)
            self.model = ResNet(self.model_config, OUTPUT_DIM)
            self.model.load_state_dict(self.pretrained_model.state_dict())
        if self.verbose:
            print(f'The model has {count_parameters(self.model):,} trainable parameters.')
        
        if optimizer.lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def accuracy_score(self, y, yt, top=1):
        y = y.cpu().detach().numpy()
        yt = yt.cpu().detach().numpy()

        if yt.ndim != 1:
            yt = np.argmax(yt, axis=1)
        if top == 1:
            y = np.argmax(y, axis=1)
            acc = np.array(y == yt).mean()
        else:
            y = np.argsort(y, axis=1)[:,-top:]
            lst = []
            for i in range(len(yt)):
                lst.append(yt[i] in y[i,:])
            acc = np.array(lst).mean()

        return acc

    def evaluation_when_training(self, epoch:int, print_result_per_epochs:int):
        self.model.eval()
        
        # Compute the top-1 and top-2 accuracy scores on the training set
        acc1_tr = self.accuracy_score(self.output, self.y_batch, top=1)
        acc5_tr = self.accuracy_score(self.output, self.y_batch, top=2)
        self.accuracy_tr['Top-1'].append(acc1_tr)
        self.accuracy_tr['Top-2'].append(acc5_tr)
            
        # Compute the top-1 and top-2 accuracy scores on the validation set
        with torch.no_grad():
            for (x, y) in self.dataloader_va:
                x = x.to(self.device)
                y = y.to(self.device)
                if 'resnet' in self.model_name:
                    yp, _ = self.model(x)
                else:
                    yp = self.model(x)
                acc1_va = self.accuracy_score(yp, y, top=1)
                acc5_va = self.accuracy_score(yp, y, top=2)
                self.accuracy_va['Top-1'].append(acc1_va)
                self.accuracy_va['Top-2'].append(acc5_va)
                
        # Print the validation accuracy scores once per given no. of epochs
        if epoch % print_result_per_epochs == 0:
            print(f'Epoch {epoch:4d}')
            print(f'  Training accuracy: {acc1_tr:.4f} (top-1) {acc5_tr:.4f} (top-2)')
            print(f'Validation accuracy: {acc1_va:.4f} (top-1) {acc5_va:.4f} (top-2)')

        # Find the best performance
        if acc1_va > self.best_acc['best_acc1']:
            self.best_acc['best_acc1'] = acc1_va
            self.best_acc['best_acc1_epoch'] = epoch
            torch.save(self.model.state_dict(), self.folder_name + 'model.pt')
        if acc5_va > self.best_acc['best_acc5']:
            self.best_acc['best_acc5'] = acc5_va
            self.best_acc['best_acc5_epoch'] = epoch

    def train_single_epoch(self):
        self.model.train()
        for (x, y) in tqdm(self.dataloader_tr):
            self.X_batch = x.to(self.device)
            self.y_batch = y.to(self.device)
            if 'resnet' in self.model_name:
                self.output, _ = self.model(self.X_batch)
            else:
                self.output = self.model(self.X_batch)
            loss = self.criterion(self.output, self.y_batch)
            loss.backward()
            self.optimizer.step()
            self.loss_tr.append(loss.cpu().detach().numpy())

    def train(self, print_result_per_epochs=10):
        '''
        1. Initialize
        2. Load the pretrained model (if the pretrained model name is given) or train a new model
        3. Training for no. of epochs
        4. Report that the model have been finished training and 
           Save the model performances: acc_tr/acc_va, acc_te, best_acc, and time_cost
        '''
        
        # 1. Intialize
        self.dt = datetime.now().strftime('%y-%m-%d-%H-%M-%S')
        self.folder_name = self.savePATH + self.dt + '_' + self.model_name + '_' + '_bs=' + str(self.batch_size) + '_epochs=' + str(self.epochs) + '/'
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)
        self.accuracy_tr = {'Top-1': [], 'Top-2': []}
        self.accuracy_va = {'Top-1': [], 'Top-2': []}
        self.best_acc = {'best_acc1': 0, 'best_acc1_epoch': 0, 'best_acc5': 0, 'best_acc5_epoch': 0,}
        if self.verbose:
            print('\nTraining a new', self.model_name, 'model ...', end='  ')
            print(f'(Batch size = {self.batch_size}, No. of epochs = {self.epochs})\n')
            print(self.model)

        # 3. Training! 
        self.time_cost = []
        t0 = time.time()
        for epoch in range(1, self.epochs+1):
            tEpoch = time.time()
            self.train_single_epoch()
            self.evaluation_when_training(epoch, print_result_per_epochs)
            tStep = time.time()
            self.time_cost.append(tStep - tEpoch)

        # 4. Report that the model have been finished training and 
        #    Save the model performances: acc_tr/acc_va, acc_te, best_acc, and time_cost
        tdiff = time.time() - t0
        if self.verbose:
            print('\nFinish training! Total time cost for %d epochs: %.2f s' % (self.epochs, tdiff))
            print('The best validation performance: Top-1 accuracy=%.4f at epoch %d' % (self.best_acc['best_acc1'], self.best_acc['best_acc1_epoch']))
    
    def evaluate(self):
        # Compute the top-1 and top-2 accuracy scores on the testing set
        if self.verbose:
            print('\nEvaluate the trained model on the testing set ...')
        self.model.eval()
        with torch.no_grad():
            for (x, y) in self.dataloader_te:
                x = x.to(self.device)
                y = y.to(self.device)
                if 'resnet' in self.model_name:
                    yp, _ = self.model(x)
                else:
                    yp = self.model(x)
                acc1_te = self.accuracy_score(yp, y, top=1)
                acc5_te = self.accuracy_score(yp, y, top=2)
        if self.verbose:
            print(f'Testing performance: Top-1 accuracy={acc1_te:.4f}, Top-2 accuracy={acc5_te:.4f}')
            print('\nModel performances are saved as the following files:')
        
        self.fn = self.folder_name + 'Accuracy.txt'
        arr = [self.accuracy_tr['Top-1'], self.accuracy_tr['Top-2'], self.accuracy_va['Top-1'], self.accuracy_va['Top-2']] 
        np.savetxt(self.fn, arr)
        if self.verbose:
            print('-->', self.fn)

        fn = self.fn.replace('Acc', 'TestAcc')
        pd.DataFrame({'Top-1': acc1_te, 'Top-2': acc5_te}, index=['score']).T.to_csv(fn)
        if self.verbose:
            print('-->', fn)

        fn = self.fn.replace('Acc', 'BestAcc')
        pd.DataFrame(self.best_acc, index=[0]).T.to_csv(fn)
        if self.verbose:
            print('-->', fn)

        fn = self.fn.replace('Accuracy', 'TimeCost')
        np.savetxt(fn, np.array(self.time_cost))
        if self.verbose:
            print('-->', fn)

    def predict(self, fn_test, label, data_test):
        dataloader_test = DataLoader(data_test, shuffle=False, batch_size=self.batch_size)
        yout = []
        self.model.eval()
        with torch.no_grad():
            for (x, _) in dataloader_test:
                x = x.to(self.device)
                if 'resnet' in self.model_name:
                    yp, _ = self.model(x)
                else:
                    yp = self.model(x)
                yout += list(np.argmax(yp.cpu().numpy(), axis=1))
        
        if self.verbose:
            print('\nDistribution of predicted labels:')
            for i, n in enumerate(np.bincount(yout)):
                p = n/len(yout)
                print(f'{i}: {n:4d} ({p:.4f})', end='  ')
                if i % 3 == 2:
                    print()
        
        df_out = pd.read_csv(fn_test, header=0)
        df_out[label] = yout
        fn = self.folder_name + 'prediction_out_' + self.model_name + '.csv'
        df_out.to_csv(fn, index=False)
        if self.verbose:
            print('*** The prediction output is saved as', fn)

    def plot_training(self, type_:str, figsize:tuple, save_plot=True):
        plt.figure(figsize=figsize)
        if type_ == 'loss':
            plt.plot(np.arange(len(self.loss_tr)), self.loss_tr, label=type_+'(train)')
        elif type_ == 'accuracy':
            x = np.arange(self.epochs)
            plt.plot(x, self.accuracy_tr['Top-1'], label=type_+'(train)')
            plt.plot(x, self.accuracy_va['Top-1'], label=type_+'(val)')
            plt.ylim(.8, 1)
            plt.legend()
        plt.title('Plot of ' + type_.capitalize() + ' of ' + self.model_name)
        plt.xlabel('Epoch')
        plt.ylabel(type_.capitalize())
        plt.grid()
            

        if save_plot:
            fn = self.folder_name + type_ + '_plot.png'
            plt.savefig(fn)
            print('The', type_, 'plot is saved as', fn)
