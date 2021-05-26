import os, shutil, torch
import pandas as pd
from tqdm import tqdm
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split 
from aoi_torch.args import init_arguments


def _files_preprocess(table, label, source_folder, target_folder):
    
    for idx in tqdm(table.index):
        tmp_folder_path = os.path.join(target_folder, str(table.loc[idx, label]))
        if not os.path.exists(tmp_folder_path):
            os.makedirs(tmp_folder_path)
        tmp_source_path = os.path.join(source_folder, table.loc[idx, 'ID'])
        tmp_target_path = os.path.join(tmp_folder_path, table.loc[idx, 'ID'])
        shutil.copyfile(tmp_source_path, tmp_target_path)


def files_preprocess(subset:str, label:str, val_size:float, test_size:float, seed:int):
    print('Preprocessing the original', subset + 'ing', 'images ...')
    table = pd.read_csv(subset + '.csv', sep = ',')
    source_folder = subset + '_images'
    
    if subset == 'train':
        print('Train/Val/Test = ', end='')
        print(1 - val_size - test_size, val_size, test_size, sep='/')
        table_tr, table_ = train_test_split(table, test_size=val_size + test_size, stratify=table[label], random_state=seed)
        table_va, table_te = train_test_split(table_, test_size=test_size / (val_size + test_size), stratify=table_[label], random_state=seed)
        _files_preprocess(table_tr, label, source_folder, 'images_tr')
        _files_preprocess(table_va, label, source_folder, 'images_va')
        _files_preprocess(table_te, label, source_folder, 'images_te')
    else:
        _files_preprocess(table, label, source_folder, 'images_test')


def load_with_splitting(subset:str, label:str, val_size:float, test_size:float, seed:int, transforms_hyper:dict):
    if not os.path.exists('images_va'):
        files_preprocess('train', label, val_size, test_size, seed)
    if not os.path.exists('images_test'):
        files_preprocess('test', label, val_size, test_size, seed)

    data_tr = datasets.ImageFolder(root='images_tr', transform=transforms.ToTensor())
    means_ = torch.zeros(3)
    stds_ = torch.zeros(3)
    for img, label in data_tr:
        means_ += torch.mean(img, dim = (1,2))
        stds_ += torch.std(img, dim = (1,2))
    
    means_ /= len(data_tr)
    stds_ /= len(data_tr)

    train_transforms = transforms.Compose([
        transforms.Resize(transforms_hyper['pretrained_size']),
        transforms.RandomRotation(transforms_hyper['rotation']), 
        transforms.RandomHorizontalFlip(transforms_hyper['horizontal_flip']),
        transforms.RandomCrop(transforms_hyper['pretrained_size'], padding=transforms_hyper['padding']),
        transforms.ToTensor(),
        transforms.Normalize(mean=means_.cpu().numpy(), std=stds_.cpu().numpy())
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(transforms_hyper['pretrained_size']),
        transforms.CenterCrop(transforms_hyper['pretrained_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=means_.cpu().numpy(), std=stds_.cpu().numpy())
    ])

    data_tr = datasets.ImageFolder(root='images_tr', transform=train_transforms)
    data_va = datasets.ImageFolder(root='images_va', transform=test_transforms)
    data_te = datasets.ImageFolder(root='images_te', transform=test_transforms)
    data_test = datasets.ImageFolder(root='images_test', transform=test_transforms)

    return data_tr, data_va, data_te, data_test


if __name__ == '__main__':
    args = init_arguments().parse_args()
    if not os.path.exists('images_va'):
        files_preprocess('train', args.label, args.val_size, args.test_size, args.random_state)
    if not os.path.exists('images_test'):
        files_preprocess('test', args.label, args.val_size, args.test_size, args.random_state)