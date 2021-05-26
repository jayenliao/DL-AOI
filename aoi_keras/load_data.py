import numpy as np
import pandas as pd   
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split 

def load_with_preprocessing(dataPATH:str, trainDATA:str, label:str, batch_size:int, resize:int, val_size:float, test_size:float, preprocess_hyper:dict, seed:int):
    data_csv = pd.read_csv(trainDATA)
    data_csv[label] = data_csv[label].astype(str)
    print('Train/Val/Test = ', end='')
    print(1 - val_size - test_size, val_size, test_size, sep='/')
    csv_tr, csv_ = train_test_split(data_csv, test_size=val_size + test_size, stratify=data_csv[label], random_state=seed)
    csv_va, csv_te = train_test_split(csv_, test_size=test_size / (val_size + test_size), stratify=csv_[label], random_state=seed)
    # print(csv_tr.shape, csv_va.shape, csv_te.shape)
    # print(csv_va.head())
    # print(csv_te.head())
    # sys.exit()

    imgGen_tr = ImageDataGenerator(**preprocess_hyper)
    imgGen_va = ImageDataGenerator(**preprocess_hyper)
    imgGen_te = ImageDataGenerator(**preprocess_hyper)
    imgGen_out = ImageDataGenerator(rescale=preprocess_hyper['rescale'])
    
    print('\nLoading the training data ...')
    data_tr = imgGen_tr.flow_from_dataframe(
        dataframe=csv_tr, directory=dataPATH,
        x_col='ID', y_col=label, seed=seed, shuffle=False,
        batch_size=batch_size, target_size=(resize, resize),
        color_mode='grayscale', class_mode='categorical'
    )
    print('\nLoading the validation data ...')
    data_va = imgGen_va.flow_from_dataframe(
        dataframe=csv_va, directory=dataPATH,
        x_col='ID', y_col=label, seed=seed, shuffle=False,
        batch_size=batch_size, target_size=(resize, resize),
        color_mode='grayscale', class_mode='categorical'
    )
    print('\nLoading the testing data ...')
    data_te = imgGen_te.flow_from_dataframe(
        dataframe=csv_te, directory=dataPATH,
        x_col='ID', y_col=label, seed=seed, shuffle=False,
        batch_size=batch_size, target_size=(resize, resize),
        color_mode='grayscale', class_mode='categorical'
    )
    print('\nLoading the real testing data without labels ...')
    data_out = imgGen_out.flow_from_dataframe(
        dataframe=pd.read_csv(trainDATA.replace('train', 'test')),
        directory=dataPATH.replace('train', 'test'),
        x_col='ID', y_col=None, shuffle=False,
        batch_size=batch_size, target_size=(resize, resize),
        color_mode='grayscale', class_mode=None
    )

    #return imgGen_tr, imgGen_va, imgGen_te, imgGen_out
    return data_tr, data_va, data_te, data_out

