from aoi_keras.load_data import load_with_preprocessing
from tensorflow.python.ops.math_ops import DivideDelegateWithName
from datetime import datetime
import os, time
import numpy as np
import pandas as pd   
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

class Trainer:
    def __init__(self, model_name:str, savePATH:str, load_args:dict, epochs:int, l2:float, dropout:float, activation=str):
        self.imgGen_tr, self.imgGen_va, self.imgGen_te, self.imgGen_out = load_with_preprocessing(**load_args)
        for img, label in self.imgGen_tr:
            print('Dimension of one batch:', img.shape, label.shape)
            break
        self.model_name = model_name
        self.savePATH = savePATH
        self.num_classes = label.shape[1]
        self.label = load_args['label']
        self.batch_size = load_args['batch_size']
        self.epochs = epochs
        self.activation = activation 
        #self.input = layers.Input(shape=(load_args['resize'], load_args['resize'], 1))
        #self.ouput = layers.Dense(self.num_classes, activation='softmax')()
        self.model = Sequential([
            layers.Conv2D(16, 3, input_shape=(load_args['resize'], load_args['resize'], 1), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=l2)),
            layers.MaxPooling2D(),
            layers.Conv2D(24, 3, padding='same', activation=activation, kernel_regularizer=tf.keras.regularizers.l2(l=l2)),
            layers.MaxPooling2D(),
            layers.Conv2D(24, 3, padding='same', activation=activation, kernel_regularizer=tf.keras.regularizers.l2(l=l2)),
            layers.MaxPooling2D(),
            layers.Conv2D(24, 3, padding='same', activation=activation, kernel_regularizer=tf.keras.regularizers.l2(l=l2)),
            layers.MaxPooling2D(),
            layers.Conv2D(24, 3, padding='same', activation=activation, kernel_regularizer=tf.keras.regularizers.l2(l=l2)),
            layers.MaxPooling2D(),
            layers.Dropout(dropout),
            layers.Flatten(),
            layers.Dense(32, activation=activation),
            layers.Dense(self.num_classes, activation='softmax')
        ])

    def train(self):
        self.model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        self.model.summary()
        t0 = time.time()
        self.history = self.model.fit(self.imgGen_tr, validation_data=self.imgGen_va, epochs=self.epochs)
        tEnd = time.time()
        print('\nFinish training! Time cost:', round(tEnd - t0, 2), 's.')
        
    def evaluate(self):
        self.dt = datetime.now().strftime('%y-%m-%d-%H-%M-%S')
        self.folder_name = self.savePATH + self.dt + '_' + self.model_name + '_' + self.activation + '_bs=' + str(self.batch_size) + '_epochs=' + str(self.epochs) + '/'
        try:
            os.makedirs(self.folder_name)
        except FileExistsError:
            pass

        print('Testing performance:')
        testing_performance = self.model.evaluate(self.imgGen_te)
        with open(self.folder_name + 'testing_performance.txt', 'w') as f:
            for row in testing_performance:
                f.write(str(row) + '\n')
        
        print('\nThe model is saved as:')
        fn = self.folder_name + 'model.h5'
        self.model.save(fn)
        print('-->', fn)
            
        print('The model history is saved as:')
        fn = fn.replace('model.h5', 'history.csv')
        pd.DataFrame(self.history.history).to_csv(fn)
        print('-->', fn)

    def predict(self, testDATA:str):
        yp = np.argmax(self.model.predict(self.imgGen_out), axis=1)
        print('\nDistribution of predicted labels:')
        for i, n in enumerate(np.bincount(yp)):
            p = n/len(yp)
            print(f'{i}: {n:4d} ({p:.4f})', end='  ')
            if i % 3 == 2:
                print()
        df_out = pd.read_csv(testDATA, header=0)
        df_out[self.label] = yp
        df_out.to_csv(self.folder_name + 'prediction_out.csv', index=False)

    def plot_training(self, type_:str, figsize:tuple, save_plot=True):
        x = np.arange(len(self.history.epoch))
        plt.figure(figsize=figsize)
        plt.plot(x, self.history.history[type_], label=type_+'(train)')
        plt.plot(x, self.history.history['val_'+type_], label=type_+'(val)')
        plt.legend()
        plt.title('Plot of' + type_.capitalize() + ' of ' + self.model_name)
        plt.xlabel('Epoch')
        if type_ == 'loss':
            plt.ylabel('Loss')
        elif type_ == 'accuracy':
            plt.ylabel('Accuracy')
            plt.ylim(0, 1)
        plt.grid()

        if save_plot:
            fn = self.folder_name + type_ + '_plot.png'
            plt.savefig(fn)
            print('The', type_, 'plot is saved as', fn)