# Deep Learning - HW5: Automated Optical Inspection (AOI)

Author: Jay Liao (re6094028@gs.ncku.edu.tw)

This is assignment 5 of Deep Learning, a course at Institute of Data Science, National Cheng Kung University. This project aims to utilize deep learning techniques to perform defect classification for AOI images.

## Data

- Images: please go to the page of the competition on the AIdea [here](https://aidea-web.tw/topic/285ef3be-44eb-43dd-85cc-f0388bf85ea4?focus=intro) to download raw image files with two folders, `./train_images/` and `./test_images/`.

- File name lists of images: `./train.csv` and `./test.txt`.

## Code

- `main_keras.py`: the main program for training a 5-layered CNN with Keras. The following codes demonstrate the model structute.

```
raw_model = Sequential([
    layers.Conv2D(16, 3, input_shape=(size[0], size[1], 1), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.01)),
    layers.MaxPooling2D(),
    layers.Conv2D(24, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.01)),
    layers.MaxPooling2D(),
    layers.Conv2D(24, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.01)),
    layers.MaxPooling2D(),
    layers.Conv2D(24, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.01)),
    layers.MaxPooling2D(),
    layers.Conv2D(24, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.01)),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
])
```

- Source codes for training a 5-layered CNN with Keras:

    -  `./lenet_keras/args.py`: define the arguments parser
    
    -  `./lenet_keras/trainer.py`: class for training, predicting, and evaluating the models
    
    -  `./lenet_keras/load_data.py`: functions for loading images with train/val/test splitting

## Folders

- `./train_images/` should contain 2,528 raw training image files (please go [here](https://aidea-web.tw/topic/285ef3be-44eb-43dd-85cc-f0388bf85ea4?focus=intro) to download). They will be splitted into three subsets after running the main program (folders `./images_tr/`, `./images_va/`, and `./images_te/` will be created in torch version).

- `./test_images/` should contain 10,142 testing image files without ground truth lables (please go [here](https://aidea-web.tw/topic/285ef3be-44eb-43dd-85cc-f0388bf85ea4?focus=intro) to download). The folder `./images_test/` will be created after running the main program of torch version.

- `./output_torch/` and `./output_keras/` will contain trained models, model performances, and experiments results after running. 

## Requirements

```
numpy==1.16.3
pandas==0.24.2
tqdm==4.50.0
opencv-python==3.4.2.16
matplotlib==3.1.3
torch==1.7.1
keras==2.4.3
tensorflow==2.3.1
tensorflow-gpu==2.3.1
```

## Usage

1. Clone this repo.

```
git clone https://github.com/jayenliao/DL-AOI.git
```

2. Set up the required packages.

```
cd DL-AOI
pip3 install -H requirements.txt
```

3. Run the experiments.

```
python3 main_keras.py
python3 main_torch.py
```

## Reference

1. https://github.com/bentrevett/pytorch-image-classification
2. https://github.com/pytorch/tutorials
3. https://github.com/pytorch/examples
4. https://colah.github.io/posts/2014-10-Visualizing-MNIST/
5. https://distill.pub/2016/misread-tsne/
6. https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
7. https://github.com/activatedgeek/LeNet-5
8. https://github.com/ChawDoe/LeNet5-MNIST-PyTorch
9. https://github.com/kuangliu/pytorch-cifar
10. https://github.com/akamaster/pytorch_resnet_cifar10
11. https://sgugger.github.io/the-1cycle-policy.html
