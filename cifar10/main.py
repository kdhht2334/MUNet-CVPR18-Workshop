__author__ = "DaeHaKim"
# -*- coding: utf-8 -*-
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
app = argparse.ArgumentParser()
app.add_argument("-g", "--gpus", type=int, default=1,
                 help="# of GPUs to use for training")
args = vars(app.parse_args())

G = args["gpus"]

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(G)
set_session(tf.Session(config=config))

print()
print("[INFO] training with {} GPUs ...".format(G))
print()

import numpy as np
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import backend as K
from keras.engine import Input, Model
from keras.callbacks import CSVLogger, LearningRateScheduler
import keras

from layers import MUNET

# training parameters
batch_size = 128
maxepoches = 300
num_classes = 10
learning_rate = 0.1
lr_decay = 1e-6
lrf = learning_rate
initial_conv_depth = 6
conv_depth = 4

csv_logger = CSVLogger('MUNET_cifar10.csv', append=True)

lr_schedule = [0.5, 0.75]
def schedule(epoch_idx):
    if (epoch_idx + 1) < (300 * lr_schedule[0]):
        return 0.1
    elif (epoch_idx + 1) < (300 * lr_schedule[1]):
        return 0.01

    return 0.001


def normalize(X_train,X_test):
    mean = np.mean(X_train,axis=(0,1,2,3))
    std = np.std(X_train, axis=(0,1,2,3))
    X_train = (X_train-mean)/(std+1e-7)
    X_test = (X_test-mean)/(std+1e-7)
    return X_train, X_test

# Keras specific
if K.image_dim_ordering() == "th":
    channel_axis = 1
    input_shape = (3, 32, 32)
else:
    channel_axis = -1
    input_shape = (32, 32, 3)


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = normalize(x_train, x_test)
    
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)
    
    
    # MUNET model
    inputs = Input(input_shape)
    
    prediction = MUNET(inputs)
    model_star = Model(inputs=[inputs], outputs=[prediction])
    model_star.summary()
    params = model_star.count_params()
    params /= 1E6
    print()
    print("Number of params of model: %f M" % (params))
    
    # optimization details
    sgd = optimizers.SGD(lr=lrf, decay=lr_decay, momentum=0.9, nesterov=True)
    model_star.compile(loss='categorical_crossentropy',
                       optimizer=sgd,
                       metrics=['accuracy'])
    
    
    # data augmentation
    datagen = ImageDataGenerator(
        width_shift_range=4./32,   # randomly shift images horizontally (fraction of total width)
        height_shift_range=4./32,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True)      # randomly flip images
    datagen.fit(x_train, augment=True, rounds=2, seed=0)
    
    print()
    print("Training start!!")
    print()
    
    historystarnet = model_star.fit_generator(
            datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=True),
                         steps_per_epoch=x_train.shape[0] // batch_size,
                         epochs=300,
                         validation_data=(x_test, y_test),
                         callbacks=[LearningRateScheduler(schedule)])
    
    #model_star.save_weights('MUNET_cifar10.h5')
    

