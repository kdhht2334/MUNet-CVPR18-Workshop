__author__ = "DaeHaKim"
# -*- coding: utf-8 -*-
from keras.layers import Activation, concatenate, add, Dense
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, \
                         GlobalAveragePooling2D
from keras.layers.advanced_activations import ELU
from keras import regularizers

num_classes = 10
initial_conv_depth = 6
conv_depth = 4

num_MU_layer1 = 4
num_MU_layer2 = 8
num_MU_layer3 = 16

def Initial_MU(depth, inp):
    
    x1 = Conv2D(depth, (3, 3), padding='same', 
                kernel_initializer="he_uniform", 
                kernel_regularizer=regularizers.l2(0.0001), 
                data_format='channels_last')(inp)
    x1 = Activation(ELU())(x1)
    x1 = BatchNormalization()(x1)

    x2 = Conv2D(depth, (1, 3), padding='same', 
                kernel_initializer="he_uniform", 
                kernel_regularizer=regularizers.l2(0.0001), 
                data_format='channels_last')(x1)
    x2 = Activation(ELU())(x2)
    x2 = BatchNormalization()(x2)
    
    x3 = Conv2D(depth, (3, 1), padding='same', 
                kernel_initializer="he_uniform", 
                kernel_regularizer=regularizers.l2(0.0001), 
                data_format='channels_last')(x1)
    x3 = Activation(ELU())(x3)
    x3 = BatchNormalization()(x3)
    
    out1 = add( [x1, x2] )
    out2 = add( [x1, x3] )
    return out1, out2


def MU_2_path(depth, nodes):
    
    con = concatenate(nodes)

    x1 = Conv2D(depth, (3, 3), padding='same', 
                kernel_initializer="he_uniform", 
                kernel_regularizer=regularizers.l2(0.0001), 
                data_format='channels_last')(con)
    x1 = Activation(ELU())(x1)
    x1 = BatchNormalization()(x1)

    x2 = Conv2D(depth, (1, 3), padding='same', 
                kernel_initializer="he_uniform", 
                kernel_regularizer=regularizers.l2(0.0001), 
                data_format='channels_last')(x1)
    x2 = Activation(ELU())(x2)
    x2 = BatchNormalization()(x2)
    
    x3 = Conv2D(depth, (3, 1), padding='same', 
                kernel_initializer="he_uniform", 
                kernel_regularizer=regularizers.l2(0.0001), 
                data_format='channels_last')(x1)
    x3 = Activation(ELU())(x3)
    x3 = BatchNormalization()(x3)
    
    out1 = add( [x1, x2] )
    out2 = add( [x1, x3] )
    return out1, out2


def MU_1_path(depth, nodes):
    
    con = concatenate(nodes)

    x1 = Conv2D(depth, (3, 3), padding='same', 
                kernel_initializer="he_uniform", 
                kernel_regularizer=regularizers.l2(0.0001), 
                data_format='channels_last')(con)
    x1 = Activation(ELU())(x1)
    x1 = BatchNormalization()(x1)

    x2 = Conv2D(depth, (1, 3), padding='same', 
                kernel_initializer="he_uniform", 
                kernel_regularizer=regularizers.l2(0.0001), 
                data_format='channels_last')(x1)
    x2 = Activation(ELU())(x2)
    x2 = BatchNormalization()(x2)
    
    x3 = Conv2D(depth, (3, 1), padding='same', 
                kernel_initializer="he_uniform", 
                kernel_regularizer=regularizers.l2(0.0001), 
                data_format='channels_last')(x1)
    x3 = Activation(ELU())(x3)
    x3 = BatchNormalization()(x3)
    
    out = add( [x2, x3] )
    return out


def pooling(depth, x):
    x = Conv2D(depth, (1, 1), padding='same', 
               kernel_initializer="he_uniform", 
               kernel_regularizer=regularizers.l2(0.0001), 
               data_format='channels_last')(x)
    x = Activation(ELU())(x)
    x = BatchNormalization()(x)
    
    out = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(x)
    return out


def MU1(inp1, inp2):
    inp1 = Dropout(0.2)(inp1)
    inp2 = Dropout(0.2)(inp2)
    
    node1 = []; node2 = []
    for i in range(num_MU_layer1):
        node1.append(MU_1_path(initial_conv_depth, 
                               [inp1, inp2, inp1, inp2]))
    for i in range(num_MU_layer1):
        node2.append(MU_1_path(initial_conv_depth, 
                               [node1[0], node1[1], node1[2], node1[3]]))
        
    for i in range(num_MU_layer1):
        node2[i] = pooling(conv_depth, node2[i])
    
    node3_1 = []; node3_2 = []
    for i in range(num_MU_layer1):
        node3_1.append(MU_2_path(conv_depth, 
                                 [node2[0], node2[1], node2[2],
                                 node2[3], node2[1], node2[3]])[0])
        node3_2.append(MU_2_path(conv_depth, 
                                 [node2[0], node2[1], node2[2],
                                 node2[3], node2[1], node2[3]])[1])
    return node3_1, node3_2


def MU2(inp1, inp2):
    for i in range(num_MU_layer1):
        inp1[i] = Dropout(0.5)(inp1[i])
        inp2[i] = Dropout(0.5)(inp2[i])
        
    node4 = []; node5 = []
    for i in range(num_MU_layer2):
        node4.append(MU_1_path(conv_depth, 
                               [inp1[0], inp2[0], inp1[1], inp2[1],
                               inp1[2], inp2[2], inp1[3], inp2[3]]))
    for i in range(num_MU_layer2):
        node5.append(MU_1_path(conv_depth, 
                               [node4[0], node4[1], node4[2], node4[3], 
                               node4[4], node4[5], node4[6], node4[7]]))
    
    for i in range(num_MU_layer2):
        node5[i] = pooling(conv_depth, node5[i])
        
    node6_1 = []; node6_2 = []
    for i in range(num_MU_layer2):
        node6_1.append(MU_2_path(conv_depth, 
                                 [node5[0], node5[1], node5[2], node5[3], node5[4],
                                 node5[5], node5[6], node5[7], node5[0], node5[1]])
                                 [0])
        node6_2.append(MU_2_path(conv_depth, 
                                 [node5[0], node5[1], node5[2], node5[3], node5[4],
                                 node5[5], node5[6], node5[7], node5[0], node5[1]])
                                 [1])
    return node6_1, node6_2


def MU3(inp1, inp2):
    for i in range(num_MU_layer2):
        inp1[i] = Dropout(0.5)(inp1[i])
        inp2[i] = Dropout(0.5)(inp2[i])
        
    node7 = []
    for i in range(num_MU_layer3):
        node7.append(MU_1_path(conv_depth, 
                               [inp1[0], inp2[0], inp1[1], inp2[1],
                               inp1[2], inp2[2], inp1[3], inp2[3],
                               inp1[4], inp2[4], inp1[5], inp2[5],
                               inp1[6], inp2[6], inp1[7], inp2[7]]))
    return node7


def MUNET(inp):
    node0_1, node0_2 = Initial_MU(initial_conv_depth, inp)
    out1_1, out1_2 = MU1(node0_1, node0_2)
    out2_1, out2_2 = MU2(out1_1, out1_2)
    out3 = MU3(out2_1, out2_2)
    
    bottleneck = concatenate( [out3[0], out3[1], out3[2], out3[3],
                               out3[4], out3[5], out3[6], out3[7],
                               out3[8], out3[9], out3[10], out3[11],
                               out3[12], out3[13], out3[14], out3[15]] )
    gap = GlobalAveragePooling2D()(bottleneck)
    
    prediction = Dense(num_classes)(gap)
    prediction = Activation('softmax')(prediction)

    return prediction        

