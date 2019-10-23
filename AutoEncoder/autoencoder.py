# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 13:05:48 2019

@author: Guo
"""
import keras
from keras.layers import Input, Conv1D,Activation,MaxPooling1D,Dense,Flatten,UpSampling1D
from keras.layers.normalization import BatchNormalization
from keras.models import Model,load_model
import tensorflow as tf
import numpy as np
import os

def get_sub_model(myinputs,channel=3,sensors=1,multi=1):
    x     = Conv1D(filters =8*sensors*multi, kernel_size=3,strides = 1, padding = 'same')(myinputs)
    x     = BatchNormalization()(x) 
    x     = Activation('relu')(x)
    x     = MaxPooling1D(pool_size=3,strides = 2,padding='same')(x)
    #  x shape is 64X16 =3072
    x     = Conv1D(filters =12*sensors*multi, kernel_size=3,strides = 1, padding = 'same')(x)
    x     = BatchNormalization()(x) 
    x     = Activation('relu')(x)
    x     = MaxPooling1D(pool_size=2,strides = 2,padding='same')(x)
    #  x shape is 32X24 = 2048
    x     = Conv1D(filters =16*sensors*multi, kernel_size=3,strides = 1, padding = 'same')(x)
    x     = BatchNormalization()(x) 
    x     = Activation('relu')(x)
    x     = MaxPooling1D(pool_size=2,strides = 2,padding='same')(x)
    # x shape is 16*32 = 1536
    #x     = Conv1D(filters =20*sensors*multi, kernel_size=3,strides = 1, padding = 'same')(x)
    #x     = BatchNormalization()(x) 
    #x     = Activation('relu')(x)
    #x     = MaxPooling1D(pool_size=2,strides = 2,padding='same')(x)
    # x shape is 8*32 = 1024
    x     = Conv1D(filters =24*sensors, kernel_size=3,strides = 1, padding = 'same')(x)
    # x shape is 4*32 = 512

    latent_vector = Flatten()(x)
    # decoder x = 4*128
    x     = Conv1D(filters =24*sensors*multi, kernel_size=3,strides = 1, padding = 'same')(x)
    x     = BatchNormalization()(x) 
    x     = Activation('relu')(x)
    x     = UpSampling1D(2)(x)
    # x shape is 8*128
    #x     = Conv1D(filters =20*sensors*multi, kernel_size=3,strides = 1, padding = 'same')(x)
    #x     = BatchNormalization()(x) 
    #x     = Activation('relu')(x)
    #x     = UpSampling1D(2)(x)
    # x shape is 16*96
    x     = Conv1D(filters =16*sensors*multi, kernel_size=3,strides = 1, padding = 'same')(x)
    x     = BatchNormalization()(x) 
    x     = Activation('relu')(x)
    x     = UpSampling1D(2)(x)
    # x shape is 32*64
    x     = Conv1D(filters =12*sensors*multi, kernel_size=3,strides = 1, padding = 'same')(x)
    x     = BatchNormalization()(x) 
    x     = Activation('relu')(x)
    x     = UpSampling1D(2)(x)
    # x shape is 64*48
    x     = Conv1D(filters =8*sensors*multi, kernel_size=3,strides = 1, padding = 'same')(x)
    x     = BatchNormalization()(x) 
    x     = Activation('relu')(x)
    #x     = UpSampling1D(2)(x)
    # x shape is 128*48
    y     = Conv1D(filters =channel, kernel_size=3,strides = 1, padding = 'same')(x)
    #y     = Activation('relu')(x)
    return latent_vector,y

def build_model(name="push_detect",channel=3,sensors=1,multi=1):
    # input shape 128X26 =3382
    with tf.name_scope(name):
        myInputs = Input(shape=(32,channel))
                
        LV,Y = get_sub_model(myInputs,channel=channel,sensors=sensors,multi=multi)
        
        autoencoder = Model (inputs=myInputs,
                             outputs =Y)
    
    return autoencoder

if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    #fullModel=load_model('fullModel.h5')
    fullModel=build_model(channel=18,sensors=6,multi=4)
    opt = keras.optimizers.SGD(lr=0.3, momentum=0.0, decay=0.000, nesterov=False)
    fullModel.compile(optimizer=opt,loss='mse',metrics=['accuracy'])
    data= np.load('watch_norm_32_4_29.npy')
    np.random.shuffle(data)
    p=0.85
    train_num =int(data.shape[0]*p)
    data_train=data[:train_num,:,:]
    data_valid=data[train_num:,:,:]
    train_round =50
    epochs_per_round=50
    for i in range(train_round):
        print('start training round '+str(i))
        print('training full model')
        fullModel.fit(data_train,data_train,
                      batch_size=256,
                      epochs=(i+1)*epochs_per_round,
                      validation_data=[data_valid,data_valid],verbose=1,initial_epoch=i*epochs_per_round)
        fullModel.save('./models/25hz/fullModel.h5')
