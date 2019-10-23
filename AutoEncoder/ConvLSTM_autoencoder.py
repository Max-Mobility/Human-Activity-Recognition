# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 13:36:07 2019

@author: Guo
"""

from numpy import array
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input,Conv1D,UpSampling1D,BatchNormalization,Activation,MaxPooling1D
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot
import os

def build_model(channel=12,sensors=4,multi=3):
    
    inputs = Input(shape=(128, channel))
    x0=Conv1D(filters =8*sensors*multi, kernel_size=5,strides = 1, padding = 'same')(inputs)
    x0     = BatchNormalization()(x0) 
    x0     = Activation('relu')(x0)
    x0     = MaxPooling1D(pool_size=3,strides = 2,padding='same')(x0)
    
    x0     = Conv1D(filters =12*sensors*multi, kernel_size=3,strides = 1, padding = 'same')(x0)
    x0     = BatchNormalization()(x0) 
    x0     = Activation('relu')(x0)
    x0     = MaxPooling1D(pool_size=2,strides = 2,padding='same')(x0)
    
    x1=LSTM(768, activation='relu')(x0)
    
    x2=RepeatVector(32)(x1)
    
    x3=LSTM(768, activation='relu', return_sequences=True)(x2)
        
    x4 = UpSampling1D(2)(x3)
    x5     = Conv1D(filters =8*sensors*multi, kernel_size=3,strides = 1, padding = 'same')(x4)
    x5     = BatchNormalization()(x5) 
    x5     = Activation('relu')(x5)
    x5     = UpSampling1D(2)(x5)
    
    x6=Conv1D(filters=channel,kernel_size=5,strides=1,padding = 'same')(x5)
        
    model = Model(inputs=inputs, outputs=x6)
    return model


if __name__=='__main__':
    
    os.environ["CUDA_VISIBLE_DEVICES"]="6"
    data= np.load('watch_norm.npy')
    channel=12
    sensors=2
    multi=6
    saveStr='ConvLstmModel_'+str(sensors)+'.h5'
    np.random.shuffle(data)
    p=0.85
    train_num =int(data.shape[0]*p)
    index=[0,1,2,3,4,5,13,14,15,16,17,18]
    data_train=data[:train_num,:,index]
    model=build_model(channel=channel,sensors=sensors,multi=multi)
    #model=load_model(saveStr)
    opt=keras.optimizers.Adam(lr=0.0001)
    model.compile(optimizer=opt, loss='mse',metrics=['accuracy'])
    train_round =10
    epochs_per_round=20
    for i in range(train_round):
        print('start training round '+str(i))
        print('training full LSTM model')
        model.fit(data_train,data_train,
                  batch_size=256,
                  epochs=(i+1)*epochs_per_round,
                  validation_split=0.1,verbose=1,initial_epoch=i*epochs_per_round
                  )
        
        model.save(saveStr)