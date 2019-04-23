# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 13:36:07 2019
build models that train each channel seperately 

@author: Guo
"""

from numpy import array
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot
import os

def build_model(inputLength=128,channel=1,reduction_rate=4):
    inputs = Input(shape=(inputLength, channel))

    x1=LSTM(int(inputLength/reduction_rate), activation='relu')(inputs)
    
    x2=RepeatVector(inputLength)(x1)
    
    x3=LSTM(int(inputLength/reduction_rate), activation='relu', return_sequences=True)(x2)
       
    x4=TimeDistributed(Dense(channel))(x3)
    
    return Model(inputs=inputs, outputs=x4)


if __name__=='__main__':
    
    os.environ["CUDA_VISIBLE_DEVICES"]="3"
    data= np.load('watch_norm.npy')
    np.random.shuffle(data)
    p=0.85
    train_num =int(data.shape[0]*p)
    gyro_train=data[:train_num,:,0:3]
    linearAcc_train=data[:train_num,:,3:6]
    gravity_train=data[:train_num,:,6:9]
    gameVec_train=data[:train_num,:,9:13]
    data_train=data[:train_num,:,:]
    
    gyro_test =data[train_num:,:,0:3]
    linearAcc_test =data[train_num:,:,3:6]
    gravity_test =data[train_num:,:,6:9]
    gameVec_test =data[train_num:,:,9:13]
    
    
    opt=keras.optimizers.Adam(lr=0.01)
    channels=26
    models={}
    for i in range(channels):
        models[i]=build_model()
        models[i].compile(optimizer=opt, loss='mse',metrics=['accuracy'])
    
    for i in range(500):
        for j in range(channels):
            models[j].fit(data_train[:,:,j],data_train[:,:,j], epochs=20,batch_size=256)
            saveStr='lstmModel_channel_'+str(j)+'.h5'
            model.save(saveStr)