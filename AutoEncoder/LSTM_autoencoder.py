# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 13:36:07 2019

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

if __name__=='__main__':
    
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
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
    
    
    inputs = Input(shape=(128, 26))

    x1=LSTM(768, activation='relu')(inputs)
    
    x2=RepeatVector(128)(x1)
    
    x3=LSTM(768, activation='relu', return_sequences=True)(x2)
       
    x4=TimeDistributed(Dense(26))(x3)
    
    model = Model(inputs=inputs, outputs=x4)
    
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
                  validation_split=0.1,verbose=1,initial_epoch=i*epochs_per_round)
        model.save('lstmModel.h5')