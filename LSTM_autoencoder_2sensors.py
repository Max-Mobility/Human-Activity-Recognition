# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 13:36:07 2019

@author: Guo
"""

from numpy import array
import numpy as np
import keras
from keras.models import Sequential, Model,load_model
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot
import os

if __name__=='__main__':
    
    os.environ["CUDA_VISIBLE_DEVICES"]="3"
    data= np.load('watch_norm.npy')
    np.random.shuffle(data)
    p=0.85
    train_num =int(data.shape[0]*p)
    index=[0,1,2,3,4,5,13,14,15,16,17,18]
    data_train=data[:train_num,:,index]
    
    
    
    inputs = Input(shape=(128, 12))

    x1=LSTM(384, activation='relu')(inputs)
    
    x2=RepeatVector(128)(x1)
    
    x3=LSTM(384, activation='relu', return_sequences=True)(x2)
       
    x4=TimeDistributed(Dense(12))(x3)
    
    model = Model(inputs=inputs, outputs=x4)
    #model = load_model('lstmModel_2sensors.h5')
    
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
        model.save('lstmModel_2sensors.h5')