# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:00:47 2019

@author: Guo
"""

import keras
from keras.layers import LSTM,Dense,Input,Flatten,Concatenate
from keras.models import Model,load_model
import numpy as np
import os


def build_model(n_units):
    x_in = Input(shape=(32,9))
    lstm_out,lstm_h,lstm_c = LSTM(n_units, return_state = True)(x_in)
    #x = Flatten()(x_in)
    #lstm_con = Concatenate()([lstm_h,lstm_c])
    
    y_out = Dense(1,activation = 'relu')(lstm_out)
    #y_out = Dense(1,activation = 'relu')(y)
    return Model(inputs=x_in,outputs=y_out)


def build_model_v2(n_units,num_inputs=3):
    x_in = Input(batch_shape=(1,1,num_inputs))
    lstm_out,lstm_h,lstm_c = LSTM(n_units, return_state = True,stateful=True)(x_in)
    '''
    h_in = Input(shape=(n_units,))
    c_in = Input(shape=(n_units,))
    lstm = LSTM(n_units, return_state = True)
    lstm_out,lstm_h,lstm_c = lstm(x_in,initial_state =[h_in,c_in])
    '''
    #x = Flatten()(x_in)
    #lstm_con = Concatenate()([lstm_h,lstm_c])
    
    y_out = Dense(1,activation = 'relu')(lstm_out)
    #y_out = Dense(1,activation = 'relu')(y)
    return Model(inputs=x_in,outputs=y_out)



if __name__== '__main__':
# =============================================================================
#     os.environ["CUDA_VISIBLE_DEVICES"]="7"
#     data_x=np.load('watch_4_29_raw.npy')
#     data_x=data_x.reshape((len(data_x),1,9))
#     data_y=np.load('watch_4_29_y.npy')
#     data_y=np.append(np.zeros(31),data_y)
# =============================================================================
    #os.environ["CUDA_VISIBLE_DEVICES"]="7"
    num_inputs=3
    data_x=np.loadtxt('./TapData/data_raw_active.csv',delimiter=",")
    data_y=np.loadtxt('./TapData/data_y_active.csv',delimiter=",")
    data_x=data_x[:,(6-num_inputs):6].reshape((len(data_x),1,num_inputs))
    preY = np.loadtxt('./TapData/data_preY_3C_fullData.active.csv',delimiter=",")
    '''
    # for model V1
    index = np.random.permutation(len(data_x))
    x=data_x[index]
    y=data_y[index]
    p=0.80
    train_num =int(len(x)*p)
    x_train=x[:train_num,:,:]
    x_valid=x[train_num:,:,:]
    y_train=y[:train_num]
    y_valid=y[train_num:]
    
    sample_weight=y_train*15+1
    realtime_LSTM= build_model(256)
    opt = keras.optimizers.Adam(lr=0.000001)
    realtime_LSTM.compile(optimizer =opt, loss='mse',metrics=['acc'])
    realtime_LSTM.load_weights('realtime_LSTM_useOutepoch_5000loss_0.01_weights.h5')
    realtime_LSTM_Ypredict = realtime_LSTM.predict(data_x[:,:,0:9])
    realtime_LSTM_Ypredict = np.append(np.zeros(31),realtime_LSTM_Ypredict)
    plot(realtime_LSTM_Ypredict)
    np.savetxt('realtime_LSTM_Ypredict.csv',realtime_LSTM_Ypredict)
    his=realtime_LSTM.evaluate(data_x[:,:,0:9],data_y)
    print(his)
    
    '''
    
    # for model V2
    sample_weight=preY*len(preY)/preY.sum()+data_y*len(data_y)/data_y.sum()+1
    realtime_LSTM= build_model_v2(32,num_inputs=num_inputs)
    opt = keras.optimizers.Adam(lr=0.000001)
    realtime_LSTM.compile(optimizer =opt, loss='mse',metrics=['acc'])
    train_round =50
    epochs_per_round=20
    for i in range(train_round):
        print('start training round '+str(i))
        print('training realtime_LSTM model')
        his=realtime_LSTM.fit(data_x,data_y,
                      epochs=(i+1)*epochs_per_round,
                      initial_epoch=i*epochs_per_round,
                      shuffle=False,
                      batch_size=1,sample_weight=sample_weight)
        pre_y=realtime_LSTM.predict(data_x,batch_size=1)
        sample_weight=preY*len(preY)/preY.sum()+data_y*len(data_y)/data_y.sum()+1
        saveStr = 'epoch_'+str((i+1)*epochs_per_round)+'_loss_'+'{:.2f}'.format(his.history['loss'][-1])
        realtime_LSTM.save_weights('./models/25hz/realtime_LSTM_fullActiveData_Tap_32U_3C_DynamicWpreYMix_'+saveStr+'_weights.h5')
        realtime_LSTM.save('./models/25hz/realtime_LSTM_fullActiveData_Tap_32U_3C_DynamicWpreYMix_'+saveStr+'.h5')
    
    
    
    '''
    train_round =100
    epochs_per_round=50
    for i in range(train_round):
        print('start training round '+str(i))
        print('training realtime_LSTM model')
        his=realtime_LSTM.fit(x_train[:,:,0:9],y_train,
                      validation_data=[x_valid[:,:,0:9],y_valid],
                      epochs=(i+1)*epochs_per_round,
                      initial_epoch=i*epochs_per_round,
                      batch_size=64,sample_weight=sample_weight)
        saveStr = 'epoch_'+str((i+1)*epochs_per_round)+'loss_'+'{:.2f}'.format(his.history['loss'][-1])
        realtime_LSTM.save_weights('./models/25hz/realtime_LSTM'+saveStr+'_weights.h5')
        realtime_LSTM.save('./models/25hz/realtime_LSTM'+saveStr+'.h5')
    '''