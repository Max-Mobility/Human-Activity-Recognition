#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 15:21:57 2019

@author: xing
"""
#from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
#import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM



#from math import sqrt
#from matplotlib import pyplot

import numpy as np
import os
#import pandas as pd

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def fit_lstm(Dx,Dy, nb_epoch, neurons,load_dir=None,load_old_model=False,lr=0.000001,start_epoch=0):
    if (load_old_model):
        print("loading old model.")
        model=tf.keras.models.load_model(load_dir)
    else:
        model = Sequential()
        model.add(LSTM(neurons, batch_input_shape=(1, 1,Dx.shape[2]), stateful=True))
        model.add(Dense(units=16,activation = 'relu'))
        model.add(Dense(units=1,activation = 'relu'))    
    
    
    opt = tf.keras.optimizers.Adam(lr = lr)
    model.compile(loss='mean_squared_error', optimizer=opt,metrics=['accuracy'])
    
    X,y = data_shuffle(Dx,Dy)
    sample_weight = y*len(y)/y.sum()+1
    for i in range(start_epoch+1,start_epoch+nb_epoch):
        print("epoch"+str(i))
        model.fit(X, y, epochs=1, batch_size=1, verbose=1, shuffle=False,sample_weight=sample_weight)
        #model.train_on_batch(X, y, epochs=1, batch_size=1, verbose=1, shuffle=False,sample_weight=sample_weight)
        model.reset_states()
        saveStr = './model/25hz/keras_LSTM_combinedData_sample_'+str(neurons)+'_i'+str(i)+'.hdf5'
        model.save(saveStr)
        if (i%10==0):
            X,y = data_shuffle(Dx,Dy)
            predict = model.predict(X,batch_size=1,verbose=1)
            predict = predict.reshape([len(predict)])
            print(predict.sum())
            if (predict.sum()<(y.sum()/10)):
                sample_weight = y*len(y)/y.sum()*1+1
            else:
                sample_weight = y*len(y)/y.sum()*2+1 + predict*len(predict)/predict.sum()
        
    return model

def data_shuffle(x,y,gap=150,nonActGap=500):
    
    label_index= np.where(y==1)[0]
    label_index = np.append(label_index,[len(y)-1])
    label_index = np.append([0],label_index)
    segments=np.zeros([0,2],dtype=int)
    last_seg = 0

    for i in range(len(label_index) -1):
        if ((label_index[i+1]-label_index[i])>gap):
            num_seg = int((label_index[i+1]-label_index[i])/nonActGap)+2
            seg_gap = int((label_index[i+1]-label_index[i])/num_seg)
            for j in range(num_seg):
                if j == num_seg-1:  # last segment
                    seg = label_index[i+1]
                else:
                    seg = seg_gap*(j+1)+label_index[i]
                segments = np.append(segments,[[last_seg,seg]],axis=0)
                last_seg=seg
    
    newSeg = np.arange(len(segments))
    np.random.shuffle(newSeg)
    if (len(x.shape)==3):
        newx = np.zeros([0,x.shape[1],x.shape[2]])
    else:
        newx = np.zeros([0,x.shape[1]])
    newy = np.zeros([0])
    #print("number of segments:"+str(len(newSeg)))
    for i in range(len(newSeg)):
        newx = np.append(newx,x[segments[newSeg[i]][0]:segments[newSeg[i]][1]],axis=0)
        newy = np.append(newy,y[segments[newSeg[i]][0]:segments[newSeg[i]][1]],axis=0)
    return newx,newy

if __name__ =='__main__':
    print("gpu disabled")
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    
    ##Use GPU
# =============================================================================
#     os.environ["CUDA_VISIBLE_DEVICES"]="0"
#     gpus = tf.config.experimental.list_physical_devices('GPU')
#     tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)])
# =============================================================================
    
# =============================================================================
#     print("Num GPUs Available: ", get_available_gpus())
#     physical_devices = tf.config.experimental.list_physical_devices('CPU')
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
# =============================================================================
    
    Dx1 = np.loadtxt('./data/PSDSData (17_20)_raw.csv',delimiter=',')
    Dy1 = np.loadtxt('./data/push_finish(17_20)_1.csv',delimiter=',')
    Dx2 = np.loadtxt('./data/PSDSData (30)_raw.csv',delimiter=',')
    Dy2 = np.loadtxt('./data/push_30.csv',delimiter=',')
    Dx3 = np.loadtxt('./data/PSDSData (32)_raw.csv',delimiter=',')
    Dy3 = np.loadtxt('./data/push_32.csv',delimiter=',')
    DataNonAct = np.loadtxt('./data/NonActivityData.csv',delimiter=',')
    DataNonActY = np.zeros_like(DataNonAct[:,1])
    
    DataNonAct_shuffle,DataNonActY_shuffle=data_shuffle(DataNonAct,DataNonActY,nonActGap=1000)
    DataNonAct_sample =DataNonAct_shuffle[0:int(len(DataNonAct_shuffle)/10)]
    DataNonActy_sample =DataNonActY_shuffle[0:int(len(DataNonAct_shuffle)/10)]
    
    Dx3 = Dx3[:,0:9]
    
    useSample =True
    
    Datax = np.empty([0,9])
    Datax = np.append(Datax,Dx1,axis=0)
    Datax = np.append(Datax,Dx2,axis=0)
    Datax = np.append(Datax,Dx3,axis=0)
    if useSample==True:
        Datax = np.append(Datax,DataNonAct_sample,axis=0)
# =============================================================================
#     else:
#         Datax = np.append(Datax,DataNonAct,axis=0)
# =============================================================================
    
    Datay = np.empty([0])
    Datay = np.append(Datay,Dy1,axis=0)
    Datay = np.append(Datay,Dy2,axis=0)
    Datay = np.append(Datay,Dy3,axis=0)
    if useSample==True:
        Datay = np.append(Datay,DataNonActy_sample,axis=0)    
# =============================================================================
#     else:
#         Datay = np.append(Datay,DataNonActY,axis=0)
# =============================================================================
    
    if useSample==True:
        np.savetxt('./data/train/CombineDataX_sample.csv',Datax,delimiter=',')
        np.savetxt('./data/train/CombineDatay_sample.csv',Datay,delimiter=',')
# =============================================================================
#     else:
#         np.savetxt('./data/train/CombineDataX.csv',Datax,delimiter=',')
#         np.savetxt('./data/train/CombineDatay.csv',Datay,delimiter=',')
# =============================================================================
        
    Datax = Datax[:,3:9].reshape(Datax.shape[0],1,6)
    Datax[:,:,3:6] = np.arccos(Datax[:,:,3:6]/10.0)
    Datax = Datax[:10000]
    Datay = Datay[:10000]
    
    start_epoch=0
    model_dir = './model/25hz/keras_LSTM_combinedData_'+str(64)+'_i'+str(start_epoch)+'.hdf5'

    lstm_model=fit_lstm(Datax,Datay,2,64,load_dir=model_dir,load_old_model=False,lr=0.000001,start_epoch=start_epoch)
