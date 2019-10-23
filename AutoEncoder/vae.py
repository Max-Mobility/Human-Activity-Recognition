#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 10:17:49 2019

@author: xing
"""

from keras.layers import Lambda, Input, Dense, Conv1D, Flatten, Reshape, MaxPooling1D,UpSampling1D
from keras.models import Model
from keras.losses import mse

from keras import optimizers
from keras import backend as K
import keras.callbacks as kc
import os
import numpy as np

"""Reparameterization trick by sampling from an isotropic unit Gaussian.
# Arguments
    args (tensor): mean and log of variance of Q(z|X)
# Returns
    z (tensor): sampled latent vector
"""
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

class NewCallback(kc.Callback):
    def __init__(self, alpha):
        self.alpha=alpha

    def on_epoch_begin(self, epoch, logs={}):
        #K.set_value(self.alpha, 1/(1+np.exp((3000-epoch)/200)))
        #K.set_value(self.alpha, 1)#/(1+np.exp((3000-epoch)/200)))
        a=K.get_value(self.alpha)
        if (epoch%1000==0):
            K.set_value(self.alpha, a/2)
        else:
            K.set_value(self.alpha, a)
        print(K.get_value(self.alpha))


def vae(input_shape,latent_dim,intermediate_dim,alpha):
    # VAE model

    inputs= Input(shape=input_shape, name='encoder_input')
    x1=Conv1D(128,3,padding='same',activation='relu')(inputs)
    x1=MaxPooling1D(pool_size=2, strides=2, padding='valid')(x1)
    x2=Conv1D(256,3,padding='same',activation='relu',strides=1)(x1)
    x2=MaxPooling1D(pool_size=2, strides=2, padding='valid')(x2)
    x2=Conv1D(384,3,padding='same',activation='relu',strides=1)(x2)
    x2=Flatten()(x2)
    x=Dense(intermediate_dim, activation='relu')(x2)
    
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var',activation='softplus')(x)
    
    # Sampling layer
    z=Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    
    # Encoder
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    #encoder.summary()
    
    # Decoder
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(3072, activation='relu')(latent_inputs)#
    x = Reshape((8,384))(x)
    x3= Conv1D(384,3,padding='same',activation='relu', strides=1)(x)
    x3= UpSampling1D(2)(x3)
    x4= Conv1D(256,3,padding='same',activation='relu', strides=1)(x3)
    x4= UpSampling1D(2)(x4)
    x4= Conv1D(128,3,padding='same',activation='relu',strides=1)(x4)
    
    
    outputs= Conv1D(18,3,padding='same', activation='linear')(x4)
    #outputs = Dense(original_dim, activation='tanh')(x4)
    
    decoder = Model(latent_inputs, outputs, name='decoder')
    #decoder.summary()
    
    
    outputs=decoder(encoder(inputs)[2])
    vae= Model(inputs, outputs, name='vae')
    '''
    reconstruction_loss = K.mean(mse(K.batch_flatten(inputs), K.batch_flatten(outputs)), 
                                axis=-1)
    kl_loss = 1+z_log_var-K.square(z_mean)- K.exp(z_log_var)
    kl_loss = K.mean(kl_loss, axis=-1)
    kl_loss *= -0.5
    '''
    
    reconstruction_loss = K.mean(mse(K.flatten(inputs), K.flatten(outputs))) 
                                #axis=-1)
    kl_loss = 1+z_log_var-K.square(z_mean)- K.exp(z_log_var)
    kl_loss = K.mean(K.sum(kl_loss, axis=-1))
    kl_loss *= -0.5
    
    vae_loss = (reconstruction_loss + alpha*kl_loss)
    vae.add_loss(vae_loss)
    opt=optimizers.adam(lr=1e-4)
    vae.compile(optimizer=opt)
    vae.metrics_tensors.append(reconstruction_loss)
    vae.metrics_names.append("rc_loss")
    vae.metrics_tensors.append(kl_loss)
    vae.metrics_names.append("kl_loss")
    
    return vae,encoder,decoder
        
if __name__ =='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="4"
    data=np.load('./watch_norm_32_4_29.npy')
    x_train=data[:20736,:,:]
    print(x_train.shape)
    
    original_dim=data.shape[1] * data.shape[2]
    input_shape=(data.shape[1],data.shape[2])
    intermediate_dim = 1024
    batch_size = 256
    latent_dim = 256
    epochs=10000
    
    alpha = K.variable(0.00001)
    
    vae,encoder,decoder=vae(input_shape, latent_dim,intermediate_dim,alpha)
    
    vae.load_weights('./models/25hz/vae_epoch_10000rc_0.02_KL_367.35_weights.h5')
    train_round =10
    epochs_per_round=1000
    
    for i in range(train_round):
        print('start training round '+str(i))
        print('training lstm_auto model')
        his=vae.fit(x_train, epochs=(i+1)*epochs_per_round+10000,
                    initial_epoch=i*epochs_per_round+10000,
                    batch_size=batch_size,callbacks=[NewCallback(alpha),])
        saveStr = 'epoch_'+str((i+1)*epochs_per_round)+'rc_'+'{:.2f}'.format(his.history['rc_loss'][-1])+'_KL_'+'{:.2f}'.format(his.history['kl_loss'][-1])
        vae.save_weights('./models/25hz/vae_'+saveStr+'_weights.h5')
        vae.save('./models/25hz/vae_'+saveStr+'.h5')
    
