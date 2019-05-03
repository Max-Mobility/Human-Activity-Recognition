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
        K.set_value(self.alpha, 1/(1+np.exp((2500-epoch)/200)))
        print(K.get_value(self.alpha))


def vae(input_shape,latent_dim):
    # VAE model

    inputs= Input(shape=input_shape, name='encoder_input')
    x1=Conv1D(128,3,padding='same',activation='tanh')(inputs)
    x1=MaxPooling1D(pool_size=2, strides=2, padding='valid')(x1)
    x2=Conv1D(256,3,padding='same',activation='tanh',strides=1)(x1)
    x2=MaxPooling1D(pool_size=2, strides=2, padding='valid')(x2)
    x2=Conv1D(384,3,padding='same',activation='tanh',strides=1)(x2)
    x2=Flatten()(x2)
    x=Dense(intermediate_dim, activation='tanh')(x2)
    
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var',activation='relu')(x)
    
    # Sampling layer
    z=Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    
    # Encoder
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    #encoder.summary()
    
    # Decoder
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(3072, activation='relu')(latent_inputs)#
    x = Reshape((8,384))(x)
    x3= Conv1D(384,3,padding='same',activation='tanh', strides=1)(x)
    x3= UpSampling1D(2)(x3)
    x4= Conv1D(256,3,padding='same',activation='tanh', strides=1)(x3)
    x4= UpSampling1D(2)(x4)
    x4= Conv1D(128,3,padding='same',activation='tanh',strides=1)(x4)
    
    
    outputs= Conv1D(18,3,padding='same', activation='tanh')(x4)
    #outputs = Dense(original_dim, activation='tanh')(x4)
    
    decoder = Model(latent_inputs, outputs, name='decoder')
    #decoder.summary()
    
    
    outputs=decoder(encoder(inputs)[2])
    vae= Model(inputs, outputs, name='vae')
    
    reconstruction_loss = K.mean(mse(K.batch_flatten(inputs), K.batch_flatten(outputs)), 
                                axis=-1)
    kl_loss = 1+z_log_var-K.square(z_mean)- K.exp(z_log_var)
    kl_loss = K.mean(kl_loss, axis=-1)
    kl_loss *= -0.5
    
    
    vae_loss = K.mean(reconstruction_loss + alpha*kl_loss)
    vae.add_loss(vae_loss)
    opt=optimizers.adam(lr=1e-4)
    vae.compile(optimizer=opt)
    vae.metrics_tensors.append(reconstruction_loss)
    vae.metrics_names.append("rc_loss")
    vae.metrics_tensors.append(kl_loss)
    vae.metrics_names.append("kl_loss")
    
    return vae,encoder,decoder
        
if __name__ =='__main__':
    data=np.load('./watch_norm_128_4_29_.npy')
    x_train=data[:20928,:,:]
    
    original_dim=data.shape[1] * data.shape[2]
    input_shape=(data.shape[1],data.shape[2])
    intermediate_dim = 1024
    batch_size = 32
    latent_dim = 128
    epochs=10000
    
    alpha = K.variable(0.)
    
    vae,encoder,decoder=vae(input_shape, latent_dim)
      
    his=vae.fit(x_train, epochs=epochs, batch_size=batch_size,callbacks=[NewCallback(alpha),])
    vae.save_weights('vae_mlp_weights.h5')
    
