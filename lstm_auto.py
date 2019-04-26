#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:50:48 2019

@author: xing
"""

from keras.models import Model
from keras.layers import Input, LSTM, Dense
#from keras.utils.vis_utils import plot_model

import numpy as np


def define_models(n_input, n_output, n_units):
    
    # Define on input sequence and process it.
    encoder_inputs=Input(shape=(128, n_input))
    encoder=LSTM(n_units, return_state = True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)

    encoder_states = [state_h, state_c]

    # Set up the decoder, using 'encoder_states' as initial state
    decoder_inputs = Input(shape=(128, n_output))

    # Set up decoder to return full output sequences,
    # and to return internal states as well.
    # Don't use the return states in the training model, but will use them in inference.
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _, = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn 'encoder_input_date' & 'decoder_input_data'
    # into 'decoder_target_data'
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Plot model
    #plot_model(model, to_file='model.png', show_shapes=True)

    # Define encoder inference model
    encoder_model = Model(encoder_inputs, encoder_states)

    # Define decoder inference model
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]

    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs]+decoder_states_inputs, [decoder_outputs]+decoder_states)

    # Summarize model
    #plot_model(encoder_model, to_file='encoder_model.png', show_shapes=True)
    #plot_model(decoder_model, to_file='decoder_model.png', show_shapes=True)

    # Note that the encoder LSTM does not directly pass its outputs as inputs to the decoder LSTM; 
    # as noted above, the decoder uses the final hidden and cell states as the initial state for the decoder.

    # Also note that the decoder LSTM only passes the sequence of hidden states to the Dense for output, 
    # not the final hidden and cell states as suggested by the output shape information.
    
    return model, encoder_model, decoder_model

if __name__ =='__main__':
    data=np.load('watch_norm_128.npy')
    y=np.flip(data, 1)
    z=np.zeros((y.shape[0],1,y.shape[2]))
    X2=np.concatenate((z,y[:,:-1,:]),axis=1)

    # Configure
    n_steps_in = 128
    n_steps_out = 128
    n_features = 26
    n_units=512
    
    # Define model
    train, infenc, infdec = define_models(n_features, n_features,n_units)
    train.compile(optimizer='adam', loss='mse', metrics=['acc'])
    train.fit([data, X2],y,epochs=100)