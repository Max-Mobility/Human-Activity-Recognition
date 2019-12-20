#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 14:00:44 2019

@author: xing
"""

import numpy as np
import tensorflow as tf
from tqdm import tqdm

interpreter=tf.lite.Interpreter(model_path="keras_acos_1024.tflite")
interpreter.allocate_tensors()

input_details=interpreter.get_input_details()
output_details=interpreter.get_output_details()

data_x=np.loadtxt('./watch0.csv',delimiter=",",dtype=np.float32)
input_data=data_x#[:,3:9]
input_data[:,3:6]=np.arccos(input_data[:,3:6]/10)
#input_data[:,0:3]= input_data[:,0:3]*1.5 #np.array([ -0.06250018, 0.524807, 0.40479565, -2.744695, -9.367873, 0.9380921], dtype=np.float32)#
x_states=np.zeros([1,128],dtype=np.float32)
#x_states=np.loadtxt('./x_previous_states.csv', dtype=np.float32)
#x_states=x_states.reshape([1,128])
output_data=np.zeros(len(input_data))

print(input_data.shape,x_states.shape)

for i in tqdm(range(len(input_data))):
    input_x=input_data[i,:].reshape([1,6])
#input_x=input_data.reshape([1,6])
    interpreter.set_tensor(input_details[0]['index'],input_x)
    interpreter.set_tensor(input_details[1]['index'],x_states)
    interpreter.invoke()
    output_data[i]=interpreter.get_tensor(output_details[0]['index'])
#output_data=interpreter.get_tensor(output_details[0]['index'])
    x_states=interpreter.get_tensor(output_details[1]['index'])
    
    
np.savetxt('watch0_predict_lite.csv',output_data)
#print(output_data)
#np.savetxt('pred_states.csv',x_states)