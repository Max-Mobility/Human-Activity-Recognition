#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 15:50:38 2019

@author: xing
"""

import tensorflow as tf

graph_def_file="./retrain/data17_20/realLSTM_data_U6_5P_64_i1.pb"
input_arrays=["input_tensor", "input_state"]
output_arrays=["output", "output_state"]

converter=tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays)
tflite_model=converter.convert()
open("./keras_acos_1024.tflite", "wb").write(tflite_model)
