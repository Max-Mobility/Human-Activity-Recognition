#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 09:22:18 2019

@author: xing
"""

import numpy as np

data_x=np.loadtxt('./PSDSData (32)_raw.csv',delimiter=",",dtype=np.float32)

max_g=np.max(abs(data_x[:,6:9]))
print(max_g)

theta_x=np.arccos(data_x[:,6]/max_g).reshape([len(data_x),1])
theta_y=np.arccos(data_x[:,7]/max_g).reshape([len(data_x),1])
theta_z=np.arccos(data_x[:,8]/max_g).reshape([len(data_x),1])
print(data_x.shape,theta_x.shape,theta_y.shape,theta_z.shape)

data=np.concatenate([data_x,theta_x,theta_y,theta_z],axis=1)
print(data.shape)
np.savetxt("./PSDSData (32)_acos.csv",data,delimiter=',')