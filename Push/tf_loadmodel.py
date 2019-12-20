#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 14:30:21 2019

@author: xing
"""

import tensorflow as tf
import numpy as np
from tqdm import tqdm

data_x=np.loadtxt('./PSDSData (17_20)_raw.csv',delimiter=",",dtype=np.float32)
input_data=data_x[:,3:9]
data_y=np.loadtxt('./push_finish(17_20)_1.csv',delimiter=",")
datay=data_y
data_y=data_y.reshape(len(data_y),1)
data_state=np.loadtxt('./retrain/data17_20/realLSTM_data_U6_5P_64_i8175_mse4.54_21.94_data_state.csv')
data_state=data_state.reshape(1,len(data_state))
loss_weight=np.loadtxt('./retrain/data17_20/realLSTM_data_U6_5P_64_i8175_mse4.54_21.94_loss_weight.csv')
loss_weight=loss_weight.reshape(len(loss_weight),1)

#x_states=np.zeros([1,128],dtype=np.float32)
#output_data=np.zeros(len(input_data))

#GRAPH_PB_PATH = './model/25hz/6t/5p/realLSTM_data_U6_5P_64_i100_mse8.38_1.92.pb'
with tf.compat.v1.Session() as sess1:
    init=tf.global_variables_initializer()
    sess1.run(init)
    new_saver =tf.train.import_meta_graph('./retrain/data17_20/realLSTM_data_U6_5P_64_i1-1.meta')
    new_saver.restore(sess1,('./retrain/data17_20/realLSTM_data_U6_5P_64_i8175-8175'))
    graph=tf.get_default_graph()
    #print([n.name for n in tf.get_default_graph().as_graph_def().node])
    train_step=graph.get_operation_by_name("train_op")
    
    
    iters=8175
    predictY=np.zeros([len(data_x)])
    while iters<10001:
        for i in tqdm(range(len(data_x))):
            feed={"input_tensor:0":input_data[i:i+1],
              "input_state:0":data_state,
              "labels:0":data_y[i:i+1],
              "loss_weight:0":loss_weight[i:i+1]
               }
            sess1.run(train_step,feed_dict=feed)
            data_state,_predictY = sess1.run(["output_state:0","output:0"],feed_dict=feed)

            #print(data_h.shape)
            #data_c = sess.run(c,feed_dict=feed)
            predictY[i]=_predictY[0]
            #print(i)
        
        if iters%1==0:
            if iters==1:
                new_saver.save(sess1,'./retrain/data17_20/realLSTM_data_U6_5P_64'+'_i'+str(iters),global_step=iters)
            else:
                new_saver.save(sess1,'./retrain/data17_20/realLSTM_data_U6_5P_64'+'_i'+str(iters),global_step=iters,write_meta_graph=False)
                
            print(iters)
            mse = (np.square(predictY-data_y.reshape([len(data_x)]))).mean(axis=0)
            loss_weight = loss_weight.reshape([len(loss_weight)])
            mse_weight =np.square(np. multiply((predictY-data_y.reshape([len(data_x)])),loss_weight)).mean(axis=0)
            loss_weight = loss_weight.reshape([len(loss_weight),1])
            #taps_predict = detect(predictY,raw=data_x)
            #FP,FN,TP = verify(taps_predict,taps_label)
            #file = open(logFile,'a+')
            #file.write('\niters:'+str(iters))
            #file.write('\nmse:'+str(mse))
            #file.write('\nmse_weight:'+str(mse_weight))
            #file.write('\ntotal taps:'+str(taps_predict.sum()))
            #file.write('\nFP,FN,TP:'+str(FP.sum())+', '+str(FN.sum())+', '+str(TP.sum()))
            #file.close()
            print('mse:'+str(mse))
            print('mse_weight:'+str(mse_weight))
            #print('total taps:'+str(taps_predict.sum()))
            #print('FP,FN,TP:'+str(FP.sum())+', '+str(FN.sum())+', '+str(TP.sum()))
            if iters%5==0:
    
                loss_weight = predictY*len(predictY)/predictY.sum()*50+datay*len(datay)/datay.sum()+1
                loss_weight = loss_weight.reshape([len(loss_weight),1])
                print('noneTapWeight'+str(len(predictY)/predictY.sum()))

        
        #output_graph='./push.pb'
        output_graph='./retrain/data17_20/realLSTM_data_U6_5P_64'+'_i'+str(iters)+'_mse'+'{0:.2f}'.format(mse*100)+'_'+'{0:.2f}'.format(mse_weight)+'.pb'
        output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
                sess1, # The session is used to retrieve the weights
                tf.compat.v1.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
                ['output','output_state'] # The output node names are used to select the usefull nodes
            ) 
        with tf.io.gfile.GFile(output_graph, "wb") as f:
                f.write(output_graph_def.SerializeToString())
        iters=iters+1
        out_preY = output_graph.split('.pb')[0]+'_preY.csv'
        np.savetxt(out_preY,predictY)
        out_data_state = output_graph.split('.pb')[0]+'_data_state.csv'
        np.savetxt(out_data_state,data_state)
        out_loss_weight = output_graph.split('.pb')[0]+'_loss_weight.csv'
        np.savetxt(out_loss_weight,loss_weight)
        
       
       