# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 08:28:32 2019

@author: Guo
"""

from sklearn.cluster import KMeans
import keras
from keras.layers import Input, Conv1D,Activation,MaxPooling1D,Dense,Flatten,UpSampling1D
from keras.layers.normalization import BatchNormalization
from keras.models import Model,load_model
import tensorflow as tf
import numpy as np
from matplotlib.pyplot import plot
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle
from mpl_toolkits.mplot3d import axes3d


if __name__ == '__main__':
    data= np.load('watch_norm_32_4_29.npy')
    useLSTM = False
    #data= np.load('./WatchData (28).npy')
    timeStep=1
    indexs = np.arange(0,len(data),timeStep)
    data= data[indexs]
    if useLSTM:
        y=np.flip(data,1)
        z=np.zeros((y.shape[0],1,y.shape[2]))
        X2 = np.concatenate((z,y[:,:-1,:]),axis=1)
        Mymodel = load_model('lstm_auto_encoder_128Units.h5')
    else:
        Mymodel = load_model('./models/25hz/vae_epoch_7000rc_0.02_KL_99.82.h5')
    
    LV=Mymodel.predict(data)
    full_LV = np.append(LV[0],LV[1],axis=1)
    n_clusters=3
    
    #index = np.append(index,np.arange(96*4,96*6))
    kmean=KMeans(n_clusters=n_clusters,max_iter=300000)
    kmean.fit(full_LV[:,:])
    kmean_fileName='kmean.pkl'
    with open(kmean_fileName,'wb') as f:
        pickle.dump(kmean,f)
    
    with open(kmean_fileName,'rb') as f1:
        print(f1)
        kmean1=pickle.load(f1)
    print(kmean.labels_,kmean.inertia_,kmean.n_iter_)
    plot(kmean.labels_)
    print(kmean.cluster_centers_.shape)
    y_kmeans=kmean.predict(full_LV[:,:])
    pca=PCA(n_components=3)
    pcaComponents=pca.fit_transform(full_LV[:,:])
    
    print(pca.explained_variance_ratio_  )
    np.save('pca.npy',pcaComponents)
    np.savetxt('pca.csv',pcaComponents,delimiter=",")
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(221, projection='3d')
    ax.scatter(pcaComponents[:,0],pcaComponents[:,1],pcaComponents[:,2],c=y_kmeans)
    plt.show
    #ax.view_init(45, -45)
    y_kmeans=np.append(np.zeros((32,1)),y_kmeans)
    #saveStr='./kmeans_4_29_cluster_128Units_'+str(n_clusters)+'.csv'
    saveStr='./WatchData (28)_128Units_cluster_'+str(n_clusters)+'.csv'
    np.savetxt(saveStr,np.append(y_kmeans.reshape((len(y_kmeans),1)),pcaComponents,axis=1),delimiter=",")
    
    # testing
    data_test= np.load('watch_norm_32_4_25.npy')
    LV_test=Mymodel.predict(data_test)
    full_LV_test = np.append(LV_test[0],LV_test[1],axis=1)
    y_kmeans_test=kmean.predict(full_LV_test[:,:])
    y_kmeans_test=np.append(np.zeros((32,1)),y_kmeans_test)
    saveStr='./kmeans_4_25_cluster_'+str(n_clusters)+'.csv'
    np.savetxt(saveStr,y_kmeans_test)
    print('finished testing')
    
    
    
    kmean_raw=KMeans(n_clusters=n_clusters,max_iter=300000)
    kmean_raw.fit(data.reshape((len(data),-1)))
    y_kmean_raw=kmean_raw.predict(data.reshape((len(data),-1)))
    y_kmean_raw=np.append(np.zeros((32,1)),y_kmean_raw)
    saveStr='./kmeans_4_29_raw_cluster_'+str(n_clusters)+'.csv'
    np.savetxt(saveStr,y_kmean_raw)