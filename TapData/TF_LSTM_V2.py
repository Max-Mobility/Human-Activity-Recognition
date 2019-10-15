import tensorflow as tf
#import tensorflow.nn.rnn_cell as rnn
import numpy as np
#import tensorflow.keras as keras
#from tensorflow.keras.initializers import Constant
import tensorflow.keras.backend as K


from tqdm import tqdm
import os

def detect(predicts,raw,windows=16,threshold=0.8,raw_threshold=-10):
    taps=np.zeros_like(predicts)
    windows_count=0
    buff_size=25
    
    for i in np.arange(buff_size,len(taps)):
        raw_buff=raw[i-buff_size:i+1,0,2]# only z acc
        #print(raw_buff.min())
        diff=abs(raw_buff.max()-raw_buff.min())
        buff_sum = 0
        for x in raw_buff:
            if x <0:
                buff_sum+= x*-1
        
        if predicts[i]>threshold and windows_count==0  and buff_sum>raw_threshold:# and diff>6:# and predicts[i-1]>threshold:
            #print(diff)
            taps[i]=1
            windows_count=windows
            #print(buff_sum)
        else:
            windows_count=max(windows_count-1,0)
    return taps


def verify(taps,label_taps):
    False_positive= np.zeros_like(taps)
    False_negtive = np.zeros_like(taps)
    True_positive = np.zeros_like(taps)
    #taps=np.append(np.zeros([1]),taps)
    #taps=np.append(taps,np.zeros([1]))
    #label_taps=np.append(np.zeros([1]),label_taps)
    #label_taps=np.append(label_taps,np.zeros([1]))
    for i in range(len(taps)):
        if label_taps[i] ==1:
            if taps[i-2:i+3].sum()==0:
                False_negtive[i]=1
            else:
                True_positive[i]=1

        if taps[i] ==1:
            if label_taps[i-2:i+4].sum()==0:
                False_positive[i]=1
                #raw_buff=raw[i-4:i+1,2]# only z acc
                #diff=abs(raw_buff.max()-raw_buff.min())
                #print(i,raw_buff,diff)
            
    return [False_positive,  False_negtive ,    True_positive ]

if __name__ =='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    re_train=True
    time_steps=1
    n_input =3
    num_units=64
    n_classes=1
    lr =0.00001
    logFile = './traing.txt'
    
    #data_x=np.loadtxt('./data_raw_v2.csv',delimiter=",")
    #data_y=np.loadtxt('./data_y_v2.csv',delimiter=",")
    #preY = np.loadtxt('./data_preY_v2.csv',delimiter=",")
    data_x=np.loadtxt('./data_raw_100Hz.csv',delimiter=",")
    data_y=np.loadtxt('./data_100Hz_y.csv',delimiter=",")
    preY = np.loadtxt('./data_100Hz_y.csv',delimiter=",")
    
    datay=data_y
    data_y=data_y.reshape(len(data_y),1)
    data_x=data_x[:,(6-n_input):6].reshape((len(data_x),3))
    data_x=data_x.reshape([len(data_x),1,n_input])
    data_state = np.zeros([1,num_units*2],dtype='float')
    taps_label = detect(data_y,raw=data_x)
    
    loss_weight = preY*len(preY)/preY.sum()+datay*len(datay)/datay.sum()+1
    loss_weight = loss_weight.reshape([len(loss_weight),1])
    
    x= tf.placeholder('float',[None,1,n_input],name='input_tensor')
    y = tf.placeholder('float',[None,n_classes],name='labels')
    #inputs = tf.unstack(x,time_steps,1,name='input_tensor')
    #state_h = tf.placeholder('float',[None,num_units],name='input_h')
    #state_c = tf.placeholder('float',[None,num_units],name='input_c')
    state_in   = tf.placeholder('float',[None,num_units*2],name='input_state')
    loss_weight_in = tf.placeholder('float',[None,1],name='loss_weight')


# =============================================================================
#     wi=tf.Variable(name='wi',initial_value=tf.random_normal([n_input, num_units], stddev=0.35))
#     wf=tf.Variable(name='wf',initial_value=tf.random_normal([n_input, num_units], stddev=0.35))
#     wc=tf.Variable(name='wc',initial_value=tf.random_normal([n_input, num_units], stddev=0.35))
#     wo=tf.Variable(name='wo',initial_value=tf.random_normal([n_input, num_units], stddev=0.35))
#     ui=tf.Variable(name='ui',initial_value=tf.random_normal([num_units, num_units], stddev=0.35))
#     uf=tf.Variable(name='uf',initial_value=tf.random_normal([num_units, num_units], stddev=0.35))
#     uc=tf.Variable(name='uc',initial_value=tf.random_normal([num_units, num_units], stddev=0.35))
#     uo=tf.Variable(name='uo',initial_value=tf.random_normal([num_units, num_units], stddev=0.35))
#     
#     bi=tf.Variable(name='bi',initial_value=tf.random_normal([num_units], stddev=0.35))
#     bf=tf.Variable(name='bf',initial_value=tf.random_normal([num_units], stddev=0.35))
#     bc=tf.Variable(name='bc',initial_value=tf.random_normal([num_units], stddev=0.35))
#     bo=tf.Variable(name='bo',initial_value=tf.random_normal([num_units], stddev=0.35))
#     
# =============================================================================
    
    out_weights1=tf.Variable(name="weights1",initial_value=tf.random.normal([num_units, 16], stddev=0.35))
    out_weights2=tf.Variable(name="weights2",initial_value=tf.random.normal([16, 1], stddev=0.35))
    out_bias1=tf.Variable(name="bias1",initial_value=tf.random.normal([16], stddev=0.35))
    out_bias2=tf.Variable(name="bias2",initial_value=tf.random.normal([1], stddev=0.35))
    
    
    state_h,state_c = tf.split(state_in,num_or_size_splits=2,axis=1)
# =============================================================================
#     x_i = K.dot(x,wi)
#     x_f = K.dot(x,wf)
#     x_c = K.dot(x,wc)
#     x_o = K.dot(x,wo)
#     
#     i = K.hard_sigmoid(tf.add(tf.add(x_i,bi),K.dot(state_h,ui)))
#     f = K.hard_sigmoid(tf.add(tf.add(x_f,bf),K.dot(state_h,uf)))
#     c = tf.add(tf.multiply(f,state_c),K.tanh(tf.add(tf.add(x_c,bc),K.dot(state_h,uc))),name='output_c')
#     o = K.hard_sigmoid(tf.add(tf.add(x_o,bo),K.dot(state_h,uo)))
#     h = tf.multiply(o,tf.tanh(c),name='output_h')
# =============================================================================
    inputX=tf.unstack(x ,1,1)
    lstm_layer=tf.nn.rnn_cell.BasicLSTMCell(num_units)
    outputs,state_output=tf.nn.static_rnn(lstm_layer,inputX,dtype="float32",initial_state = [state_h,state_c] ,sequence_length=[1])

    
    dense1  = tf.nn.relu(tf.add(tf.matmul(outputs,out_weights1), out_bias1))
    prediction=tf.nn.relu(tf.add(tf.matmul(dense1,out_weights2), out_bias2,name='output_forRelu'), name="output")
    #prediction=tf.nn.relu(tf.add(tf.matmul(h,out_weights), out_bias,name='output_forRelu'), name="output")
    state_out = tf.concat(state_output,axis=1,name='output_state')
    prediction_weight=tf.matmul(prediction,loss_weight_in)
    y_weight=tf.matmul(y,loss_weight_in)
    
    loss=tf.losses.mean_squared_error(predictions=prediction_weight,labels=y_weight)
    opt=tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    
    tf.train.Saver()
    iters=1
    predictY=np.zeros([len(data_x)])
    while iters<100:
        for i in tqdm(range(len(data_x))):
            feed={x:data_x[i:i+1],
                  state_in:data_state,
                    y:data_y[i:i+1],
                  loss_weight_in:loss_weight[i:i+1]
                   }
            if re_train:
                sess.run(opt,feed_dict=feed)
    
            data_state,_predictY = sess.run([state_out,prediction],feed_dict=feed)
            #print(data_h.shape)
            #data_c = sess.run(c,feed_dict=feed)
            predictY[i]=_predictY[0]
            #print(i)
        if iters%1==0:
            print(iters)
            mse = (np.square(predictY-data_y.reshape([len(data_x)]))).mean(axis=0)
            loss_weight = loss_weight.reshape([len(loss_weight)])
            mse_weight =np.square(np. multiply((predictY-data_y.reshape([len(data_x)])),loss_weight)).mean(axis=0)
            loss_weight = loss_weight.reshape([len(loss_weight),1])
            taps_predict = detect(predictY,raw=data_x)
            FP,FN,TP = verify(taps_predict,taps_label)
            file = open(logFile,'a+')
            file.write('\niters:'+str(iters))
            file.write('\nmse:'+str(mse))
            file.write('\nmse_weight:'+str(mse_weight))
            file.write('\ntotal taps:'+str(taps_predict.sum()))
            file.write('\nFP,FN,TP:'+str(FP.sum())+', '+str(FN.sum())+', '+str(TP.sum()))
            file.close()
            print('mse:'+str(mse))
            print('mse_weight:'+str(mse_weight))
            print('total taps:'+str(taps_predict.sum()))
            print('FP,FN,TP:'+str(FP.sum())+', '+str(FN.sum())+', '+str(TP.sum()))
            if iters%10==0:
    
                loss_weight = predictY*len(predictY)/predictY.sum()/100+datay*len(datay)/datay.sum()+1
                loss_weight = loss_weight.reshape([len(loss_weight),1])

        
        #output_graph='../models/100hz/TapModels/3C/realLSTM_dataV2_U'+str(num_units)+'_i'+str(iters)+'_mse'+'{0:.2f}'.format(mse*100)+'_'+'{0:.2f}'.format(mse_weight)+'.pb'
        output_graph='../models/100hz/realLSTM_data_U'+str(num_units)+'_i'+str(iters)+'_mse'+'{0:.2f}'.format(mse*100)+'_'+'{0:.2f}'.format(mse_weight)+'.pb'
        output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, # The session is used to retrieve the weights
                tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
                ['output','output_state'] # The output node names are used to select the usefull nodes
            ) 
        with tf.gfile.GFile(output_graph, "wb") as f:
                f.write(output_graph_def.SerializeToString())
        iters=iters+1
        out_preY = output_graph.split('.pb')[0]+'_preY.csv'
        np.savetxt(out_preY,predictY)