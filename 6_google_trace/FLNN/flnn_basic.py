#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 22:44:02 2017
@author: thieunv

    ## 1.Thu vs du lieu CPU
    
    ## 2. Thu vs du lieu RAM
    
    ## 3. Thu vs du lieu CPU va RAM
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import sin, cos, pi, exp, sqrt
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


#full_path_name="/mnt/volume/ggcluster/spark-2.1.1-bin-hadoop2.7/thieunv/machine_learning/6_google_trace/data/"
#full_path= "/mnt/volume/ggcluster/spark-2.1.1-bin-hadoop2.7/thieunv/machine_learning/6_google_trace/FLNN/results/notDecompose/data10minutes/univariate/cpu/"
file_name = "Fuzzy_data_sampling_617685_metric_10min_datetime_origin.csv"
full_path_name = "/home/thieunv/university/LabThayMinh/code/6_google_trace/data/"
full_path = "/home/thieunv/university/LabThayMinh/code/6_google_trace/FLNN/"

colnames = ['cpu_rate','mem_usage','disk_io_time','disk_space'] 
df = read_csv(full_path_name+ file_name, header=None, index_col=False, names=colnames, usecols=[0], engine='python')
dataset_original = df.values

dataset = []
for val in dataset_original:
    data = []
    data.append(val)
    
    data.append(sin(pi*val))
    data.append(sin(2*pi*val))
    data.append(cos(pi*val))
    data.append(cos(2*pi*val))
    
    """
    data.append(1)
    data.append(2 * val**2 - 1)
    data.append(4 * val**3 - 3*val)
    data.append(8 * val**3 - 8 * val**2 + 1)
    """
    """
    data.append(1)
    data.append( (3 * val**2 - 1) / 2 )
    data.append( (5 * val**3 - 3*val) / 2)
    data.append( (35 * val**4 - 30*val + 3) / 8)
    """
    dataset.append(data)
dataset = (np.asarray(dataset)).astype(np.float64)


# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
CPU_orin_unnormal = dataset[:, 0].reshape(-1, 1)


CPU_sinpi = scaler.fit_transform(dataset[:,1].reshape(-1, 1))
CPU_sin2pi = scaler.fit_transform(dataset[:, 2].reshape(-1, 1))
CPU_cospi = scaler.fit_transform(dataset[:, 3].reshape(-1, 1))
CPU_cos2pi = scaler.fit_transform(dataset[:, 4].reshape(-1, 1))
CPU_orin = scaler.fit_transform(dataset[:, 0].reshape(-1, 1))

length = len(dataset)
num_features = 1
num_expand = 5
sliding = 2
weight_size = num_features*num_expand*sliding
data = []
for i in range(length-sliding):
    detail=[]
    for j in range(sliding):
        detail.append(CPU_orin[i+j])
    for j in range(sliding):
        detail.append(CPU_sinpi[i+j])
    for j in range(sliding):
        detail.append(CPU_sin2pi[i+j])
    for j in range(sliding):
        detail.append(CPU_cospi[i+j])
    for j in range(sliding):
        detail.append(CPU_cos2pi[i+j])
    data.append(detail)
data = np.reshape(np.array(data), (len(data), weight_size) )


train_size = 2880
test_size = length - train_size
epoch = 100
batch_size = 16
valid = 0.25
learning_rate = 0.01
epsilon = 0.1   

trainX, trainY = data[0:train_size], CPU_orin[sliding:train_size+sliding]
testX = data[train_size:length-sliding]
testY =  CPU_orin_unnormal[train_size+sliding:length]

transformTestY = CPU_orin[train_size:length-sliding]


### Helper functions
def sigmoid(x):
    return 1 / (1 + exp(-x))

def relu(x):
    return max(0, x)

def get_batch_data_next(trainX, trainY, index, batch_size):
    real_index = index*batch_size
    if (len(trainX) % batch_size != 0 and real_index == len(trainX)):
        return (trainX[real_index:], trainY[real_index:])
    elif (real_index == len(trainX)):
        return ([], [])
    else:
        return (trainX[real_index: (real_index+batch_size)], trainY[real_index: (real_index+batch_size)])
    

#### My model here
def myFLNN(trainX, trainY, testX, epoch, batch_size, validation, weight_size, epsilon=0.5, eta = 0.01):

    print "epoch: {0}, batch_size: {1}, learning-rate: {2}".format(epoch, batch_size, eta)
    ## 1. Init weight
    w = np.random.rand(weight_size, 1) - 0.5
    print "init weight: "
    print w
    
    ## 2. Split validation dataset
    valid_len = int(len(trainX)*validation)
    valid_X = trainX[(len(trainX) - valid_len):]
    valid_y = trainY[(len(trainX) - valid_len):]
    train_X = trainX[0:(len(trainX) - valid_len)]
    train_y = trainY[0:(len(trainX) - valid_len)]
    
    valid_list_loss = [100]
    
    ## 3. Loop in epochs
    for i in range(epoch):
        print "Epoch thu: {0}".format(i+1)
        
        ## 3.1 Update w after 1 batch
        num_loop = int(len(trainX) / batch_size)
        for ind in range(num_loop):
            
            ## 3.1.1 Get next batch
            X_train_next, y_train_next = get_batch_data_next(train_X, train_y, ind, batch_size)
            if (len(X_train_next) == 0):
                break
            
            ## 3.1.2 Calculate all delta weight in 1 batch 
            delta_ws = []
            for j in range(len(X_train_next)):
                y_output = np.dot(X_train_next[j], w)
                y_output = relu(y_output)
                ek = y_train_next[j] - y_output
                delta_w = eta * ek * X_train_next[j]
                
                delta_ws.append(delta_w)
            
            ## 3.1.3 Sum all delta weight to get mean delta weight
            delta_wbar = np.array(np.sum(delta_ws, axis = 0) / len(X_train_next))
            
            ## 3.1.4 Change new weight with delta_wbar (mean delta weight)
            w += np.reshape(delta_wbar, (weight_size, 1))
        
        ## 3.2 Calculate validation loss after each epoch train to adjust hyperparameter
        valid_predict = []
        for j in range(len(valid_X)):
            valid_ybar = np.dot(valid_X[j], w)
            valid_ybar = relu(valid_ybar)
            valid_predict.append(valid_ybar)
        valid_RMSE = mean_squared_error(valid_y, np.array(valid_predict))
        if valid_RMSE > valid_list_loss[-1] + epsilon:
            print "Break epoch to finish training."
            print w
            break
        valid_list_loss.append(valid_RMSE)
        print w
        
    print "Final weight: "
    print w
    print valid_list_loss
  
    ## 4. Predict test dataset
    predict = []
    for i in range(len(testX)):
        ybar = np.dot(testX[i], w)
        ybar = relu(ybar)
        predict.append(ybar)
     
    return (w, np.reshape(predict, (len(predict), 1)))
            
w, predict = myFLNN(trainX, trainY, testX, epoch=epoch, batch_size = batch_size, validation=valid, weight_size=weight_size, epsilon=epsilon, eta=learning_rate)

# invert predictions        
testPredictInverse = scaler.inverse_transform(predict)
print 'len(testY): {0}, len(testPredict): {1}'.format(len(testY[0]), len(testPredictInverse))

# calculate root mean squared error
testScoreRMSE = sqrt(mean_squared_error(testY, testPredictInverse))
testScoreMAE = mean_absolute_error(testY, testPredictInverse)

#testScoreRMSE = sqrt(mean_squared_error(transformTestY, predict))
#testScoreMAE = mean_absolute_error(transformTestY, predict)
print('Test Score: %f RMSE' % (testScoreRMSE))
print('Test Score: %f MAE' % (testScoreMAE))

# summarize history for loss

plt.plot(testY)
plt.plot(testPredictInverse)

#plt.plot(transformTestY)
#plt.plot(predict)
plt.title('model predict')
plt.ylabel('real value')
plt.xlabel('point')
plt.legend(['realY', 'predictY'], loc='upper right')
#plt.close()
#plt.show()


        