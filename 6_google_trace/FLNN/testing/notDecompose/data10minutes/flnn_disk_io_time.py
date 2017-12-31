#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 02:14:02 2017

@author: thieunv
- Cac tham so co the thay doi: 
    + (1): so dung cac function khac nhau: 0, 1, 2, 3, 4
    + (2): activation function: 1-sigmoid, 2-relu, else-ko dung 
    + (3): sliding window. 1, 2, 4, ...
    + (4): epoch : 100, 200, 500, 1000, ...
    + (5): batch_size: 8, 16, 32, 64, ...
    + (6): validation: 0.1, 0.2, 0.25, ... < 0.5
    + (7): learning rate: 0.01, 0.02, 0.03, ...
    + (8): epsilon: 0.001, 0.002, 0.005, 0,01...
    + (9): training size
    + (10): 0,1-CPU,RAM, 2-Disk-ioTime, 3-Disk-space
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import sqrt
from pandas import read_csv
from sklearn.metrics import mean_squared_error, mean_absolute_error
## import helper function
from flnn_functions import get_batch_data_next, my_invert_min_max_scaler, my_min_max_scaler
## get list functions expand
import flnn_functions as gen_fun

## Load data frame
full_path_name="/mnt/volume/ggcluster/spark-2.1.1-bin-hadoop2.7/thieunv/machine_learning/6_google_trace/data/"
full_path= "/mnt/volume/ggcluster/spark-2.1.1-bin-hadoop2.7/thieunv/machine_learning/6_google_trace/FLNN/testing/notDecompose/data10minutes/univariate/disk_io_time/"
file_name = "Fuzzy_data_sampling_617685_metric_10min_datetime_origin.csv"
#full_path_name = "/home/thieunv/university/LabThayMinh/code/6_google_trace/data/"
#full_path = "/home/thieunv/university/LabThayMinh/code/6_google_trace/FLNN/results/notDecompose/data10minutes/univariate/cpu/"
df = read_csv(full_path_name+ file_name, header=None, index_col=False, usecols=[2], engine='python')    # (10)
dataset_original = df.values

list_function_expand = [1, 2, 4, 8, 9]                             # (1)
list_activation = [0, 1, 2]                                                 # (2) 0 dung, sigmoid, relu  
sliding_windows = [1, 2, 3, 5]                                           # (3)

length = len(dataset_original)
num_features = df.shape[1]

train_size = 2880                                                                       # (9)
test_size = length - train_size
epochs = [250, 500, 1000, 1500, 2000]                        # (4)
batch_sizes = [8, 16, 32, 64, 128]                                                     # (5)

valid = 0.25                                                                                # (6)
learning_rates = [0.025, 0.05, 0.1, 0.15, 0.20, 0.25]                # (7)
epsilons = [0.00001, 0.00005, 0.0001, 0.0005]                     # (8)


#### My model here
def myFLNN(trainX, trainY, testX, epoch, batch_size, validation, activation, weight_size, epsilon=0.001, eta = 0.01):

#    training_detail_file_name = full_path + 'FE=' + str(func_index) + '_Acti=' + str(act_index) + '_Sliding=' + str(sliding) + '_Epoch=' + str(epoch) + '_BatchS=' + str(batch_size) + '_Eta=' + str(eta) + '_Epsilon=' + str(epsilon) +'_NetworkDetail.txt'
    
#    print "epoch: {0}, batch_size: {1}, learning-rate: {2}".format(epoch, batch_size, eta)
    ## 1. Init weight [-0.5, +0.5]
    w = np.random.rand(weight_size, 1) - 0.5
#    print "init weight: "
#    print w
    
#    with open(training_detail_file_name, 'a') as f:
#        print >> f, "epoch: {0}, batch_size: {1}, learning-rate: {2}".format(epoch, batch_size, eta)
#        print >> f, "init weight: ", w
                    
    
    ## 2. Split validation dataset
    valid_len = int(len(trainX)*validation)
    valid_X = trainX[(len(trainX) - valid_len):]
    valid_y = trainY[(len(trainX) - valid_len):]
    train_X = trainX[0:(len(trainX) - valid_len)]
    train_y = trainY[0:(len(trainX) - valid_len)]
    
    valid_list_loss = [0.1]
    train_list_loss = [0.1]
    
    ## 3. Loop in epochs
    for i in range(epoch):
#        if (i % 50 == 0):
##            print "Epoch thu: {0}".format(i)
#            with open(training_detail_file_name, 'a') as f:
#                print >> f, "Epoch thu: {0}".format(i)
        
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
                y_output = activation_func(y_output)
                ek = y_train_next[j] - y_output
                delta_w = eta * ek * X_train_next[j]
                
                delta_ws.append(delta_w)
            
            ## 3.1.3 Sum all delta weight to get mean delta weight
            delta_wbar = np.array(np.sum(delta_ws, axis = 0) / len(X_train_next))
            
            ## 3.1.4 Change new weight with delta_wbar (mean delta weight)
            w += np.reshape(delta_wbar, (weight_size, 1))
        
        ## 3.2 Calculate validation loss after each epoch train to adjust hyperparameter
        
        # Calculate validation error
        valid_predict = []
        for j in range(len(valid_X)):
            valid_ybar = np.dot(valid_X[j], w)
            valid_ybar = activation_func(valid_ybar)
            valid_predict.append(valid_ybar)
        valid_MSE = mean_squared_error(valid_y, np.array(valid_predict))
        
        # Calculate train error
        train_predict = []
        for j in range(len(train_X)): 
            train_ybar = np.dot(train_X[j], w)
            train_ybar = activation_func(valid_ybar)
            train_predict.append(train_ybar)
        train_MSE = mean_squared_error(train_y, np.array(train_predict))
        train_list_loss.append(train_MSE)
        
        # Avoiding overfitting
        if (valid_MSE > (valid_list_loss[-1] + epsilon)):
#            print "Break epoch to finish training."
#            print w
            
#            with open(training_detail_file_name, 'a') as f:
#                print >> f, "Break epoch to finish training.", w
                
            break
        valid_list_loss.append(valid_MSE)
#        if (i % 50 == 0):
#            with open(training_detail_file_name, 'a') as f:
#                print >> f, w
#                print w
    
#    with open(training_detail_file_name, 'a') as f:
#        print >> f, "Final weight: ", w
#    print "Final weight: "
#    print w
  
    ## 4. Predict test dataset
    predict = []
    for i in range(len(testX)):
        ybar = np.dot(testX[i], w)
        ybar = activation_func(ybar)
        predict.append(ybar)
     
    return (w, np.reshape(predict, (len(predict), 1)), valid_list_loss, train_list_loss)


pl1 = 1
pl2 = 100000
couting_number_loop = 0
### Loop to test 5 way expand
for func_index in list_function_expand:       # 5 way to expand 
    
    ### Loop to test 3 way use activation function (sigmoid, relu, 0 dung)
    for act_index in list_activation:
        
        list_label_func_expand = gen_fun.get_list_function(func_index)       #(1)
        activation_func = gen_fun.get_activation_func(act_index)            #(9)
        
        ## expand input by list functions expand
        dataset = []
        for val in dataset_original:
            data = []
            for dimen_index in range(dataset_original.shape[1]):
                for func in list_label_func_expand:
                    data.append(func(val[dimen_index]))
            dataset.append(data)
        dataset = (np.asarray(dataset)).astype(np.float64)
        
        # normalize the dataset
        GoogleTrace_orin_unnormal = dataset[:, 0].reshape(-1, 1)    # keep orginal data to test
        min_GT = min(GoogleTrace_orin_unnormal)
        max_GT = max(GoogleTrace_orin_unnormal)
        
        ## Scaling min max
        data_scaler = []
        for i in range( dataset.shape[1] ):
            data_scaler.append( my_min_max_scaler(dataset[:, i].reshape(-1, 1)) ) 

        for sliding in sliding_windows:
            
            num_expand = len(list_label_func_expand)
            weight_size = num_features*num_expand*sliding
            
            for epoch in epochs: 
                
                for batch_size in batch_sizes:
                    
                    for learning_rate in learning_rates:
                        
                        for epsilon in epsilons:
            
                            
                            data = []
                            for i in range(length-sliding):
                                detail=[]
                                for ds in data_scaler:
                                    for j in range(sliding):
                                        detail.append(ds[i+j])
                                data.append(detail)
                            data = np.reshape(np.array(data), (len(data), weight_size) )
                            
                            trainX, trainY = data[0:train_size], data[sliding:train_size+sliding, 0:1]
                            testX = data[train_size:length-sliding]
                            testY =  GoogleTrace_orin_unnormal[train_size+sliding:length]
                            #transformTestY = data[train_size:length-sliding, 0:1]
                            
                                
                            w, predict, valid_list_loss, train_list_loss = myFLNN(trainX, trainY, testX, epoch=epoch, batch_size = batch_size, 
                                                validation=valid, activation=activation_func, weight_size=weight_size, epsilon=epsilon, eta=learning_rate)
                            
                            # invert predictions        
                            testPredictInverse = my_invert_min_max_scaler(predict, min_GT, max_GT)
#                            with open(training_detail_file_name, 'a') as f:
#                                print >> f, 'len(testY): {0}, len(testPredict): {1}'.format(len(testY[0]), len(testPredictInverse))
#                            print 'len(testY): {0}, len(testPredict): {1}'.format(len(testY[0]), len(testPredictInverse))
                            
                            # calculate root mean squared error
                            testScoreRMSE = sqrt(mean_squared_error(testY, testPredictInverse))
                            testScoreMAE = mean_absolute_error(testY, testPredictInverse)
#                            print('Test Score: %f RMSE' % (testScoreRMSE))
#                            print('Test Score: %f MAE' % (testScoreMAE))
                            
                            if testScoreMAE < 0.5:
                                # summarize history for point prediction
                                plt.figure(pl1)
                                plt.plot(testY)
                                plt.plot(testPredictInverse)
                                
                                plt.title('model predict')
                                plt.ylabel('real value')
                                plt.xlabel('point')
                                plt.legend(['realY...Test Score RMSE= ' + str(testScoreRMSE), 'predictY...Test Score MAE= ' + str(testScoreMAE) ], loc='upper right')
                                pic1_fn = full_path + 'pointPredict_FE=' + str(func_index) + '_Acti=' + str(act_index) + '_Sliding=' + str(sliding) + '_Epoch=' + str(epoch) + '_BatchS=' + str(batch_size) + '_Eta=' + str(learning_rate) + '_Epsilon=' + str(epsilon) +'.png'
                                plt.savefig(pic1_fn)
                                plt.close()
                                
                                plt.figure(pl2)
                                plt.plot(valid_list_loss)
                                plt.plot(train_list_loss)
                                
                                plt.title('model loss')
                                plt.ylabel('real value')
                                plt.xlabel('epoch')
                                plt.legend(['validation error...Test Score RMSE= ' + str(testScoreRMSE), 'train error...Test Score MAE= ' + str(testScoreMAE) ], loc='upper right')
                                pic2_fn = full_path + 'validAndTrainLoss_FE=' + str(func_index) + '_Acti=' + str(act_index) + '_Sliding=' + str(sliding) + '_Epoch=' + str(epoch) + '_BatchS=' + str(batch_size) + '_Eta=' + str(learning_rate) + '_Epsilon=' + str(epsilon) +'.png'
                                plt.savefig(pic2_fn)
                                plt.close()
                                
                                pl1 += 1
                                pl2 += 1
                            
                            couting_number_loop += 1
                            if couting_number_loop % 5000 == 0:
                                print "Vong lap thu: {0}".format(couting_number_loop)

print "Processing DONE!!!"

