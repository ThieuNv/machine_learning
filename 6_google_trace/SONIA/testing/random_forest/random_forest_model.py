#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 12:43:17 2018

@author: thieunv
"""


from random import uniform, randint
from math import exp, sqrt
from operator import itemgetter
import numpy as np
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.metrics import mean_squared_error, mean_absolute_error
### Helper functions
def sigmoid_activation(x):
    return 1.0 / (1.0 + exp(-x))

def relu(x):
    return max(0, x)

def self_activation(x):
    return x
def hyperbolic_tangent_sigmoid_activation(x):   # -1 <= output <= 1
    return (2.0 / (1.0 + exp(-2.0*x)) - 1.0 )

def my_min_max_scaler(data):
    minx = min(data)
    maxx = max(data)
    return (np.array(data).astype(np.float64) - minx) / (maxx - minx)

def my_invert_min_max_scaler(data, minx, maxx):
    return np.array(data).astype(np.float64) * (maxx-minx) + minx

def get_random_input_vector(train_X):
    temp = copy.deepcopy(train_X)
    return temp[randint(0, len(train_X) - 1)]

def get_random_vector_weight(wHa, wHb):
    temp = []
    for i in range(len(wHa)):
        temp.append(uniform(wHa[i], wHb[i]))
    return np.array(temp)

def get_batch_data_next(trainX, trainY, index, batch_size):
    real_index = index*batch_size
    if (len(trainX) % batch_size != 0 and real_index == len(trainX)):
        return (trainX[real_index:], trainY[real_index:])
    elif (real_index == len(trainX)):
        return ([], [])
    else:
        return (trainX[real_index: (real_index+batch_size)], trainY[real_index: (real_index+batch_size)])


def my_sorted_list(my_list):
    list_elem = my_list[4][len(my_list[4])-1]
    return list_elem

## Load data frame
#full_path_name="/mnt/volume/ggcluster/spark-2.1.1-bin-hadoop2.7/thieunv/machine_learning/6_google_trace/data/"
#full_path= "/mnt/volume/ggcluster/spark-2.1.1-bin-hadoop2.7/thieunv/machine_learning/6_google_trace/SONIA/testing/no_mutation/no_update/no_pruning/result/cpu/"
file_name = "Fuzzy_data_sampling_617685_metric_10min_datetime_origin.csv"

full_path_name = "/home/thieunv/university/LabThayMinh/code/6_google_trace/data/"
full_path = "/home/thieunv/university/LabThayMinh/code/6_google_trace/SONIA/testing/random_forest/result/cpu/"
df = read_csv(full_path_name+ file_name, header=None, index_col=False, usecols=[0], engine='python')   
dataset_original = df.values


distance_level = 0.1  #[0.1, 0.15, 0.2]
threshold_number = 2 #[2, 3, 4]

stimulation_level = [0.15] #[0.10, 0.2, 0.25, 0.50, 1.0, 1.5, 2.0]  # [0.20]    
positive_numbers = [0.005]  #[0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.20]     # [0.1]   

learning_rates = [0.01] #[0.005, 0.01, 0.025, 0.05, 0.10, 0.12, 0.15]   # [0.2]    
sliding_windows = [3] #[ 2, 3, 5]           # [3]  
     
epochs = [500] #[100, 250, 500, 1000, 1500, 2000]   # [500]                       
batch_sizes = [16] #[8, 16, 32, 64, 128]      # [16]     

length = dataset_original.shape[0]
num_features = dataset_original.shape[1]

train_size = 700       #2880                   
test_size = length - train_size
valid = 0.25        # Hien tai chua dung den tham so nay
epsilon = 0.00001   # Hien tai chua dung den tham so nay

list_num = [(2500, 3000)]


def build_model(train_X, train_y, test_X, epoch, batch_size, validation,sliding, learning_rate = 0.01, 
            positive_number = 0.1, stimulation_level=0.05, distance_level=0.1, threshold_number=2):
    ### Qua trinh train va dong thoi tao cac hidden unit (Pha 1 - cluster data)
    # 2. Khoi tao hidden thu 1
    hu1 = [0, get_random_input_vector(train_X)]   # hidden unit 1 (t1, wH)
    list_hu = [copy.deepcopy(hu1)]         # list hidden units
    matrix_Wih = copy.deepcopy(hu1[1]).reshape(hu1[1].shape[0], 1)     # Mang 2 chieu 
    ### +++ Technical use to trace back matrix weight
    trace_back_list_matrix_Wih = [copy.deepcopy(matrix_Wih)]
    trace_back_list_hu = [copy.deepcopy(list_hu)]
#    training_detail_file_name = full_path + 'SL=' + str(stimulation_level) + '_Slid=' + str(sliding) + '_Epoch=' + str(epoch) + '_BS=' + str(batch_size) + '_LR=' + str(learning_rate) + '_PN=' + str(positive_number) + '_CreateHU.txt'
    
    m = 0
    while m < len(train_X):
        list_dist_mj = []      # Danh sach cac dist(mj)
         # number of hidden units
        for j in range(0, len(list_hu)):                # j: la chi so cua hidden thu j
            dist_sum = 0.0
            for i in range(0, len(train_X[0])):        # i: la chi so cua input unit thu i
                dist_sum += pow(train_X[m][i] - matrix_Wih[i][j], 2.0)
            list_dist_mj.append([j, sqrt(dist_sum)])
        list_dist_mj = sorted(list_dist_mj, key=itemgetter(1))        # Sap xep tu be den lon
        
        c = list_dist_mj[0][0]      # c: Chi so (index) cua hidden unit thu c ma dat khoang cach min
        distmc = list_dist_mj[0][1] # distmc: Gia tri khoang cach nho nhat
        
        if distmc < stimulation_level:
            list_hu[c][0] += 1                  # update hidden unit cth
            # Just update vector W(c)
            list_hu[c][1] += positive_number * distmc
            matrix_Wih = np.transpose(matrix_Wih)    # Phai o dang numpy thi ms update toan bo duoc
            matrix_Wih[c] += positive_number * distmc
            matrix_Wih = np.transpose(matrix_Wih)
            ## +++ Save the matrix_wih 
            trace_back_list_matrix_Wih.append(copy.deepcopy(matrix_Wih))
            trace_back_list_hu.append(copy.deepcopy(list_hu))
            # Tiep tuc vs cac example khac
            m += 1
#            if m % 100 == 0:
#                with open(training_detail_file_name, 'a') as f:
#                    print >> f, 'distmc = :', distmc  
#                    print >> f, 'Example thu :', m  
#                print "distmc = {0}".format(distmc)
#                print "m = {0}".format(m)
        else:
            ## +++ Get the first matrix weight hasn't been customize
            matrix_Wih = copy.deepcopy(trace_back_list_matrix_Wih[0])
            list_hu = copy.deepcopy(trace_back_list_hu[0])
            ## +++ Del all trace back matrix weight except the first one
            del trace_back_list_matrix_Wih[1:]
            del trace_back_list_hu[1:]
#            with open(training_detail_file_name, 'a') as f:
#                print >> f, 'Failed !!!. distmc = ', distmc  
#            print "Failed !!!. distmc = {0}".format(distmc)
            list_hu.append([0, copy.deepcopy(train_X[m]) ])
#            with open(training_detail_file_name, 'a') as f:
#                print >> f, 'Hidden unit thu:', len(list_hu), ' duoc tao ra.'
#            print "Hidden unit thu: {0} duoc tao ra.".format(len(list_hu))
            matrix_Wih = np.append(matrix_Wih, copy.deepcopy(train_X[m]).reshape((matrix_Wih.shape[0], 1)), axis = 1)
            for hu in list_hu:
                hu[0] = 0
            # then go to step 1
            m = 0
            ### +++
            trace_back_list_matrix_Wih[0] = copy.deepcopy(matrix_Wih)
            trace_back_list_hu[0] = copy.deepcopy(list_hu)    
            
    ### +++ Get the last matrix weight 
    sorted_list_hu = copy.deepcopy(trace_back_list_hu[-1])
    sorted_matrix_Wih = copy.deepcopy(trace_back_list_matrix_Wih[-1])
    ### +++ Delete trace back
    del trace_back_list_matrix_Wih
    del trace_back_list_hu
            
    ### Building set of weights between hidden layer and output layer
    ## Initialize weights and bias

    matrix_Who = np.zeros(len(sorted_list_hu))
    bias = 1
#    print "Random bias is: {0}".format(bias)
    ## Training weights and bias based on backpropagation
    list_loss_RMSE = []
    list_loss_AMSE = []
    for t in range(epoch):    # 100: epoch 
#        print "Epoch thu: {0}".format(t+1) 
        loss1 = 0.0
        loss2 = 0.0
        ### Update w after 1 batch
        num_loop = int(len(trainX) / batch_size)
        for ind in range(num_loop):
            ## Get next batch
            X_train_next, y_train_next = get_batch_data_next(train_X, train_y, ind, batch_size)
            if len(X_train_next) == 0:
                break
            ## Calculate all delta weight in 1 batch 
            delta_ws = []
            for k in range(0, len(X_train_next)):        # training with 1 example at a time
                
                # Calculate output of hidden layer to put it in input of output layer
                output_hidden_layer = []     
                for i in range(0, len(np.transpose(sorted_matrix_Wih))):
                    xHj_sum = 0.0
                    for j in range(0, len(X_train_next[0])):
                        xHj_sum += pow(sorted_matrix_Wih[j][i] - X_train_next[k][j], 2.0)
                    output_hidden_layer.append(hyperbolic_tangent_sigmoid_activation(sqrt(xHj_sum)))
                    
                # Right now we have: output hidden, weights hidden and output, bias
                # Next: Calculate y_output
                y_output = 0 #     bias
                for i in range(0, len(matrix_Who)):
                    y_output += matrix_Who[i] * output_hidden_layer[i]
                y_output = sigmoid_activation(y_output)
                loss1 += abs(y_output - y_train_next[k])
                loss2 += pow(y_output - y_train_next[k], 2.0)
                ### Next: Update weight and bias using backpropagation
                ##  update weights and bias hidden and output
                delta_weights_ho = -2 * learning_rate * y_output * (1 - y_output) * (y_output - y_train_next[k]) * np.array(output_hidden_layer) 
    #            delta_bias = -1 * learning_rate * y_output * (1 - y_output) * (y_output - train_y[k]) * 0.00001
                delta_ws.append(delta_weights_ho)   
            ## Sum all delta weight to get mean delta weight
            delta_wbar = np.array(np.sum(delta_ws, axis = 0) / len(X_train_next))
            matrix_Who += delta_wbar
    #       bias += delta_bias
        
        list_loss_AMSE.append(loss1/len(train_X))
        list_loss_RMSE.append(loss2/len(train_X))
    ## Ending backpropagation
    
    ## Right now, we have all we need: sorted_matrix_Wih, sorted_list_hu, matrix_Who
    ### Predict test dataset
    predict = []
    for k in range(len(test_X)):
        pre_output_hl = []
        for i in range(0, len(np.transpose(sorted_matrix_Wih))):
            xHj_sum = 0.0
            for j in range(0, len(test_X[0])):
                xHj_sum += pow(sorted_matrix_Wih[j][i] - test_X[k][j], 2.0)
            pre_output_hl.append(hyperbolic_tangent_sigmoid_activation(sqrt(xHj_sum)))
        
        pre_y_output = 0#bias
        for i in range(0, len(matrix_Who)):
            pre_y_output += matrix_Who[i] * pre_output_hl[i]
        pre_y_output = sigmoid_activation(pre_y_output)
        predict.append(pre_y_output)
     
    return [sorted_matrix_Wih, matrix_Who , bias, np.array(predict), list_loss_AMSE, list_loss_RMSE]




#### My model here
def mySONIA(train_X, train_y, test_X, epoch, batch_size, validation,sliding, learning_rate = 0.01, 
            positive_number = 0.1, stimulation_level=0.05, distance_level=0.1, threshold_number=2):
    
    #### Chu y:
    # 1. Tat ca cac input trainX phai normalize ve doan [0, 1]
    
    ## 1. Split validation dataset
    #valid_len = int(len(trainX)*validation)
    #valid_X = trainX[(len(trainX) - valid_len):]
    #valid_y = trainY[(len(trainX) - valid_len):]
   
    #valid_list_loss = [0.1]
    #train_list_loss = [0.1]
    
#    sorted_matrix_Wih, matrix_Who , bias, predict, list_loss_AMSE, list_loss_RMSE = build_model(train_X, train_y, test_X, epoch, batch_size, validation, sliding, learning_rate, positive_number, stimulation_level, distance_level, threshold_number)
#    return sorted_matrix_Wih, matrix_Who , bias, predict, list_loss_AMSE, list_loss_RMSE

    list_outcome = []
    for i in range(len(train_X)/500):
        if i == len(train_X) / 500:
            X_train = copy.deepcopy(train_X[i*500:])
            y_train = copy.deepcopy(train_y[i*500:])
        else:
            X_train = copy.deepcopy(train_X[i*500:(i+1)*500])
            y_train = copy.deepcopy(train_y[i*500:(i+1)*500])
        list_outcome.append(build_model(X_train, y_train, test_X, epoch, batch_size, validation, sliding, learning_rate, positive_number, stimulation_level, distance_level, threshold_number))
    
    list_outcome = sorted(list_outcome, key=my_sorted_list)
    
    predictions = np.zeros(len(test_X))
    for i in range(0, 3):
        predictions += list_outcome[i][3]
    predictions = predictions / 3
    return list_outcome, predictions

pl1 = 1         # Use to draw figure
#pl2 = 1000
so_vong_lap = 0

for u_num in list_num:
    for sliding in sliding_windows:
        ## Load data
        dataset_split = dataset_original[:u_num[1]+sliding]
        GoogleTrace_orin_unnormal = copy.deepcopy(dataset_split)    # keep orginal data to test
        # normalize the dataset
        min_GT = min(GoogleTrace_orin_unnormal[:u_num[0]])
        max_GT = max(GoogleTrace_orin_unnormal[:u_num[0]])
        ## Scaling min max
        dataset_scale = my_min_max_scaler(dataset_split)
        ## Handle data with sliding
        dataset_sliding = dataset_scale[:len(dataset_scale)-sliding]
        for i in range(sliding-1):
            dddd = np.array(dataset_scale[i+1: len(dataset_scale)-sliding+i+1])
            dataset_sliding = np.concatenate((dataset_sliding, dddd), axis=1)
        ## Split data to set train and set test
        trainX, trainY = dataset_sliding[0:u_num[0]], dataset_sliding[sliding:u_num[0]+sliding, 0:1]
        testX = dataset_sliding[u_num[0]:u_num[1]-sliding]
        testY = GoogleTrace_orin_unnormal[u_num[0]+sliding:u_num[1]]
        
        for sti_level in stimulation_level:
            for epoch in epochs:
                for batch_size in batch_sizes:
                    for learning_rate in learning_rates:
                        for positive_number in positive_numbers:
                                
#                            sorted_matrix_Wih, matrix_Who , bias, predict, list_loss_AMSE, list_loss_RMSE= mySONIA(trainX, trainY, testX, epoch=epoch, 
#                                        batch_size=batch_size, validation=valid, sliding=sliding, learning_rate=learning_rate, 
#                                        positive_number=positive_number, stimulation_level=sti_level, distance_level=distance_level, threshold_number=threshold_number)
#                            
                            list_outcome, predict= mySONIA(trainX, trainY, testX, epoch=epoch, 
                                        batch_size=batch_size, validation=valid, sliding=sliding, learning_rate=learning_rate, 
                                        positive_number=positive_number, stimulation_level=sti_level, distance_level=distance_level, threshold_number=threshold_number)
                            
                            
    #                        print "bias: {0}".format(bias)
    #                        print "Weight input and hidden: "
    #                        print matrix_Wih
    #                        print "Weight hidden and output: "
    #                        print vector_Who
    #                        print "Predict " 
    #                        print predict
                        
                            # invert predictions        
                            testPredictInverse = my_invert_min_max_scaler(predict, min_GT, max_GT)
    #                        print testPredictInverse
    #                        print 'len(testY): {0}, len(testPredict): {1}'.format(len(testY[0]), len(testPredictInverse))
                            
                            # calculate root mean squared error
                            testScoreRMSE = sqrt(mean_squared_error(testY, testPredictInverse))
                            testScoreMAE = mean_absolute_error(testY, testPredictInverse)
    #                        print('Test Score: %f RMSE' % (testScoreRMSE))
    #                        print('Test Score: %f MAE' % (testScoreMAE))
                            
    #                        detail_network_file_name = full_path + 'SL=' + str(sti_level) + '_Slid=' + str(sliding) + '_Epoch=' + str(epoch) + '_BS=' + str(batch_size) + '_LR=' + str(learning_rate) + '_PN=' + str(positive_number) + '_NetworkDetail.txt'
    #                        with open(detail_network_file_name, 'a') as f:
    #                            print >> f, 'Weight input and hidden: ', matrix_Wih 
    #                            print >> f, 'Weight hidden and output: ', vector_Who 
    #                            print >> f, 'Predict normalize', predict
    #                            print >> f, 'Predict unnormalize', testPredictInverse
    #                            print >> f, 'len(testY): {0}, len(testPredict): {1}'.format(len(testY[0]), len(testPredictInverse))
                            
                            # summarize history for point prediction
#                            if testScoreMAE < 0.4:
                            plt.figure(pl1)
                            plt.plot(testY)
                            plt.plot(testPredictInverse)
                            plt.title('model predict')
                            plt.ylabel('real value')
                            plt.xlabel('point')
                            plt.legend(['realY... Test Score MAE= ' + str(testScoreMAE) , 'predictY... Test Score RMSE= '+ str(testScoreRMSE)], loc='upper left')
                            pic1_file_name = full_path + 'Train=' +str(u_num[0]) + '_Test=' + str(u_num[1]) + '_SL=' + str(sti_level) + '_Slid=' + str(sliding) + '_Epoch=' + str(epoch) + '_BS=' + str(batch_size) + '_LR=' + str(learning_rate) + '_PN=' + str(positive_number) + '_PointPredict.png'
                            plt.savefig(pic1_file_name)
#                                plt.close()
                            pl1 += 1
                    
    #                        plt.figure(pl2)
    #                        plt.plot(list_loss_AMSE)
    #                        plt.plot(list_loss_RMSE)
    #                        plt.ylabel('Real training loss')
    #                        plt.xlabel('Epoch:')
    #                        plt.legend(['Test Score MAE= ' + str(testScoreMAE) , 'Test Score RMSE= ' + str(testScoreRMSE) ], loc='upper left')
    #                        pic2_file_name = full_path + 'SL=' + str(sti_level) + '_Slid=' + str(sliding) + '_Epoch=' + str(epoch) + '_BS=' + str(batch_size) + '_LR=' + str(learning_rate) + '_PN=' + str(positive_number) + '_TrainingLoss.png'
    #                        plt.savefig(pic2_file_name)
    #                        plt.close()
    #                        pl2 += 1
                            
                            so_vong_lap += 1
                            if so_vong_lap % 5000 == 0:
                                print "Vong lap thu : {0}".format(so_vong_lap)

print "Processing DONE !!!"