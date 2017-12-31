#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 12:54:15 2017

@author: thieunv

Hien tai mang nay la tot nhat. Nhan xet:
    - Da~ loai bo bias, va update weight input and hidden in backpropagation process
    - chi lam duoc vs du lieu nho (200 - 500)
    - Cac tham so anh huong rat lon
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
    return temp[randint(0, len(train_X)-1 )]

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

# t = [[9, 1], [2, 6], [3, 7], [8, 5], [7, 1]]
def square(list_elem):
    temp = 0.0
    for val in list_elem:
        temp += pow(val, 2.0)
    return temp
#sorted(t, key=square)
#Out[9]: [[2, 6], [7, 1], [3, 7], [9, 1], [8, 5]]
    
# list_temp = [ [132, np.array([0, 0, 0]) ], [181, np.array([1, 1, 1]) ], [15, np.array([0.5, 0.55, 0.555])] ]
def take_second_and_square(elem):
    list_elem = elem[1]
    temp = 0.0
    for val in list_elem:
        temp += pow(val, 2.0)
    return temp
# sorted(list_temp, key=square)
# [[132, array([0, 0, 0])], [15, array([ 0.5  ,  0.55 ,  0.555])], [181, array([1, 1, 1])]]

## C2: list_temp.sort(key=lambda item: sum(map(lambda e: e*e, item[1])))
    

def get_mutate_vector_weight(wHa, wHb, mutation_id = 1):
    temp = []
    if mutation_id == 1:    # Lay trung binh cong
        for i in range(len(wHa)):
            temp.append( (wHa[i] + wHb[i]) / 2 )
    elif mutation_id == 2:    # Lay uniform
        for i in range(len(wHa)):
            temp.append(uniform(wHa[i], wHb[i]))
    
    return np.array(temp)

def my_sorted(list_elem, key_function, key_id):
    if key_id == 1:     # Sort the matrix 2-D
        kk = list_elem.transpose()
        temp = sorted(kk, key=key_function)
        return np.transpose(np.array(temp))
    if key_id == 2:     # Sort the matrix 2-D inside the list
        return sorted(list_elem, key=key_function)

def decrease_list_hidden_unit(list_hus, matrix_weights, percent=0.13, sort=True, how=1):
    if sort == True:
        ind1 = int(percent* len(list_hus) / 2)
        ind2 = len(list_hus) - ind1
        matrix_weights = my_sorted(matrix_weights, key_function=square, key_id=1)
        list_hus = my_sorted(list_hus, key_function=take_second_and_square, key_id=2) 
        
        temp1 = list_hus[:][ind1:ind2]
        temp2 = [matrix_weights[i][ind1:ind2] for i in range(matrix_weights.shape[0])]
        
        return (temp1, np.array(temp2))
        

## Load data frame
full_path_name="/mnt/volume/ggcluster/spark-2.1.1-bin-hadoop2.7/thieunv/machine_learning/6_google_trace/data/"
full_path= "/mnt/volume/ggcluster/spark-2.1.1-bin-hadoop2.7/thieunv/machine_learning/6_google_trace/SONIA/testing/mutation/update/pruning/result/cpu/"
file_name = "Fuzzy_data_sampling_617685_metric_10min_datetime_origin.csv"

#full_path_name = "/home/thieunv/university/LabThayMinh/code/6_google_trace/data/"
#full_path = "/home/thieunv/university/LabThayMinh/code/6_google_trace/SONIA/results/notDecompose/data10minutes/univariate/cpu/"
df = read_csv(full_path_name+ file_name, header=None, index_col=False, usecols=[0], engine='python')   
dataset_original = df.values

distance_levels = [0.1, 0.25, 0.5]
threshold_number = 100   # 50 example                #[2, 3, 4]

stimulation_level = [0.10, 0.2, 0.25, 0.50, 1.0, 1.5, 2.0]  # [0.20]    
positive_numbers = [0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.20]     # [0.1]   

learning_rates = [0.005, 0.01, 0.025, 0.05, 0.10, 0.12, 0.15]   # [0.2]    
sliding_windows = [1, 2, 3, 5]           # [3]  
     
epochs = [100, 250, 500, 1000, 1500, 2000]   # [500]                       
batch_sizes = [8, 16, 32, 64, 128]      # [16]     

length = dataset_original.shape[0]
num_features = dataset_original.shape[1]

train_size = 700       #2880                   
test_size = length - train_size
valid = 0.25        # Hien tai chua dung den tham so nay
epsilon = 0.00001   # Hien tai chua dung den tham so nay

constant_error = 0.0001

list_percent_decreases = [0.10, 0.13, 0.15, 0.20, 0.25]

list_num = [(500, 1000), (500, 1500), (500, 2000), (750, 1250), (750, 1750), (750, 2250),
            (1500, 2000), (1500, 2500), (1500, 3000), (1500, 2000), (1500, 2500), (1500, 3000),
            (2000, 2500), (2000, 3000), (2000, 3500), (2500, 3000), (2500, 3500), (2500, 4000)
]
                             
#### My model here
def mySONIA(train_X, train_y, test_X, epoch, batch_size, validation,sliding, decrease=0.13, learning_rate = 0.01, 
            positive_number = 0.1, stimulation_level=0.05, distance_level=0.1, threshold_number=2):
    
    #### Chu y:
    # 1. Tat ca cac input trainX phai normalize ve doan [0, 1]
    
    ## 1. Split validation dataset
    #valid_len = int(len(trainX)*validation)
    #valid_X = trainX[(len(trainX) - valid_len):]
    #valid_y = trainY[(len(trainX) - valid_len):]
   
    #valid_list_loss = [0.1]
    #train_list_loss = [0.1]
    
    
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
            
#            if m % 1000 == 0:
##                with open(training_detail_file_name, 'a') as f:
##                    print >> f, 'distmc = :', distmc  
##                    print >> f, 'Example thu :', m  
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
    matrix_Wih = copy.deepcopy(trace_back_list_matrix_Wih[-1])
    list_hu = copy.deepcopy(trace_back_list_hu[-1])
    ### +++ Delete trace back
    del trace_back_list_matrix_Wih
    del trace_back_list_hu
            
    
    ### Qua trinh mutated hidden unit (Pha 2- Adding artificial local data)
    # Adding 2 hidden unit in begining and ending points of input space
    t1 = np.zeros(num_features * sliding)
    t2 = np.ones(num_features * sliding)
    list_hu.append([0, t1])
    list_hu.append([0, t2])
    matrix_Wih = np.append(matrix_Wih, t1.reshape(t1.shape[0], 1), axis=1)
    matrix_Wih = np.append(matrix_Wih, t2.reshape(t2.shape[0], 1), axis=1)

#    # Sort matrix weights input and hidden, Sort list hidden unit by list weights
    sorted_matrix_Wih = my_sorted(matrix_Wih, key_function=square, key_id=1)
    sorted_list_hu = my_sorted(list_hu, key_function=take_second_and_square, key_id=2) 

#    # Now working on both sorted matrix weights and sorted list hidden units     
    for i in range(len(sorted_list_hu) - 1):
        ta, wHa = sorted_list_hu[i][0], sorted_list_hu[i][1]
        tb, wHb = sorted_list_hu[i+1][0], sorted_list_hu[i+1][1]
        
        dab_sum = 0.0
        for j in range(0, len(wHa)):
            dab_sum += pow(wHa[j] - wHb[j], 2.0)
        dab = sqrt(dab_sum)
        
        if dab > distance_level and ta < threshold_number and tb < threshold_number:
            # Create new hidden unit (Lai ghep)
            #t1 = get_random_vector_weight(wHa, wHb)
            
            # Create new mutated hidden unit (Dot Bien)
            temp_node = get_mutate_vector_weight(wHa, wHb, mutation_id=1)
            sorted_list_hu.insert(i+1, [0, temp_node])
            sorted_matrix_Wih = np.insert(sorted_matrix_Wih, [i+1], temp_node.reshape(temp_node.shape[0], 1), axis=1)
#            print "New hidden unit created. {0}".format(len(sorted_list_hu))
                    
    # Ending phrase 2
    
    ### Building set of weights between hidden layer and output layer
    ## Initialize weights and bias
    
#    sorted_list_hu = copy.deepcopy(sorted_list_hu)
#    sorted_matrix_Wih = copy.deepcopy(sorted_matrix_Wih)
#    
    
    # Avoid 1.0 / 0 when update weights between input and hidden
    for i in range(len(sorted_matrix_Wih[0])):
        if sorted_list_hu[i][0] == 1:
            sorted_list_hu[i][1] += constant_error
            sorted_matrix_Wih[i] += constant_error
    
    sorted_list_hu, sorted_matrix_Wih = decrease_list_hidden_unit(sorted_list_hu, sorted_matrix_Wih, percent=decrease)
    
    matrix_Who = np.zeros(len(sorted_list_hu))
    bias = 1
#    print "Random bias is: {0}".format(bias)

    ## Training weights and bias based on backpropagation
    
    list_loss_RMSE = []
    list_loss_AMSE = []
    for t in range(epoch):   
        
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
#            delta_bias = []
            delta_weights_ih = []
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
#                delta_bias_temp = -2 * learning_rate * y_output * (1 - y_output) * (y_output - train_y[k]) 
                
                delta_ws.append(delta_weights_ho)
#                delta_bias.append(delta_bias_temp)
                
                
                
                
                 ##2. update weights input and hidden
            
                distance_out_hl = []        # Tinh khoang cach tu 1 hidden unit den example
                list_xHj = []
                for i in range(0, len(np.transpose(sorted_matrix_Wih))):
                    xHj_sum = 0.0
                    for j in range(0, len(train_X[0])):
                        xHj_sum += pow(sorted_matrix_Wih[j][i] - train_X[k][j], 2.0)
                    distance_out_hl.append(1.0 / sqrt(xHj_sum))
                    list_xHj.append(1 - hyperbolic_tangent_sigmoid_activation(sqrt(xHj_sum)))
                    
                # a. Calculate matrix input Xi
                
                temp = []           # Vd: [x1, x2]
                for j in range(0, len(train_X[0])):
                    temp.append(train_X[k][j])
                matrix_input_Xi = np.array(temp)
                
                # VD: (w11 w12 w13; w21 w22 w23) --> (w11 w21; w12 w22; w13 w23) - (x1 x2) = (w11-x1 w21-x2; w12-x1 w22-x2; w13-x1 w23-x2)
                matrix_input_Xi = (sorted_matrix_Wih.transpose() - matrix_input_Xi).transpose()
                
                distance_out_hl = np.array([distance_out_hl])   # make it to matrix 2D
                list_xHj = np.array([list_xHj])
                part_three_1 = distance_out_hl
                part_three_2 = list_xHj
                for j in range(0, len(train_X[0])-1):
                    part_three_1 = np.concatenate((part_three_1, distance_out_hl), axis = 0)
                    part_three_2 = np.concatenate((part_three_2, list_xHj), axis = 0)
               
                # Dot product two matrix
                part_three = part_three_1 * matrix_input_Xi * part_three_2
                
                # Calculate part_two
                part_two_temp = np.array([matrix_Who * y_output * (1- y_output)])   # Make 2-D array
                part_two = part_two_temp
                # dulicate row to make it to make
                for j in range(0, len(train_X[0])-1):
                    part_two = np.concatenate((part_two, part_two_temp), axis = 0)
                
                delta_weights_ih_temp = -2*(train_y[k] - y_output)* part_two * part_three
                delta_weights_ih.append(delta_weights_ih_temp)
                
                
            ## Sum all delta weight to get mean delta weight
            delta_wbar = np.array(np.sum(delta_ws, axis = 0) / len(X_train_next))
#            delta_b = np.array(np.sum(delta_bias, axis = 0) / len(X_train_next))
            delta_wbar_ih = np.zeros( (delta_weights_ih[0].shape[0], delta_weights_ih[0].shape[1]) )
            for dw in delta_weights_ih:
                delta_wbar_ih += dw
            
            matrix_Who += delta_wbar
#            bias += delta_b
            sorted_matrix_Wih += np.array(delta_wbar_ih) / len(X_train_next)
        
        if t % 2 == 0:
            print "Epoch thu: {0}".format(t)
            print "MASE loss = {0}".format(loss1/len(train_X))
            print "RMSE loss = {0}".format(loss2/len(train_X))
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
        
        pre_y_output = 0 #bias
        for i in range(0, len(matrix_Who)):
            pre_y_output += matrix_Who[i] * pre_output_hl[i]
        pre_y_output = sigmoid_activation(pre_y_output)
        predict.append(pre_y_output)
     
    return (sorted_matrix_Wih, matrix_Who , bias, np.array(predict), list_loss_AMSE, list_loss_RMSE)


pl1 = 1         # Use to draw figure
#pl2 = 10000000
counting_number_loop = 0

for u_num in list_num:
    
    for sliding in sliding_windows:
        
        ## Load data
        dataset = []
        for val in dataset_original[:u_num[1]+sliding]:
            dataset.append(val)
        dataset = (np.asarray(dataset)).astype(np.float64)
        
        # normalize the dataset
        GoogleTrace_orin_unnormal = dataset[:, 0].reshape(-1, 1)    # keep orginal data to test
        min_GT = min(GoogleTrace_orin_unnormal)
        max_GT = max(GoogleTrace_orin_unnormal)
    
        ## Scaling min max
        data_scaler = []
        for i in range( dataset.shape[1] ):
            data_scaler.append( my_min_max_scaler(dataset[:, i].reshape(-1, 1)) ) 
        
        ## Handle data with sliding
        data = []
        for i in range(len(GoogleTrace_orin_unnormal)-sliding):
            detail=[]
            for ds in data_scaler:
                for j in range(sliding):
                    detail.append(ds[i+j])
            data.append(detail)
        data = np.reshape(np.array(data), (len(data), num_features*sliding ) )
        
        ## Split data to set train and set test
        trainX, trainY = data[0:u_num[0]], data[sliding:u_num[0]+sliding, 0:1]
        testX = data[u_num[0]:u_num[1]-sliding]
        testY = GoogleTrace_orin_unnormal[u_num[0]+sliding:u_num[1]]
        
        
        for sti_level in stimulation_level:
    
            for epoch in epochs:
                
                for batch_size in batch_sizes:
                    
                    for learning_rate in learning_rates:
                        
                        for positive_number in positive_numbers:
                            
                            for distance_level in distance_levels:
                                
                                for decrease_percent in list_percent_decreases:
                                    
                                    matrix_Wih, vector_Who, bias, predict, list_loss_AMSE, list_loss_RMSE = mySONIA(trainX, trainY, testX, epoch=epoch, 
                                                batch_size=batch_size, validation=valid, sliding=sliding, decrease=decrease_percent, learning_rate=learning_rate, 
                                                positive_number=positive_number, stimulation_level=sti_level, distance_level=distance_level, threshold_number=threshold_number)
                                    
                                    
#                                    print "bias: {0}".format(bias)
#                                    print "Weight input and hidden: "
#                                    print matrix_Wih
#                                    print "Weight hidden and output: "
#                                    print vector_Who
#                                    print "Predict " 
#                                    print predict
                                
                                    # invert predictions        
                                    testPredictInverse = my_invert_min_max_scaler(predict, min_GT, max_GT)
#                                    print testPredictInverse
#                                    print 'len(testY): {0}, len(testPredict): {1}'.format(len(testY[0]), len(testPredictInverse))
                                    
                                    # calculate root mean squared error
                                    testScoreRMSE = sqrt(mean_squared_error(testY, testPredictInverse))
                                    testScoreMAE = mean_absolute_error(testY, testPredictInverse)
#                                    print('Test Score: %f RMSE' % (testScoreRMSE))
#                                    print('Test Score: %f MAE' % (testScoreMAE))
                                    
            #                        detail_network_file_name = full_path + 'SL=' + str(sti_level) + '_Slid=' + str(sliding) + '_Epoch=' + str(epoch) + '_BS=' + str(batch_size) + '_LR=' + str(learning_rate) + '_PN=' + str(positive_number) + '_NetworkDetail.txt'
            #                        with open(detail_network_file_name, 'a') as f:
            #                            print >> f, 'Weight input and hidden: ', matrix_Wih 
            #                            print >> f, 'Weight hidden and output: ', vector_Who 
            #                            print >> f, 'Predict normalize', predict
            #                            print >> f, 'Predict unnormalize', testPredictInverse
            #                            print >> f, 'len(testY): {0}, len(testPredict): {1}'.format(len(testY[0]), len(testPredictInverse))
                                    
                                    # summarize history for point prediction
                                    if testScoreMAE < 0.5:
                                        plt.figure(pl1)
                                        plt.plot(testY)
                                        plt.plot(testPredictInverse)
                                        plt.title('model predict')
                                        plt.ylabel('real value')
                                        plt.xlabel('point')
                                        plt.legend(['realY... Test Score MAE= ' + str(testScoreMAE), 'predictY... Test Score RMSE= ' + str(testScoreRMSE)], loc='upper right')
                                        pic1_file_name = full_path + 'Train=' +str(u_num[0]) + '_Test=' + str(u_num[1]) + '_SL=' + str(sti_level) + '_Slid=' + str(sliding) + '_Epoch=' + str(epoch) + '_BS=' + str(batch_size) + '_LR=' + str(learning_rate) + '_PN=' + str(positive_number) + '_DL=' + str(distance_level) + '_Decrease=' + str(decrease_percent) + '_PointPredict.png'
                                        plt.savefig(pic1_file_name)
                                        plt.close()
                                        pl1 += 1
                                
#                                        plt.figure(pl2)
#                                        plt.plot(list_loss_AMSE)
#                                        plt.plot(list_loss_RMSE)
#                                        plt.ylabel('Real training loss')
#                                        plt.xlabel('Epoch:')
#                                        plt.legend(['Test Score MAE= ' + str(testScoreMAE) , 'Test Score RMSE= ' + str(testScoreRMSE) ], loc='upper left')
#                                        pic2_file_name = full_path + 'SL=' + str(sti_level) + '_Slid=' + str(sliding) + '_Epoch=' + str(epoch) + '_BS=' + str(batch_size) + '_LR=' + str(learning_rate) + '_PN=' + str(positive_number) + '_TrainingLoss.png'
#                                        plt.savefig(pic2_file_name)
#                                        plt.close()
#                                        pl2 += 1

                                    counting_number_loop += 1
                                    if counting_number_loop % 5000 == 0:
                                        print "Vong lap thu: {0}".format(counting_number_loop)

print "Processing DONE!!!"
