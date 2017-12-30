 #!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 07:39:13 2017

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
    return temp[randint(0, len(train_X))]

def get_random_vector_weight(wHa, wHb):
    temp = []
    for i in range(len(wHa)):
        temp.append(uniform(wHa[i], wHb[i]))
    return np.array(temp)


## Load data frame
full_path_name = "/home/thieunv/university/LabThayMinh/code/6_google_trace/data/"
file_name = "Fuzzy_data_sampling_617685_metric_10min_datetime_origin.csv"
full_path = "/home/thieunv/university/LabThayMinh/code/6_google_trace/FLNN/results/notDecompose/data10minutes/univariate/ram/"
df = read_csv(full_path_name+ file_name, header=None, index_col=False, usecols=[0], engine='python')   
dataset_original = df.values

stimulation_level = [0.5]
distance_level = 0.1    # [0.1, 0.15, 0.2]
threshold_number = 2     #[2, 3, 4]
learning_rate = 0.01    #[0.02, 0.05, 0.1, 0.15, 0.2]
positive_number = 0.005      # [0.02, 0.05, 0.1, 0.15, 0.2]
length = dataset_original.shape[0]
num_features = dataset_original.shape[1]
sliding_windows = [3]  #[2, 3, 4, 5]        
train_size = 2880                   #
test_size = length - train_size
valid = 0.25        # Hien tai chua dung den tham so nay
epsilon = 0.00001   # Hien tai chua dung den tham so nay

                             
#### My model here
def mySONIA(train_X, train_y, test_X, validation,sliding, eta = 0.01, stimulation_level=0.05, distance_level=0.1, threshold_number=2):
    
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
    list_hu = [hu1]         # list hidden units
    matrix_Wih = hu1[1].reshape(hu1[1].shape[0], 1)     # Mang 2 chieu 
    
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
            print "distmc = {0}".format(distmc)
            list_hu[c][0] += 1                  # update hidden unit cth
            
            # update all vector W
#            for hu in list_hu:
#                hu[1] += eta * distmc
#            matrix_Wih += eta * distmc    # Phai o dang numpy thi ms update toan bo duoc
            
            # Just update vector W(c)
            list_hu[c][1] += eta * distmc
            matrix_Wih = np.transpose(matrix_Wih)    # Phai o dang numpy thi ms update toan bo duoc
            matrix_Wih[c] += eta * distmc
            matrix_Wih = np.transpose(matrix_Wih)
            
            
            # Tiep tuc vs cac example khac
            m += 1
            if m % 100 == 0:
                print "m = {0}".format(m)
        else:
            list_hu.append([0, copy.deepcopy(train_X[m]) ])
            print "Hidden unit thu: {0} duoc tao ra.".format(len(list_hu))
            matrix_Wih = np.append(matrix_Wih, copy.deepcopy(train_X)[m].reshape((matrix_Wih.shape[0], 1)), axis = 1)
            for hu in list_hu:
                hu[0] = 0
            # then go to step 1
            m = 0
    
    ### Qua trinh mutated hidden unit (Pha 2- Adding artificial local data)
    # Adding 2 hidden unit in begining and ending points of input space
    t1 = np.zeros(num_features * sliding)
    t2 = np.ones(num_features * sliding)
    list_hu.append([0, t1])
    list_hu.append([0, t2])
    matrix_Wih = np.append(matrix_Wih, t1.reshape(t1.shape[0], 1), axis=1)
    matrix_Wih = np.append(matrix_Wih, t2.reshape(t2.shape[0], 1), axis=1)
    
    # Sort matrix weights input and hidden
    sorted_matrix_Wih = []
    for i in range(0, len(train_X[0])):
        sorted_matrix_Wih.append(sorted(matrix_Wih[i]))
    sorted_matrix_Wih = np.array(sorted_matrix_Wih)
    
    # Sort list hidden unit by list weights
    sorted_list_hu = []
    for i in range(0, len(list_hu)):
        sorted_list_hu.append([list_hu[i][0], np.transpose(sorted_matrix_Wih)[i]])
        
    # Now working on both sorted matrix weights and sorted list hidden units     
    for i in range(0, len(list_hu)-1):
        ta, wHa = list_hu[i][0], list_hu[i][1]
        tb, wHb = list_hu[i+1][0], list_hu[i+1][1]
        
        dab_sum = 0.0
        for j in range(0, len(wHa)):
            dab_sum += pow(wHa[j] - wHb[j], 2.0)
        dab = sqrt(dab_sum)
        
        if dab > distance_level and ta < threshold_number and tb < threshold_number:
            # Create new mutated hidden unit
            t1 = get_random_vector_weight(wHa, wHb)
            sorted_list_hu.append([0, t1])
            sorted_matrix_Wih = np.append(sorted_matrix_Wih, t1.reshape(t1.shape[0], 1), axis=1)
    # Ending phrase 2
    
    ### Building set of weights between hidden layer and output layer
    ## Initialize weights and bias
    
    #sorted_list_hu = list_hu
    #sorted_matrix_Wih = matrix_Wih

    matrix_Who = np.zeros(len(sorted_list_hu))
    bias = 1
    print "Random bias is: {0}".format(bias)

    ## Training weights and bias based on backpropagation
    
    list_loss = []
    for t in range(1):    # 100: epoch
        
        loss = 0.0
        
        for k in range(0, len(train_X)):        # training with 1 example at a time
            
            # Calculate output of hidden layer to put it in input of output layer
            output_hidden_layer = []     
            for i in range(0, len(np.transpose(sorted_matrix_Wih))):
                xHj_sum = 0.0
                for j in range(0, len(train_X[0])):
                    xHj_sum += pow(sorted_matrix_Wih[j][i] - train_X[k][j], 2.0)
                output_hidden_layer.append(hyperbolic_tangent_sigmoid_activation(sqrt(xHj_sum)))
#            with open('out_hidden_layer.txt', 'a') as f:
#                print >> f, 'Output hidden layer:', output_hidden_layer  # Python 2.x
            # Right now we have: output hidden, weights hidden and output, bias
            
            # Next: Calculate y_output
            y_output = bias
            for i in range(0, len(matrix_Who)):
                y_output += matrix_Who[i] * output_hidden_layer[i]
            y_output = sigmoid_activation(y_output)
            
            loss += abs(y_output - train_y[k])
   
            ### Next: Update weight and bias using backpropagation
            ## 1. update weights and bias hidden and output
            delta_weights_ho = -1 * positive_number * y_output * (1 - y_output) * (y_output - train_y[k]) * np.array(output_hidden_layer) 
            delta_bias = -1 * positive_number * y_output * (1 - y_output) * (y_output - train_y[k])
            
#            with open('out_delta_weights.txt', 'a') as f:
#                print >> f, 'Delta weights:', delta_weights_ho  
#            with open('out_delta_bias.txt', 'a') as f:
#                print >> f, 'Delta bias:', delta_bias  
            
            matrix_Who += delta_weights_ho
            bias += delta_bias
            
            ##2. update weights input and hidden
            
#            distance_out_hl = []        # Tinh khoang cach tu 1 hidden unit den example
#            list_xHj = []
#            for i in range(0, len(np.transpose(sorted_matrix_Wih))):
#                xHj_sum = 0.0
#                for j in range(0, len(train_X[0])):
#                    xHj_sum += pow(sorted_matrix_Wih[j][i] - train_X[k][j], 2.0)
#                distance_out_hl.append(1.0 / sqrt(xHj_sum))
#                list_xHj.append(1 - hyperbolic_tangent_sigmoid_activation(sqrt(xHj_sum)))
#                
#            # a. Calculate matrix input Xi
#            
#            temp = []           # Vd: [x1, x2]
#            for j in range(0, len(train_X[0])):
#                temp.append(train_X[k][j])
#            matrix_input_Xi = np.array(temp)
#            
#            # VD: (w11 w12 w13; w21 w22 w23) --> (w11 w21; w12 w22; w13 w23) - (x1 x2) = (w11-x1 w21-x2; w12-x1 w22-x2; w13-x1 w23-x2)
#            matrix_input_Xi = (sorted_matrix_Wih.transpose() - matrix_input_Xi).transpose()
#            
#            distance_out_hl = np.array([distance_out_hl])   # make it to matrix 2D
#            list_xHj = np.array([list_xHj])
#            part_three_1 = distance_out_hl
#            part_three_2 = list_xHj
#            for j in range(0, len(train_X[0])-1):
#                part_three_1 = np.concatenate((part_three_1, distance_out_hl), axis = 0)
#                part_three_2 = np.concatenate((part_three_2, list_xHj), axis = 0)
#           
#            # Dot product two matrix
#            part_three = part_three_1 * matrix_input_Xi * part_three_2
#            
#            # Calculate part_two
#            part_two_temp = np.array([matrix_Who * y_output * (1- y_output)])   # Make 2-D array
#            part_two = part_two_temp
#            # dulicate row to make it to make
#            for j in range(0, len(train_X[0])-1):
#                part_two = np.concatenate((part_two, part_two_temp), axis = 0)
#            
#            
#            delta_weights_ih = -2*(train_y[k] - y_output)* part_two * part_three
#            sorted_matrix_Wih += delta_weights_ih
            
        list_loss.append(loss/len(train_X))
        
    print "loss"
    print list_loss
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
        
        pre_y_output = bias
        for i in range(0, len(matrix_Who)):
            pre_y_output += matrix_Who[i] * pre_output_hl[i]
        pre_y_output = sigmoid_activation(pre_y_output)
        predict.append(pre_y_output)
     
    return (sorted_matrix_Wih, matrix_Who , bias, np.array(predict))

pl1 = 1         # Use to draw figure
### Loop to test effect of difference stimulation level 
for sti_level in stimulation_level:
    
    ## Load data
    dataset = []
    for val in dataset_original:
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
    
    for sliding in sliding_windows:
        data = []
        for i in range(length-sliding):
            detail=[]
            for ds in data_scaler:
                for j in range(sliding):
                    detail.append(ds[i+j])
            data.append(detail)
        data = np.reshape(np.array(data), (len(data), num_features*sliding ) )
        
        
#        trainX, trainY = data[0:train_size], data[sliding:train_size+sliding, 0:1]
#        testX = data[train_size:length-sliding]
#        testY =  GoogleTrace_orin_unnormal[train_size+sliding:length]
        #transformTestY = data[train_size:length-sliding, 0:1]
        
        trainX, trainY = data[0:200], data[sliding:200+sliding, 0:1]
        testX = data[200:300-sliding]
        testY = GoogleTrace_orin_unnormal[200+sliding:300]
            
        matrix_Wih, vector_Who, bias, predict = mySONIA(trainX, trainY, testX, validation=valid, sliding=sliding,
                eta=learning_rate, stimulation_level=sti_level, distance_level=distance_level, threshold_number=threshold_number)
        
        print "bias: {0}".format(bias)
        print "Weight input and hidden: "
        print matrix_Wih
        print "Weight hidden and output: "
        print vector_Who
        print "Predict " 
        print predict
    
        # invert predictions        
        testPredictInverse = my_invert_min_max_scaler(predict, min_GT, max_GT)
        print testPredictInverse
        print 'len(testY): {0}, len(testPredict): {1}'.format(len(testY[0]), len(testPredictInverse))
        
        # calculate root mean squared error
        testScoreRMSE = sqrt(mean_squared_error(testY, testPredictInverse))
        testScoreMAE = mean_absolute_error(testY, testPredictInverse)
        print('Test Score: %f RMSE' % (testScoreRMSE))
        print('Test Score: %f MAE' % (testScoreMAE))
        
        # summarize history for point prediction
        plt.figure(pl1)
        plt.plot(testY)
        plt.plot(testPredictInverse)
        
        plt.title('model predict')
        plt.ylabel('real value')
        plt.xlabel('point')
        plt.legend(['realY', 'predictY'], loc='upper left')
        plt.savefig(full_path+'pointPredict_stimulationLevel=%s_sliding_window=%s.png'%(sti_level, sliding))
        pl1 += 1


