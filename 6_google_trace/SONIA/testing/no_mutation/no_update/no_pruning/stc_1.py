#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:44:56 2018

@author: thieunv

SONIA without normalization input data
"""

# Import the needed libraries
import numpy as np  
from pandas import read_csv
from scipy.spatial import distance
from operator import itemgetter
from math import exp, sqrt
from random import randint
import copy
import tensorflow as tf  
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


def distance_func(a, b):
    return distance.euclidean(a, b)
def sigmoid_activation(x):
    return 1.0 / (1.0 + exp(-x))

def relu_activation(x):
    return max(0, x)

def get_random_input_vector(array):
    return copy.deepcopy(array[randint(0, len(array)-1)])

def my_min_max_scaler(data):
    minx = min(data)
    maxx = max(data)
    return (np.array(data).astype(np.float64) - minx) / (maxx - minx)
def my_invert_min_max_scaler(data, minx, maxx):
    return np.array(data).astype(np.float64) * (maxx-minx) + minx

def get_batch_data_next(trainX, trainY, index, batch_size):
    real_index = index*batch_size
    if (len(trainX) % batch_size != 0 and index == (len(trainX)/batch_size +1) ):
        return (trainX[real_index:], trainY[real_index:])
    elif (real_index == len(trainX)):
        return ([], [])
    else:
        return (trainX[real_index: (real_index+batch_size)], trainY[real_index: (real_index+batch_size)])


class SoniaModel(object):
    def __init__(self, dataset_original, list_idx, stimulation_level, positive_number, epoch, batch_size, sliding, learning_rate):
        self.dataset_original = dataset_original
        self.train_idx = list_idx[0]
        self.test_idx = list_idx[1]
        self.stimulation_level = stimulation_level
        self.positive_number = positive_number
        self.epoch = epoch
        self.batch_size = batch_size
        self.sliding = sliding
        self.learning_rate = learning_rate
        
    def preprocessing_data(self):
        train_idx, test_idx, dataset_original, sliding = self.train_idx, self.test_idx, self.dataset_original, self.sliding
        ## Split data
        dataset_split = dataset_original[:test_idx + sliding]
#        GoogleTrace_orin_unnormal = copy.deepcopy(dataset_split)    # keep orginal data to test
        # normalize the dataset
#        self.min_GT = min(GoogleTrace_orin_unnormal[:train_idx])
#        self.max_GT = max(GoogleTrace_orin_unnormal[:train_idx])
        ## Scaling min max
#        dataset_scale = my_min_max_scaler(dataset_split)
        ## Handle data with sliding
        dataset_sliding = dataset_split[:len(dataset_split)-sliding]
        for i in range(sliding-1):
            dddd = np.array(dataset_split[i+1: len(dataset_split)-sliding+i+1])
            dataset_sliding = np.concatenate((dataset_sliding, dddd), axis=1)
        ## Split data to set train and set test
        self.X_train, self.y_train = dataset_sliding[0:train_idx], dataset_sliding[sliding:train_idx+sliding, 0:1]
        self.X_test, self.y_test = dataset_sliding[train_idx:test_idx-sliding], dataset_sliding[train_idx+sliding:test_idx, 0:1]
        #        self.y_test = GoogleTrace_orin_unnormal[train_idx+sliding:test_idx]
        print("Processing data done!!!")
        
    def cluster_data(self):
        train_X = copy.deepcopy(self.X_train)
        stimulation_level, positive_number = self.stimulation_level, self.positive_number
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
                
                # update all vector W
#                for hu in list_hu:
#                    hu[1] += positive_number * distmc
#                matrix_Wih += positive_number * distmc    # Phai o dang numpy thi ms update toan bo duoc
                
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
                if m % 100 == 0:
                    print "distmc = {0}".format(distmc)
                    print "m = {0}".format(m)
            else:
                ## +++ Get the first matrix weight hasn't been customize
                matrix_Wih = copy.deepcopy(trace_back_list_matrix_Wih[0])
                list_hu = copy.deepcopy(trace_back_list_hu[0])
                ## +++ Del all trace back matrix weight except the first one
                del trace_back_list_matrix_Wih[1:]
                del trace_back_list_hu[1:]
                print "Failed !!!. distmc = {0}".format(distmc)
                    
                list_hu.append([0, copy.deepcopy(train_X[m]) ])
                print "Hidden unit thu: {0} duoc tao ra.".format(len(list_hu))
                
                matrix_Wih = np.append(matrix_Wih, copy.deepcopy(train_X[m]).reshape((matrix_Wih.shape[0], 1)), axis = 1)
                for hu in list_hu:
                    hu[0] = 0
                # then go to step 1
                m = 0
                ### +++
                trace_back_list_matrix_Wih[0] = copy.deepcopy(matrix_Wih)
                trace_back_list_hu[0] = copy.deepcopy(list_hu)    
        ### +++ Get the last matrix weight 
        self.matrix_Wih = copy.deepcopy(np.transpose(trace_back_list_matrix_Wih[-1]))
        self.list_hu_1 = copy.deepcopy(trace_back_list_hu[-1])
        ### +++ Delete trace back
        del trace_back_list_matrix_Wih
        del trace_back_list_hu
        
        print("Cluster Data DONE!!!")
    
    def transform_features(self):
        temp1 = []
        for i in range(0, len(self.X_train)):  
            Sih = []
            for j in range(0, len(self.matrix_Wih)):     # (w11, w21) (w12, w22), (w13, w23)
                Sih.append(np.tanh(sqrt(distance_func(self.matrix_Wih[j], self.X_train[i]))))
            temp1.append(np.array(Sih))
        
        temp2 = []
        for i in range(0, len(self.X_test)):  
            Sih = []
            for j in range(0, len(self.matrix_Wih)):     # (w11, w21) (w12, w22), (w13, w23)
                Sih.append(np.tanh(sqrt(distance_func(self.matrix_Wih[j], self.X_train[i]))))
            temp2.append(np.array(Sih))
            
        self.S_train = np.array(temp1)
        self.S_test = np.array(temp2)
        
        print("Transform Features DONE!!!")
        
    # Create and train a tensorflow model of a neural network
    def build_model_and_train(self):
        learning_rate, epoch, batch_size, X_train, y_train = self.learning_rate, self.epoch, self.batch_size, copy.deepcopy(self.S_train), copy.deepcopy(self.y_train)
        w2 = np.random.rand(len(self.list_hu_1), 1) - 0.5
        
         ## 3. Loop in epochs
        total_batch = int(len(X_train) / self.batch_size) + 1
        for i in range(epoch):
            print "Epoch thu: {0}".format(i+1)
            ## 3.1 Update w after 1 batch
            for ind in range(0, total_batch):
                
                ## 3.1.1 Get next batch
                X_train_next, y_train_next = get_batch_data_next(X_train, y_train, ind, batch_size)
                if (len(X_train_next) == 0):
                    break
                
                ## 3.1.2 Calculate all delta weight in 1 batch 
                delta_ws = []
                for j in range(len(X_train_next)):
                    y_output = np.dot(X_train_next[j], w2)
                    y_output = relu_activation(y_output)
                    ek = y_train_next[j] - y_output
                    delta_w = learning_rate * ek * X_train_next[j]
                    
                    delta_ws.append(delta_w)
                
                ## 3.1.3 Sum all delta weight to get mean delta weight
                delta_wbar = np.array(np.sum(delta_ws, axis = 0) / len(X_train_next))
                
                ## 3.1.4 Change new weight with delta_wbar (mean delta weight)
                w2 += np.reshape(delta_wbar, (len(self.list_hu_1), 1))
        
        self.w2 = copy.deepcopy(w2)
        print "Final weight: "
        print w2
    
        print("Build model and train done!!!")
        
    def draw_loss(self):
        plt.figure(1)
        plt.plot(range(self.epoch), self.loss_train, label="Loss on training per epoch")
        plt.xlabel('Iteration', fontsize=12)  
        plt.ylabel('Loss', fontsize=12)  
        
        
    def draw_predict(self):
        plt.figure(2)
        plt.plot(self.y_test_inverse)
        plt.plot(self.y_pred_inverse)
        plt.title('Model predict')
        plt.ylabel('Real value')
        plt.xlabel('Point')
        plt.legend(['realY... Test Score RMSE= ' + str(self.score_test_RMSE) , 'predictY... Test Score MAE= '+ str(self.score_test_MAE)], loc='upper right')
    
    def predict(self):
        # Evaluate models on the test set
        X_test, y_test, w2 = self.S_test, self.y_test, self.w2
        
        predict = []
        for i in range(len(X_test)):
            ybar = np.dot(X_test[i], w2)
            ybar = relu_activation(ybar)
            predict.append(ybar)
        y_pred_inverse = copy.deepcopy(np.reshape(predict, (len(predict), 1)))
        y_test_inverse = y_test
        
        testScoreRMSE = sqrt(mean_squared_error(y_test_inverse, y_pred_inverse))
        testScoreMAE = mean_absolute_error(y_test_inverse, y_pred_inverse)
            
        self.y_predict, self.score_test_RMSE, self.score_test_MAE = y_pred_inverse, testScoreRMSE, testScoreMAE
        self.y_test_inverse, self.y_pred_inverse = y_test_inverse, y_pred_inverse
            
        print('DONE - RMSE: %.5f, MAE: %.5f' % (testScoreRMSE, testScoreMAE))
        print(list(y_pred_inverse))
        print("Predict done!!!")
    
    def fit(self):
        self.preprocessing_data()
        self.cluster_data()
        self.transform_features()
        self.build_model_and_train()
        self.predict()
#        self.draw_loss()
        self.draw_predict()
    

## Load data frame
#full_path_name="/mnt/volume/ggcluster/spark-2.1.1-bin-hadoop2.7/thieunv/machine_learning/6_google_trace/data/"
#full_path= "/mnt/volume/ggcluster/spark-2.1.1-bin-hadoop2.7/thieunv/machine_learning/6_google_trace/FLNN/results/notDecompose/data10minutes/univariate/cpu/"

full_path_name = "/home/thieunv/university/LabThayMinh/code/6_google_trace/data/"
full_path = "/home/thieunv/university/LabThayMinh/code/6_google_trace/tensorflow/testing/"

file_name = "Fuzzy_data_sampling_617685_metric_10min_datetime_origin.csv"
df = read_csv(full_path_name+ file_name, header=None, index_col=False, usecols=[0], engine='python')  

#file_name = "wc98_workload_hour.csv"
#df = read_csv(full_path_name+ file_name, header=None, index_col=False, usecols=[1], engine='python')   

dataset_original = np.array(df.values, dtype=np.float64)

learning_rates = [0.25]     #[0.005, 0.01, 0.025, 0.05, 0.10, 0.12, 0.15]   # [0.2]    
sliding_windows = [2] #[ 2, 3, 5]           # [3]  
epochs = [1200] #[100, 250, 500, 1000, 1500, 2000]   # [500]                       
batch_sizes = [32] #[8, 16, 32, 64, 128]      # [16]     
list_idx = (3000, 3700)

stimulation_levels = [5]      
positive_numbers = [0.0001]    


so_vong_lap = 0
for sliding in sliding_windows:
    for epoch in epochs:
        for batch_size in batch_sizes:
            for learning_rate in learning_rates:
                for stimulation_level in stimulation_levels:
                    for positive_number in positive_numbers:

                        ann = SoniaModel(dataset_original, list_idx, stimulation_level, positive_number, epoch, batch_size, sliding, learning_rate)
                        ann.fit()
                        
                        so_vong_lap += 1
                        if so_vong_lap % 5000 == 0:
                            print "Vong lap thu : {0}".format(so_vong_lap)

print "Processing DONE !!!"