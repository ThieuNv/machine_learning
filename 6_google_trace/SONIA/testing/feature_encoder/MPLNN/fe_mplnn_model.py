#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 15:36:18 2018

@author: thieunv


"""

import tensorflow as tf
import numpy as np
from scipy.spatial import distance
from math import exp, sqrt
import copy
from random import randint
from operator import itemgetter

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pandas import read_csv
from sklearn.metrics import mean_squared_error, mean_absolute_error


class FeBpnnModel(object):
    def __init__(self, dataset_original, list_idx, epoch, batch_size, sliding, learning_rate, positive_number, stimulation_level):
        self.dataset_original = dataset_original
        self.train_idx = list_idx[0]
        self.test_idx = list_idx[1]
        self.epoch = epoch
        self.batch_size = batch_size
        self.sliding = sliding
        self.learning_rate = learning_rate
        self.positive_number = positive_number
        self.stimulation_level = stimulation_level
        
    def preprocessing_data(self):
        train_idx, test_idx, dataset_original, sliding = self.train_idx, self.test_idx, self.dataset_original, self.sliding
        ## Split data
        dataset_split = dataset_original[:test_idx + sliding]
        GoogleTrace_orin_unnormal = copy.deepcopy(dataset_split)    # keep orginal data to test
        # normalize the dataset
        self.min_GT = min(GoogleTrace_orin_unnormal[:train_idx])
        self.max_GT = max(GoogleTrace_orin_unnormal[:train_idx])
        ## Scaling min max
        dataset_scale = FeBpnnModel.my_min_max_scaler(dataset_split)
        ## Handle data with sliding
        dataset_sliding = dataset_scale[:len(dataset_scale)-sliding]
        for i in range(sliding-1):
            dddd = np.array(dataset_scale[i+1: len(dataset_scale)-sliding+i+1])
            dataset_sliding = np.concatenate((dataset_sliding, dddd), axis=1)
        ## Split data to set train and set test
        self.X_train, self.y_train = dataset_sliding[0:train_idx], dataset_sliding[sliding:train_idx+sliding, 0:1]
        self.X_test, self.y_test = dataset_sliding[train_idx:test_idx-sliding], dataset_sliding[train_idx+sliding:test_idx, 0:1]
#        self.y_test = GoogleTrace_orin_unnormal[train_idx+sliding:test_idx]
    
    
    def encoder_features(self):
        train_X = copy.deepcopy(self.X_train)
        stimulation_level = copy.deepcopy(self.stimulation_level)
        positive_number = self.positive_number
         ### Qua trinh train va dong thoi tao cac hidden unit (Pha 1 - cluster data)
        # 2. Khoi tao hidden thu 1
        hu1 = [0, FeBpnnModel.get_random_input_vector(train_X)]   # hidden unit 1 (t1, wH)
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
    
    def transform_features(self):
        temp1 = []
        for i in range(0, len(self.X_train)):  
            Sih = []
            for j in range(0, len(self.matrix_Wih)):     # (w11, w21) (w12, w22), (w13, w23)
                Sih.append(np.tanh(FeBpnnModel.distance_func(self.matrix_Wih[j], self.X_train[i])))
            temp1.append(np.array(Sih))
        
        temp2 = []
        for i in range(0, len(self.X_test)):  
            Sih = []
            for j in range(0, len(self.matrix_Wih)):     # (w11, w21) (w12, w22), (w13, w23)
                Sih.append(np.tanh(FeBpnnModel.distance_func(self.matrix_Wih[j], self.X_test[i])))
            temp2.append(np.array(Sih))
            
        self.S_train = np.array(temp1)
        self.S_test = np.array(temp2)
    
    
    def build_bpnn_and_train(self):
        ## Build layer's sizes
        X_train = self.S_train
        y_train = self.y_train
        
        X_size = X_train.shape[1]   
        h1_size = 2 * X_size
        h2_size = 10
        y_size = y_train.shape[1]
        ## Symbols
        X = tf.placeholder("float64", shape=[None, X_size])
        y = tf.placeholder("float64", shape=[None, y_size])
        
        # now declare the weights connecting the input to the hidden layer
        W1 = tf.Variable(tf.random_normal([X_size, h1_size], stddev=0.03, dtype=tf.float64), name="W1")
        b1 = tf.Variable(tf.random_normal([h1_size], dtype=tf.float64), name="b1")
        # and the weights connecting the hidden layer to the output layer
        W2 = tf.Variable(tf.random_normal([h1_size, h2_size], stddev=0.03, dtype=tf.float64), name='W2')
        b2 = tf.Variable(tf.random_normal([h2_size], dtype=tf.float64), name='b2')
        
        W3 = tf.Variable(tf.random_normal([h2_size, y_size], stddev=0.03, dtype=tf.float64), name='W3')
        b3 = tf.Variable(tf.random_normal([y_size], dtype=tf.float64), name='b3')
        
        # calculate the output of the hidden layer
        hidden_out_1 = tf.sigmoid( tf.add(tf.matmul(X, W1), b1) )
        hidden_out_2 = tf.sigmoid( tf.add(tf.matmul(hidden_out_1, W2), b2) )
        y_ = tf.sigmoid(tf.add(tf.matmul(hidden_out_2, W3), b3))
        
        # Backward propagation
        cost    = tf.reduce_mean( tf.square(y_ - y) )
        updates = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
        
        loss_plot = []
        # start the session
        with tf.Session() as sess:
            # initialise the variables
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            
            total_batch = int(len(X_train) / self.batch_size) + 1
            for epoch in range(self.epoch):
               
                for i in range(total_batch):
                     ## Get next batch
                    batch_x, batch_y = FeBpnnModel.get_batch_data_next(X_train, y_train, i, batch_size)
                    if len(batch_x) == 0:
                        break
                    sess.run(updates, feed_dict={X: batch_x, y: batch_y})
                
                loss_epoch = sess.run(cost, feed_dict={X: X_train, y: y_train})
                weights1, bias1, weights2, bias2, weights3, bias3 = sess.run([W1, b1, W2, b2, W3, b3])
                loss_plot.append(loss_epoch)
                print("Epoch: {}".format(epoch + 1), "cost = {}".format(loss_epoch))
                
        self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.loss_train = weights1, bias1, weights2, bias2, weights3, bias3, loss_plot
        print("Build model and train done!!!")
        

    def fit(self):
        self.preprocessing_data()
        self.encoder_features()
        self.transform_features()
        self.build_bpnn_and_train()
        self.predict()
        self.draw_loss()
        self.draw_predict()
    
    def draw_loss(self):
        plt.figure(1)
        plt.plot(range(self.epoch), self.loss_train, label="Loss on training per epoch")
        plt.xlabel('Iteration', fontsize=12)  
        plt.ylabel('Loss', fontsize=12)  
        
        
    def draw_predict(self):
        plt.figure(2)
        plt.plot(FeBpnnModel.my_invert_min_max_scaler(self.y_test, self.min_GT, self.max_GT))
        plt.plot(FeBpnnModel.my_invert_min_max_scaler(self.y_predict, self.min_GT, self.max_GT))
        plt.title('Model predict')
        plt.ylabel('Real value')
        plt.xlabel('Point')
        plt.legend(['realY... Test Score RMSE= ' + str(self.score_test_RMSE) , 'predictY... Test Score MAE= '+ str(self.score_test_MAE)], loc='upper right')
    
    def predict(self):
        # Evaluate models on the test set
        X_test, y_test = self.S_test, self.y_test
        
        X_size = X_test.shape[1]   
        y_size = y_test.shape[1]
        
        X = tf.placeholder("float64", shape=[None, X_size], name='X')
        y = tf.placeholder("float64", shape=[None, y_size], name='y')
        
        W1 = tf.Variable(self.w1)
        b1 = tf.Variable(self.b1)
        W2 = tf.Variable(self.w2)
        b2 = tf.Variable(self.b2)
        W3 = tf.Variable(self.w3)
        b3 = tf.Variable(self.b3)
        
        hidden_out_1 = tf.sigmoid( tf.add(tf.matmul(X, W1), b1) )
        hidden_out_2 = tf.sigmoid( tf.add(tf.matmul(hidden_out_1, W2), b2) )
        y_ = tf.sigmoid(tf.add(tf.matmul(hidden_out_2, W3), b3))
        
        # Calculate the predicted outputs
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            y_est_np = sess.run(y_, feed_dict={X: X_test, y: y_test})
            
            testScoreRMSE = sqrt(mean_squared_error(y_test, y_est_np))
            testScoreMAE = mean_absolute_error(y_test, y_est_np)
            
            self.y_predict, self.score_test_RMSE, self.score_test_MAE = y_est_np, testScoreRMSE, testScoreMAE
            print('DONE - RMSE: %.5f, MAE: %.5f' % (testScoreRMSE, testScoreMAE))
            print(list(y_est_np))
        
        print("Predict done!!!")
        
    @staticmethod
    def distance_func(a, b):
        return distance.euclidean(a, b)
    
    @staticmethod
    def sigmoid_activation(x):
        return 1.*0 / (1.0 + exp(-x))
    
    @staticmethod
    def get_random_input_vector(train_X):
        return copy.deepcopy(train_X[randint(0, len(train_X)-1)])
    
    @staticmethod
    def my_min_max_scaler(data):
        minx = min(data)
        maxx = max(data)
        return (np.array(data).astype(np.float64) - minx) / (maxx - minx)
    
    @staticmethod
    def my_invert_min_max_scaler(data, minx, maxx):
        return np.array(data).astype(np.float64) * (maxx-minx) + minx
    
    @staticmethod
    def get_batch_data_next(trainX, trainY, index, batch_size):
        real_index = index*batch_size
        if (len(trainX) % batch_size != 0 and index == (len(trainX)/batch_size +1) ):
            return (trainX[real_index:], trainY[real_index:])
        elif (real_index == len(trainX)):
            return ([], [])
        else:
            return (trainX[real_index: (real_index+batch_size)], trainY[real_index: (real_index+batch_size)])


## Load data frame
#full_path_name="/mnt/volume/ggcluster/spark-2.1.1-bin-hadoop2.7/thieunv/machine_learning/6_google_trace/data/"
#full_path= "/mnt/volume/ggcluster/spark-2.1.1-bin-hadoop2.7/thieunv/machine_learning/6_google_trace/FLNN/results/notDecompose/data10minutes/univariate/cpu/"
file_name = "Fuzzy_data_sampling_617685_metric_10min_datetime_origin.csv"

full_path_name = "/home/thieunv/university/LabThayMinh/code/6_google_trace/data/"
full_path = "/home/thieunv/university/LabThayMinh/code/6_google_trace/tensorflow/testing/"
df = read_csv(full_path_name+ file_name, header=None, index_col=False, usecols=[0], engine='python')   
dataset_original = df.values

stimulation_level = [0.35]  #[0.10, 0.2, 0.25, 0.50, 1.0, 1.5, 2.0]  # [0.20]    
positive_numbers = [0.001] #[0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.20]     # [0.1]   
learning_rates = [0.25] #[0.005, 0.01, 0.025, 0.05, 0.10, 0.12, 0.15]   # [0.2]    
sliding_windows = [3] #[ 2, 3, 5]           # [3]  
epochs = [93] #[100, 250, 500, 1000, 1500, 2000]   # [500]                       
batch_sizes = [32] #[8, 16, 32, 64, 128]      # [16]     
list_num = [(3000, 3500)]


pl1 = 1         # Use to draw figure
#pl2 = 1000
so_vong_lap = 0

for list_idx in list_num:
    for sliding in sliding_windows:
        for sti_level in stimulation_level:
            for epoch in epochs:
                for batch_size in batch_sizes:
                    for learning_rate in learning_rates:
                        for positive_number in positive_numbers:

                            febpnn = FeBpnnModel(dataset_original, list_idx, epoch, batch_size, sliding, learning_rate, positive_number, sti_level)
                            febpnn.fit()
                            
                            so_vong_lap += 1
                            if so_vong_lap % 5000 == 0:
                                print "Vong lap thu : {0}".format(so_vong_lap)

print "Processing DONE !!!"
    
    