#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 09:15:03 2018

@author: thieunv

*) GradientDescentOptimizer (MinMaxScaler - 2)

    elu / relu/ tanh/ sigmoid ===> 0.35 (nhin figure thi xau) / failed/  / failed 
    
*) AdamOptimizer 
    
     elu / relu/ tanh/ sigmoid ===> 0.54/ failed / 0.42 / failed
    
*) AdagradOptimizer   
    
     elu / relu/  tanh/  sigmoid ===> 0.46/ 0.45 /  / failed

*) AdadeltaOptimizer
    
     elu / relu/ tanh/ sigmoid ===> 0.41/ 0.41 / 0.41 / failed
    
=====> No Tech, Gauss
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
from sklearn import preprocessing


class Model(object):
    def __init__(self, dataset_original, list_idx, epoch, batch_size, sliding, learning_rate, positive_number, stimulation_level, alpha=0.6, beta=0.2):
        self.dataset_original = dataset_original
        self.train_idx = list_idx[0]
        self.test_idx = list_idx[1]
        self.epoch = epoch
        self.batch_size = batch_size
        self.sliding = sliding
        self.learning_rate = learning_rate
        self.positive_number = positive_number
        self.stimulation_level = stimulation_level
        self.alpha = alpha
        self.beta = beta
        self.min_max_scaler = preprocessing.MinMaxScaler()
        self.standard_scaler = preprocessing.StandardScaler()
        
    def preprocessing_data(self):
        train_idx, test_idx, dataset_original, sliding = self.train_idx, self.test_idx, self.dataset_original, self.sliding
        
        ## Get original dataset
        dataset_split = dataset_original[:test_idx + sliding]
        
        training_set = dataset_original[0:train_idx+sliding]
        testing_set = dataset_original[train_idx+sliding:test_idx+sliding]
        
        training_set_transform = self.min_max_scaler.fit_transform(training_set)
        testing_set_transform = self.min_max_scaler.transform(testing_set)
        
        dataset_transform = np.concatenate( (training_set_transform, testing_set_transform), axis=0 )
        ## Handle data with sliding
        dataset_sliding = dataset_transform[:len(dataset_transform)-sliding]
        for i in range(sliding-1):
            dddd = np.array(dataset_transform[i+1: len(dataset_transform)-sliding+i+1])
            dataset_sliding = np.concatenate((dataset_sliding, dddd), axis=1)
            
        ## Split data to set train and set test
        self.X_train, self.y_train = dataset_sliding[0:train_idx], dataset_sliding[sliding:train_idx+sliding, 0:1]
        self.X_test = dataset_sliding[train_idx:test_idx-sliding]
        self.y_test = dataset_split[train_idx+sliding:test_idx]
        
        print("Processing data done!!!")
    
    
    def encoder_features(self):
        train_X, train_y = copy.deepcopy(self.X_train), copy.deepcopy(self.y_train)
        stimulation_level, positive_number, alpha, beta = self.stimulation_level, self.positive_number, self.alpha, self.beta 
        ### Qua trinh train va dong thoi tao cac hidden unit (Pha 1 - cluster data)
        # 2. Khoi tao hidden thu 1
        temp = randint(0, len(train_X)-1)
        hu1 = [0, copy.deepcopy(train_X[temp]), copy.deepcopy(train_y[temp])]   # hidden unit 1 (t1, wH)
        list_hu = [copy.deepcopy(hu1)]         # list hidden units
        weight_hji = copy.deepcopy(hu1[1]).reshape(1, hu1[1].shape[0])     
        weight_hjk = copy.deepcopy(hu1[2]).reshape(1, hu1[2].shape[0])
        #    training_detail_file_name = full_path + 'SL=' + str(stimulation_level) + '_Slid=' + str(sliding) + '_Epoch=' + str(epoch) + '_BS=' + str(batch_size) + '_LR=' + str(learning_rate) + '_PN=' + str(positive_number) + '_CreateHU.txt'
        m = 0
        while m < len(train_X):
            list_dist_mj = []      # Danh sach cac dist(mj)
             # number of hidden units
            for j in range(0, len(list_hu)):                # j: la chi so cua hidden thu j
                dist_sum, dist_1, dist_2 = 0.0, 0.0, 0.0
                for i in range(0, len(train_X[0])):        # i: la chi so cua input unit thu i
                    dist_1 += pow(train_X[m][i] - weight_hji[j][i], 2.0)
                for i in range(0, len(train_y[0])):
                    dist_2 += pow(train_y[m][i] - weight_hjk[j][i], 2.0)
                dist_sum = alpha * sqrt(dist_1) + beta * sqrt(dist_2)
                list_dist_mj.append([j, dist_sum])
            list_dist_mj_sorted = sorted(list_dist_mj, key=itemgetter(1))        # Sap xep tu be den lon
            
            c = list_dist_mj_sorted[0][0]      # c: Chi so (index) cua hidden unit thu c ma dat khoang cach min
            distmc = list_dist_mj_sorted[0][1] # distmc: Gia tri khoang cach nho nhat
            if distmc < stimulation_level:
                list_hu[c][0] += 1                  # update hidden unit cth
                hic = exp(- (distmc * distmc) )
                
                neighbourhood_node = 1 + int( 0.10 * len(list_hu) )
                for i in range(0, neighbourhood_node ):
                    c_temp = list_dist_mj_sorted[i][0]
                    delta_1 = (positive_number * hic) * (train_X[m] - list_hu[c_temp][1])
                    delta_2 = (positive_number * hic) * (train_y[m] - list_hu[c_temp][2])
                    
                    list_hu[c_temp][1] += delta_1
                    list_hu[c_temp][2] += delta_2
                    weight_hji[c_temp] += delta_1
                    weight_hjk[c_temp] += delta_2
                # Tiep tuc vs cac example khac
                m += 1
                if m % 100 == 0:
                    print "distmc = {0}".format(distmc)
                    print "m = {0}".format(m)
            else:
                print "Failed !!!. distmc = {0}".format(distmc)
                list_hu.append([0, copy.deepcopy(train_X[m]), copy.deepcopy(train_y[m]) ])
                print "Hidden unit thu: {0} duoc tao ra.".format(len(list_hu))
                weight_hji = np.append(weight_hji, [copy.deepcopy(train_X[m])], axis = 0)
                weight_hjk = np.append(weight_hjk, [copy.deepcopy(train_y[m])], axis = 0)
                for hu in list_hu:
                    hu[0] = 0
                # then go to step 1
                m = 0
                ### +++
        ### +++ Get the last matrix weight 
        self.weight_hji, self.weight_hjk, self.list_hu = copy.deepcopy(weight_hji), copy.deepcopy(weight_hjk), copy.deepcopy(list_hu)
        print("Encoder features done!!!")
    
    def transform_features(self):
        temp1 = []
        for m in range(0, len(self.X_train)):  
            Xhj = []
            for j in range(0, len(self.weight_hji)):     # (w11, w21) (w12, w22), (w13, w23)
                Vhj = Model.distance_func(self.weight_hji[j], self.X_train[m])
                Zhj = Model.distance_func(self.weight_hjk[j], self.y_train[m])
                Dhj = self.alpha * Vhj + self.beta * Zhj
                Xhj.append(np.tanh(Dhj))
            temp1.append(Xhj)
        self.S_train = np.array(temp1)
        print("Transform features done!!!")
    
    def build_bpnn_and_train(self):
        ## Build layer's sizes
        X_train, y_train = self.S_train, self.y_train

        X_size = X_train.shape[1]   
        h_size = len(self.list_hu)
        y_size = y_train.shape[1]
        ## Symbols
        X = tf.placeholder("float64", shape=[None, X_size])
        y = tf.placeholder("float64", shape=[None, y_size])
        
        W = tf.Variable(tf.random_normal([h_size, y_size], stddev=0.03, dtype=tf.float64), name="W")
        y_ = tf.nn.elu( tf.matmul(X, W) )
        # Backward propagation
        cost    = tf.reduce_mean( tf.square(y_ - y) )
        updates = tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(cost)
        
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
                    batch_x, batch_y = Model.get_batch_data_next(X_train, y_train, i, batch_size)
                    if len(batch_x) == 0:
                        break
                    sess.run(updates, feed_dict={X: batch_x, y: batch_y})
                
                loss_epoch, weight = sess.run(cost, feed_dict={X: X_train, y: y_train}), sess.run([W])
                loss_plot.append(loss_epoch)
                print("Epoch: {}".format(epoch + 1), "cost = {}".format(loss_epoch))
                
        self.weight_ok, self.loss_train = weight, loss_plot
        print("Build model and train done!!!")
        
    def predict_y(self, x, y_prev):
        Xhj = []
        for j in range(0, len(self.list_hu)):
            Vhj = Model.distance_func(self.weight_hji[j], x)
            Zhj = Model.distance_func(self.weight_hjk[j], y_prev)
            Dhj = self.alpha * Vhj + self.beta * Zhj
            Xhj.append(np.tanh(Dhj))
        y_temp = np.dot(np.array(Xhj), self.weight_ok ) 
        y = Model.elu_activation(y_temp)
        return y
    
    def predict(self):
        # Evaluate models on the test set
        y_prev = self.y_train[-1]
        y_pred = [ y_prev ]
        
        for i in range(0, len(self.X_test)):
            x = self.X_test[i]
            y_prev = y_pred[-1]
            
            y = self.predict_y(x, y_prev)
            y_pred.append(y)
        
        y_pred = np.array([copy.deepcopy(y_pred[1:])])
        y_pred = np.reshape( y_pred, (-1, 1))
        
        y_test_inverse = self.y_test
        y_pred_inverse = self.min_max_scaler.inverse_transform(y_pred)
        
        testScoreRMSE = sqrt(mean_squared_error(y_test_inverse, y_pred_inverse))
        testScoreMAE = mean_absolute_error(y_test_inverse, y_pred_inverse)
        
        self.y_predict, self.score_test_RMSE, self.score_test_MAE = y_pred, testScoreRMSE, testScoreMAE
        self.y_test_inverse, self.y_pred_inverse = y_test_inverse, y_pred_inverse
        
        print('DONE - RMSE: %.5f, MAE: %.5f' % (testScoreRMSE, testScoreMAE))
        

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
    
    def draw_data_train(self):
        plt.figure(3)
        plt.plot(self.X_train[:, 0], self.X_train[:, 1], 'ro')
        plt.title('Train Dataset')
        plt.ylabel('Real value')
        plt.xlabel('Real value')
        
    def draw_data_test(self):
        plt.figure(4)
        plt.plot(self.X_test[:, 0], self.X_test[:, 1], 'ro')
        plt.title('Test Dataset')
        plt.ylabel('Real value')
        plt.xlabel('Real value')
        
        
    def fit(self):
        self.preprocessing_data()
        self.encoder_features()
        self.transform_features()
        self.build_bpnn_and_train()
        self.predict()
        self.draw_loss()
        self.draw_predict()
        
    @staticmethod
    def distance_func(a, b):
        return distance.euclidean(a, b)
    
    @staticmethod
    def elu_activation(x):
        if x > 0:
            return exp(x) - 1
        return x
    
    @staticmethod
    def sigmoid_activation(x):
        return 1.0 / (1.0 + exp(-x))
    
    @staticmethod
    def relu_activation(x):
        return max(x, 0)
    
    @staticmethod
    def get_random_input_vector(train_X, train_y):
        temp = randint(0, len(train_X)-1)
        return copy.deepcopy(train_X[temp]), copy.deepcopy(train_y[temp])
    
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

stimulation_level = [0.30]  #[0.10, 0.2, 0.25, 0.50, 1.0, 1.5, 2.0]  # [0.20]    
positive_numbers = [0.25] #[0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.20]     # [0.1]   
learning_rates = [0.5] #[0.005, 0.01, 0.025, 0.05, 0.10, 0.12, 0.15]   # [0.2]    
hyper_params = [ (0.85, 0.15) ]
sliding_windows = [2] #[ 2, 3, 5]           # [3]  
epochs = [3800] #[100, 250, 500, 1000, 1500, 2000]   # [500]                       
batch_sizes = [32] #[8, 16, 32, 64, 128]      # [16]     
list_num = [(2800, 4170)]


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
                            for hyper_param in hyper_params:
                                
                                febpnn = Model(dataset_original, list_idx, epoch, batch_size, sliding, learning_rate, positive_number, sti_level, hyper_param[0], hyper_param[1])
                                febpnn.fit()
                            
                                so_vong_lap += 1
                                if so_vong_lap % 5000 == 0:
                                    print "Vong lap thu : {0}".format(so_vong_lap)

print "Processing DONE !!!"
    
    