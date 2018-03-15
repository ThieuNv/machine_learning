#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 14:13:51 2018

@author: thieunv

*) GradientDescentOptimizer (MinMaxScaler - 2)

    elu / relu/ tanh/ sigmoid ===> 0.368/ 0.44 / 0.42 / failed
    
*) AdamOptimizer 
    
     elu / relu/ tanh/ sigmoid ===> failed/ failed / failed / failed
    
*) AdagradOptimizer   
    
     elu / relu/  tanh/  sigmoid ===> 0.48/ failed / 0.53 / 

*) AdadeltaOptimizer
    
     elu / relu/ tanh/ sigmoid ===> 0.53/ failed / 0.48 / failed
    
    
"""


import tensorflow as tf
import numpy as np
from scipy.spatial import distance
from math import sqrt
import copy
from random import randint

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pandas import read_csv
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing
from sklearn.cluster import KMeans


class Model(object):
    def __init__(self, dataset_original, list_idx, epoch, batch_size, sliding, learning_rate, number_cluster):
        self.dataset_original = dataset_original
        self.train_idx = list_idx[0]
        self.test_idx = list_idx[1]
        self.epoch = epoch
        self.batch_size = batch_size
        self.sliding = sliding
        self.learning_rate = learning_rate
        self.number_cluster = number_cluster
        self.min_max_scaler = preprocessing.MinMaxScaler()
        self.min_max_scaler2 = preprocessing.MinMaxScaler()
        self.standard_scaler = preprocessing.StandardScaler()
        
        
    def preprocessing_data(self):
        train_idx, test_idx, dataset_original, sliding = self.train_idx, self.test_idx, self.dataset_original, self.sliding
        ## Transform all dataset
        dataset_split = dataset_original[:test_idx + sliding]
        
        mem = self.min_max_scaler.fit_transform(np.reshape(dataset_split[:, 1], (-1, 2) ))
        cpu = self.min_max_scaler.fit_transform(np.reshape(dataset_split[:, 0], (-1, 2) ))
        dataset_transform =  np.concatenate((cpu, mem), axis=1)
        
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
        train_X = copy.deepcopy(self.X_train)

        kmeans = KMeans(n_clusters=self.number_cluster, random_state=0).fit(train_X)
        labelX = kmeans.predict(train_X).tolist()
        matrix_Wih = kmeans.cluster_centers_
        
        list_hu = []
        for i in range(len(matrix_Wih)):
            temp = labelX.count(i)
            list_hu.append([temp, matrix_Wih[i]])
        
        self.matrix_Wih = copy.deepcopy(matrix_Wih)
        self.list_hu_1 = copy.deepcopy(list_hu)
        
    
    def transform_features(self):
        temp1 = []
        for i in range(0, len(self.X_train)):  
            Sih = []
            for j in range(0, len(self.matrix_Wih)):     # (w11, w21) (w12, w22), (w13, w23)
#                Sih.append(np.tanh(distance_func(self.matrix_Wih[j], self.X_train[i])))
                Sih.append(np.tanh( Model.distance_func(self.matrix_Wih[j], self.X_train[i]) ))
            temp1.append(np.array(Sih))
        
        temp2 = []
        for i in range(0, len(self.X_test)):  
            Sih = []
            for j in range(0, len(self.matrix_Wih)):     # (w11, w21) (w12, w22), (w13, w23)
#                Sih.append(np.tanh(distance_func(self.matrix_Wih[j], self.X_train[i])))
                Sih.append(np.tanh( Model.distance_func(self.matrix_Wih[j], self.X_test[i])))
            temp2.append(np.array(Sih))
            
        self.S_train = np.array(temp1)
        self.S_test = np.array(temp2)
    
    
    def build_bpnn_and_train(self):
        ## Build layer's sizes
        X_train, y_train = self.S_train, self.y_train

        X_size = X_train.shape[1]   
        h_size = len(self.list_hu_1)
        y_size = y_train.shape[1]
        ## Symbols
        X = tf.placeholder("float64", shape=[None, X_size])
        y = tf.placeholder("float64", shape=[None, y_size])
        
        W = tf.Variable(tf.random_normal([h_size, y_size], stddev=0.03, dtype=tf.float64), name="W")
        b = tf.Variable(tf.random_normal([y_size], dtype=tf.float64), name="b")
      
        y_ = tf.nn.elu(tf.add(tf.matmul(X, W), b))
        # Backward propagation
        cost    = tf.reduce_mean( tf.square(y_ - y) )
        updates = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(cost)
        
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
                
                loss_epoch = sess.run(cost, feed_dict={X: X_train, y: y_train})
                weight, bias = sess.run([W, b])
                loss_plot.append(loss_epoch)
                print("Epoch: {}".format(epoch + 1), "cost = {}".format(loss_epoch))
                
        self.weight, self.bias, self.loss_train = weight, bias, loss_plot
        print("Build model and train done!!!")
        
        
    def predict(self):
        # Evaluate models on the test set
        X_test, y_test = self.S_test, self.y_test
        
        X_size = X_test.shape[1]   
        y_size = y_test.shape[1]
        
        X = tf.placeholder("float64", shape=[None, X_size], name='X')
        y = tf.placeholder("float64", shape=[None, y_size], name='y')
        
        W = tf.Variable(self.weight)
        b = tf.Variable(self.bias)
     
        y_ = tf.nn.elu(tf.add(tf.matmul(X, W), b))
        
        # Calculate the predicted outputs
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            y_est_np = sess.run(y_, feed_dict={X: X_test, y: y_test})
            
            y_test_inverse = y_test
            y_pred_inverse = self.min_max_scaler.inverse_transform(y_est_np)
            
            testScoreRMSE = sqrt(mean_squared_error(y_test_inverse, y_pred_inverse))
            testScoreMAE = mean_absolute_error(y_test_inverse, y_pred_inverse)
            
            self.y_predict, self.score_test_RMSE, self.score_test_MAE = y_est_np, testScoreRMSE, testScoreMAE
            self.y_test_inverse, self.y_pred_inverse = y_test_inverse, y_pred_inverse
            
            print('DONE - RMSE: %.5f, MAE: %.5f' % (testScoreRMSE, testScoreMAE))
        
        print("Predict done!!!")
        
    def draw_loss(self):
        plt.figure(1)
        plt.plot(range(self.epoch), self.loss_train, label="Loss on training per epoch")
        plt.xlabel('Iteration', fontsize=12)  
        plt.ylabel('Loss', fontsize=12)  
        
        
    def draw_predict(self):
        plt.figure(2, figsize=(8.0, 5.0))
        plt.plot(self.y_test_inverse)
        plt.plot(self.y_pred_inverse)
        plt.title('Model predict')
        plt.ylabel('Real value')
        plt.xlabel('Point')
        plt.legend(['realY... Test Score RMSE= ' + str(self.score_test_RMSE) , 'predictY... Test Score MAE= '+ str(self.score_test_MAE)], loc='upper right')
        pic1_file_name = full_path + 'minmax_Slid=' + str(sliding) + '_Epoch=' + str(epoch) + '_BS=' + str(batch_size) + '_LR=' + str(learning_rate)  + '_PointPredict.png'
        plt.savefig(pic1_file_name)
                        
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
    def get_random_input_vector(train_X):
        return copy.deepcopy(train_X[randint(0, len(train_X)-1)])
    
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
full_path = "/home/thieunv/university/LabThayMinh/code/6_google_trace/SONIA/testing/feature_encoder/BPNN/kmeans/ann/scale2_chuan/"
df = read_csv(full_path_name+ file_name, header=None, index_col=False, usecols=[0,1], engine='python')   
dataset_original = df.values

number_clusters = [7]  #[0.10, 0.2, 0.25, 0.50, 1.0, 1.5, 2.0]  # [0.20]    
learning_rates = [0.25] #[0.005, 0.01, 0.025, 0.05, 0.10, 0.12, 0.15]   # [0.2]    
sliding_windows = [2] #[ 2, 3, 5]           # [3]  
epochs = [1500] #[100, 250, 500, 1000, 1500, 2000]   # [500]                       
batch_sizes = [32] #[8, 16, 32, 64, 128]      # [16]     
list_num = [(2800, 4170)]


pl1 = 1         # Use to draw figure
#pl2 = 1000
so_vong_lap = 0

for list_idx in list_num:
    for sliding in sliding_windows:
        for epoch in epochs:
            for batch_size in batch_sizes:
                for learning_rate in learning_rates:
                    for number_cluster in number_clusters:

                        febpnn = Model(dataset_original, list_idx, epoch, batch_size, sliding, learning_rate, number_cluster)
                        febpnn.fit()
                        
                        so_vong_lap += 1
                        if so_vong_lap % 5000 == 0:
                            print "Vong lap thu : {0}".format(so_vong_lap)

print "Processing DONE !!!"
    
    