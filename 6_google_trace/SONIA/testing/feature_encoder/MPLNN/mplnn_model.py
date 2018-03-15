#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 11:11:07 2018

@author: thieunv

- MPLNN - Multiple Layer Neuron Network

"""


# Import the needed libraries
import numpy as np  
from pandas import read_csv
from scipy.spatial import distance
from math import exp, sqrt
from random import randint
import copy
import tensorflow as tf  
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


class MplnnModel(object):
    def __init__(self, dataset_original, list_idx, epoch, batch_size, sliding, learning_rate):
        self.dataset_original = dataset_original
        self.train_idx = list_idx[0]
        self.test_idx = list_idx[1]
        self.epoch = epoch
        self.batch_size = batch_size
        self.sliding = sliding
        self.learning_rate = learning_rate
        
    def preprocessing_data(self):
        train_idx, test_idx, dataset_original, sliding = self.train_idx, self.test_idx, self.dataset_original, self.sliding
        ## Split data
        dataset_split = dataset_original[:test_idx + sliding]
        GoogleTrace_orin_unnormal = copy.deepcopy(dataset_split)    # keep orginal data to test
        # normalize the dataset
        self.min_GT = min(GoogleTrace_orin_unnormal[:train_idx])
        self.max_GT = max(GoogleTrace_orin_unnormal[:train_idx])
        ## Scaling min max
        dataset_scale = MplnnModel.my_min_max_scaler(dataset_split)
        ## Handle data with sliding
        dataset_sliding = dataset_scale[:len(dataset_scale)-sliding]
        for i in range(sliding-1):
            dddd = np.array(dataset_scale[i+1: len(dataset_scale)-sliding+i+1])
            dataset_sliding = np.concatenate((dataset_sliding, dddd), axis=1)
        ## Split data to set train and set test
        self.X_train, self.y_train = dataset_sliding[0:train_idx], dataset_sliding[sliding:train_idx+sliding, 0:1]
        self.X_test, self.y_test = dataset_sliding[train_idx:test_idx-sliding], dataset_sliding[train_idx+sliding:test_idx, 0:1]
        #        self.y_test = GoogleTrace_orin_unnormal[train_idx+sliding:test_idx]
        print("Processing data done!!!")
    
    # Create and train a tensorflow model of a neural network
    def build_model_and_train(self):
        # Reset the graph
        tf.reset_default_graph()
        X_train, y_train = self.X_train, self.y_train
        
        X_size = X_train.shape[1]   
        y_size = y_train.shape[1]
        # Placeholders for input and output data
        X = tf.placeholder("float64", shape=[None, X_size], name='X')
        y = tf.placeholder("float64", shape=[None, y_size], name='y')
        
        # now declare the weights connecting the input to the hidden layer
        hidden_layer =[ [0, 0, X_size, 3*X_size], [0, 0, 3*X_size, 5*X_size], [0, 0, 5*X_size, 3*X_size], [0, 0, 3*X_size, y_size] ]
        
        i = 1
        for hl in hidden_layer:
            hl[0] = tf.Variable(tf.random_normal([hl[2], hl[3]], stddev=0.03, dtype=tf.float64), name="W" + str(i))
            hl[1] = tf.Variable(tf.random_normal([hl[3]], dtype=tf.float64), name="b" + str(i))
            i += 1
            
        # calculate the output of the hidden layer
        hidden_out = tf.nn.sigmoid( tf.add(tf.matmul(X, hidden_layer[0][0]), hidden_layer[0][1]) )
        for i in range(1, len(hidden_layer)):
            hidden_out = tf.nn.sigmoid( tf.add(tf.matmul(hidden_out, hidden_layer[i][0]), hidden_layer[i][1]) )
        # Forward propagation # now calculate the hidden layer output - in this case, let's use a softmax activated output layer
        y_ = hidden_out

        # Loss function
        deltas = tf.square(y_ - y)
        loss = tf.reduce_mean(deltas)
        
        # Backward propagation
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        train = optimizer.minimize(loss)
        
        # Initialize variables and run session
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        
        # Go through num_iters iterations
        loss_plot = []
        total_batch = int(len(X_train) / self.batch_size) + 1
        for epoch in range(self.epoch):
            for i in range(total_batch):
                ## Get next batch
                batch_x, batch_y = MplnnModel.get_batch_data_next(X_train, y_train, i, batch_size)
                if len(batch_x) == 0:
                    break
                sess.run(train, feed_dict={X: batch_x, y: batch_y})
            
            loss_epoch = sess.run(loss, feed_dict={X: X_train, y: y_train})
            
            hl_train = [[0, 0], [0, 0], [0, 0], [0, 0]]
            for i in range(len(hidden_layer)):
                hl_train[i][0], hl_train[i][1] = sess.run(hidden_layer[i][0]), sess.run(hidden_layer[i][1])
            loss_plot.append(loss_epoch)
            print("Epoch: {}".format(epoch + 1), "loss = {}".format(loss_epoch))
            
        sess.close()
        self.hidden_layer = copy.deepcopy(hl_train)
        self.loss_train = loss_plot
        print("Build model and train done!!!")
        
    def draw_loss(self):
        plt.figure(1)
        plt.plot(range(self.epoch), self.loss_train, label="Loss on training per epoch")
        plt.xlabel('Iteration', fontsize=12)  
        plt.ylabel('Loss', fontsize=12)  
        
        
    def draw_predict(self):
        plt.figure(2)
        plt.plot(MplnnModel.my_invert_min_max_scaler(self.y_test, self.min_GT, self.max_GT))
        plt.plot(MplnnModel.my_invert_min_max_scaler(self.y_predict, self.min_GT, self.max_GT))
        plt.title('Model predict')
        plt.ylabel('Real value')
        plt.xlabel('Point')
        plt.legend(['realY... Test Score RMSE= ' + str(self.score_test_RMSE) , 'predictY... Test Score MAE= '+ str(self.score_test_MAE)], loc='upper right')
    
    def predict(self):
        # Evaluate models on the test set
        X_test, y_test = self.X_test, self.y_test       
        X_size = X_test.shape[1]   
        y_size = y_test.shape[1]
        
        X = tf.placeholder("float64", shape=[None, X_size], name='X')
        y = tf.placeholder("float64", shape=[None, y_size], name='y')
            
        # calculate the output of the hidden layer
        hidden_out = tf.nn.sigmoid( tf.add(tf.matmul(X, self.hidden_layer[0][0]), self.hidden_layer[0][1]) )
        for i in range(1, len(self.hidden_layer)):
            hidden_out = tf.nn.sigmoid( tf.add(tf.matmul(hidden_out, self.hidden_layer[i][0]), self.hidden_layer[i][1]) )
        y_ = hidden_out
        # Calculate the predicted outputs
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            y_est_np = sess.run(y_, feed_dict={X: X_test, y: y_test})
            
            testScoreRMSE = sqrt(mean_squared_error(y_test, y_est_np))
            testScoreMAE = mean_absolute_error(y_test, y_est_np)
            
            self.y_predict = y_est_np
            self.score_test_RMSE = testScoreRMSE
            self.score_test_MAE = testScoreMAE
            print('DONE - RMSE: %.2f, MAE: %.2f' % (testScoreRMSE, testScoreMAE))
            print(list(y_est_np))
        
        print("Predict done!!!")
    
    @staticmethod
    def distance_func(a, b):
        return distance.euclidean(a, b)
    @staticmethod
    def sigmoid_activation(x):
        return 1.0 / (1.0 + exp(-x))
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


    def fit(self):
        self.preprocessing_data()
        self.build_model_and_train()
        self.predict()
        self.draw_loss()
        self.draw_predict()
    

## Load data frame
#full_path_name="/mnt/volume/ggcluster/spark-2.1.1-bin-hadoop2.7/thieunv/machine_learning/6_google_trace/data/"
#full_path= "/mnt/volume/ggcluster/spark-2.1.1-bin-hadoop2.7/thieunv/machine_learning/6_google_trace/FLNN/results/notDecompose/data10minutes/univariate/cpu/"
file_name = "Fuzzy_data_sampling_617685_metric_10min_datetime_origin.csv"

full_path_name = "/home/thieunv/university/LabThayMinh/code/6_google_trace/data/"
full_path = "/home/thieunv/university/LabThayMinh/code/6_google_trace/tensorflow/testing/"
df = read_csv(full_path_name+ file_name, header=None, index_col=False, usecols=[0], engine='python')   
dataset_original = df.values

learning_rates = [0.05]     #[0.005, 0.01, 0.025, 0.05, 0.10, 0.12, 0.15]   # [0.2]    
sliding_windows = [3] #[ 2, 3, 5]           # [3]  
epochs = [100] #[100, 250, 500, 1000, 1500, 2000]   # [500]                       
batch_sizes = [32] #[8, 16, 32, 64, 128]      # [16]     
list_idx = (3000, 3500)

so_vong_lap = 0
for sliding in sliding_windows:
    for epoch in epochs:
        for batch_size in batch_sizes:
            for learning_rate in learning_rates:

                ann = MplnnModel(dataset_original, list_idx, epoch, batch_size, sliding, learning_rate)
                ann.fit()
                
                so_vong_lap += 1
                if so_vong_lap % 5000 == 0:
                    print "Vong lap thu : {0}".format(so_vong_lap)

print "Processing DONE !!!"
