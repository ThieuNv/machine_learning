#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 13:32:22 2018

@author: thieunv
- ANN using tensorflow

- Trong code nay` su dung: Transform toan bo dataset sau do moi tach ra training set va testing set.
- Scaler: StandardScaler

- Optimizer: AdadeltaOptimizer
    
- Activation: Relu, Relu/ Tanh, Relu / Sigmoid, Relu/ Sigmoid, sigmoid/ ...  => MSE : Failed
    + Loss se giam nhanh theo tung epoch
    
- Activation: Elu, Elu ==> MSE: 0.39 (tot hon Relu,Relu)
    + Loss giam cham hon cua Relu, Elu

- Activation: Relu, Elu ==> MSE: 0.38 
    + Loss giam rat nhanh va hieu qua

- Activation: Tanh, Elu ==> MSE: 0.38 
    + Loss giam rat nhanh va hieu qua


- Optimizer: AdadeltaOptimizer (AdamOptimizer, AdagradOptimizer)



"""

# Import the needed libraries
import numpy as np  
from pandas import read_csv
from math import sqrt
import tensorflow as tf  
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing

class AnnModel(object):
    def __init__(self, dataset_original, list_idx, epoch, batch_size, sliding, learning_rate):
        self.dataset_original = dataset_original
        self.train_idx = list_idx[0]
        self.test_idx = list_idx[1]
        self.epoch = epoch
        self.batch_size = batch_size
        self.sliding = sliding
        self.learning_rate = learning_rate
        self.min_max_scaler = preprocessing.MinMaxScaler()
        self.standard_scaler = preprocessing.StandardScaler()
        
    def preprocessing_data(self):
        train_idx, test_idx, dataset_original, sliding = self.train_idx, self.test_idx, self.dataset_original, self.sliding
        ## Transform all dataset
        dataset_split = dataset_original[:test_idx + sliding]
        dataset_transform = self.standard_scaler.fit_transform(dataset_split)
        
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
        
    
    # Create and train a tensorflow model of a neural network
    def build_model_and_train(self):
        # Reset the graph
        tf.reset_default_graph()
        
        X_train, y_train = self.X_train, self.y_train
        
        X_size = X_train.shape[1]   
        h_size = 20
        y_size = y_train.shape[1]
        
        # Placeholders for input and output data
        X = tf.placeholder("float64", shape=[None, X_size], name='X')
        y = tf.placeholder("float64", shape=[None, y_size], name='y')
        
        # now declare the weights connecting the input to the hidden layer
        W1 = tf.Variable(tf.random_normal([X_size, h_size], stddev=0.03, dtype=tf.float64), name="W1")
        b1 = tf.Variable(tf.random_normal([h_size], dtype=tf.float64), name="b1")
        # and the weights connecting the hidden layer to the output layer
        W2 = tf.Variable(tf.random_normal([h_size, y_size], stddev=0.03, dtype=tf.float64), name='W2')
        b2 = tf.Variable(tf.random_normal([y_size], dtype=tf.float64), name='b2')
    
        # calculate the output of the hidden layer
        hidden_out = tf.nn.tanh( tf.add(tf.matmul(X, W1), b1) )
        # Forward propagation # now calculate the hidden layer output
        y_ = tf.nn.relu(tf.add(tf.matmul(hidden_out, W2), b2))
        
        # Loss function
        deltas = tf.square(y_ - y)
        loss = tf.reduce_mean(deltas)
        
        # Backward propagation
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
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
                batch_x, batch_y = AnnModel.get_batch_data_next(X_train, y_train, i, batch_size)
                if len(batch_x) == 0:
                    break
                sess.run(train, feed_dict={X: batch_x, y: batch_y})
            
            loss_epoch = sess.run(loss, feed_dict={X: X_train, y: y_train})
            weights1, bias1, weights2, bias2 = sess.run([W1, b1, W2, b2])
            loss_plot.append(loss_epoch)
            print("Epoch: {}".format(epoch + 1), "loss = {}".format(loss_epoch))
            
        sess.close()
        self.w1, self.b1, self.w2, self.b2, self.loss_train = weights1, bias1, weights2, bias2, loss_plot
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
        X_test, y_test = self.X_test, self.y_test
        
        X_size = X_test.shape[1]   
        y_size = y_test.shape[1]
        
        X = tf.placeholder("float64", shape=[None, X_size], name='X')
        y = tf.placeholder("float64", shape=[None, y_size], name='y')
        
        W1 = tf.Variable(self.w1)
        b1 = tf.Variable(self.b1)
        W2 = tf.Variable(self.w2)
        b2 = tf.Variable(self.b2)
        
        hidden_out = tf.nn.tanh( tf.add(tf.matmul(X, W1), b1) )
        y_ = tf.nn.relu(tf.add(tf.matmul(hidden_out, W2), b2))
        
        # Calculate the predicted outputs
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            y_est_np = sess.run(y_, feed_dict={X: X_test, y: y_test})

            y_test_inverse = y_test
            y_pred_inverse = self.standard_scaler.inverse_transform(y_est_np)
            
            testScoreRMSE = sqrt(mean_squared_error(y_test_inverse, y_pred_inverse))
            testScoreMAE = mean_absolute_error(y_test_inverse, y_pred_inverse)
            
            self.y_predict, self.score_test_RMSE, self.score_test_MAE = y_est_np, testScoreRMSE, testScoreMAE
            self.y_test_inverse, self.y_pred_inverse = y_test_inverse, y_pred_inverse
            
            print('DONE - RMSE: %.5f, MAE: %.5f' % (testScoreRMSE, testScoreMAE))
            print(list(y_est_np))
        
        print("Predict done!!!")
    
    
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

so_vong_lap = 0
for sliding in sliding_windows:
    for epoch in epochs:
        for batch_size in batch_sizes:
            for learning_rate in learning_rates:

                ann = AnnModel(dataset_original, list_idx, epoch, batch_size, sliding, learning_rate)
                ann.fit()
                
                so_vong_lap += 1
                if so_vong_lap % 5000 == 0:
                    print "Vong lap thu : {0}".format(so_vong_lap)

print "Processing DONE !!!"


