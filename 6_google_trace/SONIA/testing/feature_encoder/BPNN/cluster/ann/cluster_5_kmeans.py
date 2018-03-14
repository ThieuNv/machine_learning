#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 16:21:33 2018

@author: thieunv

Cluster --> KMEANS


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
from sklearn.cluster import KMeans

class Model(object):
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
        train_X = copy.deepcopy(self.X_train)

        kmeans = KMeans(n_clusters= 12, random_state=0).fit(train_X)
        labelX = kmeans.predict(train_X).tolist()
        matrix_Wih = kmeans.cluster_centers_
        
        list_hu = []
        for i in range(len(matrix_Wih)):
            temp = labelX.count(i)
            list_hu.append([temp, matrix_Wih[i]])
        
        self.matrix_Wih = copy.deepcopy(matrix_Wih)
        self.list_hu_1 = copy.deepcopy(list_hu)
        
        print("Encoder features done!!!")
        
    
    def transform_features(self):
        temp1 = []
        for i in range(0, len(self.X_train)):  
            Sih = []
            for j in range(0, len(self.matrix_Wih)):     # (w11, w21) (w12, w22), (w13, w23)
                Sih.append(np.tanh( Model.distance_func(self.matrix_Wih[j], self.X_train[i])))
            temp1.append(np.array(Sih))
        
        temp2 = []
        for i in range(0, len(self.X_test)):  
            Sih = []
            for j in range(0, len(self.matrix_Wih)):     # (w11, w21) (w12, w22), (w13, w23)
                Sih.append(np.tanh( Model.distance_func(self.matrix_Wih[j], self.X_test[i])))
            temp2.append(np.array(Sih))
            
        self.S_train = np.array(temp1)
        self.S_test = np.array(temp2)
        
        print("Transform features done!!!")
    
   
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
        
        
    def draw_center(self):
        plt.figure(5)
        plt.plot(self.matrix_Wih[:, 0], self.matrix_Wih[:, 1], 'ro')
        plt.title('Centers Cluter KMEANS')
        plt.ylabel('Real value')
        plt.xlabel('Real value')
        
    def fit(self):
        self.preprocessing_data()
        self.encoder_features()
        self.transform_features()
        self.draw_data_train()
        self.draw_data_test()
        self.draw_center()
        
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

stimulation_level = [0.25]  #[0.10, 0.2, 0.25, 0.50, 1.0, 1.5, 2.0]  # [0.20]    
positive_numbers = [0.01] #[0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.20]     # [0.1]   
learning_rates = [0.25] #[0.005, 0.01, 0.025, 0.05, 0.10, 0.12, 0.15]   # [0.2]    
sliding_windows = [2] #[ 2, 3, 5]           # [3]  
epochs = [2800] #[100, 250, 500, 1000, 1500, 2000]   # [500]                       
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

                            febpnn = Model(dataset_original, list_idx, epoch, batch_size, sliding, learning_rate, positive_number, sti_level)
                            febpnn.fit()
                            
                            so_vong_lap += 1
                            if so_vong_lap % 5000 == 0:
                                print "Vong lap thu : {0}".format(so_vong_lap)

print "Processing DONE !!!"
    
    