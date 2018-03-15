#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:18:27 2018

@author: thieunv
- ANN using tensorflow, scale input data
"""

# Import the needed libraries
import numpy as np  
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn import preprocessing

class AnnModel(object):
    def __init__(self, dataset_original, list_idx,sliding):
        self.dataset_original = dataset_original
        self.train_idx = list_idx[0]
        self.test_idx = list_idx[1]
        self.sliding = sliding
        self.min_max_scaler = preprocessing.MinMaxScaler()
        self.standard_scaler = preprocessing.StandardScaler()
        
    def preprocessing_data(self):
        train_idx, test_idx, dataset_original, sliding = self.train_idx, self.test_idx, self.dataset_original, self.sliding
        
        ## Split data
        dataset_split = dataset_original[:test_idx + sliding]
        ## Handle data with sliding
        dataset_sliding = dataset_split[:len(dataset_split)-sliding]
        for i in range(sliding-1):
            dddd = np.array(dataset_split[i+1: len(dataset_split)-sliding+i+1])
            dataset_sliding = np.concatenate((dataset_sliding, dddd), axis=1)
        
        training_set = dataset_sliding[0:train_idx]
        testing_set = dataset_sliding[train_idx:test_idx]
        
        standard_training_set = self.min_max_scaler.fit_transform(training_set)
        standard_testing_set = self.min_max_scaler.transform(testing_set)
        
        
        ## Original data
#        self.X_train = X_train
#        self.y_train = y_train
#        self.X_test = X_test
#        self.y_test = y_test
        
        ## Scaling Min Max
#        self.X_train = self.min_max_scaler.fit_transform(X_train)
#        self.y_train = self.min_max_scaler.transform(y_train)
#        self.X_test = self.min_max_scaler.transform(X_test)
#        self.y_test = y_test
        
        ## Standard Data
#        self.X_train = self.standard_scaler.fit_transform(X_train)
#        self.X_train = self.standard_scaler.fit(X_train)
#        self.X_train = self.standard_scaler.transform(X_train)
#        self.y_train = self.standard_scaler.transform(y_train)
#        self.X_test = self.standard_scaler.transform(X_test)
#        self.y_test = y_test
        
## Split data to set train and set test
        self.X_train, self.y_train = standard_training_set[0:train_idx-sliding], standard_training_set[sliding:train_idx+sliding, 0:1]
#        X_test, y_test = testing_set[train_idx:test_idx-sliding], testing_set[train_idx+sliding:test_idx, 0:1]
        self.X_test, self.y_test = standard_testing_set[0:test_idx-train_idx-2*sliding], standard_testing_set[sliding:test_idx, 0:1]
        
        print("Processing data done!!!")
        
    def draw_dataset(self):
        plt.figure(0)
        plt.plot(range(len(self.X_train)), self.X_train, label="Data training")
        plt.xlabel('Iteration', fontsize=12)  
        plt.ylabel('Real value', fontsize=12)
    

    
    def fit(self):
        self.preprocessing_data()
        self.draw_dataset()
    

## Load data frame
full_path_name = "/home/thieunv/university/LabThayMinh/code/6_google_trace/data/"
full_path = "/home/thieunv/university/LabThayMinh/code/6_google_trace/tensorflow/testing/"
file_name = "Fuzzy_data_sampling_617685_metric_10min_datetime_origin.csv"
df = read_csv(full_path_name+ file_name, header=None, index_col=False, usecols=[0], engine='python')  
dataset_original = np.array(df.values, dtype=np.float64)

sliding_windows = [2] #[ 2, 3, 5]           # [3]  
list_idx = (4100, 4170)

so_vong_lap = 0
for sliding in sliding_windows:
    ann = AnnModel(dataset_original, list_idx, sliding)
    ann.fit()
    
    so_vong_lap += 1
    if so_vong_lap % 5000 == 0:
        print "Vong lap thu : {0}".format(so_vong_lap)

print "Processing DONE !!!"


