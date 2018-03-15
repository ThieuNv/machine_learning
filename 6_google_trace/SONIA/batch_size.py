#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 08:13:34 2018

@author: thieunv
"""

from pandas import read_csv
import numpy as np

def get_batch_data_next(trainX, trainY, index, batch_size):
    real_index = index*batch_size
    if (len(trainX) % batch_size != 0 and index == (len(trainX)/batch_size +1) ):
        return (trainX[real_index:], trainY[real_index:])
    elif (real_index == len(trainX)):
        return ([], [])
    else:
        return (trainX[real_index: (real_index+batch_size)], trainY[real_index: (real_index+batch_size)])
    

file_name = "Fuzzy_data_sampling_617685_metric_10min_datetime_origin.csv"

full_path_name = "/home/thieunv/university/LabThayMinh/code/6_google_trace/data/"
full_path = "/home/thieunv/university/LabThayMinh/code/6_google_trace/SONIA/results/notDecompose/data10minutes/univariate/cpu/"
df = read_csv(full_path_name+ file_name, header=None, index_col=False, usecols=[0], engine='python')   
dataset_original = df.values


## Load data
dataset = []
for val in dataset_original:
    dataset.append(val)
dataset = (np.asarray(dataset)).astype(np.float64)

trainX, trainY = dataset[0:100], dataset[:100, 0:1]

num_loop = int(len(trainX) / 50) + 1
for ind in range(num_loop):
            
    ## Get next batch
    X_train_next, y_train_next = get_batch_data_next(trainX, trainY, ind, 50)
    if len(X_train_next) == 0:
        break
    