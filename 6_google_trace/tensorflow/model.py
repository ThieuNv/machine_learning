#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 09:36:01 2018

@author: thieunv

Sigmoid python nhanh hon dung libs: https://stackoverflow.com/questions/3985619/how-to-calculate-a-logistic-sigmoid-function-in-python

"""

import numpy as np
from scipy.spatial import distance
from math import exp, sqrt
import copy
from random import randint
from operator import itemgetter
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


def distance_func(a, b):
    return distance.euclidean(a, b)
def sigmoid_activation(x):
    return 1.0 / (1.0 + exp(-x))
def get_random_input_vector(train_X):
    return copy.deepcopy(train_X[randint(0, len(train_X)-1)])

def my_min_max_scaler(data):
    minx = min(data)
    maxx = max(data)
    return (np.array(data).astype(np.float64) - minx) / (maxx - minx)

def my_invert_min_max_scaler(data, minx, maxx):
    return np.array(data).astype(np.float64) * (maxx-minx) + minx

def get_batch_data_next(trainX, trainY, index, batch_size):
    real_index = index*batch_size
    if (len(trainX) % batch_size != 0 and real_index == len(trainX)):
        return (trainX[real_index:], trainY[real_index:])
    elif (real_index == len(trainX)):
        return ([], [])
    else:
        return (trainX[real_index: (real_index+batch_size)], trainY[real_index: (real_index+batch_size)])

class SONIAModel(object):
    def __init__(self, X_train, y_train, X_test, y_test, epoch, batch_size, validation, sliding, learning_rate, positive_number, stimulation_level, distance_level, threshold_number):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.epoch = epoch
        self.batch_size = batch_size
        self.validation = validation
        self.sliding = sliding
        self.learning_rate = learning_rate
        self.positive_number = positive_number
        self.stimulation_level = stimulation_level
        self.distance_level = distance_level
        self.threshold_number = threshold_number
    
    def fit(self):
        self.construct_hidden_layer_paper_2()
        self.backpropagation()
        self.predict()
    
    def predict(self):
        self.predictions = []
         ### Predict test dataset
        for k in range(len(self.X_test)):
            pre_output_hl = []
            for i in range(0, len(self.matrix_Wih)):     # (w11, w21) (w12, w22), (w13, w23)
                pre_output_hl.append(np.tanh(sqrt(distance_func(self.X_test[k], self.matrix_Wih[i]))))
            pre_y_output = np.dot(np.array(self.matrix_Who), np.array(pre_output_hl)) + 0 # bias
            self.predictions.append(sigmoid_activation(pre_y_output))
        print("DONE - predict")
            
    def get_test_predict_inverse(self, min_value, max_value):
        self.predictions_inverse = my_invert_min_max_scaler(self.predictions, min_value, max_value)
        return self.predictions_inverse
    
    def get_loss(self):
        self.testScoreRMSE = sqrt(mean_squared_error(self.y_test, self.predictions_inverse))
        self.testScoreMAE = mean_absolute_error(self.y_test, self.predictions_inverse)
        return (testScoreRMSE, testScoreMAE)
    
    def backpropagation_2(self):      ## Training weights and bias based on backpropagation
        list_hu = copy.deepcopy(self.list_hu)
        epoch = copy.deepcopy(self.epoch)
        X_train = copy.deepcopy(self.X_train)
        batch_size = copy.deepcopy(self.batch_size)
        y_train = copy.deepcopy(self.y_train)
        matrix_Wih = copy.deepcopy(self.matrix_Wih)
        
        
        matrix_Who = tf.Variable(np.zeros(len(list_hu)), dtype=tf.float64)
        bias = tf.Variable(1, dtype=tf.float64)
     
        xHj = tf.placeholder(tf.float64)
        linear_model = matrix_Who * xHj + bias
        y = tf.placeholder(tf.float64)
        # loss
        loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
         # optimizer
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        train = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init) # reset values to wrong
        # training loop
        for t in range(epoch):
            num_loop = int(len(X_train) / batch_size)
            for ind in range(num_loop): 
                 ## Get next batch
                X_train_next, y_train_next = get_batch_data_next(X_train, y_train, ind, batch_size)
                if len(X_train_next) == 0:
                    break 
                for i in range(0, len(X_train_next)):  
                    output_hidden_layer = []
                    for j in range(0, len(matrix_Wih)):     # (w11, w21) (w12, w22), (w13, w23)
                        output_hidden_layer.append(np.tanh(sqrt(distance_func(matrix_Wih[j], X_train_next[i]))))
                    sess.run([matrix_Who, bias, loss, train], {xHj: output_hidden_layer, y: y_train_next})   
        ## Ending backpropagation
        self.matrix_Who = copy.deepcopy(matrix_Who)
        self.bias = copy.deepcopy(bias)
        self.loss = copy.deepcopy(loss)
        
        print("DONE - backpropagation")
    
    def backpropagation(self):      ## Training weights and bias based on backpropagation
        self.matrix_Who = tf.Variable(np.zeros(len(self.list_hu)), dtype=tf.float64)
        self.bias = tf.Variable(1, dtype=tf.float64)
     
        xHj = tf.placeholder(tf.float64)
        linear_model = self.matrix_Who * xHj + self.bias
        y = tf.placeholder(tf.float64)
        # loss
        loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
         # optimizer
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        train = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init) # reset values to wrong
        # training loop
        for t in range(self.epoch):
            num_loop = int(len(self.X_train) / self.batch_size)
            for ind in range(num_loop): 
                 ## Get next batch
                X_train_next, y_train_next = get_batch_data_next(self.X_train, self.y_train, ind, self.batch_size)
                if len(X_train_next) == 0:
                    break 
                for i in range(0, len(X_train_next)):  
                    output_hidden_layer = []
                    for j in range(0, len(self.matrix_Wih)):     # (w11, w21) (w12, w22), (w13, w23)
                        output_hidden_layer.append(np.tanh(sqrt(distance_func(self.matrix_Wih[j], X_train_next[i]))))
                    sess.run([self.matrix_Who, self.bias, loss, train], {xHj: output_hidden_layer, y: y_train_next})   
        ## Ending backpropagation
        print("DONE - backpropagation")
        
    def construct_hidden_layer_kmeans_cluster(self, cluster_number):
        kmeans = KMeans(n_clusters=cluster_number, random_state=0).fit(self.X_train)
        labelX = kmeans.predict(self.X_train).tolist()
        self.matrix_Wih = kmeans.cluster_centers_
        self.list_hu = []
        for i in range(len(self.matrix_Wih)):
            temp = labelX.count(i)
            self.list_hu.append([temp, self.matrix_Wih[i]])

    def construct_hidden_layer_dbscan(self, min_samples):
        # Compute DBSCAN
        db = DBSCAN(eps=0.025, min_samples=50, metric='euclidean').fit_predict(self.X_train)
        labels = db.labels_.tolist()
        # Number of clusters in labels, ignoring noise if present.
    #    clusters = [X[labels == i] for i in xrange(n_clusters_)]
    #    outliers = X[labels == -1]
        
        n_clusters_ = len(set(labels))
        matrix_Wih = db.components_
        testtt = db.core_sample_indices_
        testttt = db.metric
        list_hu = []
        for i in range(n_clusters_):
            temp = labels.count(i)
            list_hu.append([temp, matrix_Wih[i]])
        return (matrix_Wih, list_hu)
    
    def construct_hidden_layer_paper_2(self):
        train_X = copy.deepcopy(self.X_train)
        stimulation_level = copy.deepcopy(self.stimulation_level)
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
    #            if m % 100 == 0:
    #                with open(training_detail_file_name, 'a') as f:
    #                    print >> f, 'distmc = :', distmc  
    #                    print >> f, 'Example thu :', m  
    #                print "distmc = {0}".format(distmc)
    #                print "m = {0}".format(m)
            else:
                ## +++ Get the first matrix weight hasn't been customize
                matrix_Wih = copy.deepcopy(trace_back_list_matrix_Wih[0])
                list_hu = copy.deepcopy(trace_back_list_hu[0])
                ## +++ Del all trace back matrix weight except the first one
                del trace_back_list_matrix_Wih[1:]
                del trace_back_list_hu[1:]
    #            with open(training_detail_file_name, 'a') as f:
    #                print >> f, 'Failed !!!. distmc = ', distmc  
    #            print "Failed !!!. distmc = {0}".format(distmc)
                    
                list_hu.append([0, copy.deepcopy(train_X[m]) ])
                
    #            with open(training_detail_file_name, 'a') as f:
    #                print >> f, 'Hidden unit thu:', len(list_hu), ' duoc tao ra.'
    #            print "Hidden unit thu: {0} duoc tao ra.".format(len(list_hu))
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
        self.list_hu = copy.deepcopy(trace_back_list_hu[-1])
        ### +++ Delete trace back
        del trace_back_list_matrix_Wih
        del trace_back_list_hu
    
    
    def construct_hidden_layer_paper(self):
        ### Qua trinh train va dong thoi tao cac hidden unit (Pha 1 - cluster data)
        # 2. Khoi tao hidden thu 1
        vector1 = get_random_input_vector(self.X_train)
        self.list_hu = [ [0, copy.deepcopy(vector1)] ]         # list hidden units , # hidden unit 1 (t1, wH)
        self.matrix_Wih = np.array(copy.deepcopy(np.reshape(vector1, (1, len(vector1)))))     # Mang 2 chieu 
        ### +++ Technical use to trace back matrix weight
        trace_back_list_matrix_Wih = [copy.deepcopy(self.matrix_Wih)]
        trace_back_list_hu = [copy.deepcopy(self.list_hu)]
    
        m = 0
        while m < len(self.X_train):
            list_dist_mj = []      # Danh sach cac dist(mj)
            for j in range(0, len(self.list_hu)):              # number of hidden units   # j: la chi so cua hidden thu j
                list_dist_mj.append([j, sqrt(distance_func(self.X_train[m], self.matrix_Wih[j]))])
            list_dist_mj = sorted(list_dist_mj, key=itemgetter(1))        # Sap xep tu be den lon
            
            c = list_dist_mj[0][0]      # c: Chi so (index) cua hidden unit thu c ma dat khoang cach min
            distmc = list_dist_mj[0][1] # distmc: Gia tri khoang cach nho nhat
            if distmc < self.stimulation_level:
                self.list_hu[c][0] += 1                  # update hidden unit cth
                # Just update vector W(c)
                self.list_hu[c][1] += self.positive_number * distmc
                self.matrix_Wih[c] += self.positive_number * distmc
                ## +++ Save the matrix_wih 
                trace_back_list_matrix_Wih.append(copy.deepcopy(self.matrix_Wih))
                trace_back_list_hu.append(copy.deepcopy(self.list_hu))
                # Tiep tuc vs cac example khac
                m += 1
            else:
                ## +++ Get the first matrix weight hasn't been customize
                self.matrix_Wih = copy.deepcopy(trace_back_list_matrix_Wih[0])
                self.list_hu = copy.deepcopy(trace_back_list_hu[0])
                ## +++ Del all trace back matrix weight except the first one
                del trace_back_list_matrix_Wih[1:]
                del trace_back_list_hu[1:]
        
                self.list_hu.append([0, copy.deepcopy(self.X_train[m]) ])
                self.matrix_Wih = np.insert(self.matrix_Wih, len(self.matrix_Wih), copy.deepcopy(self.X_train[m]), axis=0)
                for hu in self.list_hu:
                    hu[0] = 0
                # then go to step 1
                m = 0
                ### +++
                trace_back_list_matrix_Wih[0] = copy.deepcopy(self.matrix_Wih)
                trace_back_list_hu[0] = copy.deepcopy(self.list_hu)    
                
        ### +++ Get the last matrix weight 
        self.matrix_Wih = copy.deepcopy(trace_back_list_matrix_Wih[-1])
        self.list_hu = copy.deepcopy(trace_back_list_hu[-1])
        ### +++ Delete trace back
        del trace_back_list_matrix_Wih
        del trace_back_list_hu
        print("DONE - Construct hidden layer!!!")

## Load data frame
#full_path_name="/mnt/volume/ggcluster/spark-2.1.1-bin-hadoop2.7/thieunv/machine_learning/6_google_trace/data/"
#full_path= "/mnt/volume/ggcluster/spark-2.1.1-bin-hadoop2.7/thieunv/machine_learning/6_google_trace/FLNN/results/notDecompose/data10minutes/univariate/cpu/"
file_name = "Fuzzy_data_sampling_617685_metric_10min_datetime_origin.csv"

full_path_name = "/home/thieunv/university/LabThayMinh/code/6_google_trace/data/"
full_path = "/home/thieunv/university/LabThayMinh/code/6_google_trace/tensorflow/testing/"
df = read_csv(full_path_name+ file_name, header=None, index_col=False, usecols=[0], engine='python')   
dataset_original = df.values

distance_level = 0.1  #[0.1, 0.15, 0.2]
threshold_number = 2 #[2, 3, 4]

stimulation_level = [0.15]#[0.10, 0.2, 0.25, 0.50, 1.0, 1.5, 2.0]  # [0.20]    
positive_numbers = [0.005] #[0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.20]     # [0.1]   

learning_rates = [0.05] #[0.005, 0.01, 0.025, 0.05, 0.10, 0.12, 0.15]   # [0.2]    
sliding_windows = [2] #[ 2, 3, 5]           # [3]  
     
epochs = [100, 250, 500, 1000, 1500, 2000]   # [500]                       
batch_sizes = [8, 16, 32, 64, 128]      # [16]     

length = dataset_original.shape[0]
num_features = dataset_original.shape[1]

train_size = 700       #2880                   
test_size = length - train_size
valid = 0.25        # Hien tai chua dung den tham so nay
epsilon = 0.00001   # Hien tai chua dung den tham so nay
list_num = [(3000, 3500)]


#list_num = [(500, 1000), (500, 1500), (500, 2000), (750, 1250), (750, 1750), (750, 2250),
#            (1500, 2000), (1500, 2500), (1500, 3000), (1500, 2000), (1500, 2500), (1500, 3000),
#            (2000, 2500), (2000, 3000), (2000, 3500), (2500, 3000), (2500, 3500), (2500, 4000)
#]

pl1 = 1         # Use to draw figure
#pl2 = 1000
so_vong_lap = 0

for u_num in list_num:
    for sliding in sliding_windows:
        ## Load data
        dataset_split = dataset_original[:u_num[1]+sliding]
        GoogleTrace_orin_unnormal = copy.deepcopy(dataset_split)    # keep orginal data to test
        # normalize the dataset
        min_GT = min(GoogleTrace_orin_unnormal[:u_num[0]])
        max_GT = max(GoogleTrace_orin_unnormal[:u_num[0]])
        ## Scaling min max
        dataset_scale = my_min_max_scaler(dataset_split)
        ## Handle data with sliding
        dataset_sliding = dataset_scale[:len(dataset_scale)-sliding]
        for i in range(sliding-1):
            dddd = np.array(dataset_scale[i+1: len(dataset_scale)-sliding+i+1])
            dataset_sliding = np.concatenate((dataset_sliding, dddd), axis=1)
        ## Split data to set train and set test
        trainX, trainY = dataset_sliding[0:u_num[0]], dataset_sliding[sliding:u_num[0]+sliding, 0:1]
        testX = dataset_sliding[u_num[0]:u_num[1]-sliding]
        testY = GoogleTrace_orin_unnormal[u_num[0]+sliding:u_num[1]]
        
        for sti_level in stimulation_level:
            for epoch in epochs:
                for batch_size in batch_sizes:
                    for learning_rate in learning_rates:
                        for positive_number in positive_numbers:
                            sonia = SONIAModel(trainX, trainY, testX, testY, epoch, batch_size, valid, sliding, learning_rate, positive_number, sti_level, distance_level, threshold_number)
                            sonia.fit()
    #                        print "bias: {0}".format(bias)
    #                        print "Weight input and hidden: "
    #                        print matrix_Wih
    #                        print "Weight hidden and output: "
    #                        print vector_Who
    #                        print "Predict " 
    #                        print predict
                        
                            # invert predictions        
                            testPredictInverse = sonia.get_test_predict_inverse(min_GT, max_GT)   
                            # calculate root mean squared error
                            testScoreRMSE, testScoreMAE = sonia.get_loss()
    #                        print('Test Score: %f RMSE' % (testScoreRMSE))
    #                        print('Test Score: %f MAE' % (testScoreMAE))
                            
                            # summarize history for point prediction
#                            if testScoreMAE < 0.4:
                            plt.figure(pl1)
                            plt.plot(testY)
                            plt.plot(testPredictInverse)
                            plt.title('model predict')
                            plt.ylabel('real value')
                            plt.xlabel('point')
                            plt.legend(['realY... Test Score MAE= ' + str(testScoreMAE) , 'predictY... Test Score RMSE= '+ str(testScoreRMSE)], loc='upper left')
                            pic1_file_name = full_path + 'Train=' +str(u_num[0]) + '_Test=' + str(u_num[1]) + '_SL=' + str(sti_level) + '_Slid=' + str(sliding) + '_Epoch=' + str(epoch) + '_BS=' + str(batch_size) + '_LR=' + str(learning_rate) + '_PN=' + str(positive_number) + '_PointPredict.png'
                            plt.savefig(pic1_file_name)
                            plt.close()
                            pl1 += 1
                    
    #                        plt.figure(pl2)
    #                        plt.plot(list_loss_AMSE)
    #                        plt.plot(list_loss_RMSE)
    #                        plt.ylabel('Real training loss')
    #                        plt.xlabel('Epoch:')
    #                        plt.legend(['Test Score MAE= ' + str(testScoreMAE) , 'Test Score RMSE= ' + str(testScoreRMSE) ], loc='upper left')
    #                        pic2_file_name = full_path + 'SL=' + str(sti_level) + '_Slid=' + str(sliding) + '_Epoch=' + str(epoch) + '_BS=' + str(batch_size) + '_LR=' + str(learning_rate) + '_PN=' + str(positive_number) + '_TrainingLoss.png'
    #                        plt.savefig(pic2_file_name)
    #                        plt.close()
    #                        pl2 += 1
                            
                            so_vong_lap += 1
                            if so_vong_lap % 5000 == 0:
                                print "Vong lap thu : {0}".format(so_vong_lap)

print "Processing DONE !!!"
    
    