#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 09:26:55 2017

@author: thieunv
"""

# https://machinelearningmastery.com/time-series-forecast-study-python-monthly-sales-french-champagne/

from pandas import Series
from sklearn.metrics import mean_squared_error
from math import sqrt

# load data
series = Series.from_csv('dataset.csv')

# prepare data
X = series.values                       # 105 value
X = X.astype('float32')             
train_size = int(len(X) * 0.50)         # train: 52, test: 53
train, test = X[0:train_size], X[train_size:]

# walk-forward validation
history = [x for x in train]            # list of history of training set
predictions = list()
for i in range(len(test)):
	# predict
	yhat = history[-1]
	predictions.append(yhat)
	# observation
	obs = test[i]
	history.append(obs)
	print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
# report performance
mse = mean_squared_error(test, predictions)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)




