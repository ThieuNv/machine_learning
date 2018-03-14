#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 09:26:25 2018

@author: thieunv
"""
import pandas as pd
import numpy as np

#import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential()
model.add(Dense(20, input_dim=1, kernel_initializer='lecun_uniform'))
model.add(Activation('relu'))
model.add(Dense(1, kernel_initializer='lecun_uniform'))
model.add(Activation('relu'))

model.compile(optimizer='rmsprop', loss='mse')

df_train = pd.read_csv('Train_CPU.csv', header=None, names=['x', 'y1', 'y2'])
df1_train = df_train[np.isfinite(df_train['y1'])]
df2_train = df_train[np.isfinite(df_train['y2'])]

x_train = df1_train['x'].values.reshape(len(df1_train), 1)
y_train = df1_train['y1'].values.reshape(len(df1_train), 1)

df_test = pd.read_csv('Test_CPU.csv', header=None, names=['x', 'y1', 'y2'])
df1_test = df_test[np.isfinite(df_test['y1'])]
df2_test = df_test[np.isfinite(df_test['y2'])]
x_test = df1_test['x'].values.reshape(len(df1_test),1)
y_test = df1_test['y1'].values.reshape(len(df1_test),1)

epochs = 50

for i in range(epochs):
    hist = model.fit(x_train, y_train, batch_size=32, verbose=2)
    score = model.evaluate(x_test, y_test, batch_size=256)
    print hist.history['loss'][0], score

y_predict = model.predict(x_test, batch_size=256)
print len(y_predict)

import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(np.arange(len(x_test)), y_predict)
plt.show()