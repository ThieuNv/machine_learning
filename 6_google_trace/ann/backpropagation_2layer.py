#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 18:06:09 2017

@author: thieunv
"""


import numpy as np

### Step 1: Collect data
X_train = np.array([ [0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1] ])
y_train = np.array([ [0],
              [1],
              [1],
              [0] ])
### Step 2: build model
num_epochs = 600000

# init weights
syn1 = 2*np.random.random((3, 1)) - 1   # matrix(3, 1)

def nonlin(x, deriv = False):
    if(deriv == True):
        return x*(1-x)
    return 1 / (1 + np.exp(-x))

### Step 3: Train model
for j in range(num_epochs):
    # feed forward through layer 0, 1, 2
    k0 = X_train                        # matrix(4, 3)
    k1 = nonlin(np.dot(k0, syn1))       # matrix(4, 1)

    # how much did we miss the target value?
    k1_error = y_train - k1             # matrix(4, 1)
    
    if(j % 1000 == 0):
        print "Error: " + str( np.mean(np.abs(k1_error)) )
        
    # in what direction is the target value?
    k1_delta = k1_error * nonlin(k1, deriv = True)
    
    syn1 += k1.T.dot(k1_delta)

    
    
    



