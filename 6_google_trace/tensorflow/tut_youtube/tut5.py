#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 15:21:48 2018

@author: thieunv
"""

import tensorflow as tf

# Model parameters
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)

# Inputs and Outputs
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

linear_model = W * x + b

# Loss
squared_delta = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_delta)


# Optimize
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# Run in a session
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    
    for i in range(1000):
        sess.run([train], feed_dict={x:[1, 2, 3, 4], y:[0, -1, -2, -3]})
        print(sess.run([loss], feed_dict={x:[1, 2, 3, 4], y:[0, -1, -2, -3]}))
        
    print(sess.run([W, b]))