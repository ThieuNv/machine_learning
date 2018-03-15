#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 13:05:36 2018

@author: thieunv
"""

import tensorflow as tf

a = tf.placeholder(tf.float64)
b = tf.placeholder(tf.float64)

c = a * b

with tf.Session() as sess:
    a, b, output = sess.run([a, b, c], feed_dict={a:[3, 4, 5], b:[1,5,6]})
    
    print(output)
    print(a)
    print(b)