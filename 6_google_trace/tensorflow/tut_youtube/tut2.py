#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 13:01:41 2018

@author: thieunv
"""

import tensorflow as tf

a = tf.constant(3.0, tf.float64)
b = tf.constant(4.0, tf.float64)

c = a * b

with tf.Session() as sess:
    a, b, output = sess.run([a, b, c])
    
    print(output)
    print(a)
    print(b)