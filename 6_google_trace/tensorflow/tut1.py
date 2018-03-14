#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 08:06:29 2018

@author: thieunv

http://stackabuse.com/tensorflow-neural-network-tutorial/
"""

import tensorflow as tf

tf.reset_default_graph()

# Define a placeholder
a = tf.placeholder("float", name='pholdA')  
print("a:", a)

# Define a variable 
b = tf.Variable(2.0, name='varB')  
print("b:", b)

# Define a constant
c = tf.constant([1., 2., 3., 4.], name='consC')  
print("c:", c)  

d = a * b + c  
print d  

# call the default graph
graph = tf.get_default_graph()

# print operations in the graph
for op in graph.get_operations():  
    print(op.name)
    
# Initialize variables
init = tf.global_variables_initializer()

# Run a session and calculate d
sess = tf.Session()  
sess.run(init)  
print(sess.run(d, feed_dict={a: [[0.5], [2], [3]]}))  
print(sess.run(b))
print(sess.run(a, feed_dict={a: [[0.5], [2], [3]]}))
print(sess.run(c))
sess.close()  