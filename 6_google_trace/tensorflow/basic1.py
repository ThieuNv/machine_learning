#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 07:40:03 2018

@author: thieunv
"""
from __future__ import print_function
import tensorflow as tf

# Core : 
# Building the computational graph.
# Running the computational graph.

# A computational graph is a series of TensorFlow operations arranged into a graph of nodes.
# Simple graph : Each node takes zero or more tensors as inputs and produces a tensor as an output. 
# One type of node is a constant. Like all TensorFlow constants, it takes no inputs, and it outputs a value it stores internally. 
#

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)

# To actually evaluate the nodes, we must run the computational graph within a session. 
# A session encapsulates the control and state of the TensorFlow runtime.
# The following code creates a Session object and then invokes its run method 
#    to run enough of the computational graph to evaluate node1 and node2
sess = tf.Session()
print(sess.run([node1, node2]))

# We can build more complicated computations by combining Tensor nodes with operations (Operations are also nodes). 
# For example, we can add our two constant nodes and produce a new graph as follows

node3 = tf.add(node1, node2)
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))

# TensorBoard : utility, display a picture of the computational graph. Output always constant.
# Parameter of graph is placeholders. This is like function or lambda which define 2 input para (a, b) 
#   and then operation on them. 
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)

# We can evaluate this graph with multiple inputs by using the 
#   feed_dict argument to the run method to feed concrete values to the placeholders
print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

# Make graph more complex.
add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b: 4.5}))

### Using variable to make model trainable
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + b

### Constant always initilize when call: tf.constant
### Variable is different, it will init when call: 
init = tf.global_variables_initializer()
sess.run(init)

### We need to init all global variable before run
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

### Init y : desired value
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

### We can change variable by : tf.assign
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

### tf.train API
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init)          # reset values to incorrect defaults.
for i in range(1000):
  sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

print(sess.run([W, b]))





