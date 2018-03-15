#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 12:55:49 2018

@author: thieunv
"""

import tensorflow as tf

node1 = tf.constant(3.0, tf.float64)
node2 = tf.constant(4.0)

print(node1, node2)

#sess = tf.Session()
#print(sess.run([node1, node2]))
#sess.close()


with tf.Session() as sess:
    output = sess.run([node1, node2])
    print(output)