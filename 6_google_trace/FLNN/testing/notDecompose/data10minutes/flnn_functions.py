#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 20:07:39 2017

@author: thieunv

A Novel Neural Network Approach For Software Cost Estimation Using Functional Link Artificial Neural Network (FLANN)
"""

from math import sin, cos, pi, exp, tan, log
from random import randint
import numpy as np

def selfx(x):
    return x
def one(x):
    return randint(-9, 9)*1.0

def sinpix(x):
    return sin(pi*x)
def cospix(x):
    return cos(pi*x)
def sin2pix(x):
    return sin(2*pi*x)
def cos2pix(x):
    return cos(2*pi*x)
def get_list_orthogonals():
    return [selfx, sinpix, cospix, sin2pix, cos2pix]


def c2x2(x):
    return 2 * x**2- 1
def c4x3(x):
    return 4 * x**3 - 3*x
def c8x3(x):
    return 8 * x**3 - 8 * x**2 + 1
def get_list_chebyshev_polynomials():
    return [selfx, one, c2x2, c4x3, c8x3]


def l3x2(x):
    return (3 * x**2 - 1) / 2 
def l5x3(x):
    return (5 * x**3 - 3*x) / 2
def l35x4(x):
    (5 * x**4 - 8*x**2 + 16*x - 3) / 8
def get_list_legendre_polynomials():
    return [selfx, one, l3x2, l5x3, l35x4]


def p2(x):
    return x**2
def p3(x):
    return x**3
def p4(x):
    return x**4
def get_list_power_series():
    return [selfx, one, p2, p3, p4]

def rp2(x):
    return randint(-29, 29) * x**2
def rp3(x):
    return randint(-19, 19) * x**3
def rp4(x):
    return randint(-9, 9) * x**4
def get_list_random_power_series():
    return [selfx, one, rp2, rp3, rp4]


def t2(x):
    return (cos(pi* x**2) + sin(pi * x) + x)
def t3(x):
    return tan(x)
def t4(x):
    return sin(pi*x)*cos(pi*x)
def get_list_t_orthogonals():
    return [selfx, one, t2, t3, t4]

def th2(x):
    return sin(pi* (cos(x**2 + log(x)))**(-1) )
def th3(x):
    return (sin(pi*x) + cos(pi*x))
def th4(x):
    return cos(pi*log(sin(pi*x)))
def get_list_th_orthogonals():
    return [selfx, one, th2, th3, th4]


def thi2(x):
    return sin(pi*x / 2) 
def thi3(x):
    return cos(pi*x / 2)
def thi4(x):
    return sin(pi*x / 2) * cos(pi*x / 2)
def thi5(x):
    return sin(pi*x / 2) + cos(pi*x / 2)
def get_list_thi_orthogonals():
    return [selfx, thi2, thi3, thi4, thi5]


def thie2(x):
    return sin(pi*x / 4) * sin(pi*x / 2) 
def thie3(x):
    return cos(pi*x / 4) * cos(pi*x / 2)
def thie4(x):
    return sin(pi*x / 4) + sin(pi*x / 2)
def thie5(x):
    return cos(pi*x / 4) + cos(pi*x / 2)
def get_list_thie_orthogonals():
    return [selfx, thie2, thie3, thie4, thie5]



def get_list_function(index):
    if index == 2:
        return get_list_chebyshev_polynomials()
    elif index == 3:
        return get_list_legendre_polynomials()
    elif index == 4:
        return get_list_power_series()
    elif index == 5:
        return get_list_random_power_series()
    elif index == 6:
        return get_list_t_orthogonals()
    elif index == 7:
        return get_list_th_orthogonals()
    elif index == 8:
        return get_list_thi_orthogonals()
    elif index == 9:
        return get_list_thie_orthogonals()
    else:
        return get_list_orthogonals()


### Helper functions
def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))
def relu(x):
    return max(0, x)
def self_activation(x):
    return x
def get_activation_func(index):
    if index == 1:
        return sigmoid
    elif index == 2:
        return relu
    else:
        return self_activation

def get_batch_data_next(trainX, trainY, index, batch_size):
    real_index = index*batch_size
    if (len(trainX) % batch_size != 0 and real_index == len(trainX)):
        return (trainX[real_index:], trainY[real_index:])
    elif (real_index == len(trainX)):
        return ([], [])
    else:
        return (trainX[real_index: (real_index+batch_size)], trainY[real_index: (real_index+batch_size)])


def my_min_max_scaler(data):
    minx = min(data)
    maxx = max(data)
    return (np.array(data).astype(np.float64) - minx) / (maxx - minx)

def my_invert_min_max_scaler(data, minx, maxx):
    return np.array(data).astype(np.float64) * (maxx-minx) + minx


