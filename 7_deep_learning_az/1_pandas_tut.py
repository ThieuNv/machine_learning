#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 23:02:23 2018

@author: thieunv

1. Pandas tut
"""

import numpy as np
import pandas as pd

filepath = "/home/thieunv/university/LabThayMinh/code/7_deep_learning_az/data/csv/"
filename = "PastHires.csv"

df = pd.read_csv(filepath + filename)
df.head()
