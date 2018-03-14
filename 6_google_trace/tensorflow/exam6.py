#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 10:32:04 2018

@author: thieunv

Getter and Setter method: https://www.youtube.com/watch?v=jCzT9XFZ5bw
    
If we want use decorator in python 2. The class must inheritance from object
"""


class Employee(object):
    
    def __init__(self, first, last):
        self.first = first
        self.last = last
       
    @property       # Using attribute like a method, getter
    def email(self):
        return "{}.{}@gmail.com".format(self.first, self.last)
    
    @property
    def fullname(self):
        return "{} {}".format(self.first, self.last)
        
    @fullname.setter
    def fullname(self, name):
        first, last = name.split(" ")
        self.first = first
        self.last = last
    
    @fullname.deleter
    def fullname(self):
        print("Delete Name!")
        self.first = None
        self.last = None
    
# Instance
e1 = Employee("nguyen", "thieu")
e1.fullname = "truong khang"

print (e1.first)
print (e1.email)
print (e1.fullname)

del e1.fullname
print (e1.first)
print (e1.email)
print (e1.fullname)


















