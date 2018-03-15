#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 13:19:24 2018

@author: thieunv
1. init
2. Instance of class (object)
3. Class variable
"""

class Employee:
    # class variable
    num_of_emps = 0
    raise_amount = 1.04
    
    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.pay = pay
        self.email = first + "." + last + "@company.com"
        Employee.num_of_emps += 1
        
    def fullname(self):
        return "{} {}".format(self.first, self.last)
    
    def apply_raise(self):
        self.pay = int(self.pay * self.raise_amount)


# Instance
e1 = Employee("thieu", "nguyen", 5000)
e2 = Employee("tien", "pham", 3000)

print e1
print e1.email

## Access to method of instance : self --> refer to object
print(e1.fullname())
print(Employee.fullname(e1))

## Using class variable
print e1.__dict__
print Employee.__dict__

e1.raise_amount = 1.10

print e1.raise_amount
print Employee.raise_amount
print e2.raise_amount       # take from class

print Employee.num_of_emps






















