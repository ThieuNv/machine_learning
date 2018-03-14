#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 10:17:15 2018

@author: thieunv

Magic method:   https://www.youtube.com/watch?v=3ohzBxoFHAY

"""


class Employee:
    # class variable
    raise_amount = 1.04
    
    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.pay = pay
        self.email = first + "." + last + "@company.com"
        
    def fullname(self):
        return "{} {}".format(self.first, self.last)
    def apply_raise(self):
        self.pay = int(self.pay * self.raise_amount)
        
    def __repr__(self):     # representation method
        return "Employee('{}', '{}', '{}')".format(self.first, self.last, self.pay)
        
    def __str__(self):      # string method, stronger than __repr__
        return "{} - {}".format(self.fullname(), self.email)
    
    def __add__(self, other):
        return self.pay + other.pay
        
    def __len__(self):
        return len(self.fullname())
        
        
        
# Instance
e1 = Employee("thieu", "nguyen", 5000)
e2 = Employee("tien", "pham", 3000)

print e1

print repr(e2)      # Same: print e2.__repr__()
print str(e2)       # Same: print e2.__str__()


## Add method
print(1 + 2)
print(int.__add__(1, 2))
print(str.__add__('a', 'b'))

print(e1 + e2)      # Understand how to add using __add__

## Len method
print(len('test'))
print('test'.__len__())

print(len(e1))




















