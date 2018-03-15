#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 13:51:27 2018

@author: thieunv

1. Class methods and static methods  (https://www.youtube.com/watch?v=rq8cL2XMM5M)
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
        
    @classmethod
    def set_raise_amount(cls, amount):      # class, amount
        cls.raise_amount = amount

    @classmethod
    def from_string(cls, emp_str):
        first, last, pay = emp_str.split('-')
        return cls(first, last, pay)

    @staticmethod
    def is_workday(day):
        if day.weekday() == 5 or day.weekday() == 6:
            return False
        return True

# Instance
e1 = Employee("thieu", "nguyen", 5000)
e2 = Employee("tien", "pham", 3000)

print e1.raise_amount
print e2.raise_amount

Employee.set_raise_amount(1.15)

print e1.raise_amount
print e2.raise_amount

## Extract constructor class method
e3_str = 'John-Doe-7000'
e4_str = 'Jane-Lee-3000'

e3 = Employee.from_string(e3_str)
e4 = Employee.from_string(e4_str)

print e3.fullname()
print e4.raise_amount

print Employee.num_of_emps

### Using static method
import datetime
my_date = datetime.date(2017, 1, 13)

print(Employee.is_workday(my_date))










