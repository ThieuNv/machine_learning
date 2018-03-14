#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 09:55:14 2018

@author: thieunv

Inheritance : https://www.youtube.com/watch?v=RSl87lqOXDE

## Super:  https://stackoverflow.com/questions/38963018/typeerror-super-takes-at-least-1-argument-0-given-error-is-specific-to-any

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


class Developer(Employee):
    raise_amount = 1.10

    def __init__(self, first, last, pay, prog_lang):
        # super().__init__(first, last, pay)        # Just in python 3
        Employee.__init__(self, first, last, pay)
        self.prog_lang = prog_lang



class Manager(Employee):
    raise_amount = 1.15

    def __init__(self, first, last, pay, employees = None):
        Employee.__init__(self, first, last, pay)
        if employees is None:
            self.employees = []
        else:
            self.employees = employees
    
    def add_emp(self, emp):
        if emp not in self.employees:
            self.employees.append(emp)
    
    def remove_emp(self, emp):
        if emp in self.employees:
            self.employees.remove(emp)
    
    def print_emps(self):
        for emp in self.employees:
            print("-->", emp.fullname())
        
        
        
        
# Instance
e1 = Employee("thieu", "nguyen", 5000)
e2 = Employee("tien", "pham", 3000)

dev1 = Developer('khang', 'truong', 1000, 'Python')
dev2 = Developer('Hung', 'duy', 2500, 'Java')

print(dev1.email)
print dev2.prog_lang

print dev1.pay
dev1.apply_raise()
print dev1.pay

# print(help(Developer))


man1 = Manager('Sue', 'Smith', 10000, [dev1])
print man1.email

man1.add_emp(dev2)
man1.print_emps()

# Built-in function
print isinstance(man1, Manager)
print isinstance(man1, Employee)
print isinstance(man1, Developer)

print issubclass(Manager, Manager)
print issubclass(Manager, Employee)
print issubclass(Manager, Developer)




