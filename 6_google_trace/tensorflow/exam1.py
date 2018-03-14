#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 12:46:46 2018

@author: thieunv
"""

class exam:
    def createName(self, name):
        self.name = name
    def displayName(self):
        return self.name
    def saying(self):
        print "Hello, {0}".format(self.name)

first = exam()
second = exam()

first.createName("ThieuNv")
second.createName("Hello")

first.saying()
second.saying()

