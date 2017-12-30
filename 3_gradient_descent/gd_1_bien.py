#### BT: f(x) = x^2 + 5sin(x)

import math
import numpy as np
import matplotlib.pyplot as plt

def grad(x):
    """ Ham tinh dao ham cua f(x) = 2x + 5cos(x) """
    return 2*x + 5*np.cos(x)

def cost(x):
    """ Ham tinh gia tri cua f(x) """
    return x**2 + 5*np.sin(x)

def GD1(x0, eta = 0.1, epxilon = 1e-3):
    """ Ham thuc hien thuat toan GD
     eta: learning rate
     x0: thoi diem bat dau
     epxilon: Gia tri du nho de khi dao ham du nho
     """
    x = [x0]
    temp = 0
    for it in range(100):
        x_new = x[-1] - eta * grad(x[-1])
        temp += 1
        if abs( grad(x_new) ) < epxilon:
            break
        x.append(x_new)
    return (x, temp)

(x1, it1) = GD1(-5, 0.1)
(x2, it2) = GD1(5, 0.1)
print('Solution x1 = %f, cost = %f, obtained after %d iterations'%(x1[-1], cost(x1[-1]), it1))
print('Solution x2 = %f, cost = %f, obtained after %d iterations'%(x2[-1], cost(x2[-1]), it2))