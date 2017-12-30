# Bai toan: Dua vao chieu cao de du doan can nang
#   f(x) = w1.x1 + w2.x2 + w3.x3 + w0
#       w1, w2, w3: const, w0: bias
#       y ~ f(x): quan he tuyen tinh --> Linear Regression --> Toi uu: {w1, w2, w3, w0}
#       y: Gia tri thuc te
#       f(x): Gia tri du doan (predict)

# CT:   w = inverse(A). b
#   Voi:    A = Tranpose(X) . X
#           b = Tranpose(X) . y
#           w = A^-1 . b
# Trong bai nay thi:    (weight) = w_1*(height) + w_0
# Ta can tim` w_1 va w_0

import numpy as np
import matplotlib.pyplot as plt

# height (cm)
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T

# weight (kg)
y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T

# Building Xbar
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)
print Xbar

# Calculating weights of the fitting line
A = np.dot(Xbar.T, Xbar)
print A
b = np.dot(Xbar.T, y)
print np.linalg.pinv(A)
w = np.dot(np.linalg.pinv(A), b)

print('w = ', w)

# Preparing the fitting line
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(145, 185, 2)
y0 = w_0 + w_1*x0

# Du doan (2 du~ lieu 155, 160 ta chua dua vao mo hinh train)
y1 = w_1*155 + w_0
y2 = w_1*160 + w_0

print( u'Predict weight of person with height 155 cm: %.2f (kg), real number: 52 (kg)'  %(y1) )
print( u'Predict weight of person with height 160 cm: %.2f (kg), real number: 56 (kg)'  %(y2) )


# Drawing the fitting line
plt.plot(X.T, y.T, 'ro')     # data
plt.plot(x0, y0)               # the fitting line
plt.axis([140, 190, 45, 75])
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()






