# -*- coding: utf-8 -*-
"""
Migrate MATLIB code to Python 3.5, NumPy, SciPy code
Created on Tue Mar 15 23:02:18 2016

@author: 
"""
import numpy as np

import matplotlib as mplot
from matplotlib import pyplot as plt

#User functions
#Cost Function
def compute_cost(x, y, theta):
    m = len(y)
    hypothesis = np.dot(x, theta)
    loss = hypothesis - y
    cost = np.sum(loss ** 2) / (2 * m)
    return cost
            
#Gradient descent
def gradientDescent(x, y, theta, alpha, m, numIterations):
    xTrans = x.transpose()
    j_history = np.zeros(numIterations, dtype=float)
    
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        cost = np.sum(loss ** 2) / (2 * m)
        j_history[i] = cost
        gradient = np.dot(xTrans, loss) / m
        theta = theta - alpha * gradient
    return theta, j_history

## Main Program
#Load data and plot data
vdata = np.genfromtxt('ex1data1.txt',delimiter=',')
m = len(vdata)
x = vdata[:,0]
y = vdata[:,1]

plt.scatter(x,y)

#Gradient Descent
#Add x0 column
v = np.ones((m, 1))
X = np.c_[v, x]

#Some gradient descent settings
init_theta = np.zeros(2,dtype=float)
iterations = 1500
alpha = 0.01

init_cost = compute_cost(X, y, init_theta)
#Define Cost function for gradient descent

#Perform Gradient Descent
(xtheta, cost_history) = gradientDescent(X, y, init_theta, alpha, m, iterations)


#Plot Prediction against X
yPredict = np.dot(X, xtheta)

plt.plot(x,yPredict,'r-')

