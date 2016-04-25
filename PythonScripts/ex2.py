# -*- coding: utf-8 -*-
"""
Migrate MATLIB code to Python 3.5, NumPy, SciPy code
Created on Tue Mar 15 23:02:18 2016


Logistic Regression using Scipy


"""
#%% Import Libraries
import numpy as np
from numpy import loadtxt, where, exp, log, zeros
from matplotlib import pyplot as plt
from pylab import scatter, show, legend, xlabel, ylabel
from sklearn import linear_model
import scipy as sp
from scipy import special
from sklearn import linear_model

#%%Defined user functions
#Sigmoid function
def sigmoid(X):
    z = 1.0 /( 1.0 + exp(-1.0 * X))
    return z

# Cost and Gradient function
def costFunction(theta, x, y):
    m = len(y)
    J = 0
    hX = sigmoid(np.dot(x, theta))
    
    J = (1./m) * (-(y.T).dot(log(hX)) - (((1 - y).T).dot(log(1 - hX))))
    
    grad = (1./m) * (((hX - y).T).dot(x))  

    return J, grad
        
#%% Main Processing begins here
#load the dataset
data = loadtxt('ex2data1.txt', delimiter=',')

#Note Python slicing for X 0:2 includes only first 2 columns 
x = data[:, 0:2]
y = data[:, 2]
#Need to reshape y as a matrix for Cost Function Calculation
m = len(y)
#y = np.reshape(y,[m,1])

#Plot data
plt.figure(0)
pos = where(y == 1)
neg = where(y == 0)
plt.scatter(x[pos, 0], x[pos, 1], marker='o', c='b')
plt.scatter(x[neg, 0], x[neg, 1], marker='x', c='r')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(['Admitted', 'Not Admitted'])
plt.show()

#%%Cost functon calculation
#Set Initial parameters
#Dimensions of X
m, n = np.shape(x)
#Add x0 term to X
X = np.c_[np.ones((m, 1)), x]

#Initial theta - Note that theta has shape (n+1,) and NOT (n+1,1)
initial_theta = np.zeros(n + 1)

#Compute inital cost, grad for initial theta i.e. zero
#init_cost = cost_function(initial_theta, x, y)
#init_grad = grad_function(initial_theta, x, y)
init_cost, init_grad = costFunction(initial_theta, X, y)

print('Cost at initial theta (zeros):\n', init_cost)
print('Gradient at initial theta (zeros): \n', init_grad)


#%%Logistic Regression 
h = .02  # step size in the mesh

logreg = linear_model.LogisticRegression(C=1e5)
# Create an instance of Neighbours Classifier and fit the data.
# using liblinear as the solver
logreg.fit(x, y)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())


#Predictions using Fitted Log Reg
theta0 = logreg.fit(x,y).intercept_
theta1 = logreg.fit(x,y).coef_

#Predict for single scora set
v = np.array([1 ,45, 85],dtype=float)
w = np.append(theta0, theta1).ravel()
#sp.speciat.expit is the built in sigmoid function 
prob = sp.special.expit(v.dot(w.T))

#Compare prediction against whole data set. Strictly speaking, we
# should have cross validation and test data sets to compute accuracy

xscore = logreg.score(x,y)
print('Log reg score/ accuracy', xscore * 100)
