# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 01:36:05 2016

@author:
"""
#%% Import Libraries
import numpy as np
import scipy as sp
from numpy import log, exp, where
import matplotlib.pyplot as plt
from scipy import special
from sklearn import linear_model

#%%Defined user functions
#%%Sigmoid function
def sigmoid(X):
    z = 1.0 /( 1.0 + exp(-1.0 * X))
    return z

#%% Cost and Gradient function
def costFunction(theta, x, y):
    m = len(y)
    J = 0
    hX = sigmoid(np.dot(x, theta))
    
    J = (1./m) * (-(y.T).dot(log(hX)) - (((1 - y).T).dot(log(1 - hX))))
    grad = (1./m) * (((hX - y).T).dot(x))  

    return J, grad
    
#%% Main program
#load the dataset
data = np.loadtxt('ex2data1.txt', delimiter=',')
#Note Python slicing for X 0:2 includes only first 2 columns 
X = data[:, 0:2]
Y = data[:, 2]

#%% PLot data
#Plot data
plt.figure(0)
pos = where(Y == 1)
neg = where(Y == 0)
plt.scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
plt.scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(['Admitted', 'Not Admitted'])
plt.show()

#%% COmpute initial cost, gradient
#%%Cost functon calculation
#Set Initial parameters
#Dimensions of X
m, n = np.shape(X)
#Add x0 term to X
Xcalc = np.c_[np.ones((m, 1)), X]

#Initial theta - Note that theta has shape (n+1,) and NOT (n+1,1)
initial_theta = np.zeros(n + 1)

#Compute inital cost, grad for initial theta i.e. zero
#init_cost = cost_function(initial_theta, x, y)
#init_grad = grad_function(initial_theta, x, y)
init_cost, init_grad = costFunction(initial_theta, Xcalc, Y)

print('Cost at initial theta (zeros):\n', init_cost)
print('Gradient at initial theta (zeros): \n', init_grad)


#%%Logistic Regression/ Decision Boundary 
h = .02  # step size in the mesh

logreg = linear_model.LogisticRegression(C=1e5)
# Create an instance of Neighbours Classifier and fit the data.
# using liblinear as the solver
logreg.fit(X, Y)

#%% Plot the decision boundary. For that, we will assign a color to each
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
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())


#%%Predictions using Fitted Log Reg
theta0 = logreg.fit(X,Y).intercept_
theta1 = logreg.fit(X,Y).coef_
w = np.append(theta0, theta1)

print('Min theta found by log reg: \n', w)
xcost, xgrad = costFunction(w, Xcalc, Y)
print('Min cost = \n', xcost)

#Predict for single score set
v = np.array([1 ,45, 85],dtype=float)
#sp.speciat.expit is the built in sigmoid function 
prob = sp.special.expit(v.dot(w.T))
print('For a student with scores 45 and 85, admission probability', prob)

#Compare prediction against whole data set. Strictly speaking, we
# should have cross validation and test data sets to compute accuracy
xscore = logreg.score(X,Y)
print('Log reg score/ accuracy', xscore * 100)

