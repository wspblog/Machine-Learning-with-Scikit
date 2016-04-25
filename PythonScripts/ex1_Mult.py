# -*- coding: utf-8 -*-
"""
Migrate MATLIB code to Python 3.5, NumPy, SciPy code
Created on Tue Mar 15 23:02:18 2016


Multivariate Linear Regression
"""
#%% Import Libraries
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge

#%%User-Defined functions
#Cost, gradient Function
def computeCost(X, y, theta):
    m = len(y)
    hypothesis = np.dot(X, theta)
    loss = hypothesis - y

    cost = np.sum(loss ** 2.0)/(2.0 * m)  #cost
    gradient = np.dot(X.T, loss)/m   # gradient

    return cost, gradient
    
#%% Gradient descent
def gradientDescent(X, y, theta, alpha, m, numIterations):
    j_history = np.zeros(numIterations, dtype=float)
    
    for i in range(0, numIterations):
        vcost, vgrad = computeCost(X, y, theta)
        j_history[i] = vcost
        theta = theta - alpha * vgrad

    return theta, j_history

#%%Gradient Descent - old version
'''
def gradientDescent(x, y, theta, alpha, m, numIterations):
    xTrans = x.transpose()
    j_history = np.zeros(numIterations, dtype=float)
    
    for i in range(1, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        cost = np.sum(loss ** 2) / (2 * m)
        j_history[i] = cost
        gradient = np.dot(xTrans, loss) / m
        theta = theta - alpha * gradient
    return theta, j_history
'''    
#%% Normalize features
def featureNorm(X):
    #initialise mean,sigma arrays
    cols = X.shape[1]
    mu = np.zeros([1,cols],dtype=float)
    sigma = np.zeros([1,cols],dtype=float)
     
     #calculate column means,std dev
    mu = np.mean(X,0,dtype=float)
    #Note MATLAB std calculation based on 1 degree of freedom, Numpy is 0  
    sigma = np.std(X,0,dtype=float,ddof=1)
    
    X_Norm = (X - mu)/sigma
   
    return X_Norm, mu, sigma 
    
#%% Normal Equation
def normalEqn(X,y):
    XTX = np.dot(X.T,X)
    Xinv = np.linalg.inv(XTX)
    Xc = np.dot(Xinv,X.T)
    ntheta = np.dot(Xc,y)
    
    return ntheta    
    
#%% Main Program
#Load data and plot data
#First 2 columns are features, third column is output
data = np.genfromtxt('ex1data2.txt',delimiter=',')
m = len(data)
x = data[:,[0,1]]
y = data[:,2]

#%%Normalize data
#X_norm = preprocessing.scale(x)
X_Norm, mu, sigma = featureNorm(x)

#%%Perform Gradient Descent
print('Gradient Descent Solution')
#Append x0
v = np.ones((m, 1))
X = np.c_[v, X_Norm]

#Gradient Descent Parameters
init_theta = np.zeros(3,dtype=float)
alpha = 0.1
num_iters = 400

# Init Theta and Run Gradient Descent 
xtheta, cost_history = gradientDescent(X, y, init_theta, alpha, m, num_iters)
jels = np.size(cost_history)

#plot cost function
plt.figure(1)
plt.plot(np.arange(0,jels),cost_history,'-b', LineWidth=2)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')

print('Theta computed from gradient descent: \n',xtheta)

# Estimate the price of a 1650 sq-ft, 3 br house
#House Price Prediction - 3 room, 1650 sq ft house
pr0 = xtheta[0] * 1 
pr1 = xtheta[1] * (1650 - mu[0])/sigma[0]
pr2 = xtheta[2] * (3 - mu[1])/sigma[1]
price = pr0 + pr1 + pr2
 
print('\n Predicted price of a 1650 sq-ft, 3 br house \n', price)

#%% Perform normal equation solution
# Note that X is not normalized for the normal equation calculation
# but we need to add x0s column
print('Normal Equation Solution')
Xeq = np.c_[v, x]
ntheta = normalEqn(Xeq, y)

print('\n Theta computed from the normal equations (X not normalized:) \n', ntheta)

#Estimate the price of a 1650 sq-ft, 3 br house
nprice = np.dot(ntheta.T,[1, 1650, 3])

print('\n Predicted price of a 1650 sq-ft, 3 br house \n', nprice)

#%% solution using sklearn linear regression model
print('\nSolution using sklearn linear regression Ridge model. Data not normalized')
lrf = Ridge(alpha=alpha, fit_intercept=True,copy_X=True, max_iter=num_iters, solver='auto') 
lrf.fit(x,y)
print('Theta 0 = ', lrf.intercept_, '\nTheta 1, 2 = ', lrf.coef_)    

ltheta = [lrf.intercept_, lrf.coef_[0], lrf.coef_[1]]
ltheta = np.atleast_1d(ltheta)

xpred = np.reshape(([1650],[3]),(1,2))
#Estimate the price of a 1650 sq-ft, 3 br house
nprice = lrf.predict(xpred)
print('\n Predicted price of a 1650 sq-ft, 3 br house \n', nprice)
