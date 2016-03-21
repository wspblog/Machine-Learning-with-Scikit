# -*- coding: utf-8 -*-
"""

Multi-class Logistic Regression
Image recognition

Created on Sun Mar 20 23:09:44 2016

@author:

"""
#Python libraries used in this program
import scipy.io as sio
from numpy import dot, argmax, mean, ones, c_
from sklearn import linear_model
from scipy import special
from scipy.stats import itemfreq


#Load test data from MATLAB data file
datamat = sio.loadmat('ex3data1.mat')
X = datamat['X']
y = datamat['y']
yrave = y.ravel() #required form for scikit model
m = len(y)


#Note C is inverse of regularization strength lambda
#Add columnn of 1's to X i.e. Xlogreg has 5000 x 401 shape
v = ones((m, 1))
Xlogreg = c_[v, X]

#multi-class set to 'ovr' i.e. One-Vs-Rest multi classification
logreg = linear_model.LogisticRegression(C=1,multi_class='ovr')
logres= logreg.fit(Xlogreg, yrave)
theta = logres.coef_  #theta is 10 x 401 array 

#Compare accuracy of logistic regression
res = special.expit(dot(Xlogreg,theta.T))  #expit is the sigmoid function
p = argmax(res,axis=1)
#Note that column index is from 0 to 9 in sckit. Need to increment by 1
#to match MATLAB indices that start from 1 to 10
p = p + 1

paccu = (p ==yrave)
print(itemfreq(paccu))

accu = mean(p == yrave) * 100