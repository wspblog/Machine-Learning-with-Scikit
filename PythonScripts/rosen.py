# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 23:50:33 2016

@author: pamwong
"""
import numpy as np
import scipy as sp
from scipy.optimize import fmin_bfgs


def rosen(x):
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def rosen_der(x):
        xm = x[1:-1]
        xm_m1 = x[:-2]
        xm_p1 = x[2:]
        der = sp.zeros(x.shape,x.typecode())
        der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
        der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
        der[-1] = 200*(x[-1]-x[-2]**2)
        return der


x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
xopt = fmin_bfgs(rosen, x0, fprime=rosen_der)
print(xopt)

