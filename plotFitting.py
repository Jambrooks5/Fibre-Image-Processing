import numpy as np
from scipy.optimize import curve_fit as fit

def sinFunc(x, amplitude, shift, intercept):
    return (amplitude*np.sin(x + shift) + intercept)

#data parameter must be format [[x1,x2,x3...],[y1,y2,y3...]]
def fitSinFunc(data):
    params, params_covariance = fit(sinFunc, data[0], data[1])
    #print (params)
    return params

