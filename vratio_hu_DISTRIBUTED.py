

"""@
Author: Z Hu, MIT
2016-02-19
Rewrite R's Lo.Mac funtion in vrtest package in Python.
https://github.com/hiemal/vratiotest/blob/master/vratiotest.py
see p.45 in Chan's Algorithmic trading
because vr1 = np.sum((y-m)**2)/n, which is just the variance of the input series
means y should be the actual log returns not the prices or the log of the prices

in R:
https://cran.r-project.org/web/packages/vrtest/vrtest.pdf
http://www.turingfinance.com/testing-the-efficient-market-hypothesis-with-r/
http://www.turingfinance.com/stock-market-prices-do-not-follow-random-walks/

see also for octave and python:
https://github.com/benfulcher/hctsa

M2 values (for uncorrelated series) are not correct

"""

from __future__ import division

import numpy as np
import scipy as sp
import pandas as pd


#Define functions for vratiotest


def LM_stat(y, k):
    y1 = (y - np.mean(y))**2
    n = len(y)
    m = np.mean(y)
    vr1 = np.sum((y-m)**2)/n
    
    # use the convolve function
    flt = np.convolve(y, np.repeat(1,k), mode = 'valid')
    summ = np.sum((flt-k*m)**2)
    
    vr2 = summ/(n*k)
    vr = vr2/vr1
    
    tem1 = 2*(2*k-1)*(k-1)
    tem2 = 3*k
    
    m1 = np.sqrt(n)*(vr-1)/np.sqrt(tem1/tem2)
    w = 4*(1-np.arange(1,k)/k)**2
    dvec = np.zeros([k-1,1])
    for j in range(k-1):
        dvec[j] = np.sum(y1[(j+1):(n+1)] * y1[0:(n-j-1)])/(np.sum(y1)**2)
    summ = np.sum(w* np.ravel(dvec))
    m2 = np.sqrt(n)*(vr-1)*((n*summ)**(-0.5))
    return (m1,m2,vr)
    
    

def LoMac(y, kvec):
    """
    The function returns M1 and M2 statistics of Lo and MacKinlay (1998).
    M1: tests for iid series; M2: for uncorrelated series with possible heteroskedasticity; VR: variance ratio.
    """
    n = len(y)
    mq = np.zeros([len(kvec),3])
    for i in range(len(kvec)):
        k = kvec[i]
        LM = LM_stat(y,k)
        mq[i] = np.array(LM)
    VR = pd.DataFrame(mq, columns = ["M1", "M2", "VR"], index = ['k='+str(k) for k in kvec])
    return VR