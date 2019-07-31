# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 23:29:39 2018

@author: User
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from hurst_Mottl import random_walk
from oct2py import octave

#NOTE: vratiotest in Matlab takes the log of prices as input whereas vratio in Python takes the log returns as input

b = pd.Series(data=random_walk(4000, proba=0.65, cumprod=True))+100
b.plot()
c = np.log(b.values)

df = pd.read_csv(r"Data\BTC-USD.csv", parse_dates=['Date']) 

df = df.sort_values(by='Date')
df.set_index('Date', inplace = True)
#df['ret'] = np.log(df['Adj Close']/df['Adj Close'].shift(1)) #matlab vratiotest takes the log of prices not the log returns
df['logP'] = np.log(df['Adj Close'])
df.dropna(inplace = True)
c2 = np.array(df['logP'].values)

octave.addpath(r'C:\Users\ThinkPad\Documents\OctaveMFiles')

"""
##Long version to test whether a series is a random walk.
q = [2,3,4,5,6,7,8,9,10,15,20,40,60,80,120]
flag = [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True]
flag2 =[False,False,False,False,False,False,False,False,False,False,False,False,False,False,False]

##Long version to test whether a series is a random walk.
#either of these calls will work
h, pValue, stat, cValue, ratio = octave.feval('vratiotest',c,'period',q,'IID',flag2, nout=5)
#h, pValue, stat, cValue, ratio = octave.vratiotest(c,'period',q,'IID',flag2,nout=5)
#In matlab the fn call would be
#[h,pValue,stat,cValue,ratio] = vratiotest(c,'period',q,'IID',flag)

print(ratio)
print(pValue)
"""

#Short version to test whether a series is a random walk, assumes 'IID' is set to false by default
#in other words null hypothesis = heteroskedastic random walk is a reasonable model for the series (to reject need small value)
# In  matlab the fn call would be: [h,pValue,stat,cValue,ratio] = vratiotest(c)
h, pValue, stat, cValue, ratio = octave.feval('vratiotest',c, nout=5)

print(ratio)
print(pValue)

#Test null hypotheses = i.i.d. random walk is a reasonable model for the series (to reject, need small pvalue)
# in matlab it would be [h,pValue,stat,cValue,ratio] = vratiotest(c,'IID',true)
h, pValue, stat, cValue, ratio = octave.feval('vratiotest',c, 'IID', True, nout=5)

print(ratio)
print(pValue)

