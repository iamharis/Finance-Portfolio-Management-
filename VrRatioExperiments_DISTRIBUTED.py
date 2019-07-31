import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from hurst_Mottl_DISTRIBUTED import random_walk


#possible random walks
brownian = pd.Series(data=random_walk(4000, proba=0.5, cumprod=False))+100
persistent = pd.Series(data=random_walk(4000, proba=0.99, cumprod=False))+100
antipersistent = pd.Series(data=random_walk(4000, proba=0.01, cumprod=False))+100


y = brownian #or persistent or antipersistent
choice = 1

y1 = np.log(y/y.shift(1))
y1.dropna(inplace = True)
arr_sim = np.array(y1.values)


df = pd.read_csv(r"SPY.csv", parse_dates=['Date']) 


df = df.sort_values(by='Date')
df.set_index('Date', inplace = True)
df['ret'] = np.log(df['Adj Close']/df['Adj Close'].shift(1))
df.dropna(inplace = True)
arr_real = np.array(df['ret'].values)

if choice == 1:
    arr = arr_real
    df['Adj Close'].plot()
else:
    arr = arr_sim
    y.plot()

#needs to have as input the log returns of the prices 
from vratio_hu_DISTRIBUTED import LoMac

l_k = [2,4,6,8,10,12,14,16,18,20,25,30,35,40,60,80,120,200,240,300]
arr_k = np.array(l_k)
print("If series is a rw, VR = 1, M1 = N(0,1) for iid series,  M2 = N(0,1) for non iid series i.e. M1 = 0,  M2 = 0")
print("If VR != 1, e.g. .4 (<1) and abs(M1) or abs(M2) greater than 1.96 for 2-tailed @ 95% cert, series is probably mean reverting")
print("If VR != 1, e.g. 1.2 (>1) and abs(M1) or abs(M2) greater than 1.96  for 2-tailed @ 95% cert, probably a persistent series")
vr = LoMac(arr,arr_k)
print(vr)

