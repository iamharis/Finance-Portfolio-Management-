# -*- coding: utf-8 -*-
"""

"""

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from hurst_Mottl import compute_Hc, random_walk

#possible random walks
brownian = pd.Series(data=random_walk(400, proba=0.5, cumprod=False))+1000
persistent = pd.Series(data=random_walk(400, proba=0.999, cumprod=False))+1000
antipersistent = pd.Series(data=random_walk(400, proba=0.01, cumprod=False))+1000

dfP = pd.DataFrame()

dfP = pd.read_csv(r'SPY.csv', parse_dates=['Date'])
#dfP = pd.read_csv(r'Jonathan_BTCUSD_BBO_1minute.csv', parse_dates=['Date'])
dfP = dfP.sort_values(by='Date')
dfP = dfP.sort_values(by='Date')
dfP.set_index('Date', inplace = True)
security = dfP['Adj Close'][-1000:]

#select the series
#1 for brownian
#2 for persistent
#3 for antipersistent
#4 for security

choice = 4

if choice == 1:
    y = brownian
elif choice == 2:
    y = persistent
elif choice == 3:
    y = antipersistent
elif choice == 4:
    y = security

#plot the series
y.plot()

#plot the series autocorrelation
plot_acf(y, lags= 20, alpha=0.05)
plt.show()

# Evaluate Hurst equation
#trend = 1, random walk = .5, pink noise = 0, white noise = -.5 
H, c, data = compute_Hc(y, kind='price', simplified=True)
print("If hurst_Mottl.compute_HC > .5, means trend, else <.5 means mean reversion.  But stdev is .1")
print("H={:.4f}, c={:.4f}".format(H,c))



