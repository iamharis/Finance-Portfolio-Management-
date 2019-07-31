# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 20:05:45 2018
http://www.pythonforfinance.net/2016/09/01/moving-average-crossover-trading-strategy-backtest-in-python/
@author: Rosario
"""

import WhiteRealityCheckFor1
import detrendPrice
import pandas as pd
import numpy as np
from datetime import datetime
pd.core.common.is_list_like = pd.api.types.is_list_like #datareader problem probably fixed in next version of datareader
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
yf.pdr_override() # <== that's all it takes :-)

start_date = '2000-01-01' 
end_date = datetime.now() 

symbol = '^GSPC' 
msg = "" 
address = symbol + '.csv'

try:
    dfP = pdr.get_data_yahoo(symbol, start=start_date, end=end_date)
    dfP.to_csv(address, header = True, index=True, encoding='utf-8')
except Exception:
    msg = "yahoo problem"
    dfP = pd.DataFrame()

dfP = pd.read_csv(address, parse_dates=['Date'])
dfP = dfP.sort_values(by='Date')
dfP.set_index('Date', inplace = True)

#dfP['Close'].plot(grid=True,figsize=(8,5))
dfP['42d'] = np.round(dfP['Close'].rolling(window=42).mean(),2)
dfP['252d'] = np.round(dfP['Close'].rolling(window=252).mean(),2)
#print(dfP.tail)

#dfP[['Close','42d','252d']].plot(grid=True,figsize=(8,5))
dfP['42-252'] = dfP['42d'] - dfP['252d']

dfP['Market_Returns'] = np.log(dfP['Close'] / dfP['Close'].shift(1))

dfP['MMI'] = dfP.Market_Returns.rolling(300).apply(lambda s: market_meanness_index.market_meanness_index(s))

X = 0
dfP['Close_200MA'] = dfP.Close.rolling(100).mean()
dfP['Stance'] = np.where((dfP['42-252'] > X), 1, 0)
dfP['Stance'] = np.where(dfP['42-252'] < X, -1, dfP['Stance'])
#print(dfP['Stance'].value_counts())

#dfP['Stance'].plot(lw=1.5,ylim=[-1.1,1.1])
dfP['Strategy'] = dfP['Market_Returns'] * dfP['Stance'].shift(1)
dfP['Strategy_Cummul'] = dfP['Strategy'].cumsum()+1
dfP['I'] = dfP['Strategy_Cummul']
dfP['Market_Returns_Cummul'] = dfP['Market_Returns'].cumsum()+1

dfP[['Market_Returns_Cummul','Strategy_Cummul']].plot(grid=True,figsize=(8,5))
print ('APR: AnnualPercentageRate in %')
print (dfP.Strategy_Cummul.last('1D').values[0]/((len(dfP)/360))*100)
print ('Sharpe')
print (np.sqrt(360.)*np.mean(dfP.Strategy)/np.std(dfP.Strategy))

#Detrend prices before calculating detrended returns
dfP['DetClose'] = detrendPrice.detrendPrice(dfP.Close).values
#these are the detrended returns to be fed to White's Reality Check
dfP['DetRet']= np.log(dfP['DetClose'] / dfP['DetClose'].shift(1))
dfP['DetRet_Cummul'] = dfP['DetRet'].cumsum()+1
WhiteRealityCheckFor1.bootstrap(dfP.DetRet)

dfP.to_csv(r'Results\dfP.csv')

#dfP[['Close','42d','252d']].plot(grid=True,figsize=(8,5))



