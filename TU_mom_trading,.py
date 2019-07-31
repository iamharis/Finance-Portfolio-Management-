

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import detrendPrice
import WhiteRealityCheckFor1
from datetime import datetime
pd.core.common.is_list_like = pd.api.types.is_list_like #datareader problem probably fixed in next version of datareader
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
yf.pdr_override() # <== that's all it takes :-)

start_date = '2000-01-01' 
end_date = datetime.now() 

symbol = 'SPY' 
msg = "" 
address = symbol + '.csv'
trade_lookback = 220
trade_holddays = 220


try:
    dfP = pdr.get_data_yahoo(symbol, start=start_date, end=end_date)
    dfP.to_csv(address, header = True, index=True, encoding='utf-8')
except Exception:
    msg = "yahoo problem"
    dfP = pd.DataFrame()

dfP = pd.read_csv(address, parse_dates=['Date'])
dfP = dfP.sort_values(by='Date')
dfP.set_index('Date', inplace = True)

from scipy.stats.stats import pearsonr

lkback =  [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 80, 120, 200, 220, 250]
#lkback = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,40]
hldays = lkback

res = []
for lookback in lkback:
   for holddays in hldays:
       df_Close_lookback = dfP.Close.shift(lookback)
       df_Close_holddays = dfP.Close.shift(-holddays)
       dfP['ret_lag'] = (dfP.Close-df_Close_lookback)/df_Close_lookback
       dfP['ret_fut'] = (df_Close_holddays-dfP.Close)/dfP.Close
       dfc = dfP[['ret_lag','ret_fut']].dropna()
       idx = None
       if lookback >= holddays: 
           idx = np.array(range(0,len(dfc.ret_lag), holddays))
       else: 
           idx = np.array(range(0,len(dfc.ret_lag), lookback))
       dfc = dfc.ix[idx]
       df = dfc.dropna(inplace=False)
       t, p = pearsonr(df.ret_lag.values, df.ret_fut.values)
       res.append([lookback, holddays,  t, p])
res = pd.DataFrame(res,columns=['lookback','holddays','corr','pvalue'])
res.sort_values(by="pvalue", inplace=True)
print (res[res['corr'] >= 0][:20])

#buy (sell) the asset if it has a positive (negative) 12-month return (assuming that 1 year is the lookback), and hold the
#position for 1 month (=20 trading days, assuming that 1 month is the holding) (similar to Moskowitz, Yao, and Pedersen, 2012). We will modify
#one detail of the original strategy: Instead of making a trading decision every
#month (assuming that is the holding), we will make it every day, each day investing only one twentieth 
#of the total capital. Also, we commented out the shorts.

def trade(df,lookback,holddays):

    df["longs"] = (df.Close > df.Close.shift(lookback))
    df["shorts"] = (df.Close < df.Close.shift(lookback)) 
    
    df['pos'] = 0.
    for h in range(holddays):
       df["long_lag"] = df["longs"].shift(h).fillna(False)
       df["short_lag"]= df["shorts"].shift(h).fillna(False)
       df.loc[df["long_lag"],'pos'] += 1
       #df.loc[df["short_lag"],'pos'] -= 1 # no shorts
    
    ret=(df.pos.shift(1)* (df.Close-df.Close.shift(1)) / df.Close.shift(1))/ holddays

    cumret=np.cumprod(1+ret)-1

    print ('APR: AnnualPercentageRate in %')
    print ((((np.prod(1.+ret))**(360./len(ret)))-1)*100)
    print ('Sharpe')
    print (np.sqrt(360.)*np.mean(ret)/np.std(ret))
    
    return cumret+1

cumret=trade(dfP,lookback = trade_lookback, holddays = trade_holddays)
dfP['I'] = cumret
plt.plot(cumret)
plt.show()

#Detrend prices before calculating detrended returns
dfP['DetClose'] = detrendPrice.detrendPrice(dfP.Close).values
#these are the detrended returns to be fed to White's Reality Check
dfP['DetRet']=(dfP.pos.shift(1)*(dfP.DetClose-dfP.DetClose.shift(1)) / dfP.DetClose.shift(1))/ holddays
dfP['DetCumret']=np.cumprod(1+dfP.DetRet)-1

WhiteRealityCheckFor1.bootstrap(dfP.DetRet)

dfP.to_csv(r'Results\dfP.csv')
res.to_csv(r'Results\res.csv')


