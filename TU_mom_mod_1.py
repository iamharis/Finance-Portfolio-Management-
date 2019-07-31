import matplotlib.pylab as plt
import numpy as np
import pandas as pd

dftu = pd.read_csv('SPY.csv')

#import corr
from scipy.stats.stats import pearsonr

res = []
for lookback in [1, 5, 10, 20, 25, 40, 60, 120, 250]:
   for holddays in [1, 5, 10, 20, 25, 40, 60, 120, 250]:
       df_Close_lookback = dftu.Close.shift(lookback)
       df_Close_holddays = dftu.Close.shift(-holddays)
       dftu['ret_lag'] = (dftu.Close-df_Close_lookback)/df_Close_lookback
       dftu['ret_fut'] = (df_Close_holddays-dftu.Close)/dftu.Close
       dfc = dftu[['ret_lag','ret_fut']].dropna()
       idx = None
       if lookback >= holddays: 
           idx = np.array(range(0,len(dfc.ret_lag), holddays))
       else: 
           idx = np.array(range(0,len(dfc.ret_lag), lookback))
       dfc = dfc.ix[idx]
       #t, x, p = corr.p_corr(dfc.ret_lag, dfc.ret_fut)
       df = dfc.dropna(inplace=False)
       t, p = pearsonr(df.ret_lag.values, df.ret_fut.values)
       res.append([lookback, holddays,  t, p])
res = pd.DataFrame(res,columns=['lookback','holddays','corr','pvalue'])
print (res[res['lookback'] >= 0])

res.to_csv(r'Results\res.csv', header = True, index=True, encoding='utf-8')
