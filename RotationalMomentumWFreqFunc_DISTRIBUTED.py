# -*- coding: utf-8 -*-
"""



"""

import pandas as pd
import numpy as np
import math
import datetime
import matplotlib.pyplot as plt
from matplotlib import style

def getDate(dt):
    if type(dt) != str:
        return dt
    try:
        datetime_object = datetime.datetime.strptime(dt, '%Y-%m-%d')
    except Exception:
        datetime_object = datetime.datetime.strptime(dt, '%m/%d/%Y')
        return datetime_object
    else:
        return datetime_object
       
#All COLUMN_NAMES are capitalized
#IMPORTANT: ALL ETF LISTS  MUST END WITH SHY ETF (=rightmost column of the dataframe)
#dfP are the prices, used for the calculation of trading signals
#dfAP are the adjusted prices, used for the calculation of portfolio returns


dfP = pd.read_csv('SPY.TLT.csv', parse_dates=['Date'])
dfAP = pd.read_csv('SPY.TLT.AP.csv', parse_dates=['Date'])
title = "SPY.TLT"


dfP = dfP.sort_values(by='Date')
dfAP = dfAP.sort_values(by='Date')
dfP.set_index('Date', inplace = True)
dfAP.set_index('Date', inplace = True)

#logReturns = 1 means log returns will be used in the calculation of portfolio returns, 0 means pct_changes
#momentum = 1 means A and B returns are ranked in increasing order (momentum), 0 in decreasing order (antimomentum)
#volmomentum = 1 volatility ranked in increasing order (momentum), 0 in decreasing order (antimomentum)
#the selection of the ETF is based on maximum weighted score of: A returns, B returns and volatility
#Frequency="W" every weeek, "2W" for every 2 weeks, "3W" every 3 weeks etc
#Frequency="W-TUE" every Tuesday, "2W-TUE" for every 2 Tuesdats, "3W-TUE" every 3 Tudsdays etc
#Frequency= "BM" every month, "2BM" for every 2 months, "3BM" every 3 months etc; B relates to business days; 31 or previous business day if necessary
#Frequency="SM" on the middle (15) and end (31) of the month, or previous business day if necessary
#Delay = 1 if the trade occurs instantaneously with the signal, 2 if the trade occurs 1 day after the signal

logReturns = 0
momentum = 1
volmomentum = 0 #do not change
Aperiods = 20
Bperiods = 63
Speriods = 20
Zperiods = 200
CashFilter = 0
MAperiods = 200 #for the cash filter, 200MA
Zboundary = -1.5
Frequency = "M"
Delay = 1
#Frequency = "3BM" #good result

#dfA contains a short moving average of the daily percent changes, calculated for each ETF
#dfB contains a long moving average of the daily percent changes, calculated for each ETF
#dfS contains the annualized volatility, calculated for each ETF
#dfMA contains 200 MA of price

dfA = dfP.drop(labels=None, axis=1, columns=dfP.columns)
dfB = dfP.drop(labels=None, axis=1, columns=dfP.columns)
dfS = dfP.drop(labels=None, axis=1, columns=dfP.columns)
dfZ = dfP.drop(labels=None, axis=1, columns=dfP.columns)
dfMA = dfP.drop(labels=None, axis=1, columns=dfP.columns)

#calculating the three performance measures in accordance with their windows

dfA = dfP.pct_change(periods=Aperiods-1, fill_method='pad', limit=None, freq=None) #is counting window from 0
dfB = dfP.pct_change(periods=Bperiods-1, fill_method='pad', limit=None, freq=None) #is counting window from 0


columns = dfP.shape[1]
for column in range(columns):
    dfS[dfP.columns[column]] = (dfP[dfP.columns[column]].rolling(window=Speriods).std())*math.sqrt(252)
    dfZ[dfP.columns[column]] = (dfP[dfP.columns[column]]-dfP[dfP.columns[column]].rolling(window=Zperiods).mean())/dfP[dfP.columns[column]].rolling(window=Zperiods).std()
    dfMA[dfP.columns[column]] = (dfP[dfP.columns[column]].rolling(window=MAperiods).mean())

#Ranking each ETF w.r.t. short moving average of returns
dfA_ranks = dfP.copy(deep=True)
dfA_ranks[:] = 0

columns = dfA_ranks.shape[1]
rows = dfA_ranks.shape[0]

#this loop takes each row of the A dataframe, puts the row into an array, 
#within the array the contents are ranked, 
#then the ranks are placed into the A_ranks dataframe one by one
for row in range(rows):
    arr_row = dfA.iloc[row].values
    if momentum == 1:
        temp = arr_row.argsort() #sort momentum, best is ETF with largest return
    else:
        temp = (-arr_row).argsort()[:arr_row.size] #sort antimomentum, best is ETF with lowest return
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(1,len(arr_row)+1)
    for column in range(columns):
        dfA_ranks.iat[row, column] = ranks[column]

dfB_ranks = dfP.copy(deep=True)
dfB_ranks[:] = 0

columns = dfB_ranks.shape[1]
rows = dfB_ranks.shape[0]

#this loop takes each row of the B dataframe, puts the row into an array, 
#within the array the contents are ranked, 
#then the ranks are placed into the B_ranks dataframe one by one
for row in range(rows):
    arr_row = dfB.iloc[row].values
    if momentum == 1:
        temp = arr_row.argsort() #sort momentum
    else:
        temp = (-arr_row).argsort()[:arr_row.size] #sort antimomentum 
    ranks = np.empty_like(temp) 
    ranks[temp] = np.arange(1,len(arr_row)+1)
    for column in range(columns):
        dfB_ranks.iat[row, column] = ranks[column]

dfS_ranks = dfP.copy(deep=True)
dfS_ranks[:] = 0

columns = dfS_ranks.shape[1]
rows = dfS_ranks.shape[0]

#this loop takes each row of the Sinv dataframe, puts the row into an array, 
#within the array the contents are ranked, 
#then the ranks are placed into the Sinv_ranks dataframe one by one
for row in range(rows):
    arr_row = dfS.iloc[row].values
    if volmomentum == 1:
        temp = arr_row.argsort() #sort momentum, best is highest volatility
    else:
        temp = (-arr_row).argsort()[:arr_row.size] #sort antimomentum, best is lowest volatility
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(1,len(arr_row)+1)
    for column in range(columns):
        dfS_ranks.iat[row, column] = ranks[column]

#Weights of the varous ranks
dfA_ranks = dfA_ranks.multiply(.3)
dfB_ranks = dfB_ranks.multiply(.4)
dfS_ranks = dfS_ranks.multiply(.3)
dfAll_ranks = dfA_ranks.add(dfB_ranks, fill_value=0)
dfAll_ranks = dfAll_ranks.add(dfS_ranks, fill_value=0)

#Choice is the dataframe where the ETF with the maximum score is identified
dfChoice = dfP.copy(deep=True)
dfChoice[:] = 0
rows = dfChoice.shape[0]

#this loop takes each row of the All-ranks dataframe, puts the row into an array, 
#within the array the contents scanned for the maximum element 
#then the maximum element is placed into the Choice dataframe
for row in range(rows):
    arr_row = dfAll_ranks.iloc[row].values
    if momentum == 0:
         arr_row = arr_row[0:len(arr_row)-1] #don't rank SHY (the last column) if doing antimomentum trading
    max_arr_column = np.argmax(arr_row, axis=0) #gets the INDEX of the max
    if CashFilter == 1:
        if (dfP[dfP.columns[max_arr_column]][row] >= dfMA[dfMA.columns[max_arr_column]][row]): #200MA condition
        #if (dfZ[dfZ.columns[max_arr_column]][row] > Zboundary):
            dfChoice.iat[row, max_arr_column] = 1
        else:
            dfChoice.iat[row, dfP.columns.get_loc("SHY")] = 1
    else:
        dfChoice.iat[row, max_arr_column] = 1

#dfPRR is the dataframe containing the log or pct_change returns of the ETFs
#will be based on adjusted prices rather than straight prices

if logReturns == 1:
    dfPLog = dfAP.apply(np.log) 
    dfPLogShift = dfPLog.shift(1)
    dfPRR = dfPLog.subtract(dfPLogShift, fill_value=0)
else:
    dfPRR= dfAP.pct_change() 
    
#T is the dataframe where the trading day is calculated. 

dfT = dfP.drop(labels=None, axis=1, columns=dfP.columns)    
columns = dfP.shape[1] 
for column in range(columns):
    new = dfP.columns[column] + "_CHOICE"
    dfPRR[new] = pd.Series(np.zeros(rows), index=dfPRR.index)
    dfPRR[new] = dfChoice[dfChoice.columns[column]]

dfT['DateCopy'] = dfT.index
dfT1 = dfT.asfreq(freq=Frequency, method='pad')
dfT1.set_index('DateCopy', inplace=True)
dfTJoin = pd.merge(dfT,
                 dfT1,
                 left_index = True,
                 right_index = True,
                 how='outer', 
                 indicator=True)
dfPRR[Frequency+"_FREQ"] = dfTJoin["_merge"]


#_LEN means Long entry for that ETF
#_NUL means number units long of that ETF
#_LEX means long exit for that ETF
#_R means returns of that ETF (traded ETF)
#_ALL_R means returns of all ETFs traded, i.e. portfolio returns
#CUM_R means commulative returns of all ETFs, i.e. portfolio cummulative returns

columns = dfP.shape[1]
for column in range(columns):
    new = dfP.columns[column] + "_LEN"
    dfPRR[new] = ((dfPRR[Frequency+"_FREQ"] =="both") & (dfPRR[dfP.columns[column]+"_CHOICE"] ==1))
    new = dfP.columns[column] + "_LEX"
    dfPRR[new] = ((dfPRR[Frequency+"_FREQ"] =="both") & (dfPRR[dfP.columns[column]+"_CHOICE"] !=1))
    new = dfP.columns[column] + "_NUL"
    dfPRR[new] = np.nan
    dfPRR.loc[dfPRR[dfP.columns[column]+'_LEX'] == True, dfP.columns[column]+'_NUL' ] = 0
    dfPRR.loc[dfPRR[dfP.columns[column]+'_LEN'] == True, dfP.columns[column]+'_NUL' ] = 1 #this order is important
    dfPRR.iat[0,dfPRR.columns.get_loc(dfP.columns[column] + "_NUL")] = 0
    dfPRR[dfP.columns[column] + "_NUL"] = dfPRR[dfP.columns[column] + "_NUL"].fillna(method='pad') 
    new = dfP.columns[column] + "_R"
    dfPRR[new] = dfPRR[dfP.columns[column]]*dfPRR[dfP.columns[column]+'_NUL'].shift(Delay)


#calculating all returns
dfPRR = dfPRR.assign(ALL_R = pd.Series(np.zeros(rows)).values) 

#the return of the portfolio is a sequence of returns made
#by appending sequences of returns of traded ETFs
#Since non traded returns are multiplied by zero, we only need to add the columns
#of the returns of each ETF, traded or not
columns = dfP.shape[1]
for column in range(columns):
    dfPRR["ALL_R"] = dfPRR["ALL_R"] + dfPRR[dfP.columns[column]+"_R"]
    
dfPRR['CUM_R'] = dfPRR['ALL_R'].cumsum()
dfPRR['CUM_R'] = dfPRR['CUM_R'] + 1 #this is good only for log returns

#calculating portfolio investment column in a separate dataframe, using 'ALL_R' = portfolio returns

dfPI = pd.DataFrame(data=dfPRR['ALL_R'], index=dfPRR.index)
dfPI = dfPI.fillna(value=0)
dfPI = dfPI.assign(I = pd.Series(np.zeros(dfPI.shape[0])).values) 
start = 1

dfPI.iat[0,1] = start
rows = dfPI.shape[0]
for row in range(1,rows):
    dfPI.iat[row,1] = dfPI.iat[row-1,1]*dfPI.iat[row,0]+dfPI.iat[row-1,1] #this is good for pct return or log return

dfPRR = dfPRR.assign(I = dfPI['I'])

try:
    sharpe = ((dfPRR['ALL_R'].mean() / dfPRR['ALL_R'].std()) * math.sqrt(252)) 
except ZeroDivisionError:
    sharpe = 0.0

style.use('fivethirtyeight')
dfPRR['I'].plot()
plt.legend()
plt.show()
#plt.savefig(r'Results\%s.png' %(title))
#plt.close()

start_val = start
end_val = dfPRR['I'].iat[-1]
    
start_date = getDate(dfPI.iloc[0].name)
end_date = getDate(dfPI.iloc[-1].name)
days = (end_date - start_date).days

TotaAnnReturn = (end_val-start_val)/start_val/(days/360)
TotaAnnReturn_trading = (end_val-start_val)/start_val/(days/252)
    
CAGR_trading = round(((float(end_val) / float(start_val)) ** (1/(days/252.0))).real - 1,4) #when raised to an exponent I am getting a complex number, I need only the real part
CAGR = round(((float(end_val) / float(start_val)) ** (1/(days/350.0))).real - 1,4) #when raised to an exponent I am getting a complex number, I need only the real part

print ("TotaAnnReturn = %f" %(TotaAnnReturn*100))
print ("CAGR = %f" %(CAGR*100))
print ("Sharpe Ratio = %f" %(round(sharpe,2)))

    
dfPRR.to_csv(r'Results\dfPRR.csv', header = True, index=True, encoding='utf-8')



