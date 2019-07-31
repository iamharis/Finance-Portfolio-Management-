# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 01:48:35 2018

@author: Rosario
"""

#import needed modules
from datetime import datetime
import pandas as pd
import numpy as np
from numpy import log, sqrt
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3 as db

pd.core.common.is_list_like = pd.api.types.is_list_like #datareader problem probably fixed in next version of datareader
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
yf.pdr_override() # <== that's all it takes :-)
import itertools as it

def backtest(symbList, whichfirst):
    global msg
    
    entryZscore = -1 
    exitZscore = -1 
    window =  7
    regression = 1
    residuals_model = 0
    
    start_date = '2012-01-01' 
    end_date = datetime.now() 
    
  
    if whichfirst == "yfirst":
        try:
            y = pdr.get_data_yahoo(symbList[0], start=start_date, end=end_date)
            y.to_csv(symbList[0] + '.csv', header = True, index=True, encoding='utf-8')
            x = pdr.get_data_yahoo(symbList[1], start=start_date, end=end_date)
            x.to_csv(symbList[1] + '.csv', header = True, index=True, encoding='utf-8')
        except Exception:
            msg = "yahoo problem"
            y = pd.DataFrame()
            x = pd.DataFrame()
            return 0
    else:
        try:
            y = pdr.get_data_yahoo(symbList[1], start=start_date, end=end_date)
            y.to_csv(symbList[1] + '.csv', header = True, index=True, encoding='utf-8')
            x = pdr.get_data_yahoo(symbList[0], start=start_date, end=end_date)
            x.to_csv(symbList[0] + '.csv', header = True, index=True, encoding='utf-8')
        except Exception:           
            msg = "yahoo problem"
            y = pd.DataFrame()
            x = pd.DataFrame()
            return 0

    if whichfirst == "yfirst":
        y = pd.read_csv(symbList[0] + '.csv', parse_dates=['Date'])
        y = y.sort_values(by='Date')
        y.set_index('Date', inplace = True)
        x = pd.read_csv(symbList[1] + '.csv', parse_dates=['Date'])
        x = x.sort_values(by='Date')
        x.set_index('Date', inplace = True)
    else:
        y = pd.read_csv(symbList[1] + '.csv', parse_dates=['Date'])
        y = y.sort_values(by='Date')
        y.set_index('Date', inplace = True)
        x = pd.read_csv(symbList[0] + '.csv', parse_dates=['Date'])
        x = x.sort_values(by='Date')
        x.set_index('Date', inplace = True)
 
     
    #doing an inner join to make sure dates coincide and there are no NaNs
    #inner join requires distinct column names
    y.rename(columns={'Open':'y_Open','High':'y_High','Low':'y_Low','Close':'y_Close','Adj Close':'y_Adj_Close','Volume':'y_Volume'}, inplace=True) 
    x.rename(columns={'Open':'x_Open','High':'x_High','Low':'x_Low','Close':'x_Close','Adj Close':'x_Adj_Close','Volume':'x_Volume'}, inplace=True) 
    df1 = pd.merge(x, y, left_index=True, right_index=True, how='inner') #inner join
    
    
    #get rid of extra columns but keep the date index
    df1.drop(['x_Open', 'x_High','x_Low','x_Close','x_Volume','y_Open', 'y_High','y_Low','y_Close','y_Volume'], axis=1, inplace=True)
    df1.rename(columns={'y_Adj_Close':'y','x_Adj_Close':'x'}, inplace=True) 
    df1 = df1.assign(TIME = pd.Series(np.arange(df1.shape[0])).values) 

    #find the hedge ratio and the spread
    #regress the y variable against the x variable
    #the slope of the rolling linear univariate regression=the rolling hedge ratio
    window_hr_reg = 58 #smallest window for regression when using y_hat
    a = np.array([np.nan] * len(df1))
    b = [np.nan] * len(df1)  # If betas required.
    y_ = df1["y"].values
    x_ = df1[['x']].assign(constant=1).values
    for n in range(window_hr_reg, len(df1)):
        y = y_[(n - window_hr_reg):n]
        X = x_[(n - window_hr_reg):n]
        # betas = Inverse(X'.X).X'.y
        betas = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        y_hat = betas.dot(x_[n, :])
        a[n] = y_hat
        b[n] = betas.tolist()  # If betas required. b[n][0] is the slope, b[n][1] is the intercept
    
    if residuals_model:
        myList = []
        for e in range(len(b)):
            if e < window_hr_reg:
                 myList.append(0)
            else:
                myList.append(b[e][0])
        df1["rolling_hedge_ratio"] = myList
    else:
        df1 = df1.assign(rolling_hedge_ratio = pd.Series(np.ones(df1.shape[0])).values)
    
    df1["portfolio_cost"]=df1.x*np.abs(df1["rolling_hedge_ratio"])+df1.y
    
    #calculate the spread
    if residuals_model == 1:
        df1['spread'] = df1.y - df1.rolling_hedge_ratio*df1.x
    else:
        df1['spread'] = log(df1.x) - log(df1.y)
    
    #rolling regression instead of moving average
    a = np.array([np.nan] * len(df1))
    b = [np.nan] * len(df1)  # If betas required.
    y_ = df1['spread'].values
    x_ = df1[['TIME']].assign(constant=1).values
    for n in range(window, len(df1)):
        y = y_[(n - window):n]
        X = x_[(n - window):n]
        # betas = Inverse(X'.X).X'.y
        betas = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        y_hat = betas.dot(x_[n, :])
        a[n] = y_hat
        b[n] = betas.tolist()  # If betas required. b[n][0] is the slope, b[n][1] is the intercept
    
    df1 = df1.assign(y_hat = pd.Series(a).values)
        
    if regression == 1:
        mean = df1['y_hat']
    else:
        mean = df1['spread'].rolling(window=window).mean()
    
    #calculate the zScore indicator
    df1 = df1.assign(meanSpread = pd.Series(a).values)
    stdSpread = df1.spread.rolling(window=window).std()
    df1['zScore'] = (df1.spread-mean)/stdSpread
    
    """
    plt.plot(df1.spread)
    plt.show()
    df1['zScore'].plot()
    plt.show()
    """
    
    #set up num units long             
    df1['long entry'] = ((df1.zScore < - entryZscore) & ( df1.zScore.shift(1) > - entryZscore))
    df1['long exit'] = ((df1.zScore > - exitZscore) & (df1.zScore.shift(1) < - exitZscore)) 
    df1['num units long'] = np.nan 
    df1.loc[df1['long entry'],'num units long'] = 1 
    df1.loc[df1['long exit'],'num units long'] = 0 
    df1.iat[0,df1.columns.get_loc("num units long")]= 0
    
    df1['num units long'] = df1['num units long'].fillna(method='pad') 
    
    #set up num units short 
    df1['short entry'] = ((df1.zScore >  entryZscore) & ( df1.zScore.shift(1) < entryZscore))
    df1['short exit'] = ((df1.zScore < exitZscore) & (df1.zScore.shift(1) > exitZscore))
    df1['num units short'] = np.nan
    df1.loc[df1['short entry'],'num units short'] = -1 
    df1.loc[df1['short exit'],'num units short'] = 0
    df1.iat[0,df1.columns.get_loc("num units short")]= 0
    df1['num units short'] = df1['num units short'].fillna(method='pad')
    
    df1['numUnits'] = df1['num units long'] + df1['num units short']
    df1['real_spread'] = df1.x - df1.y
    df1['spread_return'] = (df1['real_spread']-df1['real_spread'].shift(1))/df1.portfolio_cost.shift(1)
    df1['port_rets'] = df1['spread_return'] * df1['numUnits'].shift(1)
    
    
    df1 = df1.assign(I =np.cumprod(1+df1['port_rets'])) #this is good for pct return or log return
    df1.iat[0,df1.columns.get_loc('I')]= 1
    
    start_val = 1
    end_val = df1['I'].iat[-1]
    
    start_date = df1.iloc[0].name
    end_date = df1.iloc[-1].name
    days = (end_date - start_date).days
    
    TotaAnnReturn = (end_val-start_val)/start_val/(days/360)
    TotaAnnReturn_trading = (end_val-start_val)/start_val/(days/252)
    
    CAGRdbl_trading = round(((float(end_val) / float(start_val)) ** (1/(days/252.0))).real - 1,4) #when raised to an exponent I am getting a complex number, I need only the real part
    CAGRdbl = round(((float(end_val) / float(start_val)) ** (1/(days/360))).real - 1,4) #when raised to an exponent I am getting a complex number, I need only the real part
    
    try:
        sharpe =  TotaAnnReturn_trading/( (df1['port_rets'].std()) * sqrt(252))
    except ZeroDivisionError:
        sharpe = 0.0
    
    if sharpe > .5:
        if whichfirst == "yfirst":
            title=symbList[0]+"."+symbList[1]
            ylabel = symbList[0]
            xlabel = symbList[1]
        else:
            title=symbList[1]+"."+symbList[0]
            ylabel = symbList[1]
            xlabel = symbList[0]
        
        print ("TotaAnnReturn = %f" %(TotaAnnReturn*100))
        print ("CAGR = %f" %(CAGRdbl*100))
        print ("Sharpe Ratio = %f" %(round(sharpe,2)))   
        plt.plot(df1['I'])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        #plt.show()  #if you show() it won't savefig()!
        plt.savefig(r'Results\%s.png' %(title))
        plt.close()
    
    return 1


##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################  

#MAIN 
    
#set the database file path we wish to connect to
#this will obviously be unique to wherever you created
#the SQLite database on your local system
database = 'C:\sqlite\PythonData.db'
 
#this is the SQL statement containing the information
#regarding which tickers we want to pull from the database
#As an example I have chosen to pull down all tickers which
#have their "Asset Class" listed as being "Currency"
#sql = 'SELECT Ticker FROM etftable WHERE "Asset Class" = "Currency";'
#sql = 'SELECT Ticker FROM etftable WHERE "Niche" = "Broad-based";'
#sql = 'SELECT Ticker FROM etftable WHERE "Focus" = "Silver";'
sql = 'SELECT Ticker FROM etftable WHERE "Focus" = "Platinum";'
 
#create a connection to the database specified above
cnx = db.connect(database)
cur = cnx.cursor()
 
#execute the SQL statement and place the results into a 
#variable called "tickers"
tickers = pd.read_sql(sql, con=cnx)
 
#create an empty list
symbList = []
 
#iterate over the DataFrame and append each item into the empty list
for i in range(len(tickers)):
    symbList.append(tickers.ix[i][0])


symbList = list(set(symbList))
#symbList = ["BZF", "FXCH"]

#get sumbol pairs
symbPairs = list(it.combinations(symbList, 2))
    
r = []
msg = ""
for i in symbPairs:
    ret = 1
    ret = backtest(i,"yfirst")
    r.append(ret)
    print(msg+" --")

for i in symbPairs:
    ret = 1
    ret = backtest(i,"xfirst")
    r.append(ret)
    print(msg+" --")