
# coding: utf-8

# In[147]:

import quandl 
import pandas_datareader.data as pdr
import numpy as np
from pandas.tseries.offsets import BDay
import pandas_datareader.data as pdr
from datetime import datetime
import matplotlib.pyplot as plt

#Apple Stock Data
today = datetime.now()
StockData=pdr.get_data_yahoo('AAPL',datetime(2013,1,1),today)

ReturnList = []
StockLength = len(StockData['Close'])
for i in range(StockLength):
    if(i < StockLength - 1):
        Previousday = list(StockData['Close'])[i]
        Nextday = list(StockData['Close'])[i+1]
        Dailyreturn = (Nextday - Previousday)/(Previousday)
        ReturnList.append(Dailyreturn)

StockMean = np.mean(ReturnList)
StockSD = np.std(ReturnList)

#Apple Sentiment Data 
NSData=(quandl.get("NS1/AAPL_US", authtoken="VV_5NSuUzyPxy8sgZkzp")).asfreq(BDay())

n = len(list(NSData.index))
SenData = [] #NSData sentiments trimmed to same number of data points as StockData according to same timestamps
for i in range(n):
    if NSData.index[i] in list(StockData.index):
         SenData.append(NSData.Sentiment[i])
            
SenMean = np.mean(SenData) #daily mean
SenSD = np.std(SenData)

#Deviation = []
#for i in range(SenLength):
#    sd = ((NSData['Sentiment'][i])-SenMean)**2
#    Deviation.append(sd)
#Sensd = (np.mean(Deviation))**0.5

def Covariance(X,Y):
    Xarray = np.array(X)
    Yarray = np.array(Y)
    Jointexpectation = np.mean(Xarray*Yarray)
    Xmean = np.mean(X)
    Ymean = np.mean(Y)
    
    return Jointexpectation - (Xmean*Ymean)

Cov = Covariance(ReturnList,SenData)
Correlation = Cov/(SenSD*StockSD)

#Linear Regression
beta = Cov/(SenSD**2)
alpha = StockMean - (beta*SenMean)

#R-square
total_variation = 0
unexplained_variation = 0

for i in range(len(ReturnList)):
    total_variation += ((ReturnList[i])-StockMean)**2
    
Estimatedreturns = []
for i in range(len(ReturnList)):
    Estimates = alpha + beta*SenData[i]
    Estimatedreturns.append(Estimates)

for i in range(len(ReturnList)):
    unexplained_variation += ((ReturnList[i])-Estimatedreturns[i])**2

explained_variation = total_variation - unexplained_variation
Rsquare = explained_variation/total_variation
#Trading Strategy Test - Trade according to day's Sentiment
Samedirection = 0
Oppdirection = 0
Notrades = 0
Combined = (np.array(ReturnList))*(np.array(SenData))
for i in range(len(Combined)):
    if Combined[i] > 0:
        Samedirection += 1
    elif Combined[i] < 0:
        Oppdirection += 1
    else: 
        Notrades += 1
Total = Samedirection + Oppdirection
wintrade = Samedirection/Total
losetrade = Oppdirection/Total

years = 0.5
datarange = int(years * 252)
fig = plt.figure(figsize=(20,10))
stockchart = fig.add_subplot(211)
senchart = fig.add_subplot(212)
stockchart.plot(list(StockData.index)[-datarange:],list(StockData['Close'])[-datarange:])
senchart.plot(list(StockData.index)[-datarange:], SenData[-datarange:])
plt.ylim((-5,5))


print('Apple stock returns (Y) vs Sentiment (X)')
print('Correlation = ', Correlation)
print('Linear Regression Line: Y =', alpha, '+', beta,'X')
print('Total Variation =', total_variation)
print('Unexplained Variation =', unexplained_variation)
print('Explained Variation =', explained_variation)
print('R-square = ',Rsquare)
print('-----------------------------------------------------')
print('Results of daily trading according to day sentiment')
print('Number of successful trades = ',Samedirection)
print('Probability of successful trade = ', wintrade*100,'%')
print('Number of unsuccessful trades = ',Oppdirection)
print('Probability of unsuccessful trade = ', losetrade*100, '%')
print('Days with no trades = ', Notrades)
print('-----------------------------------------------------')
print('Apple Stock Price and Sentiment Chart')
plt.show()
