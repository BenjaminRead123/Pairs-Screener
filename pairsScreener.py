#Statistical arbitrage with gaussian copula to determine best pairs


import pandas as pd
import numpy as np
import yfinance as yf
import scipy
import warnings

#check Pandas version - treating keys as positions.
warnings.simplefilter(action="ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None



percentSplit =[0.6,0.7] #training data takes up first 60%, then val takes up from 60% to 80%, so it's only 20%

def setUpBeforeSplit(data,time):
    df = data.history(period=time, auto_adjust = True)
    df.loc[:, "Log Returns"] = np.log(df.loc[:, "Close"]/df.loc[:,"Close"].shift(1))
    df = df.dropna()#need?
    return df

def setUpTrainingData(data,time):
    df = setUpBeforeSplit(data,time) 
    splitPoint1 = int(len(df)*percentSplit[0])
    trainingData = df.iloc[:splitPoint1]
    trainingData = trainingData.dropna()
    n = trainingData["Log Returns"].count()
    trainingData["Rank"] = trainingData["Log Returns"].rank(method="average")
    trainingData["Percentile"] = trainingData["Rank"] / (n + 1)
    trainingData.loc[:,"Normal Score"] = scipy.stats.norm.ppf(trainingData.loc[:,"Percentile"])
    #trainingData.loc[:, "T Score"] = scipy.stats.t.ppf(trainingData.loc[:, "Percentile"],dof)
    return trainingData 


def setUpRealOrValidationData(data,trainingData,time, Real = True): #very similar, but compares real/validation data against training percentiles
    df = setUpBeforeSplit(data,time) 
    splitPoint2 = int(len(df)*percentSplit[1])
    if Real:
        currentData = df.iloc[splitPoint2:]
    else:
        splitPoint1 =int(len(df)*percentSplit[0]) 
        currentData = df.iloc[splitPoint1:splitPoint2]
    sortedTrainingData = np.sort(trainingData.loc[:,"Log Returns"])
    n =len(sortedTrainingData) 
    percentiles = []
    for r in currentData.loc[:,"Log Returns"]:
        count = np.searchsorted(sortedTrainingData,r, side="right")
        percentile = (count+1)/(n+2)
        percentiles.append(percentile)
    currentData.loc[:,"Percentile"] = [p for p in percentiles] 
    currentData.loc[:,"Normal Score"] = scipy.stats.norm.ppf(currentData.loc[:,"Percentile"])
    return currentData


def Train(data1, data2):
    zx = data1["Normal Score"]
    zy = data2["Normal Score"]
    rho = zx.corr(zy)
    return rho

def checkHx(hx,i,position):
    if position[i-1] == 0:
        if hx[i] <= 0.05:
            position.append(1)
        elif hx[i] >= 0.95:
            position.append(-1)
        else:
            position.append(0)
    elif position[i-1] == 1:
        if hx[i] > 0.55:
            position.append(0)
        else:
            position.append(1)
    elif position[i-1] ==-1:
        if hx[i] < 0.45:
            position.append(0)
        else:
            position.append(-1)


def StrategyGaussian(train1,train2,data1,data2):
    tradesMade = True
    position = []
    position.append(0)
    returnsArray = []
    totalReturns = 1
    rho = Train(train1, train2)
    totalReturns = 1
    zy = data1["Normal Score"]
    zx = data2["Normal Score"]
    hx = scipy.stats.norm.cdf((zx - rho*zy)/np.sqrt(1-rho**2))
    for i in range(1, len(hx)):
        checkHx(hx,i,position)
    for i in range(1, len(hx)):
        data1Return = data1["Close"].iloc[i] / data1["Close"].iloc[i-1] - 1
        data2Return = data2["Close"].iloc[i] / data2["Close"].iloc[i-1] - 1
        r = position[i-1]* (data2Return-data1Return)
        returnsArray.append(r)
        totalReturns *= 1 + r
    aVol = np.std(returnsArray)*np.sqrt(252)
    if len(returnsArray) != 0:
        sharpeRatio = np.mean(returnsArray)*np.sqrt(252)/np.std(returnsArray)
    else:
        tradesMade = False
        sharpeRatio = None 
    return totalReturns-1, aVol, sharpeRatio, tradesMade

def main():
    commonPairs = [
        ["KO","PEP"],
        ["V", "MA"],
        ["XOM","CVX"],
        ["CBA.AX","NAB.AX"],
        ["BHP", "RIO"],
        ["WOW.AX","COL.AX"]
    ]
    while True:
        times = ["10y", "5y", "2y", "1y", "6mo"]
        while True:
            time = input("Time frame (10y, 5y, 2y, 1y, 6mo): ")
            if time not in times:
                print("Enter valid time.") 
            else:
                break
        
        for pair in commonPairs: 
            data0 = yf.Ticker(pair[0]) 
            data1 = yf.Ticker(pair[1])
            training0 = setUpTrainingData(data0,time)
            training1 = setUpTrainingData(data1,time)
            val0 = setUpRealOrValidationData(data0,training0,time, Real=False)
            val1 = setUpRealOrValidationData(data1,training1,time, Real=False)
            if not training0.index[0] == training1.index[0] and not val0.index[0] == val1.index[0]:
                print(f"START DATES DO NOT MATCH FOR {pair}")
                print("Add new date") 
                break 

            totalReturns, aVol, sharpeRatio, tradesMade = StrategyGaussian(training0, training1, val0, val1)
            print("---")
            print(f"{pair[0]} and {pair[1]}")
            if tradesMade:
                print("Total Returns: ", totalReturns*100, "%")
                print("Anuallised Volatility: ", aVol)
                print("Sharpe Ratio: ", sharpeRatio)
            else:
                print("No trades made")
            print("---")

      

main() 