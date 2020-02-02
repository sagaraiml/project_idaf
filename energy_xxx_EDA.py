# -*- coding: utf-8 -*-
"""
Created on 30 Aug 2019

@author: Sagar_Paithankar
"""
import os
path=r'G:\Anaconda_CC\Sagar\client_xxx_vsp'
#path='/root/sagar/client_xxx_gl'
os.chdir(path)
import pandas as pd
import numpy as np
from datetime import *
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

#import actuals of load combined
ori = pd.read_pickle('client_xxx_data')
ori.columns=['datetime', 'date', 'loadmet', 'tb']
combined = ori.copy()
#=========================================================================================

exp = combined[['datetime', 'loadmet']]
exp.set_index('datetime', inplace=True)
exp = exp.loadmet.resample('D').mean().to_frame()
exp['loadmet'].plot()
exp.reset_index(inplace=True)
#=========================================================================================
#checking stationarity
#define function for ADF test
from statsmodels.tsa.stattools import adfuller
def adf_test(timeseries):
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
       dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

#apply adf test on the series
adf_test(exp['loadmet'])

#define function for kpss test
from statsmodels.tsa.stattools import kpss
#define KPSS
def kpss_test(timeseries):
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c')
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print (kpss_output)
#apply KPSS test on the series
kpss_test(exp['loadmet'])

exp['diff'] = exp['loadmet'] - exp['loadmet'].shift(1)
exp['diff'].plot()


exp = exp.iloc[-1200:, :]
import statsmodels.api as sm
exp.set_index('datetime', inplace=True)
res = sm.tsa.seasonal_decompose(exp)
resplot = res.plot()
exp.reset_index(inplace=True)

#=========================================================================================
dum = combined[['datetime', 'loadmet']]
dum.set_index('datetime', inplace=True)
dum = dum.loadmet.resample('D').mean().to_frame()
import statsmodels.api as sm
res = sm.tsa.seasonal_decompose(dum)
resplot = res.plot()