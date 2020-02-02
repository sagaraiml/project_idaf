# -*- coding: utf-8 -*-
"""
Created on 8 Aug 2019 6pm

@author: Sagar_Paithankar
"""

import os
os.chdir(r'G:\Anaconda_CC\spyder\client_xxx')
import pandas as pd
import pymysql
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
fontP = FontProperties()
fontP.set_size('small')

def getConnection():
    db = pymysql.connect("139.59.42.147","pass","pass123","energy_consumption")
    return db
db1 = getConnection()
query = 'SELECT * FROM darksky_historical WHERE datetime_local >=' + "'" + str(datetime.now().date()-timedelta(days=14)) + "' " + 'AND datetime_local <=' +  "'" + str(datetime.now().date()-timedelta(days=12)) + "'" + 'AND plant_id = 13'
outer_df1 = pd.read_sql(query,db1)
outer_df1 = outer_df1[['datetime_local','apparent_temperature','humidity']]
outer_df1.set_index('datetime_local',inplace=True)
outer_df1 = outer_df1.resample('15T').mean()
outer_df1 = outer_df1.interpolate(method='time')
df1 = outer_df1[str(datetime.now().date()-timedelta(days=14))]
df1.reset_index(inplace=True)
df1['time'] = df1['datetime_local'].dt.time
df1 = df1[['time','apparent_temperature','humidity']]
df1.columns = ['time','apparent_temperature'+'('+str(datetime.now().date()-timedelta(days=14))+')','humidity'+'('+str(datetime.now().date()-timedelta(days=14))+')']
df2 = outer_df1[str(datetime.now().date()-timedelta(days=13))]
df2.reset_index(inplace=True)
df2['time'] = df2['datetime_local'].dt.time
df2 = df2[['time','apparent_temperature','humidity']]
df2.columns = ['time','apparent_temperature'+'('+str(datetime.now().date()-timedelta(days=13))+')','humidity'+'('+str(datetime.now().date()-timedelta(days=13))+')']
final = pd.merge(df1,df2,on='time',how='inner')
final.set_index('time',inplace=True)
db1 = getConnection()
query = 'SELECT * FROM darksky_historical WHERE datetime_local >=' + "'" + str(datetime.now().date()-timedelta(days=1)) + "' " + 'AND datetime_local <=' +  "'" + str(datetime.now().date()) + "'" + 'AND plant_id = 13'
outer_df1 = pd.read_sql(query,db1)
outer_df1 = outer_df1[['datetime_local','apparent_temperature','humidity']]
outer_df1.set_index('datetime_local',inplace=True)
outer_df1 = outer_df1.resample('15T').mean()
outer_df1 = outer_df1.interpolate(method='time')
outer_df1.reset_index(inplace=True)
query = 'SELECT * FROM darksky_forecast WHERE datetime_local >=' + "'" + str(datetime.now().date() + timedelta(days=1)) + "' " + 'AND datetime_local <=' +  "'" + str(datetime.now().date()+timedelta(days=2)) + "'" + 'AND plant_id = 13'
outer_df2 = pd.read_sql(query,db1)
outer_df2 = outer_df2[['datetime_local','apparent_temperature','humidity']]
outer_df2.set_index('datetime_local',inplace=True)
outer_df2 = outer_df2.resample('15T').mean()
outer_df2 = outer_df2.interpolate(method='time')
outer_df2.reset_index(inplace=True)
outer_df2 = outer_df2.iloc[0:93,:]
outer_df1['time'] = outer_df1['datetime_local'].dt.time
outer_df1 = outer_df1[['time','apparent_temperature','humidity']]
outer_df1.columns = ['time','apparent_temperature'+'('+str(datetime.now().date()-timedelta(days=1))+')','humidity'+'('+str(datetime.now().date()-timedelta(days=1))+')']
outer_df2['time'] = outer_df2['datetime_local'].dt.time
outer_df2 = outer_df2[['time','apparent_temperature','humidity']]
outer_df2.columns = ['time','apparent_temperature'+'('+str(datetime.now().date()+timedelta(days=1))+')','humidity'+'('+str(datetime.now().date()+timedelta(days=1))+')']
outer_df = pd.merge(outer_df1,outer_df2,on='time',how='inner')
outer_df.set_index('time',inplace=True)
plt.figure()
plt.subplot(2,2,1)
plt.plot(final[final.columns[0]], label=final.columns.tolist()[0])
plt.plot(final[final.columns[2]], label = final.columns.tolist()[2])
plt.legend(loc='upper left',bbox_to_anchor=(-0.05, 1.1),prop=fontP)
plt.subplot(2,2,2)
plt.plot(final[final.columns[1]],label=final.columns.tolist()[1])
plt.plot(final[final.columns[3]],label=final.columns.tolist()[3])
plt.legend(loc='upper right',prop=fontP)
plt.subplot(2,2,3)
plt.plot(outer_df[outer_df.columns[0]],label=outer_df.columns.tolist()[0])
plt.plot(outer_df[outer_df.columns[2]],label=outer_df.columns.tolist()[2])
plt.legend(loc='upper left',bbox_to_anchor=(-0.05, 1.1),prop=fontP)
plt.subplot(2,2,4)
plt.plot(outer_df[outer_df.columns[1]],label=outer_df.columns.tolist()[1])
plt.plot(outer_df[outer_df.columns[3]],label=outer_df.columns.tolist()[3])
plt.legend(loc='upper right',prop=fontP)
