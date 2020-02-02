#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sept 04 2019

@author: Sagar_Paithankar
"""

import os
path='/root/sagar/client_xxx_vsp'
#path=r'G:\Anaconda_CC\spyder\client_xxx'
os.chdir(path)
from client_xxx_support import *

#model_type=['lightgbm','catboost','xgboost']
model_type=['xgboost']
importing('yes','no')

stationarity='no'

df=preprocess().fetch_data('client_xxx_data','inference','holidays.csv')

weather_weights_dict = {13:[1]}
wdf = preprocess().weather('client_xxx_weather_data',8,weather_weights_dict,stationarity)

df = preprocess().feature_eng(df,wdf,stationarity)


#Specify which features do you want to use in the model
features = ['datetime','load','lag1','lag2','lag3','hour_mean','sin_doy','dow','tb_aptemp',\
            'sdtbrm', '3tbrm', 'sdtbrm2','temperature_ewm','apparent_temperature','dew_point',\
            'aptemp_mean_12h','aptemp_mean_6h','doy','tm4','load_wm3h','month','year','hour',\
            'sin_tb','dp_ewm','3tbrmw','lagcwm','load_wm12h',\
            'temperature','wt','load_aptemp','load_aptemp_ewm6','load_aptemp_ewm12',\
            'tm','tm2','tm3','tb_load']
# removed 'lag5','lag4wm',
shift_features = ['temperature_ewm','dp_ewm','temperature','apparent_temperature','dew_point','aptemp_mean_6h',\
                'aptemp_mean_12h','dow','tb_aptemp','doy','month',\
                'wt','tm','tm2','tm3','tm4']

categorical_features = ['dow','month','year','hour']

target_horizons = [192]
print(model_type)
combined=model().combine(df,features,shift_features,categorical_features,target_horizons,model_type,path)
#print(combined)
combined['final']['version'] = 'vsp'
result=combined['final']
result.to_excel((str((datetime.now()).date() + timedelta(days=1)))+' client_xxx DA forecast'+'.xlsx')
print('model forecast saved to excel in folder')
try:
    load().store_forecast(combined['final'],'client_xxx_dayahead_forecast')
    
    try:
        load().store_forecast_api(combined['final'],'client_xxx_dayahead_forecast','vsp')
    except Exception as e:
        print("Failure in Model storing result in API Phase due to : {a}".format(a=e))
        misc().send_error_notification("Failure in Model storing result in API Phase due to : {a}".format(a=e)) 

except Exception as e:
    print("Failure in Model storing in db Phase due to : {a}".format(a=e))
    misc().send_error_notification("Failure in Model storing in db Phase due to : {a}".format(a=e))
    exit

misc().send_notification('Congratulation Sagar_Paithankar new model generated result')