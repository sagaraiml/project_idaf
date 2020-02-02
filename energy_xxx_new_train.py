#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sept 04 2019

@author: Sagar_Paithankar
"""

#Importing the helper file which will call all required packages and all classes
import os
path='/root/sagar/client_xxx_vsp'
#path=r'G:\Anaconda_CC\spyder\client_xxx'
os.chdir(path)
from client_xxx_support import *

#model_type will specify the type of models used. 'lightgbm' and 'xgboost' are the currently usable models
model_type=['xgboost']

#This functon will import the packages necessary visualization if 'yes' is passed in the first argument. The second
#argument if passed as 'yes' here, will import secondary packages required when experimenting with decomposing
importing('no','no')

#Specifying 'yes' here will create a stationary series for all time-series variables used including weather in 
#later steps. Stationary series will be created for load variables after taking the natural log derivative
stationarity='no'

#The below step fetches the pickle file for UDM, updates it when necessary and preprocesses it with steps like
#removing missing values, interpolation etc. The first argument here is the location of the pickle file and the 
#second argument will require specifying 'train' or 'test' depending on the purpose
df=preprocess().fetch_data('client_xxx_data','train','holidays.csv')
df = df[df['datetime'] > datetime(2017,1,1,0,0,0)]
df['month']  = df['datetime'].dt.month
df = df[df['month'].between(6,10)]
df.drop(columns = ['month'], inplace=True)
df.reset_index(drop=True, inplace=True)

#The weather plant_ids will have to be specified followed by the weight assigned in the below dictionary. The second 
#step will preprocess and set up the consolidated weighted weather parameters and derivatives. The first argument
#in the second step specifies the location of the weather pickle file and the second is the state ID
weather_weights_dict = {13:[1]}
wdf = preprocess().weather('client_xxx_weather_data',8,weather_weights_dict,stationarity)

#The below step will merge the load and weather dataframes and will create the final features and cross-features
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

#Specify the target horizons below to create separate models for each horizon. e.g. 96 will mean 96 blocks (1 day)
#ahead while 192 will refer to 2 day ahead which is valid for the blocks normally post 7am when the DA model runs
target_horizons = [192]

#The step below will train separate models using all model types. The second last argument will require a number
#which will specify the number of days to be used for validation. The last argument requires the path to a location
#to store the pickled models
for i in model_type:
    model().train(df,features,shift_features,categorical_features,target_horizons,i,5,path)

#The below step will run the inference script to generate individual forecasts for all model types and a 
#consolidated one. The step to upload the consolidated forecast to the db is commented by default
#
#
#
exec(open(path+'/'+'client_xxx_new_inf.py').read())
