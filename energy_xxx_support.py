#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sept 04 2019

@author: Sagar_Paithankar
"""

import os
import requests, json
import pandas as pd
import numpy as np
from datetime import *
import pickle
import logging
from scipy import signal
import pymysql
from sqlalchemy import create_engine
import time
from tensorboard_logger import configure, log_value
import traceback
from math import ceil

pd.options.mode.chained_assignment = None

os.environ['TZ'] = 'Asia/Calcutta'

#Importing addtional graphical and statsmodels packages
def importing(graphics,extras):
    try:
        if graphics=='yes':
            import matplotlib.pyplot as plt
            import matplotlib.image as image
            import seaborn as sns
            sns.set()
            print('imported matplotlib and seaborn libraries')
        else:
            print('no visualization libraries imported')           
    except Exception as e:
        print("Error in importing library {b} due to : {a}".format(a=e,b=model)) 
    finally:
        pass    
     
    if extras=='yes':
        try:
            import statsmodels.formula.api as smf            
            import statsmodels.tsa.api as smt
            import statsmodels.api as sm
            import scipy.stats as scs
            print('imported statsmodels and scipy libraries')
        except Exception as e:
            print("Error in importing library {b} due to : {a}".format(a=e,b=model)) 
        finally:
            pass
    else:
        pass        
    
class preprocess:
    #Unpickle, update and re-pickle historical data
    def fetch_data(self,a,obj,holiday_file_location):
        print('Importing pickle file for UDM load data...')
        try:
            infile1 = open(a,'rb')
            df1 = pickle.load(infile1)
            infile1.close()
            df1 = df1.drop_duplicates('datetime', keep='first')
            print('pickle file imported')
        except Exception as e:
            print("Error in fetching load pickle file : {a}".format(a=e))
            misc().send_error_notification("Error in fetching load pickle file : {a}".format(a=e))
            
        print('Fetching recent data to check for updates...')    
        try:
            rest=load().get_urd_data(str(pd.to_datetime(datetime.now()).date() - timedelta(days=5)),str((datetime.now()+timedelta(days=1)).date()))
#            rest=load().get_sldc_data(str(pd.to_datetime(datetime.now()).date() - timedelta(days=5)),str((datetime.now()+timedelta(days=1)).date()))
        except Exception as e:
            print("Error in fetching recent data: {a}".format(a=e)) 
            misc().send_error_notification("Error in fetching recent data: {a}".format(a=e))
        finally:
            pass      
        if (df1.tail(1)['datetime']).reset_index(drop=True)[0] < ((rest.tail(1)['datetime']).reset_index(drop=True)[0]):
            print('Updating data...')
            try:   
                df=load().get_urd_data(str((df1.tail(1)['datetime']).reset_index(drop=True)[0].date()),str((datetime.now()+timedelta(days=1)).date()))
            except Exception as e:
                print("Error in fetching training data updates : {a}".format(a=e))
                misc().send_error_notification("Error in fetching training data updates : {a}".format(a=e))
            finally:
                pass    
            try:
                print('Processing and pickling updated data...')
                df=df.reset_index(drop=True)
                df.set_index('datetime', inplace=True)
                df = df.resample('15T').mean().reset_index()
                df['tb'] = df.datetime.apply(lambda x : ((x.hour*60 + x.minute)//15+1))
                df.set_index('datetime', inplace=True)
                df['load'] = df.load.interpolate(method='time')
                df.reset_index(inplace=True)
                df['date'] = df.datetime.dt.date
                df = df[['datetime','date','load','tb']]
                df=df1.append(df)
                df.reset_index(inplace=True,drop=True)
                pkl = open(a,'wb')
                pickle.dump(df,pkl)
                df['load'][df['load'] < 0] = 0
                df.loc[df.load < df.load.quantile(q=0.0005), 'load'] = df.load.quantile(q=0.0005)
                df.reset_index(inplace=True,drop=True)
                print('Finished processing updated load data')
            except Exception as e:
                print("Error in pickling updated training data : {a}".format(a=e))
                misc().send_error_notification("Error in pickling updated training data : {a}".format(a=e))
            finally:
                pass  
        
        else:
            print('Load data already updated. Proceeding without any changes to pickled file...')
            df=df1  
            df.loc[df.load < df.load.quantile(q=0.0005), 'load'] = df.load.quantile(q=0.0005)
            df.reset_index(inplace=True,drop=True)
        del df1
        if obj == 'inference':
            df=df[df.datetime>=misc().get_date(-20)].reset_index(drop=True)
            df = pd.merge(df, pd.DataFrame(pd.date_range(start=misc().get_date(-20), end=misc().get_date(2),\
            freq='15min'), columns=['datetime']), on='datetime', how='outer')
            df.reset_index(inplace=True,drop=True)
        else:
            print('Removing holidays...')
            df=misc().remove_holiday(df, holiday_file_location)
            df.reset_index(inplace=True,drop=True)
            print('Load data prepared')
        return df
    
    #Un-pickle historical weather, update and consolidate using weights assigned in dictionary and re-pickle
    def weather(self,a,plant_id,location_weights_dict,stationarity):
        print('Fetching weather pickle file')
        try:
            infile1 = open(a,'rb')
            wdf2 = pickle.load(infile1)
            infile1.close()
        except Exception as e:
            print("Error in fetching weather pickle file : {a}".format(a=e))
            misc().send_error_notification("Error in fetching weather pickle file : {a}".format(a=e))
            exit
        
        x=[]
        ea2={}
        print('Fetching weather data from api')    
        try:
            for i in list(location_weights_dict.keys()):
                wdf2.reset_index(inplace=True, drop=True)
                wdf = weather().combiner1((str(wdf2.datetime[len(wdf2)-3].date()-timedelta(days=20))), str((datetime.now() + timedelta(days=20)).strftime("%Y-%m-%d")), i, plant_id,'darksky_actual')
                x.append(str(wdf.datetime[len(wdf)-3].date()))
                wdf1=weather().combiner(str((wdf.tail(1)['datetime']).reset_index(drop=True)[0].date()), str((datetime.now() + timedelta(days=20)).strftime("%Y-%m-%d")), i,plant_id,'combiner_forecast')
                wdf=wdf.append(wdf1)
                del wdf1
                wdf.rename(columns={'apparent_temperature': 'apparent_temperature_{}'.format(i),'temperature': 'temperature_{}'.format(i),'humidity': 'humidity_{}'.format(i),'dew_point': 'dew_point_{}'.format(i),'wind_speed': 'wind_speed_{}'.format(i),'cloud_cover': 'cloud_cover_{}'.format(i)}, inplace=True)
                ea2[i]=wdf
                print('location {} data added'.format(i))
                print('last date for actual weather data for plant id {} is {}'.format(i,min(x)))  
        except Exception as e:
            print("Error in fetching weather api data for plant {b} due to : {a}".format(a=e,b=i))
            misc().send_error_notification("Error in fetching weather api data for plant {b} due to : {a}".format(a=e,b=i))
            exit
        finally:    
            pass    
    
        print('Processing load-weighted weather averages...')
        try:
            wdf=(pd.DataFrame(ea2[list(location_weights_dict.keys())[0]]['datetime'].reset_index(drop=True))).merge((ea2\
            [list(location_weights_dict.keys())[0]][[e for e in list(ea2[list(location_weights_dict.keys())[0]])\
            if e not in ('datetime')]].reset_index(drop=True)*(location_weights_dict[list(location_weights_dict.keys())[0]]\
            [0])), how='outer', left_index=True, right_index=True)
            for i in list(location_weights_dict.keys())[1:]:
                wdf=pd.merge(wdf, ((pd.DataFrame(ea2[i]['datetime'].reset_index(drop=True))).merge((ea2[i][[e for e in list(ea2[i])\
                if e not in ('datetime')]].reset_index(drop=True)*(location_weights_dict[i][0])), how='outer', left_index=True, right_index=True)),\
                on='datetime', how='inner')    
                for j in [e for e in list(wdf2) if e not in ('datetime')]:            
                    wdf[j+"_{}".format(list(location_weights_dict.keys())[0])]=\
                    wdf[j+"_{}".format(list(location_weights_dict.keys())[0])] + wdf[j+"_{}".format(i)]
                    wdf = wdf.drop_duplicates('datetime', keep='first').reset_index(drop=True)
                print('features added for location {}'.format(i))    
            wdf = wdf.drop_duplicates('datetime', keep='first').reset_index(drop=True)
            wdf.rename(columns={'apparent_temperature_{}'.format(list(location_weights_dict.keys())[0]): 'apparent_temperature',\
            'temperature_{}'.format(list(location_weights_dict.keys())[0]): 'temperature','humidity_{}'.format(list(location_weights_dict.keys())[0]): 'humidity',\
            'dew_point_{}'.format(list(location_weights_dict.keys())[0]): 'dew_point','wind_speed_{}'.format(list(location_weights_dict.keys())[0]): 'wind_speed',\
            'cloud_cover_{}'.format(list(location_weights_dict.keys())[0]): 'cloud_cover'}, inplace=True)
            wdf=wdf[['datetime','apparent_temperature','temperature','humidity','dew_point','wind_speed','cloud_cover']]
            wdf.reset_index(inplace=True, drop=True)
        except Exception as e:
            print("Error in processing load-weighted weather averages : {a}".format(a=e))
            misc().send_error_notification("Error in processing load-weighted weather averages : {a}".format(a=e))
            exit
        
        print('Pickling updated weather actuals...')
        try:
            wdf1=wdf[wdf.datetime.dt.date<=pd.to_datetime(pd.DataFrame(x).min()[0]).date()]
            wdf1=wdf2.append(wdf1).reset_index(drop=True)
            wdf1 = wdf1.drop_duplicates('datetime', keep='first').reset_index(drop=True)
            pkl = open(a,'wb')
            pickle.dump(wdf1,pkl)
            wdf=wdf2.append(wdf).reset_index(drop=True)
            wdf = wdf.drop_duplicates('datetime', keep='first').reset_index(drop=True)
        except Exception as e:
            print("Error in process of pickling updated weather actuals : {a}".format(a=e))
            misc().send_error_notification("Error in process of pickling updated weather actuals : {a}".format(a=e))
            
        print('Adding derivatives...')
        try:
            wdf['humidity']=((6.11*10.0**(7.5*wdf.dew_point/(237.7+wdf.dew_point)))*100)/((wdf.temperature + 273.15)*461.5)
            wdf['humidex'] = weather().humidex(wdf)
            wdf['RH'] = weather().calculate_RH(wdf)
            wdf['wci'] = 13.12 + (0.6215*wdf.temperature) - (11.37*(wdf.wind_speed**0.16)) + (0.3965*wdf.temperature*(wdf.wind_speed**0.16))
            if stationarity == 'yes':
                print('adding stationarity...')
                for i in list(wdf)[1:]:
                    print(i)
                    wdf[i]=(wdf[i]) - (wdf[i]).ewm(span=288).mean()
            else:
                pass
        except Exception as e:
            print("Error in adding final weather derivatives : {a}".format(a=e))
            misc().send_error_notification("Error in adding final weather derivatives : {a}".format(a=e))
        print('Final derivatives added')    
        wdf.dropna(inplace=True)
        wdf.reset_index(inplace=True, drop=True)
        print('Weather data transformation complete')
        return wdf
    
    #Merge load and weather dataframes and add features and cross-features
    def feature_eng(self,df,wdf,stationarity):
        if stationarity == 'yes':
            try:
                print('adding stationarity...')
                df['load1']=np.log(df.load)
                df['em'] = df['load1'].ewm(span=672).mean()
                df['load3']=df.load1-df.em
                df['load']=df.load3
                df = df[['datetime','date','load','tb']]
            except Exception as e:
                print("Error in pickling updated training data : {a}".format(a=e))
            finally:
                pass     
        else:
            pass    
        #Adding time based features
        print('Adding time based features...')
        try:
            df = df.drop_duplicates('datetime', keep='first')
            df['date'] = df.datetime.dt.date
            df['tb'] = df.datetime.apply(lambda x : ((x.hour*60 + x.minute)//15+1))
            df['dow'] = df.datetime.dt.dayofweek
            df['weekend'] = np.where(df['dow'] > 5, 1, 0)
            df['hour'] = df.datetime.dt.hour + 1
            df['doy'] = df.datetime.dt.dayofyear
            df['month'] = df.datetime.dt.month
            df['year']=df.datetime.dt.year
            df['wom'] = df.datetime.apply(lambda x : week_of_month(x))
            
            #Adding lags
            print('Adding lags...')
            df['lag1b'] = df.load.shift(1)
            df['lag2b'] = df.load.shift(2)
            df['lag3b'] = df.load.shift(3)
            df['lag1'] = df.load.shift(96)
            df['lag2'] = df.load.shift(192)
            df['lag3'] = df.load.shift(288)
            df['lag4'] = df.load.shift(384)
            df['lag5'] = df.load.shift(480)
            df['lag6'] = df.load.shift(576)
            
            #Adding ewms for load
            print('Adding ewms for load...')
            df['load_wm2h'] = df['load'].ewm(span=8).mean()
            df['load_wm3h'] = df['load'].ewm(span=12).mean()
            df['load_wm5h'] = df['load'].ewm(span=20).mean()
            df['load_wm24h'] = df['load'].ewm(span=96).mean()
            
            #Adding sine derivatives to time variables
            print('Adding sine derivatives to time variables...')
            df['sin_doy'] = misc().sinwave(df, 'doy', 365)
            df['sin_tb'] = misc().sinwave(df, 'tb', 96)
            df['sin_dow'] = misc().sinwave(df, 'dow', 7)
            
            #Adding the hourly running mean for load
            print('Adding the hourly running mean for load...')
            hour_mean = df.groupby(['date','hour'])['load'].mean().reset_index()
            hour_mean = hour_mean.rename(columns = {'load':'hour_mean'})
            df = pd.merge(df, hour_mean, on=['date','hour'], how='left')
        except Exception as e:
            print("Error in adding time based features for load : {a}".format(a=e))    
            misc().send_error_notification("Error in adding time based features for load : {a}".format(a=e))
            exit
        
        print('Merging load and weather databases...')    
        try:
            df = pd.merge(df, wdf, on='datetime', how='inner')
        except Exception as e:
            print("Error in merging load and weather databases : {a}".format(a=e))    
            misc().send_error_notification("Error in merging load and weather databases : {a}".format(a=e))
            exit
        
        print('Adding final features...')
        try:
            df['tb_aptemp'] = df.tb * df.apparent_temperature
            df['load_aptemp'] = df.load * df.apparent_temperature
            df['aptemp_mean_6h'] = df['apparent_temperature'].ewm(span=24).mean()
            df['aptemp_mean_12h'] = df['apparent_temperature'].ewm(span=48).mean()
            df['temperature_ewm'] = df['temperature'].ewm(span=48).mean()
            df['dp_ewm'] = df['dew_point'].ewm(span=48).mean()
            df['hm12'] = df['humidity'].ewm(span=48).mean()
            df['wsp_12'] = df['wind_speed'].ewm(span=48).mean()
            df['cc_12'] = df['cloud_cover'].ewm(span=48).mean()
            df['tb_load'] = df.tb * df.load
            df['sdtbrm'] = (df.load + df.load.shift(480))/2
            df['sdtbrm2'] = (df.load + df.load.shift(576))/2
            df['3tbrm'] = (df.load + df.lag1 + df.lag2)/3
            df['3tbrmw'] = df['3tbrm'].ewm(span=48).mean()
            df['factor'] = ((df.load-df.lag1)/1000)*(df.temperature_ewm - df.temperature_ewm.shift(96))*(df.aptemp_mean_12h - df.aptemp_mean_12h.shift(96))
            df['lag1wm'] = df['lag1'].ewm(span=48).mean()
            df['lag2wm'] = df['lag2'].ewm(span=48).mean()
            df['lag3wm'] = df['lag3'].ewm(span=48).mean()
            df['lag4wm'] = df['lag4'].ewm(span=48).mean()
            df['lag5wm'] = df['lag5'].ewm(span=48).mean()
            df['lag6wm'] = df['lag6'].ewm(span=48).mean()
            df['load_aptemp_ewm6'] = df['load_aptemp'].ewm(span=24).mean()
            df['load_aptemp_ewm12'] = df['load_aptemp'].ewm(span=48).mean()
            df['load_aptemp_ewm24'] = df['load_aptemp'].ewm(span=96).mean()
            df['lagcwm'] = (df.lag1wm + df.lag2wm + df.lag3wm + df.lag4wm)/4
            df['lag4avg'] = (df.lag1 + df.lag2 + df.lag3 + df.lag4)/4
            df['lag_df'] = ((df['lag1'] - df['lag2'])+(df['lag2'] - df['lag3']))/2
            df['lag_df1'] = ((df['lag2'] - df['lag3'])+(df['lag3'] - df['lag4']))/2
            df['load_wm12h'] = df['load'].ewm(span=48).mean()
            df['wt']=df.dow*df.hour
            df['tm']=df.temperature*df.month
            df['tm2']=(df.temperature**2)
            df['tm3']=(df.aptemp_mean_12h**2)
            df['tm4']=df.aptemp_mean_12h*df.month
            df['th']=df.temperature*df.hour
            df['th2']=(df.temperature**2)*df.hour
            df['th3']=(df.temperature**3)*df.hour
            df['tb_load']=df.load*df.tb
            print('Feature engineering completed successfully')
        except Exception as e:
            print("Error in adding final features : {a}".format(a=e))    
            misc().send_error_notification("Error in adding final features : {a}".format(a=e))
            exit
        return df
    
class model:
    
    #Train multiple specified models with arguments on stationarity, validation period etc.
    def train(self,df,features,shift_features,categorical_features,target_horizons,model_type,validation_period,path):
        df3=df.copy()
        for i in target_horizons:
            try:
                print('Training on {a} for {b} day ahead'.format(a=model_type,b=int(i/96)))
                df4 = df3[features]
                df4['target'] = df4.load.shift(-i) 
            
                for j in shift_features:
                    df4[j] = df4[j].shift(-i)
                print('Shifting selected features: by {b} blocks'.format(b=i))
            
                df4.dropna(inplace=True)
                x, y = df4.drop('target',1).copy(), df4[['datetime','target']].copy()
            
                x_train, y_train = x[x['datetime'] < misc().get_date(0) + ' 00:00:00'], y[y['datetime'] < misc().get_date(0) + ' 00:00:00']
            
                df4=df4.reset_index(drop=True)
                ind2=df4[df4['datetime'] >= pd.to_datetime((df4['datetime'][len(df4['datetime'])-1]).date()- timedelta(days=validation_period))]
                ind2=ind2.reset_index(drop=True)
                xv, yv = ind2.drop('target',1).copy(), ind2[['datetime','target']].copy()
            
                x_train=x_train[x_train['datetime']<=pd.to_datetime((df4['datetime'][len(df4['datetime'])-1]).date()- timedelta(days=validation_period))]
                y_train=y_train[y_train['datetime']<=pd.to_datetime((df4['datetime'][len(df4['datetime'])-1]).date()- timedelta(days=validation_period))]
            
                x_train, y_train = x_train.iloc[:,1:], y_train.iloc[:,-1]
                xv, yv = xv.iloc[:,1:], yv.iloc[:,-1]
                
                categorical_features_indices=np.array(list(misc().column_index(x_train, categorical_features)))
            except Exception as e:
                print("Error setting up training data structure for target horizon {b} due to : {a}".format(a=e,b=i))
                print(traceback.format_exc())
                misc().send_error_notification("Error setting up training data structure for target horizon {b} due to : {a}".format(a=e,b=i))
                exit
            
            try:
                if model_type == 'lightgbm':
                    import lightgbm as lgb
                    dtrain = lgb.Dataset(data = x_train, label=y_train.values.ravel())    
                    dval = lgb.Dataset(data = xv, label=yv.values.ravel(), reference=dtrain)
                    
                    if i == 96:
                        params = {'task': 'train', 'boosting_type': 'gbdt', 'objective': 'mape', 'metric': 'l1',\
                        'seed': 0,'num_leaves': 90, 'learning_rate': 0.08, 'feature_fraction': 0.8, 'max_depth': 10,\
                        'num_iterations': 2500, 'reg_lambda': 4}
                    else:
                        """
                        params = {'task': 'train', 'boosting_type': 'gbdt', 'objective': 'regression_l2', 'metric': 'l1','cat_smooth': 10,\
                        'seed': 0, 'min_data_in_leaf': 50,'num_leaves': 90, 'learning_rate': 0.08,\
                        'feature_fraction': 0.58, 'max_depth': 11,'num_iterations': 5000, 'reg_lambda': 5}
                        """
                        params = {'task': 'train', 'boosting_type': 'gbdt', 'objective': 'regression_l1', 'metric': 'l1','cat_smooth': 10,\
                        'seed': 0, 'min_data_in_leaf': 50,'num_leaves': 90, 'learning_rate': 0.08,\
                        'feature_fraction': 0.58, 'max_depth': 11,'num_iterations': 2000, 'reg_lambda': 5}                        
                        
                        model = lgb.train(params, dtrain, categorical_feature= categorical_features, valid_sets=dval, early_stopping_rounds=10)
                    #lgb.plot_importance(model, importance_type = 'gain')
                    feat_imp = pd.DataFrame(pd.Series(model.feature_importance(importance_type='gain'), index=x_train.columns),columns=['importance']).sort_values('importance', ascending=False)
                    feat_imp.sort_values('importance', ascending= False, inplace=True)
                    feat_imp['importance'] = (feat_imp['importance']/feat_imp['importance'].sum()) * 100
                    print(feat_imp)
            
                elif model_type == 'xgboost':
#==========================================================================================
                    import matplotlib.pyplot as plt
                    import xgboost as xgb
                    time_log = datetime.now().strftime("%y_%m_%d_%H_%M_%S")
                    print('this is the time we are running ',time_log)
                    NAME = 'xgb_{a}'.format(a=time_log)
                    configure('logs/{a}'.format(a=NAME))
                    data_matrix = xgb.DMatrix(x_train, label=y_train)
                    data_matrixv = xgb.DMatrix(xv, label=yv)
                    params = {'n_estimators':200,'objective':'reg:linear','booster':'gbtree','max_depth':10,'learning_rate':0.04,\
                    'colsample_bytree':0.6,'colsample_bylevel':0.6,'subsample':0.8,'min_child_weight':1,'gamma':1,\
                    #gblinear params\
                    'lambda':0.1, 'alpha':0.1}
                    model = xgb.train(params, data_matrix,num_boost_round=2000,evals=[(data_matrix,"Train"),(data_matrixv, "Test")],early_stopping_rounds=10, callbacks=[misc.logspy])
#                    #this is for gbtree
#                    xgb.plot_importance(model, max_num_features=50, height=0.8)
#                    feature_important = model.get_score(importance_type='weight')
#                    keys = list(feature_important.keys())
#                    values = np.sort(list(feature_important.values()))
#                    data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
#                    data.plot(kind='barh')
#                    plt.barh(keys, values)
#                    plt.savefig(f'feature_important_{datetime.now().strftime("%y_%m_%d_%H_%M_%S")}.png') #dpi=200, 
#                    plt.show()
#                    thresholds = np.sort(xgb.feature_importances_)
#                    from sklearn.feature_selection import SelectFromModel
#==========================================================================================
                elif model_type == 'catboost':
                    from catboost import CatBoostRegressor
                    from catboost import Pool
                    data = Pool(x_train, label=y_train, feature_names= list(x_train),cat_features=categorical_features_indices)
                    valid_data = Pool(xv, label=yv, feature_names= list(x_train),cat_features=categorical_features_indices)
                    
                    gb = CatBoostRegressor(loss_function='RMSE',eval_metric='MAE',learning_rate=0.02,l2_leaf_reg=3,
                        bootstrap_type='Bernoulli',subsample=0.8,depth=10,rsm=0.6,nan_mode='Forbidden',
                        allow_const_label=True)
                    
                    model = gb.fit(data,eval_set=valid_data,early_stopping_rounds=15,verbose_eval=True)     

                else:
                    print("Invalid Model")
                    misc().send_error_notification("Invalid Model Specified")
            except Exception as e:
                print("Error setting up final {c} model data matrices for target horizon {b} due to : {a}".format(a=e,b=i,c=model_type))
                misc().send_error_notification("Error setting up final {c} model data matrices for target horizon {b} due to : {a}".format(a=e,b=i,c=model_type))
            try:
                pickle.dump(model, open(path+'/'+model_type+'_'+str(i)+'.pkl', 'wb'))
                misc().send_notification('client_xxx {} model trained successfully'.format(model_type))
            except Exception as e:
                print("Error pickling final {c} model data matrices for target horizon {b} due to : {a}".format(a=e,b=i,c=model_type))
                print(traceback.format_exc())
                misc().send_error_notification("Error pickling final {c} model data matrices for target horizon {b} due to : {a}".format(a=e,b=i,c=model_type))                

    #Predict function for multiple specified models
    def predict(self,df,features,shift_features,categorical_features,target_horizons,model_type,path):
        df3=df.copy()
        result=pd.DataFrame()
        for i in target_horizons:
            try:
                print('Running inference on {a} for {b} day ahead'.format(a=model_type,b=int(i/96)))
                df4 = df3[features] 
            
                for j in shift_features:
                    df4[j] = df4[j].shift(-i)
                print('Shifting selected features: by {b} blocks'.format(b=i))
            
                df4.dropna(inplace=True)
                x, y = df4.iloc[:,:-1], df4.iloc[:,[0,-1]]
                dtlist, forecast = [], []
        
                df4 = df4[(df4['datetime'] >= misc().get_date(-1) + " 00:00:00") & (df4['datetime'] <= misc().get_date(0) + " 23:45:00")]
                dtlist.extend(df4.datetime.tolist())
                x_inf =df4.drop('datetime',1).copy()
                
                categorical_features_indices=np.array(list(misc().column_index(x_inf, categorical_features)))
            except Exception as e:
                print("Error setting up inference data structure for target horizon {b} due to : {a}".format(a=e,b=i))
                misc().send_error_notification("Error setting up inference data structure for target horizon {b} due to : {a}".format(a=e,b=i))
            
            try:
                print('Unpickling model and running prediction...')
                if model_type == 'lightgbm':
                    import lightgbm as lgb
                    model = pickle.load(open(path+'/'+model_type+'_'+str(i)+'.pkl', 'rb'))
                    y_pred = model.predict(x_inf, categorical_feature= categorical_features_indices, num_iteration=model.best_iteration)
                    forecast.extend(y_pred.tolist())

                elif model_type == 'xgboost':
                    import xgboost as xgb
                    xinf = xgb.DMatrix(x_inf)
                    model = pickle.load(open(path+'/'+model_type+'_'+str(i)+'.pkl', 'rb'))
                    y_pred = model.predict(xinf)
                    forecast.extend(y_pred.tolist())    

                elif model_type == 'catboost':
                    from catboost import CatBoostRegressor
                    from catboost import Pool
                    cinf = Pool(x_inf, cat_features=categorical_features_indices)
                    model = pickle.load(open(path+'/'+model_type+'_'+str(i)+'.pkl', 'rb'))
                    y_pred = model.predict(cinf)
                    forecast.extend(y_pred.tolist())        

                else:
                    print("Invalid Model. Forecast not generated")
            except Exception as e:
                print("Error in prediction for final {c} model for target horizon {b} due to : {a}".format(a=e,b=i,c=model_type))
                misc().send_error_notification("Error in prediction for final {c} model for target horizon {b} due to : {a}".format(a=e,b=i,c=model_type))
            
            try:
                print('Preparing final DataFrame...')
                res = pd.DataFrame()
                res['datetime'] = dtlist
                res['datetime'] = res.datetime + timedelta(days=int(i/96))
                res['date'] = res.datetime.dt.date
                res['start_time'] = res.datetime.dt.time
                res['end_time'] = (res.datetime + timedelta(minutes=15)).dt.time
                res['forecast'] = forecast
                res['horizon'] = 1
                res['version'] = 'v3'
                res['type'] = 1
                res['created_at'] = (datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
                result = result.append(res)
                print('Appending complete for horizon {}...'.format(i))
            except Exception as e:
                print("Error pre final {c} model data matrices for target horizon {b} due to : {a}".format(a=e,b=i,c=model_type))
                misc().send_error_notification("Error pre final {c} model data matrices for target horizon {b} due to : {a}".format(a=e,b=i,c=model_type))
        result = result.drop_duplicates('datetime', keep='first')
        result = result[(result['datetime'] >= misc().get_date(1) + ' 00:00:00') & (result['datetime'] <= misc().get_date(1) + ' 23:45:00')]
        result['timeblock']=result['datetime'].apply(lambda x: ((x.hour*60+x.minute)//15+1))
        result = result[['date','start_time','end_time','forecast','horizon','version', 'type', 'created_at']]
        result = result.dropna()
        if len(result)==96:
            print('Forecast generated successfully')
        else:
            print('Forecast unsucessful/partially successful. The length of the dataframe is {}'.format(len(result)))
            misc().send_error_notification('Forecast unsucessful/partially successful. The length of the dataframe is {}'.format(len(result)))
        return result             
    
    #Results in a dictionary with individual and consolidated model results
    def combine(self,df,features,shift_features,categorical_features,target_horizons,model_list,path):
        ea={}
        try:
            print('Prediction Phase Start...')
            for i in model_list:
                result = model().predict(df,features,shift_features,categorical_features,target_horizons,i,path)
                ea[i] = result
                result.to_csv('testforecast.csv')
            combined=ea[list(ea)[0]].copy()
        except Exception as e:
                print("Prediction Phase Failure due to : {a}".format(a=e,b=i))
                misc().send_error_notification("Prediction Phase Failure due to : {a}".format(a=e,b=i))
                exit
        print('Prediction Phase completed successfully')
        try:
            print('Model Combination Phase Start...')
            if len(model_list) < 2 and len(model_list) >= 1:
                combined['forecast']=combined['forecast']
            else:
                combined['forecast']=(combined['forecast'] + ea[list(ea)[1]]['forecast'])/2
            if len(model_list) > 2:
                for i in range(2,len(model_list)):
                    combined['forecast']=(combined['forecast'] + ea[list(ea)[i]]['forecast'])/2
            else:
                pass
            ea['combined'] = combined.copy()    
            ea['final'] = combined.copy()
#            ea['lightgbm']['version']='vrl'
            ea['xgboost']['version']='vrx'
#            ea['catboost']['version']='vrc'
            ea['combined']['forecast']=signal.savgol_filter(ea['final']['forecast'], 25, 5)
        except Exception as e:
            print("Failure in Model Combination Phase due to : {a}".format(a=e,b=i))    
            misc().send_error_notification("Failure in Model Combination Phase due to : {a}".format(a=e,b=i))
        print('Final forecasts generated for all models successfully')
        return ea

class DB(object):
        
    def getConnection(self):
        db = pymysql.connect("139.59.42.147","pass","pass123","energy_consumption")
        return db
    
    def getEngine(self):
        engine = create_engine('mysql+pymysql://pass:pass123@139.59.42.147:3306/energy_consumption', echo=False)
        return engine
        
class load:
    
    def get_urd_data(self,sdt, tdt):
        table_name = 'client_xxx_data'
        db_connection = pymysql.connect(host="139.59.42.147",user="pass",password="pass123",db="energy_consumption")
        SQL = "SELECT * FROM " + table_name + " WHERE date BETWEEN '" + sdt + "' AND '" + tdt + "'"
        df = pd.read_sql(SQL, con=db_connection)
        df['datetime'] = pd.to_datetime(df.date.astype(str) + ' ' + df.time_slot.str[:5])
        df['date'] = df.datetime.dt.date
        df['load']=df.block_load
        df['tb'] = df.datetime.apply(lambda x : ((x.hour*60 + x.minute)//15+1))
        df = df[['datetime','date','load','tb']]
        return df
    
    def get_sldc_data(self, sdt, tdt):
        table_name='tbl_actual_demand_met'
        db_connection = pymysql.connect(host="139.59.42.147",user="pass",password="pass123",db="energy_consumption")
        SQL = "SELECT * FROM " + table_name + " WHERE date BETWEEN '" + sdt + "' AND '" + tdt + "' AND discom_name='client_xxx'"
        df = pd.read_sql(SQL, con=db_connection)
        df['time'] = pd.to_datetime(df.timeslot).dt.time
        df['datetime'] = pd.to_datetime(df.date.astype(str) + ' ' + df.time.astype(str))
        df = df[['datetime','actual_demand_met_mw']]
        df = df.rename(columns = {'actual_demand_met_mw':'load'})
        df = df.set_index('datetime').resample('15min').mean().reset_index()
        df.drop_duplicates('datetime', inplace=True)
        return df
    
    def store_forecast(self,df,table):
        engine = DB().getEngine()
        tb = '{}'.format(table)
        try:
            df.to_sql(con=engine,name = tb,if_exists="append",index=False)
        except:
            print("Error in DB store")
            exit
        return df

    def get_token(self):
        username = "tech@dummy.com"
        password = "dentintheuniverse"
        base_url = "https://apis.dummy.com/api"          
        try:
            headers = {}
            params = {'username': username, 'password': password}
            
            api_token = base_url + "/get-token"
            
            r = requests.post(url=api_token, headers=headers, params=params)
            return r.json()
    
        except Exception as e:
            print("Error in getting token : {a}".format(a=e))
            
    def get_api_data(self, sdt, tdt, params, headers, api_get):
        
        headers['token']=load().get_token()['access_token']
        
        try:           
            r = requests.get(url=api_get, headers=headers, params=params)
            return r.json()
        
        except Exception as e:
                print("Error in fetching api data : {a}".format(a=e))
#                
    def get_urd_data_api(self,sdt, tdt):
        api_get = "https://apis.dummy.com/api/load/getActual"
        headers = {'token':load().get_token()['access_token'],'Content-Type': 'application/x-www-form-urlencoded'}
        params = { "from-date":sdt,
                "to-date":tdt,
                "client_id":51,
                "discom":"client_xxx",
                "type": "urd"
                }

        df = pd.DataFrame(load().get_api_data(sdt, tdt, params, headers, api_get)['data'])   
        df['datetime'] = pd.to_datetime(df.datetime.astype(str))
#        df['datetime'] = pd.to_datetime(df.date.astype(str) + ' ' + df.time_slot.str[:5])
        df['date'] = df.datetime.dt.date
        df['tb'] = df.datetime.apply(lambda x : ((x.hour*60 + x.minute)//15+1))
        df = df[['datetime','date','load','tb']]
        return df

    def store_forecast_api(self,df,result,version):
        try:
            token = load().get_token()
            print('Setting up final dataframe...')
            result = df[['date','start_time','end_time','forecast','horizon','version', 'type', 'created_at']]
            result['version']= version
            result['date'] = result['date'].apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))
            result['source'] = "urd"
            result['revision'] = 1
            result = result.loc[:, ['date','start_time','end_time','source','forecast','version','horizon','revision']]
            ip=result.to_json(orient = 'records')  
            json_data = {'data': ip, 'type':'dayahead', "client_id":51}
            url = "https://apis.dummy.com/api/load/setForecast"
            header={'token': load().get_token()['access_token'], 'content-type': 'application/x-www-form-urlencoded'}
            response = requests.post(url=url, data=json_data, headers=header)
            print(response.json())
            
            print("Storing forecast in API")               
        except:
            print("Error while stroing in API")

            
    def get_forecast(self,sdt,tdt,version):
        headers={'token': load().get_token()['access_token'], 'content-type': 'application/x-www-form-urlencoded'}
        params = {'client_id':str(51),'from-date':sdt,'to-date':tdt,'type':'dayahead','source':'urd',"revision":1,"version":version}
        api_get = " https://apis.dummy.com/api/load/getForecast"

        try:
            d =load().get_api_data(sdt, tdt, params, headers, api_get)['data']
            l=pd.DataFrame(d)
            l=l.sort_values(by=['date','start_time'])
            l = l[ ['date','start_time','end_time','source','forecast','version','horizon','revision']]
            print("Getting forecast in DB") 
            return l
        
        except Exception as e:
                print("Error in fetching DA forecast data : {a}".format(a=e)) 

    def get_forecas_v1(self,sdt,tdt):
        headers={'token': load().get_token()['access_token'], 'content-type': 'application/x-www-form-urlencoded'}
        params = {'client_id':str(51),'from-date':sdt,'to-date':tdt,'type':'dayahead','source':'urd',"revision":1,"version":'v1'}
        api_get = " https://apis.dummy.com/api/load/getForecast"

        try:
            d =load().get_api_data(sdt, tdt, params, headers, api_get)['data']
            l=pd.DataFrame(d)
            l=l.sort_values(by=['date','start_time'])
            l = l[ ['date','start_time','end_time','source','forecast','version','horizon','revision']]
            print("Getting forecast in DB") 
            return l
        
        except Exception as e:
                print("Error in fetching DA forecast data : {a}".format(a=e))

class misc:
#==========================================================================================
    def logspy(env):
        log_value("train", env.evaluation_result_list[0][1], step=env.iteration)
        log_value("val", env.evaluation_result_list[1][1], step=env.iteration)

    def get_lags(self, df, column, day, hour):
        lag_count = int(day*96 + hour*4)
        col = df[str(column)].shift(lag_count)
        return col
    
    def sinwave(self, df, feat, divs):
        return np.sin(df[str(feat)]*(2*np.pi/divs))
    
    def hour_mean(self, df, feature):
        hm = []
        hm.append(df.groupby(['date','hour'])[str(feature)].mean())
        hm = hm[0].reset_index()
        hm = hm.rename(columns = {str(feature):'hour_mean'})
        return hm
    
    def get_date(self, timedel):
        dt = (datetime.now() + timedelta(days=timedel)).strftime("%Y-%m-%d")
        return dt
    
    def load_bins(self, column):
        bin_map = {0:1,1:1,2:1,3:1,4:1,5:2,6:2,7:2,8:2,9:2,10:2,11:2,12:3,13:3,14:3,\
                   15:3,16:3,17:4,18:4,19:4,20:4,21:4,22:4,23:4}
        day_bin = (df.datetime.dt.hour).map(bin_map)
        return day_bin
    
    def inter(self, df):
        l=pd.DataFrame()
        for i in range(1,97):
            df1=df[df.timeblock==i]
            df1['loadmet'] = df1['loadmet'].interpolate(method ='linear')
            l=l.append(df1)
        l=l.sort_values(by=['datetime'])
        l=l.reset_index(drop=True)
        return l
    
    def holiday(self, df, df1):
        c=df[df.date==df1[0]]
        for i in range(1,len(df1)):
            c1=df[df.date==(pd.to_datetime(df1[i]).date())]
            c=c.append(c1)
        return c
    
    def remove_holiday(self, df, location):
        hold = pd.read_csv(location)
        df=misc().short(df,hold)
        return df
    
    def remove_outliers(self, df, df1):
        df.loc[df.load < df1.load.quantile(q=0.005), 'load'] = df1.load.quantile(q=0.005)
        df.loc[df.load > df.load.quantile(q=0.995), 'load'] = np.nan
        return df
    
    def short(self, df, df1):
        for i in range(0,len(df1.columns.values.tolist())):
            d=df1[df1.columns.values.tolist()[i]]
            d1= misc().holiday(df,d)
            df=df[(~df.datetime.isin(d1.datetime))]
        return df
    
    def column_index(self,df, query_cols):
        cols = df.columns.values
        sidx = np.argsort(cols)
        return sidx[np.searchsorted(cols,query_cols,sorter=sidx)]
    
    def timemerge2(self, df):
        time = pd.date_range(df.datetime[0], df.datetime[len(df)-1], freq='15min')
        time = pd.DataFrame(time,columns=['datetime'])
        time['date'] = time['datetime'].apply(lambda y: datetime.strptime(str(y)[:10], '%Y-%m-%d').date())
        df = pd.merge(left= time, right= df, on= ['datetime'], how = 'left')
        return df

    def send_notification(self,power):
        data = {"to":["sagar.paithankar@dummy.com"],"from":"mailbot@dummy.com","subject":"","body":""}
        data['subject'] = "client_xxx Day-Ahead Model Training (New) Notification"
        data['body'] = "{}".format(power)
        jsdata = json.dumps(data)
        respo = requests.post("http://api.dummy.com/index.php?r=notifications/send-email",
                headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}, data= jsdata)
        return respo    

    def send_error_notification(self,power):
        data = {"to":["sagar.paithankar@dummy.com"],"from":"error-bot@dummy.com","subject":"","body":""}
        data['subject'] = "client_xxx Model Alert! Error encountered in client_xxx Day-Ahead Model (New)"
        data['body'] = "{}".format(power)
        jsdata = json.dumps(data)
        respo = requests.post("http://api.dummy.com/index.php?r=notifications/send-email",
                headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}, data= jsdata)
        return respo       

class weather:
    
    #Weather token function
    def get_token(self):
        username = "tech@dummy.com"
        password = "dentintheuniverse"
        base_url = "http://apihub.dummy.com/api"
        try:
           headers = {}
           params = {'username': username, 'password': password}
           
           api_token = base_url + "/get-token"
           
           r = requests.post(url=api_token, headers=headers, params=params)
           return r.json()
       
        except Exception as e:
            print("Error in fetching weather token : {a}".format(a=e))

    #Fetch actual weather data and return a processed dataframe
    def combiner1(self, start_date, end_date, plant_id, client_id, source):
    
        url = "http://apihub.dummy.com/api/weather/getActual"
        querystring = {"from-date":start_date,"to-date":end_date,"plant-id":plant_id,"client_id":client_id,"source":source}
        headers = {'token': weather().get_token()['access_token']}
        
        try:
            response = requests.post(url, headers=headers, params=querystring)
            """
            headers = {'token': "$2y$10$q8SasTBbnyIqHXQ7eQpXY.N.IwpLj6txoh56.mzwAoXZYICGhcScK",
                   'cache-control': "no-cache",
                   'postman-token': "013c7a7d-eb47-b3ff-45d5-ad338724c449"}
            response = requests.request("POST", url, headers=headers, params=querystring)
            """
            data=response.json()
            d=pd.DataFrame(data['data'])
            d=d[['datetime_local', 'apparent_temperature', 'temperature', 'humidity', 'dew_point', 'wind_speed', 'cloud_cover']]
            d.rename(columns={'datetime_local': 'datetime'}, inplace=True)
            d['datetime']=pd.to_datetime(d['datetime'])
            if (type(d.temperature[1]) == str) or (type(d.apparent_temperature[1]) == str) or (type(d.humidity[1]) == str):
                d=d.replace('Null', np.nan)
                d['apparent_temperature']=d['apparent_temperature'].apply(lambda x: (float(x.replace("'", ''))))
                d['temperature']=d['temperature'].apply(lambda x: (float(x.replace("'", ''))))
                d['humidity']=d['humidity'].apply(lambda x: (float(x.replace("'", ''))))
                d['dew_point']=d['dew_point'].apply(lambda x: (float(x.replace("'", ''))))
                d['wind_speed']=d['wind_speed'].apply(lambda x: (float(x.replace("'", ''))))
                d['cloud_cover']=d['cloud_cover'].apply(lambda x: (float(x.replace("'", ''))))
            else:
                d=d
            d=d.replace('Null', np.nan)
            d.dropna(inplace=True)
            wdf=d
            wdf['datetime'] = pd.to_datetime(wdf.datetime)
            wdf.iloc[:,1:] = wdf.iloc[:,1:].astype(float)
        
            wdf = wdf.drop_duplicates('datetime')
            wdf.loc[wdf['temperature'] < 0, 'temperature'] = np.nan
            wdf.loc[wdf.temperature.isnull(), 'dew_point'] = np.nan
            wdf.loc[wdf.temperature.isnull(), 'apparent_temperature'] = np.nan
            wdf.loc[wdf.temperature.isnull(), 'humidity'] = np.nan
            wdf.loc[wdf.temperature.isnull(), 'wind_speed'] = np.nan
            wdf.loc[wdf.temperature.isnull(), 'cloud_cover'] = np.nan
            wdf.set_index('datetime', inplace=True)
            wdf = wdf.interpolate(method='time')
            wdf = wdf.resample('15min').asfreq()
#            wdf = wdf.fillna(method='ffill', limit=1)
#            wdf = wdf.fillna(method='bfill', limit=1)
            wdf = wdf.interpolate(method='time')
            wdf.reset_index(inplace=True)
            wdf = wdf.drop_duplicates('datetime')
            wdf.reset_index(inplace=True,drop=True)
            return wdf
        
        except Exception as e:
                print("Error in fetching weather actuals data : {a}".format(a=e))
    
    #Fetch weather forecast data from source specified in argument
    def combiner(self, start_date, end_date, plant_id, client_id, source):
        
        url = "http://apihub.dummy.com/api/weather/getForecast"
        if source=='accuweather_forecast':
            querystring = {"from-date":start_date,"to-date":end_date,"plant-id":plant_id,"client_id":client_id,"source":source,"type":'240_hours'}
       # elif source=='combiner_forecast':
        #    querystring = {"from-date":start_date,"to-date":end_date,"plant-id":plant_id,"client_id":client_id,"source":source,"type":'optimiser'}
        else :
            querystring = {"from-date":start_date,"to-date":end_date,"plant-id":plant_id,"client_id":client_id,"source":source}
        headers = {'token': weather().get_token()['access_token']}
        
        try:
            response = requests.post(url, headers=headers, params=querystring)
            """
            headers = {'token': "$2y$10$q8SasTBbnyIqHXQ7eQpXY.N.IwpLj6txoh56.mzwAoXZYICGhcScK",
                   'cache-control': "no-cache",
                   'postman-token': "013c7a7d-eb47-b3ff-45d5-ad338724c449"}
            response = requests.request("POST", url, headers=headers, params=querystring)
            """
            data=response.json()
            l=pd.DataFrame(data['data'])
            l['datetime']=pd.to_datetime(l['datetime_local'])
            l=l.sort_values(by=['datetime'])
            l=l.reset_index(drop=True)
            l.set_index('datetime', inplace=True)
            l = l.interpolate(method='time')
            l = l.resample('15min').asfreq()
#            l = l.fillna(method='ffill', limit=1)
#            l = l.fillna(method='bfill', limit=1)
            l = l.interpolate(method='time')
            l.reset_index(inplace=True)
            l = l.drop_duplicates('datetime')
            l = l[['datetime', 'apparent_temperature', 'temperature', 'humidity', 'dew_point', 'wind_speed', 'cloud_cover']]
            return l
        
        except Exception as e:
                print("Error in fetching weather forecast data : {a}".format(a=e))
    
    #Fetch weather optimiser data
    def optimiser(self, start_date, end_date, plant_id, client_id, source):
     
        url = "http://apihub.dummy.com/api/weather/getForecast"
        if source=='accuweather_forecast':
            querystring = {"from-date":start_date,"to-date":end_date,"plant-id":plant_id,"client_id":client_id,"source":source,"type":'240_hours'}
       # elif source=='combiner_forecast':
        #    querystring = {"from-date":start_date,"to-date":end_date,"plant-id":plant_id,"client_id":client_id,"source":source,"type":'optimiser'}
        else :
            querystring = {"from-date":start_date,"to-date":end_date,"plant-id":plant_id,"client_id":client_id,"source":source,"type":'optimiser'}
        headers = {'token':weather().get_token()['access_token']}
        
        try:
            response = requests.post(url, headers=headers, params=querystring)
            """
            headers = {'token': "$2y$10$q8SasTBbnyIqHXQ7eQpXY.N.IwpLj6txoh56.mzwAoXZYICGhcScK",
                   'cache-control': "no-cache",
                   'postman-token': "013c7a7d-eb47-b3ff-45d5-ad338724c449"}
            response = requests.request("POST", url, headers=headers, params=querystring)
            """
            data=response.json()
            l=pd.DataFrame(data['data'])
            l['datetime']=pd.to_datetime(l['datetime_local'])
            l=l.sort_values(by=['datetime'])
            l=l.reset_index(drop=True)
            l.set_index('datetime', inplace=True)
            l = l.interpolate(method='time')
            l = l.resample('15min').asfreq()
            l = l.fillna(method='ffill', limit=1)
            l = l.fillna(method='bfill', limit=1)
            l = l.interpolate(method='time')
            l.reset_index(inplace=True)
            l = l.drop_duplicates('datetime')
            l = l[['datetime', 'apparent_temperature', 'temperature', 'humidity', 'dew_point']]
            return l
        
        except Exception as e:
                print("Error in fetching weather optimiser data : {a}".format(a=e))
    
    #Humidex function
    def humidex(self, df):
        Td = df['dew_point'] + 273.15
        Tc = df['temperature']
        a = 1/273.15
        b = 1/Td
        c = Tc+0.555*((6.11*(np.exp(5417.75*(a-b))))-10)
        return c
    
    #RH function
    def calculate_RH(self, df):
        Es = 6.11*10.0**((7.5*df['temperature']) / (237.7 + df['temperature']))
        E = 6.11*10.0**((7.5*df['dew_point'])/(237.7 + df['dew_point']))
        RH = (E/Es)*100
        return RH

def week_of_month(dt):
    """ Returns the week of the month for the specified date.
    """
    first_day = dt.replace(day=1)
    dom = dt.day
    adjusted_dom = dom + first_day.weekday()
    return int(ceil(adjusted_dom/7.0))