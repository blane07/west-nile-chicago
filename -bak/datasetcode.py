# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 15:55:46 2018

@author: moisessalazar77
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

dir_path='C:\\Users\\moisessalazar77\\Desktop\\proj4'
spray=pd.read_csv(os.path.join(dir_path,'spray.csv'))

spray['Date']=pd.to_datetime(spray['Date'])
spray['day']=spray['Date'].dt.day
spray['month']=spray['Date'].dt.month
spray['year']=spray['Date'].dt.year
spray['Time']=pd.to_datetime(spray['Time']).dt.strftime('%H:%M:%S')
spray['Time']=spray['Time'].astype(str)
spray['Time']=spray['Time'].str.replace(':','')
spray['Time']=pd.to_numeric(spray['Time'],downcast='integer',errors='coerce')
median_time=np.nanmedian(spray['Time'])
spray['Time'].fillna(median_time, inplace=True)


sample_2011=spray[spray.year == 2011].sample(1472)
sample_2013=spray[spray.year == 2013].sample(1472)

spray_rsh= pd.concat([sample_2011,sample_2013], ignore_index=True)


spray_rsh.rename(columns={'Date':'Date_spry','year':'year_spry',
                      'month':'month_spry',
                      'day':'day_spry',
                      'Latitude':'Latitude_spry',
                      'Longitude':'Longitude_spry',
                      'Time':'Time_spry'}, inplace=True)

spray_rsh.to_csv(os.path.join(dir_path,'preprocessed_spray.csv'),encoding='utf-8-sig',index=False)

dir_path='C:\\Users\\moisessalazar77\\Desktop\\proj4'
train=pd.read_csv(os.path.join(dir_path,'train.csv'))

train['Date']=pd.to_datetime(train['Date'])
train['day']=train['Date'].dt.day
train['month']=train['Date'].dt.month
train['year']=train['Date'].dt.year

sample_2007=train[train.year == 2007].sample(736)
sample_2009=train[train.year == 2009].sample(736)
sample_2011=train[train.year == 2011].sample(736)
sample_2013=train[train.year == 2013].sample(736)

Merged1= pd.concat([sample_2007,sample_2009], ignore_index=True)
Merged2= pd.concat([Merged1,sample_2011], ignore_index=True)
train_rsh= pd.concat([Merged2,sample_2013], ignore_index=True)


train_rsh['Street_effect']=train_rsh['AddressNumberAndStreet']
train_rsh['Street_effect'] = (train_rsh.groupby('Street')['Street'].transform('count')>1).astype(int)

#labeling text columns
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
train_rsh['Species_lb']=le.fit_transform(train_rsh['Species'])  
train_rsh['Address_lb']=le.fit_transform(train_rsh['AddressNumberAndStreet']) 

train_rsh.rename(columns={'Date':'Date_trn',                         
                      'month':'month_trn',
                      'year':'year_trn',
                      'day':'day_trn',
                      'Latitude':'Latitude_trn',
                      'Longitude':'Longitude_trn'}
                        , inplace=True)

train_rsh.to_csv(os.path.join(dir_path,'preprocessed_train.csv'),encoding='utf-8-sig',index=False)



dir_path='C:\\Users\\moisessalazar77\\Desktop\\proj4'
weather=pd.read_csv(os.path.join(dir_path,'weather.csv'))

   
#converting Data types              
weather['Tmax']=weather['Tmax'].astype(float)
weather['Tmin']=weather['Tmin'].astype(float)
weather['Tavg']=pd.to_numeric(weather['Tavg'],errors='coerse')
weather['WetBulb']=pd.to_numeric(weather['WetBulb'],errors='coerse')
weather['Heat']=pd.to_numeric(weather['Heat'],errors='coerse')
weather['Cool']=pd.to_numeric(weather['Cool'],errors='coerse')
weather['Sunrise']=pd.to_numeric(weather['Sunrise'],errors='coerse')
weather['Sunset']=pd.to_numeric(weather['Sunset'],errors='coerse')
weather['StnPressure']=pd.to_numeric(weather['StnPressure'],errors='coerse')
weather['SeaLevel']=pd.to_numeric(weather['SeaLevel'],errors='coerse')
weather['StnPressure']=pd.to_numeric(weather['StnPressure'],errors='coerse')
weather['AvgSpeed']=pd.to_numeric(weather['AvgSpeed'],errors='coerse')
weather['Date']=pd.to_datetime(weather['Date'])
weather['day']=weather['Date'].dt.day
weather['month']=weather['Date'].dt.month
weather['year']=weather['Date'].dt.year

#only one station is recording the sunrise/sunset therefore the right value is propagated
weather['Sunrise']=weather['Sunrise'].fillna(method='ffill')
weather['Sunset']=weather['Sunset'].fillna(method='ffill')
weather['Depth']=pd.to_numeric(weather['Depth'],errors='coerse')
weather['Depth']=weather['Depth'].fillna(method='ffill')
weather['SnowFall']=pd.to_numeric(weather['SnowFall'],errors='coerse')
weather['SnowFall']=weather['SnowFall'].fillna(method='ffill')
weather['PrecipTotal']=weather['PrecipTotal'].str.replace('T','0.01')
weather['PrecipTotal']=pd.to_numeric(weather['PrecipTotal'],errors='coerse')
                   

#fill the mixing spots with the median, some were originally in the file others created after type conversion
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values="NaN",strategy='median',axis=0)
imputer=imputer.fit(weather.iloc[:,[4,7,8,9,16,17,18,21]])
weather.iloc[:,[4,7,8,9,16,17,18,21]]=imputer.transform(weather.iloc[:,[4,7,8,9,16,17,18,21]])

#labeling text columns
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
weather['CodeSum_lb']=le.fit_transform(weather['CodeSum'])  

#Feature engineering
weather['Tmax:Tmin:Sunst']=(weather['Tmax']-weather['Tmin'])*(weather['Sunset'])
weather['Tavg:WB']=(weather['Tavg'])*(weather['WetBulb'])
weather['Hot_wet']=(weather['Tmax'])*(weather['PrecipTotal'])
weather['DP:Tmin']=(weather['DewPoint'])*(weather['Tmin'])
weather['P:sea:WB']=((weather['StnPressure'])/(weather['SeaLevel']))*(weather['WetBulb'])
weather['T_diff']=((41>(weather['Tmax']-weather['Tmin']))&((weather['Tmax']-weather['Tmin']>21))).astype(int)
weather['Dry']=(weather['WetBulb']>53).astype(int)
weather['Dry_Wet_Diff'] =(weather['WetBulb']-weather['DewPoint']>5).astype(int)
weather['month_cat']=(7>(weather['month'])&((weather['month']>9))).astype(int)
weather['DewPoint:Sunrise']=(weather['DewPoint'])*(weather['Sunrise'])


colsToDrop = ['Depart','Water1']
weather.drop(colsToDrop, axis=1,inplace=True)

weather.rename(columns={'Date':'Date_wthr',
                       'year':'year_wthr',
                       'month':'month_wthr',
                       'day':'day_wthr',}, inplace=True)

weather.to_csv(os.path.join(dir_path,'preprocessed_weather.csv'),encoding='utf-8-sig',index=False)


merge1 = pd.concat([spray_rsh.reset_index(), train_rsh.reset_index()], axis=1)
Dataset = pd.concat([merge1.reset_index(), weather.reset_index()], axis=1)
Dataset.to_csv(os.path.join(dir_path,'WnV2.csv'),encoding='utf-8-sig',index=False)

