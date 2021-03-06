# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 18:31:45 2018

@author: moisessalazar77
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


file_path1='C:\\Users\\moisessalazar77\\Desktop\\proj4\\weather.csv.'
weather=pd.read_csv(file_path1)

   
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

#only one station is recording the sunrise/sunset therefore the right value is propagated
weather['Sunrise']=weather['Sunrise'].fillna(method='ffill')
weather['Sunset']=weather['Sunrise'].fillna(method='ffill')
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

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
weather['CodeSum_lb']=le.fit_transform(weather['CodeSum'])  
colsToDrop = ['Date','CodeSum','Depart','Water1']
weather.drop(colsToDrop, axis=1,inplace=True)

import os
file_path='C:\\Users\\moisessalazar77\\Desktop\\projectpics\\proj3'
weather.to_csv(os.path.join(file_path,'clean_weather2.csv'),encoding='utf-8-sig',index=False)













