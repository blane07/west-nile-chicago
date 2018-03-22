# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 16:57:33 2018

@author: moisessalazar77
"""

import os
file_path='C:\\Users\\moisessalazar77\\Desktop\\west-nile-chicago\\clean_weather.csv'
clean_weather=pd.read_csv(file_path)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
clean_weather['Date']=pd.to_datetime(clean_weather['Date'])
clean_weather['hour']=clean_weather['Date'].dt.hour
clean_weather['day']=clean_weather['Date'].dt.day
clean_weather['month']=clean_weather['Date'].dt.month
clean_weather['year']=clean_weather['Date'].dt.year
labelencoder_X=LabelEncoder()
clean_weather['CodeSum_lbr']=labelencoder_X.fit_transform(clean_weather['CodeSum'])