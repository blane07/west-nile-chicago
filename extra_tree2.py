# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 18:42:20 2018

@author: moisessalazar77
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(42)

dir_path='C:\\Users\\moisessalazar77\\Desktop\\proj4'
dataset=pd.read_csv(os.path.join(dir_path,'WnV.csv'))

dataset['hot_temp']=(dataset['Tmax']>40).astype(int)
dataset['low_temp']=(dataset['Tmax']<32).astype(int)
dataset['year_spray']=(dataset['year_spry']==dataset['year_wthr']).astype(int)
dataset['month_spray']=(dataset['month_spry']==dataset['month_wthr']).astype(int)
dataset['day_spray']=((14>dataset['day_spry']-dataset['day_wthr']) & (dataset['day_spry']-dataset['day_wthr']>0)).astype(int)
dataset['spray_effect']=dataset['year_spry'] & dataset['month_spray'] & dataset['day_spray']
dataset['wind_speed']=(dataset['AvgSpeed']>10).astype(int)
dataset['low_wsp']=(dataset['ResultSpeed']<11).astype(int) 


total_feat=['Latitude_spry', 'Longitude_spry', 'day_spry',
       'month_spry', 'year_spry', 'Block',
        'Latitude_trn',
       'Longitude_trn', 'AddressAccuracy', 'NumMosquitos',
       'day_trn', 'month_trn', 'year', 'Street_effect', 'Species_lb',
       'Address_lb', 'Station', 'Tmax', 'Tmin', 'Tavg',
       'DewPoint', 'WetBulb', 'Heat', 'Cool', 'Sunrise', 'Sunset', 
       'Depth', 'SnowFall', 'PrecipTotal', 'StnPressure', 'SeaLevel',
       'ResultSpeed', 'ResultDir', 'AvgSpeed', 'day_wthr', 'month_wthr',
       'year_wthr', 'CodeSum_lb', 'Tmax:Tmin:Sunst', 'Tavg:WB', 'Hot_web',
       'DP:Tmin', 'P:sea:WB', 'T_diff', 'Dry', 'Dry_Wet_Diff', 'month_cat',
       'DewPoint:Sunrise', 'hot_temp', 'low_temp', 'year_spray', 'month_spray',
       'day_spray', 'spray_effect', 'wind_speed', 'low_wsp']

hold_on_feat=['Latitude_spry',
               'Longitude_spry', 'day_spry', 'month_spry',
               'Longitude_trn',
               'AddressAccuracy', 'day_trn', 'month_trn',
               'year', 'Street_effect', 'Species_lb',
               'Sunrise', 'Sunset','low_wsp',
               'PrecipTotal',
               'AvgSpeed', 'day_wthr', 'month_wthr', 'year_wthr', 'CodeSum_lb',
               'Tmax:Tmin:Sunst', 'Tavg:WB', 'Hot_web', 'DP:Tmin',
               'DewPoint:Sunrise','Station','NumMosquitos',]
 
feat=[
        'Address_lb','Latitude_spry',
               'Longitude_spry', 'day_spry', 'month_spry',
               'Longitude_trn',
               'AddressAccuracy', 'day_trn', 'month_trn',
               'year', 'Street_effect', 'Species_lb',
       'spray_effect','wind_speed',
       'PrecipTotal','year', 'Street_effect', 'Species_lb',
               'AvgSpeed', 'day_wthr', 'month_wthr', 'year_wthr', 'CodeSum_lb',
               'Tmax:Tmin:Sunst', 'Tavg:WB', 'Hot_web', 'DP:Tmin',
               'DewPoint:Sunrise','Station','NumMosquitos']
        


from sklearn.cross_validation import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

extc = ExtraTreesClassifier(n_estimators=150,
                            min_impurity_decrease =0.08,
                            n_jobs=-1,
                            criterion='gini',
                            class_weight ='balanced') 
 
from sklearn.metrics import accuracy_score
 
X=dataset[total_feat]

y=dataset['WnvPresent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, stratify=y)
clf=extc
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print(accuracy_score(y_test,y_pred))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))