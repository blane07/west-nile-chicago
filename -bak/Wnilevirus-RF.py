# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 11:07:36 2018

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
dataset['low_wsp']=(dataset['ResultSpeed']>11).astype(int) 

feat=[
        'Latitude_spry',
       'Longitude_spry', 'day_spry', 'month_spry',
        'Longitude_trn',
       'AddressAccuracy', 'day_trn', 'month_trn',
       'year', 'Street_effect', 'Species_lb', 'Address_lb',
       'Station','NumMosquitos','spray_effect','wind_speed',
        'Sunrise', 'Sunset','low_wsp',
       'PrecipTotal',
       'AvgSpeed', 'day_wthr', 'month_wthr', 'year_wthr', 'CodeSum_lb',
       'Tmax:Tmin:Sunst', 'Tavg:WB', 'Hot_web', 'DP:Tmin',
        'DewPoint:Sunrise']


from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier

forest1=RandomForestClassifier(criterion='entropy',
                               max_depth=100,
                               min_impurity_decrease= 0.001,
                               min_weight_fraction_leaf=0.1,
                               min_impurity_split=0.0001,
                               random_state= 42,
                               min_samples_leaf= 1,
                               min_samples_split= 2,
                               n_estimators=60,
                               n_jobs=-1,
                               class_weight='balanced')
 
from sklearn.metrics import accuracy_score
 
X=dataset[feat]

y=dataset['WnvPresent']

import itertools
r=2
it=X.columns
models_scores=[]

for i in range(1,r):
    for j in itertools.combinations(X.columns,i):
        X_train, X_test, y_train, y_test = train_test_split(X.loc[:,j], y, test_size = 0.3, stratify=y)
        clf=forest1
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        models_scores.append([j,accuracy_score(y_test,y_pred)])
        print(accuracy_score(y_test,y_pred))











