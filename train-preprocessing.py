# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 19:51:09 2018

@author: moisessalazar77
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

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
