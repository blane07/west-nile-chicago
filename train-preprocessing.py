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
train['hour']=train['Date'].dt.hour
train['day']=train['Date'].dt.day
train['month']=train['Date'].dt.month
train['year']=train['Date'].dt.year


#labeling text columns
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
train['Species_lb']=le.fit_transform(train['Species'])  
train['Address_lb']=le.fit_transform(train['AddressNumberAndStreet'])  
colsToDrop = ['Species','Trap','Street','Trap','Address']
train.drop(colsToDrop, axis=1,inplace=True)


train.to_csv(os.path.join(dir_path,'clean_train.csv'),encoding='utf-8-sig',index=False)
