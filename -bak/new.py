# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 17:46:53 2018

@author: moisessalazar77
"""
import pandas as pd
train_plus_weather=pd.read_csv('C:\\Users\moisessalazar77\\Desktop\\west-nile-chicago\\working\\train_plus_weather.csv')
train_plus_weather.drop_duplicates(inplace=True)