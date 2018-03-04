# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 13:05:39 2018

@author: moisessalazar77
"""
import pandas as pd
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

sample_2011=spray[spray.year == 2011].sample(1472)
sample_2013=spray[spray.year == 2013].sample(1472)

spray_rsh= pd.concat([sample_2011,sample_2013], ignore_index=True)

spray_rsh.rename(columns={'Date':'Date_spry','year':'year_spry',
                      'month':'month_spry',
                      'day':'day_spry',
                      'Latitude':'Latitude_spry',
                      'Longitude':'Longitude_spry',
                      'Time':'Time_spry'}, inplace=True)

spray_rsh.to_csv(os.path.join(dir_path,'preprocessed-spray.csv'),encoding='utf-8-sig',index=False)




