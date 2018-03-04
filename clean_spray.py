# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 20:34:18 2018

@author: moisessalazar77
"""

import pandas as pd
import os

dir_path='C:\\Users\\moisessalazar77\\Desktop\\proj4'
spray=pd.read_csv(os.path.join(dir_path,'spray.csv'))

spray['Time']=spray['Time'].replace('PM','',inplace=True)



spray['Time']=pd.to_numeric(spray['Time'],errors='coerse')

from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values="NaN",strategy='mean',axis=0)
imputer=imputer.fit(spray.iloc[:,1])
spray.iloc[:,1]=imputer.transform(spray.iloc[:,1])