# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 18:43:31 2022

@author: 54651
"""
import pandas as pd

d = pd.read_csv('location_counts.csv')
d.columns = ['postal_code','customer_count']
e = pd.read_csv('all_locations.csv')
e= e.iloc[:,1:]
f = pd.merge(e,d,on= 'postal_code', how = 'outer')
f = f[['postal_code', 'country_code', 'place_name', 'state_name', 'state_code',
      'county_name', 'county_code',
      'latitude', 'longitude', 'accuracy', 'customer_count']]


f.to_csv("location_with_count.csv")