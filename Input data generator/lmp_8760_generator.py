# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 15:26:13 2022

@author: 54651
"""

import os
import pandas as pd

source_folder = r'C:\Users\54651\ICF\Consumers Energy Pilots - DER Economic Analysis Pilot\BTM\input data\lmp'
destination_folder = r'C:\Users\54651\Downloads\consumers project'

source_file_name =  '20190101-20220224 MISO Day-Ahead Energy Price.csv'

os.chdir(source_folder)

df = pd.read_csv(source_file_name)
df = df[df.node =='MICHIGAN.HUB']
df = df[['Date', 'lmp']]
df.Date = pd.to_datetime(df.Date)
df["year"] = df.Date.dt.year

os.chdir(destination_folder)

df[df.year == 2019].reset_index(drop=True).to_csv("2019_lmp.csv")
df[df.year == 2020].reset_index(drop=True).to_csv("2020_lmp.csv")
df[df.year == 2021].reset_index(drop=True).to_csv("2021_lmp.csv")

