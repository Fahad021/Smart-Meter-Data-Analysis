# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 16:30:16 2022

@author: 54651
"""

import pandas as pd
import os

source_folder = r'C:\Users\54651\ICF\Consumers Energy Pilots - DER Economic Analysis Pilot\BTM\input data\MISO Load'
destination_folder = r'C:\Users\54651\Downloads\consumers project'


os.chdir(source_folder)
source_file_names =  ['EIA930_BALANCE_2019_Jan_Jun.csv',
                     'EIA930_BALANCE_2019_Jul_Dec.csv']

frames =[]
for source_file_name in source_file_names:
    df = pd.read_csv(source_file_name)
    df = df[df['Balancing Authority'] =='MISO']

    columns = [ 'Data Date', 'Hour Number',
           'Local Time at End of Hour',
           'Demand Forecast (MW)', 'Demand (MW)', 'Net Generation (MW)',
           'Demand (MW) (Adjusted)', 'Net Generation (MW) (Adjusted)']

    df = df[columns]

    number_columns=[ 'Demand Forecast (MW)', 'Demand (MW)', 'Net Generation (MW)',
           'Demand (MW) (Adjusted)', 'Net Generation (MW) (Adjusted)']

    for col in number_columns:
        df[col] = df[col].replace(',','', regex=True)
        df[col] = df[col].astype(float)

    frames.append(df)

final_df = pd.concat(frames).reset_index(drop =True)
final_df.to_csv("MISO_load.csv")