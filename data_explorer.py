# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 15:43:03 2022

@author: 54651
"""

import pandas as pd
df = pd.read_csv("my_file_xref.csv")
df2 = pd.read_csv("my_file_xref_nwa.csv")
print(len(df.AMIDataID.unique()),len(df2.AMIDataID.unique()))

col_list = ["RowKey"]
df3 = pd.read_csv("RawHourly072019.csv", usecols=col_list)