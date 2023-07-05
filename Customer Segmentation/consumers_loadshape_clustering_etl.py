# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 21:16:29 2022

@author: 54651
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import glob

def daterange(start_date, end_date):
    delta = timedelta(hours=1)
    while start_date < end_date:
        yield start_date
        start_date += delta


def reshape2daily(df_in):
    """
    DESCRIPTION: Reshapes input dataframe with hourly resolution to a dataframe with daily resolution,
                 where every column contains data for a particular hour & different consumers are stacked verticaly.

    Input:
    df_in (pd.DataFrame): Dataframe holding time series data with hourly resolution in every column. Indices are timestamps.

    Output:
    df (pd.DataFrame): Dataframe holding time series data with daily resolution and hours in every columns.

    """
    df = (pd.pivot_table(df_in,
                         values="y",
                         index=[df_in.timestamp.dt.to_period("D"), df_in.id],
                         columns=df_in.timestamp.dt.strftime("%H:%M")
                         )
            .reset_index()
            .sort_values(["id", "timestamp"])
            .set_index("timestamp")
         )
    return df


data_list =  glob.glob("*.json")
summer_loadprofile = []
winter_loadprofile = []
start_date = datetime(2019, 1, 1, 00, 00)
end_date = datetime(2020, 1, 1, 00, 00)

timestamps = []
for single_date in daterange(start_date, end_date):
    timestamps.append(single_date.strftime("%Y-%m-%d %H:%M"))

for file_name in data_list:
    data = pd.read_json(file_name)
    data = data[['RowKey','AMIData']]
    for i in range(0,data.shape[0]):
        d = pd.DataFrame(data.AMIData[i], columns = ['y'])
        d['timestamp'] = timestamps
        d.insert(2, "id", data.RowKey.values[i])
        d = d[['timestamp', 'id', 'y']]
        d['timestamp'] = pd.to_datetime(d['timestamp'])
        d['y'] = d['y'].apply(pd.to_numeric, errors='coerce')
        X_daily = reshape2daily(d)
        X_daily['month'] = X_daily.index.month

        winter = X_daily[X_daily.month == 1]
        del winter['month']
        winter = winter.reset_index(drop= True)
        winter = pd.DataFrame(winter.mean())
        winter_loadprofile.append(winter.T.values.tolist()[0])

        summer = X_daily[X_daily.month == 7]
        del summer['month']
        summer = summer.reset_index(drop= True)
        summer = pd.DataFrame(summer.mean())
        summer_loadprofile.append(summer.T.values.tolist()[0])

summer_data = pd.DataFrame(summer_loadprofile)
winter_data = pd.DataFrame(winter_loadprofile)
summer_data.to_csv("summer_loadcurves.csv")
winter_data.to_csv("winter_loadcurves.csv")