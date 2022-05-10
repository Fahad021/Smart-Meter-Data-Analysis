# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 10:10:58 2022

@author: 54651
"""

import pandas as pd
from pypvwatts import PVWatts
df = pd.DataFrame()
dti = pd.date_range("2023-01-01", periods=8760, freq="H")

df["Start Datetime (hb)"] = dti

# PV
result = PVWatts.request(
                         system_capacity=7,
                         module_type=1,
                         array_type=1,
                         azimuth=180,
                         tilt=15,
                         dataset='tmy2',
                         losses=13,
                         lat=  42.80223327869396,
                         lon= -85.4741232413969,
                         dc_ac_ratio = 1.25,
                         timeframe='hourly')
df['PV Gen (kW/rated kW)'] = result.ac
df['PV Gen (kW/rated kW)'] = df['PV Gen (kW/rated kW)'] /1000
df['PV Gen (kW/rated kW)'] = df['PV Gen (kW/rated kW)']/7

# system load

d = pd.read_csv("MISO_load.csv")
d= d[['Demand (MW)']]
df["System Load (kW)"]= d.values



'''
df['date'] =pd.DatetimeIndex(df['datetime']).date
df['month'] =pd.DatetimeIndex(df['date']).month
df['day']=pd.DatetimeIndex(df['date']).dayofweek #The day of the week with Monday=0, Sunday=6.
df['hour']=pd.DatetimeIndex(df['datetime']).hour
df['is_weekday'] =  pd.DatetimeIndex(df['date']).dayofweek < 4
df['super_off_peak'] = ((df['hour']<=6)&(df['hour']>=1)).astype(int)
df['on_peak'] = ((df['hour']<=19)&(df['hour']>=16)).astype(int)
df['off_peak'] = ((df['super_off_peak']!=1)&(df['on_peak']!=1)).astype(int)

# DR Days selection

SATURDAY = 5
DR_days = 20
DR_event_length= 4
DR_weekend = 0
DR_day_ahead =0
DR_program_start_hour = 15
DR_program_end_hour = None


if DR_program_end_hour is None:
    DR_program_end_hour = DR_program_start_hour + DR_event_length - 1

df1 = pd.DataFrame(df, columns = ['datetime', 'date','Demand (MW)'])

df1['DR_active'] = (df['hour']>=DR_program_start_hour) & (df['hour']<=DR_program_end_hour) & (df['is_weekday']==True)

load_during_active_events = df1.loc[df1['DR_active']]
load_during_active_events = load_during_active_events[['datetime', 'date','Demand (MW)']]
sum_system_load_energy = load_during_active_events.groupby(by=load_during_active_events.date).sum()
actual_dr_days = sum_system_load_energy.sort_values(ascending=False,by=['Demand (MW)'])

actual_dr_days  =actual_dr_days.reset_index()
actual_dr_days= actual_dr_days['date'][:DR_days].to_list()


# find top_4 hours in each day

top_4_hours = []

for x in actual_dr_days:
    dummy = df[df.date == x].reset_index(drop= True)
    s = dummy['Demand (MW)'].values
    top_4_hours.append(sorted(range(len(s)), key=lambda i: s[i], reverse=True)[:4]) #get top-4 indexes


# get the count of occured numbers so that we can find top-4 hour window

from collections import Counter

counter = Counter(top_4_hours[0])
for i in top_4_hours[1:]:
    counter.update(i)

counter.most_common()


#[(15, 20), (16, 20), (14, 16), (17, 15), (13, 5), (18, 4)]
# analysing the output we get 14, so add +1 to 15 (3:00 PM)


#populate actual DR active columns in main dataframe df
df['DR_active'] = 0
temp_index = df.index
for i in actual_dr_days:
    condition = df['date'] == i
    dr_day_indices = temp_index[condition]
    dr_day_indices_list = dr_day_indices.tolist()
    for j in dr_day_indices_list:
        if (df.iloc[j]['hour']>=DR_program_start_hour) & (df.iloc[j]['hour']<=DR_program_end_hour):
            df.at[j, 'DR_active']=1
        else:
            pass



result = PVWatts.request(
                         system_capacity=7,
                         module_type=1,
                         array_type=1,
                         azimuth=180,
                         tilt=15,
                         dataset='tmy2',
                         losses=13,
                         lat=  42.80223327869396,
                         lon= -85.4741232413969,
                         dc_ac_ratio = 1.25,
                         timeframe='hourly')

df["solar (kW)"] = result.ac
df["solar (kW)"] = df["solar (kW)"]/1000

files=[
 'representative_loadshape_for_cluster_0_.csv',
 'representative_loadshape_for_cluster_1_.csv',
 'representative_loadshape_for_cluster_2_.csv',
 'representative_loadshape_for_cluster_3_.csv'
]


for i in range(len(files)):
    df['site_load'] = pd.read_csv(files[i]).iloc[:,1].values
    df.to_csv("cluster_{}_loadshape_DRevents.csv".format(i))

'''
