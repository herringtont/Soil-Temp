# -*- coding: utf-8 -*-
import os
import glob
import netCDF4
import csv
import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import numpy as np
import scipy
import pandas as pd
import xarray as xr
import seaborn as sns
import pytesmo
import math
import cftime
import pathlib
import re
import cdo
import skill_metrics as sm
from cdo import Cdo
from netCDF4 import Dataset,num2date # http://unidata.github.io/netcdf4-python/
from natsort import natsorted
from natsort import os_sorted
from calendar import isleap
from dateutil.relativedelta import *
from pathlib import Path
import seaborn as sn
from calendar import isleap
from dateutil.relativedelta import *
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from dateutil.relativedelta import *
from natsort import natsorted
from natsort import os_sorted

def str_to_datetime(column, date_fmt):

    date_list = []

    for dt_str in column:
        new_dt = datetime.datetime.strptime(dt_str, date_fmt)
        date_list.append(new_dt)
			

    return date_list


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

wkdir = '/mnt/data/users/herringtont/soil_temp/In-Situ/Nordicana/CSV/'

pathlist = os.listdir(wkdir)
pathlist_sorted = natural_sort(pathlist)

site_num = 758
for path in pathlist_sorted:
    insitu_fil = ''.join([wkdir,path])

    dframe_insitu = pd.read_csv(insitu_fil, names=['data_code','station_name','latitude','longitude','year','mon','day','ISO_Time_Delimiter','hour','minute','depth','quality_code','soil_temp'], header=None, dtype={"data_code":int,'station_name':str,'latitude':int,'longitude':int,'year':int,'mon':int,'day':int,'ISO_Time_Delimeter':str,'hour':int,'min':int,'depth':str,'quality_code':str,'soil_temp':float})
    dframe_insitu = dframe_insitu.replace(-99999,np.nan)
    #print(dframe_insitu)
    dframe_latitude = dframe_insitu['latitude'].values/1E5
    dframe_longitude = dframe_insitu['longitude'].values/1E5
    dframe_long = dframe_longitude[0]
    dframe_lat = dframe_latitude[0]
    dframe_year = dframe_insitu['year'].values
    dframe_mon = dframe_insitu['mon'].values
    dframe_day = dframe_insitu['day'].values
    dframe_hour = dframe_insitu['hour'].values
    dframe_min = dframe_insitu['minute'].values
    
    len_dframe = len(dframe_insitu)
    soil_dep = dframe_insitu['depth'].values
    soil_dep_str = [ str(i) for i in soil_dep]
    soil_temps = dframe_insitu['soil_temp'].values
    stn_nam = dframe_insitu['station_name'].iloc[0]
    soil_dep_uq = np.unique(soil_dep_str)
    soil_dep_uq_sorted = natsorted(soil_dep_uq)
    #print(stn_nam)
    #print(soil_dep_uq_sorted)


    datetime_master = []
    for i in range(0,len_dframe):
    	year_i = dframe_year[i]
    	mon_i = dframe_mon[i]
    	day_i = dframe_day[i]
    	hour_i = dframe_hour[i]
    	min_i = dframe_min[i]	
    	dtime = datetime.datetime(year=year_i, month=mon_i, day=day_i, hour=hour_i, minute=min_i, second=00)
    	datetime_master.append(dtime)

    dframe_dtime = pd.DataFrame(data=dframe_latitude,columns=['latitude'])
    dframe_dtime.insert(0,'Station',dframe_insitu['station_name'].iloc[0])
    dframe_dtime['longitude'] = dframe_longitude
    dframe_dtime['datetime'] = datetime_master
    dframe_dtime['depth'] = soil_dep_str
    dframe_dtime['soil temp'] = soil_temps
    #print(dframe_dtime)

    dt_unique = natsorted(np.unique(datetime_master))

    dframe_dtime2 = dframe_dtime.set_index(pd.DatetimeIndex(datetime_master))
        

    print('Site:',site_num)
    print('Station:',dframe_insitu['station_name'].iloc[0])
    print(soil_dep_uq)    
    dt_dup = dframe_dtime2[dframe_dtime2.index.duplicated()]
    #print(dt_dup)
    #print(dframe_dtime2)
    dframe_dep_new = []
    for dep in soil_dep_uq_sorted:
    	#print(dep)
    	dframe_dep = dframe_dtime2[dframe_dtime2['depth'] == dep]
    	dframe_dep = dframe_dep[~dframe_dep.index.duplicated()]
    	#print(dframe_dep)
    	dframe_dep_rindx = dframe_dep.reindex(dt_unique,fill_value = np.nan)
    	dframe_dep_rindx = dframe_dep_rindx[['soil temp']]
    	dframe_dep_rindx.insert(0,'Site_Number',site_num)
    	dframe_dep_rindx.insert(1,'Station',stn_nam)
    	dframe_dep_rindx.insert(2,'Latitude',dframe_lat)
    	dframe_dep_rindx.insert(3,'Longitude',dframe_long)
    	dframe_dep_rindx.insert(4,'DateTime',dframe_dep_rindx.index)
    	dframe_dep_rindx.insert(5,'Depth',dep)
    	temp_dep = dframe_dep_rindx['soil temp'].values
    	date_dep = dframe_dep_rindx['DateTime'].values
    	dep_int = int(re.search(r'\d+',dep).group())	
    	print(dframe_dep_rindx)

    	if(len(dframe_dep_new) == 0): #if new dateframe is empty
    		dframe_dep_new = pd.DataFrame(data=temp_dep, columns=[dep_int])
    	elif(len(dframe_dep_new) > 0):
    		dframe_stemp2 = temp_dep
    		dframe_dep_new[dep_int] = dframe_stemp2

    dframe_dep_new.insert(0,'Station',stn_nam)
    dframe_dep_new.insert(1,'Site Number',site_num)
    dframe_dep_new.insert(2,'Latitude',dframe_lat)
    dframe_dep_new.insert(3,'Longitude',dframe_long)
    dframe_dep_new.insert(4,'DateTime',dframe_dep_rindx.index)
    dframe_dep_new = dframe_dep_new.set_index(pd.DatetimeIndex(date_dep))    
    print(dframe_dep_new)

    dframe_dep_new_daily = dframe_dep_new.resample(rule='D', convention = 'start').mean()
    dframe_dep_new_daily.insert(3, 'Date', dframe_dep_new_daily.index)
    print(dframe_dep_new_daily)

    ofil = ''.join(['/mnt/data/users/herringtont/soil_temp/In-Situ/Nordicana/site_level/site_'+str(site_num)+'.csv'])

    dframe_dep_new_daily.to_csv(ofil,na_rep="NaN",index=False)
    site_num = site_num + 1
