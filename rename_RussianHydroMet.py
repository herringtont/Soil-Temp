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
from dateutil.parser import parse
def str_to_datetime(column, date_fmt):

    date_list = []

    for dt_str in column:
        new_dt = datetime.datetime.strptime(dt_str, date_fmt)
        date_list.append(new_dt)
			

    return date_list


#set files and directories

coord_fil = str('/mnt/data/users/herringtont/soil_temp/In-Situ/Master_Lat_Long_Obs.csv')
station_info_fil = str('/mnt/data/users/herringtont/soil_temp/In-Situ/RussianHydromet/Station_Data.csv')
wkdir = str('/mnt/data/users/herringtont/soil_temp/In-Situ/RussianHydromet/')
odir = str('/mnt/data/users/herringtont/soil_temp/In-Situ/All/')


#read in data

dframe_station_info = pd.read_csv(station_info_fil)
station_num = dframe_station_info['WMO Index'].values
station_lat = dframe_station_info['Latitude'].values
station_lon = dframe_station_info['Longitude'].values

num_stn = len(station_num)

prob_stn = []
prob_lat = []
prob_lon = []
prob_site = []
reg_stn = []
reg_lat = []
reg_lon = []
reg_site = []
for i in range(0,num_stn):
    stn_i = station_num[i]
    lat_i = station_lat[i]
    lon_i = station_lon[i]
    #print(stn_i)
    site_num = 300 + i
    stn_fil = ''.join([wkdir+'st_'+str(stn_i)+'.dat'])
    dframe_station = pd.read_csv(stn_fil, sep="\s+", names=['WMO_index','Year','Month','Day','2cm_temp','2cm_quality','5cm_temp','5cm_quality','10cm_temp','10cm_quality','15cm_temp','15cm_quality','20cm_temp','20cm_quality','40cm_temp','40cm_quality','60cm_temp','60cm_quality','80cm_temp','80cm_quality','120cm_temp','120cm_quality','160cm_temp','160cm_quality','240cm_temp','240cm_quality','320cm_temp','320cm_quality'])
    #print(dframe_station)
    #print(dframe_station)
    station_day = dframe_station['Day'].values
    station_day_0 = str(station_day[0])
    if(len(station_day_0) > 2):
    	prob_stn.append(stn_i)
    	prob_lat.append(lat_i)
    	prob_lon.append(lon_i)
    	prob_site.append(site_num)
    else:
    	reg_stn.append(stn_i)
    	reg_lat.append(lat_i)
    	reg_lon.append(lon_i)
    	reg_site.append(site_num)


len_prob_stn = len(prob_stn)
#print(len_prob_stn)
len_reg_stn = len(reg_stn)

for j in range(0,len_reg_stn):
    stn_j = reg_stn[j]
    stn_lat = reg_lat[j]
    stn_lon = reg_lon[j]
    site_num = reg_site[j]
    reg_stn_fil = ''.join([wkdir+'st_'+str(stn_j)+'.dat'])
    dframe_station2 = pd.read_csv(reg_stn_fil, sep="\s+", names=['WMO_index','Year','Month','Day','2cm_temp','2cm_quality','5cm_temp','5cm_quality','10cm_temp','10cm_quality','15cm_temp','15cm_quality','20cm_temp','20cm_quality','40cm_temp','40cm_quality','60cm_temp','60cm_quality','80cm_temp','80cm_quality','120cm_temp','120cm_quality','160cm_temp','160cm_quality','240cm_temp','240cm_quality','320cm_temp','320cm_quality'])
    #print(len(dframe_station2)) 
 
    stn_number = dframe_station2['WMO_index'].values
    stn_year = dframe_station2['Year'].values
    stn_mon = dframe_station2['Month'].values
    stn_day = dframe_station2['Day'].values

    datetime_master = []
    temp_2cm_master = []
    temp_5cm_master = []
    temp_10cm_master = []
    temp_20cm_master = []
    temp_15cm_master = []
    temp_40cm_master = []
    temp_60cm_master = []
    temp_80cm_master = []
    temp_120cm_master = []
    temp_160cm_master = []
    temp_240cm_master = []
    temp_320cm_master = []
    WMO_index_master = []
    len_date = len(stn_year)
    for k in range(0,len_date):
    	WMO_k = dframe_station2['WMO_index'].iloc[k]
    	temp_2cm_k = dframe_station2['2cm_temp'].iloc[k]
    	temp_5cm_k = dframe_station2['5cm_temp'].iloc[k]
    	temp_10cm_k = dframe_station2['10cm_temp'].iloc[k]
    	temp_15cm_k = dframe_station2['15cm_temp'].iloc[k]
    	temp_20cm_k = dframe_station2['20cm_temp'].iloc[k]
    	temp_40cm_k = dframe_station2['40cm_temp'].iloc[k]
    	temp_60cm_k = dframe_station2['60cm_temp'].iloc[k]
    	temp_80cm_k = dframe_station2['80cm_temp'].iloc[k]
    	temp_120cm_k = dframe_station2['120cm_temp'].iloc[k]
    	temp_160cm_k = dframe_station2['160cm_temp'].iloc[k]
    	temp_240cm_k = dframe_station2['240cm_temp'].iloc[k]
    	temp_320cm_k = dframe_station2['320cm_temp'].iloc[k]
    	year_k = stn_year[k]	
    	mon_k = stn_mon[k]
    	day_k = stn_day[k]
    	#print(year_k,mon_k,day_k)
    	isValidDate = True
    	try:	
    		datetime.datetime(year_k,mon_k,day_k)
    	except ValueError:
    		isValidDate = False

    	#print(isValidDate)
    	if(isValidDate == True):
    		dtime = datetime.datetime(year_k,mon_k,day_k)
    		#print(dtime)
    		datetime_master.append(dtime)
    		temp_2cm_master.append(temp_2cm_k)	
    		temp_5cm_master.append(temp_5cm_k)
    		temp_10cm_master.append(temp_10cm_k)
    		temp_15cm_master.append(temp_15cm_k)
    		temp_20cm_master.append(temp_20cm_k)
    		temp_40cm_master.append(temp_40cm_k)
    		temp_60cm_master.append(temp_60cm_k)
    		temp_80cm_master.append(temp_80cm_k)
    		temp_120cm_master.append(temp_120cm_k)
    		temp_160cm_master.append(temp_160cm_k)
    		temp_240cm_master.append(temp_240cm_k)
    		temp_320cm_master.append(temp_320cm_k)
    		WMO_index_master.append(WMO_k)
    	else:
    		continue
		
    temp_2cm_new = [n/10 for n in temp_2cm_master]
    temp_5cm_new = [n/10 for n in temp_5cm_master]
    temp_10cm_new = [n/10 for n in temp_10cm_master]   
    temp_15cm_new = [n/10 for n in temp_15cm_master]
    temp_20cm_new = [n/10 for n in temp_20cm_master]
    temp_40cm_new = [n/10 for n in temp_40cm_master]
    temp_60cm_new = [n/10 for n in temp_60cm_master]
    temp_80cm_new = [n/10 for n in temp_80cm_master]
    temp_120cm_new = [n/10 for n in temp_120cm_master]
    temp_160cm_new = [n/10 for n in temp_160cm_master]
    temp_240cm_new = [n/10 for n in temp_240cm_master]
    temp_320cm_new = [n/10 for n in temp_320cm_master]

    dframe_new = pd.DataFrame(data = WMO_index_master, columns=['WMO index'])
    dframe_new['Site Number'] = site_num
    dframe_new['Latitude'] = stn_lat
    dframe_new['Longitude'] = stn_lon
    dframe_new['Date'] = datetime_master
    dframe_new['2'] = temp_2cm_new
    dframe_new['5'] = temp_5cm_new
    dframe_new['10'] = temp_10cm_new
    dframe_new['15'] = temp_15cm_new
    dframe_new['20'] = temp_20cm_new
    dframe_new['40'] = temp_40cm_new
    dframe_new['60'] = temp_60cm_new
    dframe_new['80'] = temp_80cm_new
    dframe_new['120'] = temp_120cm_new
    dframe_new['160'] = temp_160cm_new
    dframe_new['240'] = temp_240cm_new
    dframe_new['320'] = temp_320cm_new
    dframe_new = dframe_new.replace(999.9,np.nan)
    site_num2 = dframe_new['Site Number'].iloc[0]
    dframe_new2 = dframe_new.drop(['WMO index'], axis=1)
    print(dframe_new2)

    ofil = ''.join(['/mnt/data/users/herringtont/soil_temp/In-Situ/RussianHydromet/site_level/site_'+str(site_num2)+'.csv'])
    dframe_new2.to_csv(ofil,na_rep='NaN',index=False)
