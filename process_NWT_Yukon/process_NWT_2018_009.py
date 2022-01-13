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


# set directories and files #

wkdir = '/mnt/data/users/herringtont/soil_temp/In-Situ/NWT/2018-009/'
odir = '/mnt/data/users/herringtont/soil_temp/In-Situ/NWT/2018-009/site_level/'
pathlist = os.listdir(wkdir)
pathlist_sorted = natural_sort(pathlist)

master_site_cntr = 969
for path in pathlist_sorted:
    if(path == 'site_level'):
    	continue
    insitu_fil = ''.join([wkdir,path])
    sit_nam = path.split('.')[0]
    #print(sit_nam)
    dframe_insitu = pd.read_excel(insitu_fil,engine='openpyxl')
    dframe_insitu = dframe_insitu.loc[:, ~dframe_insitu.columns.str.contains('^Unnamed')]
    #print(dframe_insitu)
    dates = dframe_insitu['date_YYYY-MM-DD']      	
    latitude = dframe_insitu['latitude']
    longitude = dframe_insitu['longitude']

    if(isinstance(dates[0],datetime.datetime) == False): #check if second row needs to be deleted
    	dates = dates[1:]
    	latitude = latitude[1:]
    	longitude = longitude[1:]
    	print(master_site_cntr,'NWT_2018_009',latitude[1],longitude[1],sit_nam)

    else:
    	print(master_site_cntr,'NWT_2018_009',latitude[0],longitude[0],sit_nam)
    len_dates = len(dates)
    datetimes = []
    for i in range(0,len_dates):
    	date_i = dates.iloc[i]
    	datetimes.append(date_i)


    datetimes2 = []
    for j in datetimes:
    	date2 = j.strftime('%Y-%m-%d %H:%M:%S')
    	datetimes2.append(date2)
    #print(datetimes2)
    datetime_a = datetimes[0]
    #print(datetime_a)
    datetime_b = datetimes[1]
    #print(datetime_b)
    delta_datetime = datetime_b - datetime_a
    #print('the deltatime is:',delta_datetime)
    dframe_final = pd.DataFrame(data=latitude, columns=['latitude'])
    dframe_final['longitude'] = longitude
    dframe_final['datetime'] = datetimes2    
    col_nam = dframe_insitu.columns
    for i in range(6,len(col_nam)):
    	col_i = col_nam[i]
    	dep_nam = col_i.split('_')[0]
    	dep_cm = float(dep_nam)*100
    	dep_cm = int(dep_cm)
    	#print(dep_cm)
    	dframe_column = dframe_insitu[col_i]
    	dframe_final[dep_cm] = dframe_column

    sit_id = ''.join(['site_'+str(master_site_cntr)])
    site_fil = ''.join([odir,sit_id+'.csv'])
    print(site_fil)


    #dframe_final.to_csv(site_fil,na_rep="NaN",index=False)

    master_site_cntr = master_site_cntr + 1    
    #print(dframe_final)
  









