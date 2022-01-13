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
wkdir = '/mnt/data/users/herringtont/soil_temp/In-Situ/Yukon/'
odir = '/mnt/data/users/herringtont/soil_temp/In-Situ/Yukon/site_level/'
pathlist = os.listdir(wkdir)
pathlist_sorted = natural_sort(pathlist)



wkfil = ''.join([wkdir+'PFT_OBSERVATIONS.CSV'])
coord_fil = ''.join([wkdir+'Location_information.csv'])

dframe_insitu = pd.read_csv(wkfil)
dframe_coord = pd.read_csv(coord_fil)

site_ID = dframe_coord['ID'].values
site_lat = dframe_coord['LATITUDE']
site_lon = dframe_coord['LONGITUDE']

#print(site_ID)

len_sites = len(site_ID)


master_site_cntr = 1228

### loop through sites ###
for i in range(0,len_sites): #loop through coordinates file and grab sites and lat/lon
    #print("site number:",master_site_cntr)
    site_i = site_ID[i]
    lat_i = site_lat[i]
    lon_i = site_lon[i]
    dframe_site = dframe_insitu[dframe_insitu['LOCATION_ID'] == site_i]
    depth_m = dframe_site['DEPTH_MIN'].values

    depths = np.unique(depth_m)

    if (len(depths) == 0):
    	master_site_cntr = master_site_cntr + 1
    	continue


    depths = depths[depths < 0] # only grab depths below surface
    depths = depths*-1 #convert to positive values
    depths = np.unique(depths)
    #print(depths)


### loop through depths ###        
    dframe_stemp = "Nothing"    
    for j in depths: #loop through depths for each site
    	depth_j = j*-1 #convert back to negative for comparison purposes
    	depth_cm = depth_j*-1*100 #convert from depths in negative metres to positive cm
    	#print('the depth is:',depth_cm,'cm')    
    	dframe_depth = dframe_site[dframe_site['DEPTH_MIN'] == depth_j]

    	dates = dframe_depth['TIME_START'].values

    	datetime = pd.to_datetime(dframe_depth['TIME_START'],format='%d-%b-%y')
    	#print('Location ID:',site_i,',site_id:',master_site_cntr,'start date:',dates[0],' end date:',dates[len(dates)-1])
    	#print(dframe_depth)

    	dframe_depth.insert(0,'DateTime',datetime)
    	#print(dframe_depth)

    	dframe_depth = dframe_depth.set_index('DateTime')
    	dframe_depth2 = dframe_depth.groupby(pd.Grouper(freq='D')).mean()
    	#print(dframe_depth2)


    	new_dt_index = pd.date_range(start= '1/1/1980',end = '2020/12/31', freq='D') #create new datetime index going from 1997 to the end of 2020
    	#print(new_dt_index)

    	dframe_depth_reindex = dframe_depth2.reindex(new_dt_index, fill_value = np.nan) # reindex to common datetime index
    	#print(dframe_depth_reindex)
    

    	dates_new = dframe_depth_reindex.index #grab dates from reindexed data
    	stemp = dframe_depth_reindex['NUMERICAL_VALUE'].values #grab soil temps


### Create Final Dataframe ###
    	if (str(dframe_stemp) == "Nothing"): #if dframe_stemp is empty, then create it
    		dframe_stemp = pd.DataFrame(data=dates_new,columns=['datetime'])
    		#print(dframe_stemp)
    		dframe_stemp[depth_cm] = stemp
    		dframe_stemp.insert(0,'Site Number', master_site_cntr)
    		dframe_stemp.insert(1,'Location ID', site_i)
    		dframe_stemp.insert(2,'latitude',lat_i)
    		dframe_stemp.insert(3,'longitude',lon_i)

    	else: #else if dframe_stemp exists, add soil temperature columns to it
    		dframe_stemp[depth_cm] = stemp


    ofil = ''.join([odir,'site_'+str(master_site_cntr)+'.csv'])		
    dframe_stemp.to_csv(ofil,index=False,na_rep=np.nan)
    #print('Location ID:',site_i,',site_id:',master_site_cntr)
    print(ofil)
    
    master_site_cntr = master_site_cntr + 1



