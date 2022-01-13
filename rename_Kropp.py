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



def str_to_datetime(column, date_fmt):

    date_list = []

    for dt_str in column:
        new_dt = datetime.datetime.strptime(dt_str, date_fmt)
        date_list.append(new_dt)
			

    return date_list


#set files and directories

filename = str("/mnt/data/users/herringtont/soil_temp/In-Situ/Kropp/data/soil_temp_date_coord2.csv")
wkdir = str("/mnt/data/users/herringtont/soil_temp/In-Situ/Kropp/data/")
odir = str("/mnt/data/users/herringtont/soil_temp/In-Situ/All/")

#read in data
dframe = pd.read_csv(filename)
dframe.replace(-999, np.nan, inplace =True)
levels = dframe.columns.values.tolist()
 ###store all unique site id values
sid = np.unique(dframe['site_id'])
sitid = sid[~np.isnan(sid)]
#print("Levels:", levels)
#print("Column types:", dframe.dtypes)
#print(dframe)
    
col1 = dframe['Date']
#print(col1)
# Sample date: 2011-06-21
date_fmt = "%Y-%m-%d"
   
datetime_column = str_to_datetime(col1, date_fmt)
# The pandas builtin seems to have issues
#datetime_column = pd.to_datetime(datcol, date_fmt)
#print("Length of datetime column:", len(datetime_column))
    
#dframe['Date'] = pd.to_datetime(dframe['Date'], format=date_fmt)
#dframe = dframe.set_index(pd.DatetimeIndex(dframe['Date']))
    
###group by site id
#for i in range(2):
for i in sitid:
    dframe_siteid = dframe[dframe['site_id'] == i]
    sdepth = np.unique(dframe_siteid['st_depth'])
    sdep = sdepth[~np.isnan(sdepth)]
    print('Kropp Site ID:',i)
    if (i <= 41):
    	i2 = i + 68
    elif (44 <= i <= 46):
    	i2 = i + 66
    elif (i >= 47):
    	i2 = i + 64
    sint = str(i2)
    nam = "site_"
    snam = nam + sint
    for j in sdep:
    	wdep = int(j)
    	strdep = str(wdep)
    	dep = "depth_"
    	sdepth = dep + strdep
    	dframe_sdep = dframe_siteid[dframe_siteid['st_depth'] == j]
	#print(dframe_sdep)
    	soil_dep = dframe_sdep.iloc[1,3]
	#print("the soil depth is: ",soil_dep)
    	soil_dep2 = int(soil_dep)
    	sdept = str(soil_dep2)
	#print("soil depth (int): ", sdept)
	#print(soil_dep2)
    	if ( soil_dep2 < 0) :
    		bins = "above_ground"
    	elif ( 0 <= soil_dep2 < 5 ):
    		bins = "0_4.9"
    	elif ( 5 <= soil_dep2 < 10 ):
    		bins = "5_9.9"
    	elif ( 10 <= soil_dep2 < 15 ):
    		bins = "10_14.9"
    	elif ( 15 <= soil_dep2 < 20 ):
    		bins = "15_19.9"
    	elif ( 20 <= soil_dep2 < 30 ):
    		bins = "20_29.9"
    	elif ( 30 <= soil_dep2 < 50 ):
    		bins = "30_49.9"
    	elif ( 50 <= soil_dep2 < 70 ):
    		bins = "50_69.9"
    	elif ( 70 <= soil_dep2 < 100 ):
    		bins = "70_99.9"
		elif ( 100 <= soil_dep2 < 150 ):
			bins = "100_149.9"
		elif ( 150 <= soil_dep2 < 200 ):
			bins = "150_199.9"
		elif ( 200 <= soil_dep2 < 300 ):
			bins = "200_299.9"
		elif ( soil_dep2 >= 300 ):
    			bins = "300_deeper"
		#print(bins)
    		####write site_level values to csv file
    		###grab filename and remove .csv extension from string
		#wk_file = filename.replace('.csv','')
		#wk_file2 = wk_file.rstrip()
		#print(dframe_sdep)
		stemp = dframe_sdep[["Date","lat", "long", "st_depth", "soil_t"]]
		stemp.insert(0,'Dataset','Kropp')
		#print(stemp)
		odir2 = [odir,"depth_level/",bins,"/","site_", sint, "_", "depth_",sdept,".csv"]
		####create new filename with _mon.csv at end
		s_fil = "".join(odir2)
		print(s_fil)
		#stemp.to_csv(s_fil,na_rep="NaN", header=['Dataset','Date','Lat','Lon','Depth_cm','Soil_Temp'], index=False)
    		
