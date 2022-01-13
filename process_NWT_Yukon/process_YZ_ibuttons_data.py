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



# set files and directories #

wkdir = str('/mnt/data/users/herringtont/soil_temp/In-Situ/YZ_ibuttons/')
sitdir = str('/mnt/data/users/herringtont/soil_temp/In-Situ/YZ_ibuttons/by_site/')
odir = str('/mnt/data/users/herringtont/soil_temp/In-Situ/YZ_ibuttons/by_site/site_level/')
coord_fil = ''.join([wkdir+'YZ_ibuttons_Site_summary.csv'])
stemp_fil = ''.join([wkdir+'YZ_ibuttons_Daily_Temp_summary.csv'])

# read in data #

coord_dframe = pd.read_csv(coord_fil)
site_data = coord_dframe['Site_ID']
lat_data = coord_dframe['Latitude'] #degrees North
lon_data = coord_dframe['Longitude'] #degrees West
stemp_dframe = pd.read_csv(stemp_fil)
date_data = stemp_dframe['Date']

cntr = 0
master_site_cntr = 789
for i in site_data:
    site_i = i
    if(site_i == 'I51' or site_i == 'T33' or site_i == 'T34'): #skip I51, T33, T34 because data is missing
    	continue

    lat_i = lat_data.iloc[cntr]
    lon_i = lon_data.iloc[cntr] * -1 #convert to degrees east


    stemp_site = stemp_dframe[site_i]
    #print(stemp_site)

    dframe_site = pd.DataFrame(date_data, columns=['Date'])
    dframe_site['Latitude'] = lat_i
    dframe_site['Longitude'] = lon_i
    dframe_site['13'] = stemp_site


    sit_fil = ''.join([sitdir+str(site_i)+'.csv'])
    dframe_site.to_csv(sit_fil,na_rep='NaN',index=False)
    ofil = ''.join([odir+'site_'+str(master_site_cntr)+'.csv'])
    dframe_site.to_csv(ofil,na_rep='NaN',index=False)
    master_site_cntr = master_site_cntr + 1
    cntr = cntr+1

