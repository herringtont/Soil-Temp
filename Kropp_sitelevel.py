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
from scipy.stats import pearsonr
from dateutil.relativedelta import *

################### define functions ################
def str_to_datetime(column, date_fmt):

    date_list = []

    for dt_str in column:
        new_dt = datetime.datetime.strptime(dt_str, date_fmt)
        date_list.append(new_dt)
			

    return date_list
    
def remove_trailing_zeros(x):
    return str(x).rstrip('0').rstrip('.')

############################## if dataset is Kropp ##################################
wfil = str("/mnt/data/users/herringtont/soil_temp/In-Situ/Kropp/data/soil_temp_date_coord2.csv")
dframe = pd.read_csv(wfil)
dframe.replace(-999, np.nan, inplace=True)
sid = np.unique(dframe['site_id'])
sitid = sid[~np.isnan(sid)]
col1 = dframe['Date']
date_fmt = "%Y-%m-%d"
datetime_column = str_to_datetime(col1, date_fmt)
    	
####### group by site #######
for j in sitid:
    print("Kropp Site ID:",j)
    dframe_siteid = dframe[dframe['site_id'] == j]
    sdepth = np.unique(dframe_siteid['st_depth'])
    sdep = sdepth[~np.isnan(sdepth)]
    site_date = dframe_siteid['Date']
    dt_col = str_to_datetime(site_date,date_fmt)
    date_uq = np.unique(site_date)
    dt_col2 = str_to_datetime(date_uq,date_fmt)
    dframe_siteid = dframe_siteid.set_index(site_date)
    if (j <= 41):
    	j2 = j + 68
    elif (44 <= j <= 46):
    	j2 = j + 66
    elif (j >= 47):
    	j2 = j + 64
    	sint = str(j2)
    print("the site is: ",j2)
