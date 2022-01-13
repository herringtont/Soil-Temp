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



################### define functions ################
def str_to_datetime(column, date_fmt):

    date_list = []

    for dt_str in column:
        new_dt = datetime.datetime.strptime(dt_str, date_fmt)
        date_list.append(new_dt)
			

    return date_list
    
def remove_trailing_zeros(x):
    return str(x).rstrip('0').rstrip('.')

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)



################## define directories ################

olr = ['outliers','zscore','IQR']
lyr = ['0_9.9','10_29.9','30_99.9','100_299.9','300_deeper','top_30cm']

basedir = '/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/'


for i in olr:
    olri = i
    
    for j in lyr:
    	lyrj = j
    	wkdir = ''.join([basedir+str(olri)+'/'+lyrj+'/'])
    	pathlist = os.listdir(wkdir)
    	pathlist_sorted = natural_sort(pathlist)
    	
    	for path in pathlist_sorted:
    		if path.endswith('.csv'):
    			wkfil = ''.join([wkdir+path])
    			print(wkfil)
    			sitid1 = os.path.basename(wkfil)
    			sitid2 = sitid1.split('site_')[1].split('.csv')[0]
    			if (int(sitid2) > 299):
    				os.remove(wkfil)
    				print('removing:',wkfil)
    
