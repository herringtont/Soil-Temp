import os
import glob
import netCDF4
import csv
import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import matplotlib.patches as mpl_patches
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
from scipy.stats import pearsonr
from matplotlib.ticker import (MultipleLocator, AutoLocator, AutoMinorLocator)



########## Define Functions ##########

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def calc_bias(y_pred, y_true):
    diff = np.array(y_pred) - np.array(y_true)
    sum_diff = sum(diff)
    N = len(y_true)
    bias = sum_diff/N
    return bias

def SDVnorm(y_pred, y_true):
    SDVp = np.std(y_pred)
    SDVt = np.std(y_true)
    SDVnorml = SDVp/SDVt
    return SDVnorml

def bias(pred,obs):
    """
    Difference of the mean values.

    Parameters
    ----------
    pred : numpy.ndarray
        Predictions.
    obs : numpy.ndarray
        Observations.

    Returns
    -------
    bias : float
        Bias between observations and predictions.
    """
    return np.mean(pred) - np.mean(obs)

def ubrmsd(o, p, ddof=0):
    """
    Unbiased root-mean-square deviation (uRMSD).

    Parameters
    ----------
    o : numpy.ndarray
        Observations.
    p : numpy.ndarray
        Predictions.
    ddof : int, optional
        Delta degree of freedom.The divisor used in calculations is N - ddof,
        where N represents the number of elements. By default ddof is zero.

    Returns
    -------
    urmsd : float
        Unbiased root-mean-square deviation (uRMSD).
    """
    return np.sqrt(np.sum(((o - np.mean(o)) -
                           (p - np.mean(p))) ** 2) / (len(o) - ddof))



def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]




############## Set Directories #################

two_year_dir = '/mnt/data/users/herringtont/soil_temp/sites_2yr/'
geom_dir = '/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/'
timeseries_dir = '/mnt/data/users/herringtont/soil_temp/naive_blend_timeseries/new_data/new_depth/'

naive_type = ['simple_average']
olr = ['zscore']#['outliers','zscore','IQR']
lyr = ['top_30cm','30cm_300cm']
thr = ['100']#['0','25','50','75','100']
rmp_type = ['con']#['nn','bil']
tmp_type = ['raw_temp']

############## Loop through files ###############

for i in rmp_type:
    rmp_type_i = i
    remap_type = ''.join(['remap'+rmp_type_i])
    
    for j in naive_type:
    	naive_type_j = j

    	for k in olr:
    		olr_k = k

    		for l in lyr:
    			lyr_l = l

    			if (lyr_l == "top_30cm"):
    				geom_layer = 'top30'
    				two_year_layer = 'top_30cm'
				
    			elif (lyr_l == "30cm_300cm"):
    				geom_layer = 'L7'
    				two_year_layer = '30_299.9'

    			for m in thr:
    				thr_m = m

    				two_year_fil = ''.join([two_year_dir+'sites_2yr_'+str(olr_k)+'_'+two_year_layer+'_'+str(thr_m)+'.csv'])
    				#print(two_year_fil)
    				dframe_two_year = pd.read_csv(two_year_fil)

    				geom_fil = ''.join([geom_dir+'geometry_'+geom_layer+'_'+rmp_type_i+'.csv'])
    				#print(geom_fil)
    				dframe_geom = pd.read_csv(geom_fil)
  
    				timeseries_fil = ''.join([timeseries_dir+str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_'+str(lyr_l)+'_'+str(thr_m)+'_data_CMOS_newdepth.csv'])
    				print(timeseries_fil)
    				dframe_timeseries = pd.read_csv(timeseries_fil)
    				gcell = dframe_timeseries['Grid Cell'].values

    				gcell_uq = np.unique(gcell)


    				gcell_master = []
    				lat_master = []
    				lon_master = []
    				sites_master = []
    				for n in gcell_uq:
    					gcell_n = n
    					gcell_master.append(gcell_n)
    					dframe_gcell = dframe_timeseries[dframe_timeseries['Grid Cell'] == gcell_n]
    					cen_lat = dframe_gcell['Central Lat'].iloc[0]
    					lat_master.append(cen_lat)
    					cen_lon = dframe_gcell['Central Lon'].iloc[0]
    					lon_master.append(cen_lon)
    					avg_sites = dframe_gcell['Sites Incl'].iloc[0]
    					sites_master.append(avg_sites)

    				dframe_master = pd.DataFrame(data=gcell_master, columns=['Grid Cell'])
    				dframe_master['Central Lat'] = lat_master
    				dframe_master['Central Lon'] = lon_master
    				dframe_master['Avg Sites Incl'] = sites_master

    				print(dframe_master)

    				summary_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_timeseries/summary_fil/new_data/'+str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_'+str(lyr_l)+'_thr'+str(thr_m)+'_gcell_summary_newdepth.csv'])
    				print(summary_fil)
    				dframe_master.to_csv(summary_fil,index=False)    				


