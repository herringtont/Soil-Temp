import os
import glob
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
from statistics import stdev
from statistics import mean

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


############# Set Directories ############

naive_type = ['simple_average']
olr = ['zscore']#['outliers','zscore','IQR']
lyr = ['top_30cm','30cm_300cm']
thr = ['100']#['0','25','50','75','100']
rmp_type = ['con']#['nn','bil','con']
tmp_type = ['raw_temp']
temp_thr_o = '-2C'#['0C','-2C','-5C','-10C']
tmp_val = -10
permafrost_type = ['RS_2002_all']

############# Loop through data ###########

for i in rmp_type:
    rmp_type_i = i
    remap_type = ''.join(['remap'+rmp_type_i])

    for j in naive_type:
    	naive_type_j = j

    	for k in olr:
    		olr_k = k

    		for l in lyr:
    			lyr_l = l


    			for m in thr:
    				thr_m = m


    				for o in permafrost_type:
    					permafrost_type_o = o

    					print("Remap Type:",remap_type)
    					print("Layer:",lyr_l)
    					print("Temp Threshold:", temp_thr_o)
    					print("Permafrost Type:", permafrost_type_o)

###### Overall (across all validation grid cells) ######

#### Cold Season ####


## Master Arrays ##

    					lat_cold_master = []
    					lon_cold_master = []
    					gcell_cold_master = []
    					naive_bias_cold_master = []
    					naive_noJRA_bias_cold_master = []
    					naive_noJRAold_bias_cold_master = []
    					naive_all_bias_cold_master = []
    					CFSR_bias_cold_master = []
    					ERAI_bias_cold_master = []
    					ERA5_bias_cold_master = []
    					ERA5_Land_bias_cold_master = []
    					JRA_bias_cold_master = []
    					MERRA2_bias_cold_master = []
    					GLDAS_bias_cold_master = []
    					GLDAS_CLSM_bias_cold_master = []

    					stn_var_cold_master = []
    					naive_var_cold_master = []
    					naive_noJRA_var_cold_master = []
    					naive_noJRAold_var_cold_master = []
    					naive_all_var_cold_master = []
    					CFSR_var_cold_master = []
    					ERAI_var_cold_master = []
    					ERA5_var_cold_master = []
    					ERA5_Land_var_cold_master = []
    					JRA_var_cold_master = []
    					MERRA2_var_cold_master = []
    					GLDAS_var_cold_master = []
    					GLDAS_CLSM_var_cold_master = []

    					stn_std_cold_master = []
    					naive_std_cold_master = []
    					naive_noJRA_std_cold_master = []
    					naive_noJRAold_std_cold_master = []
    					naive_all_std_cold_master = []
    					CFSR_std_cold_master = []
    					ERAI_std_cold_master = []
    					ERA5_std_cold_master = []
    					ERA5_Land_std_cold_master = []
    					JRA_std_cold_master = []
    					MERRA2_std_cold_master = []
    					GLDAS_std_cold_master = []
    					GLDAS_CLSM_std_cold_master = []

    					naive_rmse_cold_master = []
    					naive_noJRA_rmse_cold_master = []
    					naive_noJRAold_rmse_cold_master = []
    					naive_all_rmse_cold_master = []
    					CFSR_rmse_cold_master = []
    					ERAI_rmse_cold_master = []
    					ERA5_rmse_cold_master = []
    					ERA5_Land_rmse_cold_master = []
    					JRA_rmse_cold_master = []
    					MERRA2_rmse_cold_master = []
    					GLDAS_rmse_cold_master = []
    					GLDAS_CLSM_rmse_cold_master = []

    					naive_ubrmse_cold_master = []
    					naive_noJRA_ubrmse_cold_master = []
    					naive_noJRAold_ubrmse_cold_master = []
    					naive_all_ubrmse_cold_master = []
    					CFSR_ubrmse_cold_master = []
    					ERAI_ubrmse_cold_master = []
    					ERA5_ubrmse_cold_master = []
    					ERA5_Land_ubrmse_cold_master = []
    					JRA_ubrmse_cold_master = []
    					MERRA2_ubrmse_cold_master = []
    					GLDAS_ubrmse_cold_master = []
    					GLDAS_CLSM_ubrmse_cold_master = []

    					naive_corr_cold_master = []
    					naive_noJRA_corr_cold_master = []
    					naive_noJRAold_corr_cold_master = []
    					naive_all_corr_cold_master = []
    					CFSR_corr_cold_master = []
    					ERAI_corr_cold_master = []
    					ERA5_corr_cold_master = []
    					ERA5_Land_corr_cold_master = []
    					JRA_corr_cold_master = []
    					MERRA2_corr_cold_master = []
    					GLDAS_corr_cold_master = []
    					GLDAS_CLSM_corr_cold_master = []

## Grab Data ## 
    					fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/CFSR_res/'+str(remap_type)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_dframe_scatterplot_CMOS_CLSM_subset_permafrost_cold_warm_CRU_Sep2021_airtemp_CFSR_NWT_Yukon.csv'])
    					dframe = pd.read_csv(fil)
    					dframe_cold_season = dframe[dframe['Season'] == 'Cold']
    					dframe_warm_season = dframe[dframe['Season'] == 'Warm']
    					gcell_cold = dframe_cold_season['Grid Cell'].values
    					gcell_cold_uq = np.unique(gcell_cold)

    					dframe_cold_season_permafrost = dframe_cold_season

    					gcell_cold = dframe_cold_season_permafrost['Grid Cell'].values
    					gcell_cold_uq = np.unique(gcell_cold)

    					for p in gcell_cold_uq: # loop through grid cells
    						gcell_p = p
    						if (gcell_p == 33777):
    							continue
    						dframe_cold_season_gcell = dframe_cold_season_permafrost[dframe_cold_season_permafrost['Grid Cell'] == gcell_p]


    						if (len(dframe_cold_season_gcell) < 2):
    							print("length of grid cell", gcell_p, " is less than 2 - Skipped Grid Cell")
    							continue

    						gcell_cold_master.append(gcell_p)
    						lat_cold = dframe_cold_season_gcell['Central Lat'].iloc[0]
    						lat_cold_master.append(lat_cold)
    						lon_cold = dframe_cold_season_gcell['Central Lon'].iloc[0]
    						lon_cold_master.append(lon_cold)    						
    						station_temp_cold = dframe_cold_season_gcell['Station'].values
    						naive_all_temp_cold = dframe_cold_season_gcell['Ensemble Mean'].values
    						CFSR_temp_cold = dframe_cold_season_gcell['CFSR'].values
    						ERA5_temp_cold = dframe_cold_season_gcell['ERA5'].values
    						ERA5_Land_temp_cold = dframe_cold_season_gcell['ERA5-Land'].values
    						GLDAS_temp_cold = dframe_cold_season_gcell['GLDAS-Noah'].values



## Bias ##
    						naive_all_bias_cold = bias(naive_all_temp_cold, station_temp_cold)
    						naive_all_bias_cold_master.append(naive_all_bias_cold)
    						CFSR_bias_cold = bias(CFSR_temp_cold, station_temp_cold)
    						CFSR_bias_cold_master.append(CFSR_bias_cold)
    						ERA5_bias_cold = bias(ERA5_temp_cold, station_temp_cold)
    						ERA5_bias_cold_master.append(ERA5_bias_cold)
    						ERA5_Land_bias_cold = bias(ERA5_Land_temp_cold, station_temp_cold)
    						ERA5_Land_bias_cold_master.append(ERA5_Land_bias_cold)
    						GLDAS_bias_cold = bias(GLDAS_temp_cold, station_temp_cold)
    						GLDAS_bias_cold_master.append(GLDAS_bias_cold)

## Variance ##

    						stn_var_cold =  np.var(station_temp_cold)
    						stn_var_cold_master.append(stn_var_cold)
    						naive_all_var_cold = np.var(naive_all_temp_cold)
    						naive_all_var_cold_master.append(naive_all_var_cold)					
    						CFSR_var_cold = np.var(CFSR_temp_cold)
    						CFSR_var_cold_master.append(CFSR_var_cold)   					
    						ERA5_var_cold = np.var(ERA5_temp_cold)
    						ERA5_var_cold_master.append(ERA5_var_cold)
    						ERA5_Land_var_cold = np.var(ERA5_Land_temp_cold)
    						ERA5_Land_var_cold_master.append(ERA5_Land_var_cold)
    						GLDAS_var_cold = np.var(GLDAS_temp_cold)
    						GLDAS_var_cold_master.append(GLDAS_var_cold)

## Std Dev ##

    						stn_std_cold =  np.std(station_temp_cold)
    						stn_std_cold_master.append(stn_std_cold)
    						naive_all_std_cold = np.std(naive_all_temp_cold)
    						naive_all_std_cold_master.append(naive_all_std_cold)					
    						CFSR_std_cold = np.std(CFSR_temp_cold)
    						CFSR_std_cold_master.append(CFSR_std_cold)   					
    						ERA5_std_cold = np.std(ERA5_temp_cold)
    						ERA5_std_cold_master.append(ERA5_std_cold)
    						ERA5_Land_std_cold = np.std(ERA5_Land_temp_cold)
    						ERA5_Land_std_cold_master.append(ERA5_Land_std_cold)
    						GLDAS_std_cold = np.std(GLDAS_temp_cold)
    						GLDAS_std_cold_master.append(GLDAS_std_cold)


## RMSE and ubRMSE ##
    						naive_all_rmse_cold = mean_squared_error(station_temp_cold,naive_all_temp_cold, squared=False)
    						naive_all_rmse_cold_master.append(naive_all_rmse_cold)
    						CFSR_rmse_cold = mean_squared_error(station_temp_cold,CFSR_temp_cold, squared=False)
    						CFSR_rmse_cold_master.append(CFSR_rmse_cold)
    						ERA5_rmse_cold = mean_squared_error(station_temp_cold,ERA5_temp_cold, squared=False)
    						ERA5_rmse_cold_master.append(ERA5_rmse_cold)
    						ERA5_Land_rmse_cold = mean_squared_error(station_temp_cold,ERA5_Land_temp_cold, squared=False)
    						ERA5_Land_rmse_cold_master.append(ERA5_Land_rmse_cold)
    						GLDAS_rmse_cold = mean_squared_error(station_temp_cold,GLDAS_temp_cold, squared=False)
    						GLDAS_rmse_cold_master.append(GLDAS_rmse_cold)


    						naive_all_ubrmse_cold = ubrmsd(station_temp_cold,naive_all_temp_cold)
    						naive_all_ubrmse_cold_master.append(naive_all_ubrmse_cold)
    						CFSR_ubrmse_cold = ubrmsd(station_temp_cold,CFSR_temp_cold)
    						CFSR_ubrmse_cold_master.append(CFSR_ubrmse_cold)
    						ERA5_ubrmse_cold = ubrmsd(station_temp_cold,ERA5_temp_cold)
    						ERA5_ubrmse_cold_master.append(ERA5_ubrmse_cold)
    						ERA5_Land_ubrmse_cold = ubrmsd(station_temp_cold,ERA5_Land_temp_cold)
    						ERA5_Land_ubrmse_cold_master.append(ERA5_Land_ubrmse_cold)
    						GLDAS_ubrmse_cold = ubrmsd(station_temp_cold,GLDAS_temp_cold)
    						GLDAS_ubrmse_cold_master.append(GLDAS_ubrmse_cold)



## Pearson Correlations ##
    						naive_all_corr_cold,_ = pearsonr(naive_all_temp_cold, station_temp_cold)
    						naive_all_corr_cold_master.append(naive_all_corr_cold)
    						CFSR_corr_cold,_ = pearsonr(CFSR_temp_cold, station_temp_cold)
    						CFSR_corr_cold_master.append(CFSR_corr_cold)
    						ERA5_corr_cold,_ = pearsonr(ERA5_temp_cold, station_temp_cold)
    						ERA5_corr_cold_master.append(ERA5_corr_cold)
    						ERA5_Land_corr_cold,_ = pearsonr(ERA5_Land_temp_cold, station_temp_cold)
    						ERA5_Land_corr_cold_master.append(ERA5_Land_corr_cold)
    						GLDAS_corr_cold,_ = pearsonr(GLDAS_temp_cold, station_temp_cold)
    						GLDAS_corr_cold_master.append(GLDAS_corr_cold)



#### Warm Season ####


## Master Arrays ##

    					lat_warm_master = []
    					lon_warm_master = []
    					gcell_warm_master = []
    					naive_bias_warm_master = []
    					naive_noJRA_bias_warm_master = []
    					naive_noJRAold_bias_warm_master = []
    					naive_all_bias_warm_master = []
    					CFSR_bias_warm_master = []
    					ERAI_bias_warm_master = []
    					ERA5_bias_warm_master = []
    					ERA5_Land_bias_warm_master = []
    					JRA_bias_warm_master = []
    					MERRA2_bias_warm_master = []
    					GLDAS_bias_warm_master = []
    					GLDAS_CLSM_bias_warm_master = []

    					stn_var_warm_master = []
    					naive_var_warm_master = []
    					naive_noJRA_var_warm_master = []
    					naive_noJRAold_var_warm_master = []
    					naive_all_var_warm_master = []
    					CFSR_var_warm_master = []
    					ERAI_var_warm_master = []
    					ERA5_var_warm_master = []
    					ERA5_Land_var_warm_master = []
    					JRA_var_warm_master = []
    					MERRA2_var_warm_master = []
    					GLDAS_var_warm_master = []
    					GLDAS_CLSM_var_warm_master = []

    					stn_std_warm_master = []
    					naive_std_warm_master = []
    					naive_noJRA_std_warm_master = []
    					naive_noJRAold_std_warm_master = []
    					naive_all_std_warm_master = []
    					CFSR_std_warm_master = []
    					ERAI_std_warm_master = []
    					ERA5_std_warm_master = []
    					ERA5_Land_std_warm_master = []
    					JRA_std_warm_master = []
    					MERRA2_std_warm_master = []
    					GLDAS_std_warm_master = []
    					GLDAS_CLSM_std_warm_master = []

    					naive_rmse_warm_master = []
    					naive_noJRA_rmse_warm_master = []
    					naive_noJRAold_rmse_warm_master = []
    					naive_all_rmse_warm_master = []
    					CFSR_rmse_warm_master = []
    					ERAI_rmse_warm_master = []
    					ERA5_rmse_warm_master = []
    					ERA5_Land_rmse_warm_master = []
    					JRA_rmse_warm_master = []
    					MERRA2_rmse_warm_master = []
    					GLDAS_rmse_warm_master = []
    					GLDAS_CLSM_rmse_warm_master = []

    					naive_ubrmse_warm_master = []
    					naive_noJRA_ubrmse_warm_master = []
    					naive_noJRAold_ubrmse_warm_master = []
    					naive_all_ubrmse_warm_master = []
    					CFSR_ubrmse_warm_master = []
    					ERAI_ubrmse_warm_master = []
    					ERA5_ubrmse_warm_master = []
    					ERA5_Land_ubrmse_warm_master = []
    					JRA_ubrmse_warm_master = []
    					MERRA2_ubrmse_warm_master = []
    					GLDAS_ubrmse_warm_master = []
    					GLDAS_CLSM_ubrmse_warm_master = []

    					naive_corr_warm_master = []
    					naive_noJRA_corr_warm_master = []
    					naive_noJRAold_corr_warm_master = []
    					naive_all_corr_warm_master = []
    					CFSR_corr_warm_master = []
    					ERAI_corr_warm_master = []
    					ERA5_corr_warm_master = []
    					ERA5_Land_corr_warm_master = []
    					JRA_corr_warm_master = []
    					MERRA2_corr_warm_master = []
    					GLDAS_corr_warm_master = []
    					GLDAS_CLSM_corr_warm_master = []

## Grab Data ## 

    					gcell_warm = dframe_warm_season['Grid Cell'].values
    					gcell_warm_uq = np.unique(gcell_warm)

    					dframe_warm_season_permafrost = dframe_warm_season

    					gcell_warm = dframe_warm_season_permafrost['Grid Cell'].values
    					gcell_warm_uq = np.unique(gcell_warm)

    					for p in gcell_warm_uq: # loop through grid cells
    						gcell_p = p
    						if (gcell_p == 33777):
    							continue
    						dframe_warm_season_gcell = dframe_warm_season_permafrost[dframe_warm_season_permafrost['Grid Cell'] == gcell_p]

    						if (len(dframe_warm_season_gcell) < 2):
    							print("length of grid cell", gcell_p, " is less than 2 - Skipped Grid Cell")
    							continue

    						gcell_warm_master.append(gcell_p)
    						lat_warm = dframe_warm_season_gcell['Central Lat'].iloc[0]
    						lat_warm_master.append(lat_warm)
    						lon_warm = dframe_warm_season_gcell['Central Lon'].iloc[0]
    						lon_warm_master.append(lon_warm)    						
    						station_temp_warm = dframe_warm_season_gcell['Station'].values
    						naive_all_temp_warm = dframe_warm_season_gcell['Ensemble Mean'].values
    						CFSR_temp_warm = dframe_warm_season_gcell['CFSR'].values
    						ERA5_temp_warm = dframe_warm_season_gcell['ERA5'].values
    						ERA5_Land_temp_warm = dframe_warm_season_gcell['ERA5-Land'].values
    						GLDAS_temp_warm = dframe_warm_season_gcell['GLDAS-Noah'].values



## Bias ##
    						naive_all_bias_warm = bias(naive_all_temp_warm, station_temp_warm)
    						naive_all_bias_warm_master.append(naive_all_bias_warm)
    						CFSR_bias_warm = bias(CFSR_temp_warm, station_temp_warm)
    						CFSR_bias_warm_master.append(CFSR_bias_warm)
    						ERA5_bias_warm = bias(ERA5_temp_warm, station_temp_warm)
    						ERA5_bias_warm_master.append(ERA5_bias_warm)
    						ERA5_Land_bias_warm = bias(ERA5_Land_temp_warm, station_temp_warm)
    						ERA5_Land_bias_warm_master.append(ERA5_Land_bias_warm)
    						GLDAS_bias_warm = bias(GLDAS_temp_warm, station_temp_warm)
    						GLDAS_bias_warm_master.append(GLDAS_bias_warm)

## Variance ##

    						stn_var_warm =  np.var(station_temp_warm)
    						stn_var_warm_master.append(stn_var_warm)
    						naive_all_var_warm = np.var(naive_all_temp_warm)
    						naive_all_var_warm_master.append(naive_all_var_warm)					
    						CFSR_var_warm = np.var(CFSR_temp_warm)
    						CFSR_var_warm_master.append(CFSR_var_warm)   					
    						ERA5_var_warm = np.var(ERA5_temp_warm)
    						ERA5_var_warm_master.append(ERA5_var_warm)
    						ERA5_Land_var_warm = np.var(ERA5_Land_temp_warm)
    						ERA5_Land_var_warm_master.append(ERA5_Land_var_warm)
    						GLDAS_var_warm = np.var(GLDAS_temp_warm)
    						GLDAS_var_warm_master.append(GLDAS_var_warm)

## Std Dev ##

    						stn_std_warm =  np.std(station_temp_warm)
    						stn_std_warm_master.append(stn_std_warm)
    						naive_all_std_warm = np.std(naive_all_temp_warm)
    						naive_all_std_warm_master.append(naive_all_std_warm)					
    						CFSR_std_warm = np.std(CFSR_temp_warm)
    						CFSR_std_warm_master.append(CFSR_std_warm)   					
    						ERA5_std_warm = np.std(ERA5_temp_warm)
    						ERA5_std_warm_master.append(ERA5_std_warm)
    						ERA5_Land_std_warm = np.std(ERA5_Land_temp_warm)
    						ERA5_Land_std_warm_master.append(ERA5_Land_std_warm)
    						GLDAS_std_warm = np.std(GLDAS_temp_warm)
    						GLDAS_std_warm_master.append(GLDAS_std_warm)


## RMSE and ubRMSE ##
    						naive_all_rmse_warm = mean_squared_error(station_temp_warm,naive_all_temp_warm, squared=False)
    						naive_all_rmse_warm_master.append(naive_all_rmse_warm)
    						CFSR_rmse_warm = mean_squared_error(station_temp_warm,CFSR_temp_warm, squared=False)
    						CFSR_rmse_warm_master.append(CFSR_rmse_warm)
    						ERA5_rmse_warm = mean_squared_error(station_temp_warm,ERA5_temp_warm, squared=False)
    						ERA5_rmse_warm_master.append(ERA5_rmse_warm)
    						ERA5_Land_rmse_warm = mean_squared_error(station_temp_warm,ERA5_Land_temp_warm, squared=False)
    						ERA5_Land_rmse_warm_master.append(ERA5_Land_rmse_warm)
    						GLDAS_rmse_warm = mean_squared_error(station_temp_warm,GLDAS_temp_warm, squared=False)
    						GLDAS_rmse_warm_master.append(GLDAS_rmse_warm)


    						naive_all_ubrmse_warm = ubrmsd(station_temp_warm,naive_all_temp_warm)
    						naive_all_ubrmse_warm_master.append(naive_all_ubrmse_warm)
    						CFSR_ubrmse_warm = ubrmsd(station_temp_warm,CFSR_temp_warm)
    						CFSR_ubrmse_warm_master.append(CFSR_ubrmse_warm)
    						ERA5_ubrmse_warm = ubrmsd(station_temp_warm,ERA5_temp_warm)
    						ERA5_ubrmse_warm_master.append(ERA5_ubrmse_warm)
    						ERA5_Land_ubrmse_warm = ubrmsd(station_temp_warm,ERA5_Land_temp_warm)
    						ERA5_Land_ubrmse_warm_master.append(ERA5_Land_ubrmse_warm)
    						GLDAS_ubrmse_warm = ubrmsd(station_temp_warm,GLDAS_temp_warm)
    						GLDAS_ubrmse_warm_master.append(GLDAS_ubrmse_warm)



## Pearson Correlations ##
    						naive_all_corr_warm,_ = pearsonr(naive_all_temp_warm, station_temp_warm)
    						naive_all_corr_warm_master.append(naive_all_corr_warm)
    						CFSR_corr_warm,_ = pearsonr(CFSR_temp_warm, station_temp_warm)
    						CFSR_corr_warm_master.append(CFSR_corr_warm)
    						ERA5_corr_warm,_ = pearsonr(ERA5_temp_warm, station_temp_warm)
    						ERA5_corr_warm_master.append(ERA5_corr_warm)
    						ERA5_Land_corr_warm,_ = pearsonr(ERA5_Land_temp_warm, station_temp_warm)
    						ERA5_Land_corr_warm_master.append(ERA5_Land_corr_warm)
    						GLDAS_corr_warm,_ = pearsonr(GLDAS_temp_warm, station_temp_warm)
    						GLDAS_corr_warm_master.append(GLDAS_corr_warm)


########## Save CSV Files #########

    					dframe_cold_final = pd.DataFrame(data=gcell_cold_master, columns=['Grid Cell'])
    					dframe_cold_final['Lat'] = lat_cold_master
    					dframe_cold_final['Lon'] = lon_cold_master
    					dframe_cold_final['Bias'] = naive_all_bias_cold_master
    					dframe_cold_final['RMSE'] = naive_all_rmse_cold_master
    					dframe_cold_final['Correlation'] = naive_all_corr_cold_master
    					dframe_cold_final['Norm Std'] = naive_all_std_cold_master

    					metrics_cold_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CFSR_res/'+str(remap_type)+'_'+str(lyr_l)+'_naive_metrics_CLSM_EnsMean_by_grid_cell_CRU_cold_season_Sep2021_CFSR_NWT_Yukon.csv'])  					
    					dframe_cold_final.to_csv(metrics_cold_fil)
    					print(metrics_cold_fil)

    					dframe_cold_final = pd.DataFrame(data=gcell_cold_master, columns=['Grid Cell'])
    					dframe_cold_final['Lat'] = lat_cold_master
    					dframe_cold_final['Lon'] = lon_cold_master
    					dframe_cold_final['Bias'] = CFSR_bias_cold_master
    					dframe_cold_final['RMSE'] = CFSR_rmse_cold_master
    					dframe_cold_final['Correlation'] = CFSR_corr_cold_master
    					dframe_cold_final['Norm Std'] = CFSR_std_cold_master

    					metrics_cold_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CFSR_res/'+str(remap_type)+'_'+str(lyr_l)+'_naive_metrics_CLSM_CFSR_by_grid_cell_CRU_cold_season_Sep2021_CFSR_NWT_Yukon.csv'])  					
    					dframe_cold_final.to_csv(metrics_cold_fil)
    					print(metrics_cold_fil)

    					dframe_cold_final = pd.DataFrame(data=gcell_cold_master, columns=['Grid Cell'])
    					dframe_cold_final['Lat'] = lat_cold_master
    					dframe_cold_final['Lon'] = lon_cold_master
    					dframe_cold_final['Bias'] = ERA5_bias_cold_master
    					dframe_cold_final['RMSE'] = ERA5_rmse_cold_master
    					dframe_cold_final['Correlation'] = ERA5_corr_cold_master
    					dframe_cold_final['Norm Std'] = ERA5_std_cold_master

    					metrics_cold_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CFSR_res/'+str(remap_type)+'_'+str(lyr_l)+'_naive_metrics_CLSM_ERA5_by_grid_cell_CRU_cold_season_Sep2021_CFSR.csv'])  					
    					dframe_cold_final.to_csv(metrics_cold_fil)
    					print(metrics_cold_fil)


    					dframe_cold_final = pd.DataFrame(data=gcell_cold_master, columns=['Grid Cell'])
    					dframe_cold_final['Lat'] = lat_cold_master
    					dframe_cold_final['Lon'] = lon_cold_master
    					dframe_cold_final['Bias'] = ERA5_Land_bias_cold_master
    					dframe_cold_final['RMSE'] = ERA5_Land_rmse_cold_master
    					dframe_cold_final['Correlation'] = ERA5_Land_corr_cold_master
    					dframe_cold_final['Norm Std'] = ERA5_Land_std_cold_master

    					metrics_cold_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CFSR_res/'+str(remap_type)+'_'+str(lyr_l)+'_naive_metrics_CLSM_ERA5_Land_by_grid_cell_CRU_cold_season_Sep2021_CFSR.csv'])  					
    					dframe_cold_final.to_csv(metrics_cold_fil)
    					print(metrics_cold_fil)

    					dframe_cold_final = pd.DataFrame(data=gcell_cold_master, columns=['Grid Cell'])
    					dframe_cold_final['Lat'] = lat_cold_master
    					dframe_cold_final['Lon'] = lon_cold_master
    					dframe_cold_final['Bias'] = GLDAS_bias_cold_master
    					dframe_cold_final['RMSE'] = GLDAS_rmse_cold_master
    					dframe_cold_final['Correlation'] = GLDAS_corr_cold_master
    					dframe_cold_final['Norm Std'] = GLDAS_std_cold_master

    					metrics_cold_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CFSR_res/'+str(remap_type)+'_'+str(lyr_l)+'_naive_metrics_CLSM_GLDAS_by_grid_cell_CRU_cold_season_Sep2021_CFSR.csv'])  					
    					dframe_cold_final.to_csv(metrics_cold_fil)
    					print(metrics_cold_fil)


    					dframe_warm_final = pd.DataFrame(data=gcell_warm_master, columns=['Grid Cell'])
    					dframe_warm_final['Lat'] = lat_warm_master
    					dframe_warm_final['Lon'] = lon_warm_master
    					dframe_warm_final['Bias'] = naive_all_bias_warm_master
    					dframe_warm_final['RMSE'] = naive_all_rmse_warm_master
    					dframe_warm_final['Correlation'] = naive_all_corr_warm_master
    					dframe_warm_final['Norm Std'] = naive_all_std_warm_master

    					metrics_warm_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CFSR_res/'+str(remap_type)+'_'+str(lyr_l)+'_naive_metrics_CLSM_EnsMean_by_grid_cell_CRU_warm_season_Sep2021_CFSR.csv'])  					
    					dframe_warm_final.to_csv(metrics_warm_fil)
    					print(metrics_warm_fil)

    					dframe_warm_final = pd.DataFrame(data=gcell_warm_master, columns=['Grid Cell'])
    					dframe_warm_final['Lat'] = lat_warm_master
    					dframe_warm_final['Lon'] = lon_warm_master
    					dframe_warm_final['Bias'] = CFSR_bias_warm_master
    					dframe_warm_final['RMSE'] = CFSR_rmse_warm_master
    					dframe_warm_final['Correlation'] = CFSR_corr_warm_master
    					dframe_warm_final['Norm Std'] = CFSR_std_warm_master

    					metrics_warm_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CFSR_res/'+str(remap_type)+'_'+str(lyr_l)+'_naive_metrics_CLSM_CFSR_by_grid_cell_CRU_warm_season_Sep2021_CFSR.csv'])  					
    					dframe_warm_final.to_csv(metrics_warm_fil)
    					print(metrics_warm_fil)


    					dframe_warm_final = pd.DataFrame(data=gcell_warm_master, columns=['Grid Cell'])
    					dframe_warm_final['Lat'] = lat_warm_master
    					dframe_warm_final['Lon'] = lon_warm_master
    					dframe_warm_final['Bias'] = ERA5_bias_warm_master
    					dframe_warm_final['RMSE'] = ERA5_rmse_warm_master
    					dframe_warm_final['Correlation'] = ERA5_corr_warm_master
    					dframe_warm_final['Norm Std'] = ERA5_std_warm_master

    					metrics_warm_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CFSR_res/'+str(remap_type)+'_'+str(lyr_l)+'_naive_metrics_CLSM_ERA5_by_grid_cell_CRU_warm_season_Sep2021_CFSR.csv'])  					
    					dframe_warm_final.to_csv(metrics_warm_fil)
    					print(metrics_warm_fil)


    					dframe_warm_final = pd.DataFrame(data=gcell_warm_master, columns=['Grid Cell'])
    					dframe_warm_final['Lat'] = lat_warm_master
    					dframe_warm_final['Lon'] = lon_warm_master
    					dframe_warm_final['Bias'] = ERA5_Land_bias_warm_master
    					dframe_warm_final['RMSE'] = ERA5_Land_rmse_warm_master
    					dframe_warm_final['Correlation'] = ERA5_Land_corr_warm_master
    					dframe_warm_final['Norm Std'] = ERA5_Land_std_warm_master

    					metrics_warm_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CFSR_res/'+str(remap_type)+'_'+str(lyr_l)+'_naive_metrics_CLSM_ERA5_Land_by_grid_cell_CRU_warm_season_Sep2021_CFSR.csv'])  					
    					dframe_warm_final.to_csv(metrics_warm_fil)
    					print(metrics_warm_fil)


    					dframe_warm_final = pd.DataFrame(data=gcell_warm_master, columns=['Grid Cell'])
    					dframe_warm_final['Lat'] = lat_warm_master
    					dframe_warm_final['Lon'] = lon_warm_master
    					dframe_warm_final['Bias'] = GLDAS_bias_warm_master
    					dframe_warm_final['RMSE'] = GLDAS_rmse_warm_master
    					dframe_warm_final['Correlation'] = GLDAS_corr_warm_master
    					dframe_warm_final['Norm Std'] = GLDAS_std_warm_master

    					metrics_warm_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CFSR_res/'+str(remap_type)+'_'+str(lyr_l)+'_naive_metrics_CLSM_GLDAS_by_grid_cell_CRU_warm_season_Sep2021_CFSR.csv'])  					
    					dframe_warm_final.to_csv(metrics_warm_fil)
    					print(metrics_warm_fil)



import os
import csv
import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import matplotlib.cm as cm
import matplotlib.patches as mpl_patches
import numpy as np
import scipy
import pandas as pd
import geopandas as gpd
import xarray as xr
import seaborn as sns
import shapely
import math
import cftime
import re
import cdms2
from decimal import *
from calendar import isleap
from shapely.geometry import Polygon, Point, GeometryCollection
from dateutil.relativedelta import *
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable


#product = ['EnsMean', 'CFSR','ERAI','ERA5','ERA5_Land','JRA55','MERRA2','GLDAS','GLDAS_CLSM']
#product = ['EnsMean', 'CFSR','ERA5','ERA5_Land','GLDAS']
product = ['GLDAS']
########## Create Plots ###########

for i in product:
    product_i = i
    print('Product:',product_i)    	
    top_cold_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CFSR_res/remapcon_top_30cm_naive_metrics_CLSM_'+str(product_i)+'_by_grid_cell_CRU_cold_season_Sep2021_CFSR.csv'])
    top_warm_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CFSR_res/remapcon_top_30cm_naive_metrics_CLSM_'+str(product_i)+'_by_grid_cell_CRU_warm_season_Sep2021_CFSR.csv'])
    btm_cold_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CFSR_res/remapcon_30cm_300cm_naive_metrics_CLSM_'+str(product_i)+'_by_grid_cell_CRU_cold_season_Sep2021_CFSR.csv'])
    btm_warm_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CFSR_res/remapcon_30cm_300cm_naive_metrics_CLSM_'+str(product_i)+'_by_grid_cell_CRU_warm_season_Sep2021_CFSR.csv'])

    dframe_top_cold = pd.read_csv(top_cold_fil)
    lat_top_cold = dframe_top_cold['Lat'].values
    lon_top_cold = dframe_top_cold['Lon'].values
    bias_top_cold = dframe_top_cold['Bias'].values
    RMSE_top_cold = dframe_top_cold['RMSE'].values
    corr_top_cold = dframe_top_cold['Correlation'].values
    std_top_cold = dframe_top_cold['Norm Std'].values

    dframe_btm_cold = pd.read_csv(btm_cold_fil)
    lat_btm_cold = dframe_btm_cold['Lat'].values
    lon_btm_cold = dframe_btm_cold['Lon'].values
    bias_btm_cold = dframe_btm_cold['Bias'].values
    RMSE_btm_cold = dframe_btm_cold['RMSE'].values
    corr_btm_cold = dframe_btm_cold['Correlation'].values
    std_btm_cold = dframe_btm_cold['Norm Std'].values

    dframe_top_warm = pd.read_csv(top_warm_fil)
    lat_top_warm = dframe_top_warm['Lat'].values
    lon_top_warm = dframe_top_warm['Lon'].values
    bias_top_warm = dframe_top_warm['Bias'].values
    RMSE_top_warm = dframe_top_warm['RMSE'].values 
    corr_top_warm = dframe_top_warm['Correlation'].values
    std_top_warm = dframe_top_warm['Norm Std'].values

    dframe_btm_warm = pd.read_csv(btm_warm_fil)
    lat_btm_warm = dframe_btm_warm['Lat'].values
    lon_btm_warm = dframe_btm_warm['Lon'].values
    bias_btm_warm = dframe_btm_warm['Bias'].values
    RMSE_btm_warm = dframe_btm_warm['RMSE'].values
    corr_btm_warm = dframe_btm_warm['Correlation'].values
    std_btm_warm = dframe_btm_warm['Norm Std'].values
    #print(lat_top_cold)
    #print(lon_top_cold)



###### Create bias figures ######

    fig = plt.figure(figsize = (5,5))

    #create array for parallels
    parallels = np.arange(50.,91.,10.)
    meridians = np.arange(0.,351.,10.)
    # Bias Top-30cm Subplot Cold #
    ax1 = fig.add_subplot(111)
    map1 = Basemap(projection='npstere',boundinglat=45,lon_0=0,resolution='h',llcrnrlat=45,urcrnrlat=45,llcrnrlon=-45,urcrnrlon=45)
    map1.drawmapboundary(fill_color='aqua',zorder=0) # fill to edge
    map1.fillcontinents(color='darkgrey',lake_color='aqua',zorder=1)
    #map1.drawcoastlines()

    #draw parallels on map
    map1.drawparallels(parallels,labels=[False,True,True,False])
    map1.drawmeridians(meridians,labels=[False,False,False,False])

    #plot scatterplot
    x1,y1 = map1(lon_top_cold,lat_top_cold)
    GridCells1 = ax1.scatter(x1,y1,alpha=0.7,marker='.',s=120,c=bias_top_cold, cmap='bwr', vmin=-10,vmax=10,zorder=2)
    #ax1.clim(-12,12)
    #ax1.colorbar(label='Ensemble Mean Temperature Bias ($\circ$ C)') 
    plt.title('Near Surface Cold Season Bias',fontsize=14,weight='bold') 
    divider = make_axes_locatable(ax1)
    cax=divider.append_axes("right",size="5%", pad=0.05)
    cb = plt.colorbar(GridCells1,cax)
    cb.set_label(label = 'Soil Temperature Bias ($\circ$ C)', size='large')
    cb.ax.tick_params(labelsize='large')
    plt.tight_layout()
    plt_nam = ''.join(['/mnt/data/users/herringtont/soil_temp/plots/validation_sites/CFSR_res/Bias_top30_'+str(product_i)+'_cold_Sep2021.png'])
    fig.savefig(plt_nam)
    plt.close()

    # Bias Top-30cm Subplot Warm #
    fig = plt.figure(figsize = (5,5))
    ax2 = fig.add_subplot(111)
    map2 = Basemap(projection='npstere',boundinglat=45,lon_0=0,resolution='h',llcrnrlat=45,urcrnrlat=45,llcrnrlon=-45,urcrnrlon=45)
    map2.drawmapboundary(fill_color='aqua',zorder=0) # fill to edge
    map2.fillcontinents(color='darkgrey',lake_color='aqua',zorder=1)
    #map2.drawcoastlines()

    #draw parallels on map
    map2.drawparallels(parallels,labels=[False,True,True,False])
    map2.drawmeridians(meridians,labels=[False,False,False,False])

    #plot scatterplot
    x2,y2 = map2(lon_top_warm,lat_top_warm)
    GridCells2 = ax2.scatter(x2,y2,alpha=0.7,marker='.',s=120,c=bias_top_warm, cmap='bwr', vmin=-10,vmax=10,zorder=2)
    #ax2.clim(-12,12)
    #ax2.colorbar(label='Ensemble Mean Temperature Bias ($\circ$ C)') 
    plt.title('Near Surface Warm Season Bias', fontsize=14,weight='bold')
    divider = make_axes_locatable(ax2)
    cax=divider.append_axes("right",size="5%", pad=0.05)
    cb = plt.colorbar(GridCells2,cax)
    cb.set_label(label = 'Soil Temperature Bias ($\circ$ C)', size='large')
    cb.ax.tick_params(labelsize='large')
    plt.tight_layout()
    plt_nam = ''.join(['/mnt/data/users/herringtont/soil_temp/plots/validation_sites/CFSR_res/Bias_top30_'+str(product_i)+'_warm_Sep2021.png'])
    fig.savefig(plt_nam)
    plt.close()



    # Bias Depth Subplot Cold #
    fig = plt.figure(figsize = (5,5))
    ax3 = fig.add_subplot(111)
    map3 = Basemap(projection='npstere',boundinglat=45,lon_0=0,resolution='h',llcrnrlat=45,urcrnrlat=45,llcrnrlon=-45,urcrnrlon=45)
    map3.drawmapboundary(fill_color='aqua',zorder=0) # fill to edge
    map3.fillcontinents(color='darkgrey',lake_color='aqua',zorder=1)
    #map3.drawcoastlines()

    #draw parallels on map
    map3.drawparallels(parallels,labels=[True,True,True,True])
    map3.drawmeridians(meridians,labels=[False,False,False,False])

    #plot scatterplot
    x3,y3 = map3(lon_btm_cold,lat_btm_cold)
    GridCells3 = ax3.scatter(x3,y3,alpha=0.7,marker='.',s=120,c=bias_btm_cold, cmap='bwr', vmin=-10,vmax=10,zorder=2)
    #ax3.clim(-12,12)
    #ax3.colorbar(label='Ensemble Mean Temperature Bias ($\circ$ C)') 
    plt.title('Depth Cold Season Bias', fontsize=14,weight='bold')
    divider = make_axes_locatable(ax3)
    cax=divider.append_axes("right",size="5%", pad=0.05)
    cb = plt.colorbar(GridCells3,cax)
    cb.set_label(label = 'Soil Temperature Bias ($\circ$ C)', size='large')
    cb.ax.tick_params(labelsize='large')
    plt.tight_layout()
    plt_nam = ''.join(['/mnt/data/users/herringtont/soil_temp/plots/validation_sites/CFSR_res/Bias_depth_'+str(product_i)+'_cold_Sep2021.png'])
    fig.savefig(plt_nam)
    plt.close()


    # Bias Depth Subplot Warm #
    fig = plt.figure(figsize = (5,5))
    ax4 = fig.add_subplot(111)
    map4 = Basemap(projection='npstere',boundinglat=45,lon_0=0,resolution='h',llcrnrlat=45,urcrnrlat=45,llcrnrlon=-45,urcrnrlon=45)
    map4.drawmapboundary(fill_color='aqua',zorder=0) # fill to edge
    map4.fillcontinents(color='darkgrey',lake_color='aqua',zorder=1)
    #map4.drawcoastlines()

    #draw parallels on map
    map4.drawparallels(parallels,labels=[True,True,True,True])
    map4.drawmeridians(meridians,labels=[False,False,False,False])

    #plot scatterplot
    x4,y4 = map4(lon_btm_warm,lat_btm_warm)
    GridCells4 = ax4.scatter(x4,y4,alpha=0.7,marker='.',s=120,c=bias_btm_warm, cmap='bwr', vmin=-10,vmax=10,zorder=2)
    #ax4.clim(-12,12)
    plt.title('Depth Warm Season Bias', fontsize=14,weight='bold')
    divider = make_axes_locatable(ax4)
    cax=divider.append_axes("right",size="5%", pad=0.05)
    cb = plt.colorbar(GridCells4,cax)
    cb.set_label(label = 'Soil Temperature Bias ($\circ$ C)', size='large')
    cb.ax.tick_params(labelsize='large')
    plt.tight_layout()
    plt_nam = ''.join(['/mnt/data/users/herringtont/soil_temp/plots/validation_sites/CFSR_res/Bias_depth_'+str(product_i)+'_warm_Sep2021.png'])
    fig.savefig(plt_nam)
    plt.close()



###### Create RMSE figures ######
#
#    fig = plt.figure(figsize = (5,5))
#
#    #create array for parallels
#    parallels = np.arange(50.,91.,10.)
#    meridians = np.arange(0.,351.,10.)
#    # Bias Top-30cm Subplot Cold #
#    ax1 = fig.add_subplot(111)
#    map1 = Basemap(projection='npstere',boundinglat=45,lon_0=0,resolution='h',llcrnrlat=45,urcrnrlat=45,llcrnrlon=-45,urcrnrlon=45)
#    map1.drawmapboundary(fill_color='aqua',zorder=0) # fill to edge
#    map1.fillcontinents(color='darkgrey',lake_color='aqua',zorder=1)
#    #map1.drawcoastlines()
#
#    #draw parallels on map
#    map1.drawparallels(parallels,labels=[False,True,True,False])
#    map1.drawmeridians(meridians,labels=[False,False,False,False])
#
#    #plot scatterplot
#    x1,y1 = map1(lon_top_cold,lat_top_cold)
#    GridCells1 = ax1.scatter(x1,y1,alpha=0.7,marker='.',s=120,c=RMSE_top_cold, cmap='YlOrRd', vmin=0,vmax=10,zorder=2)
#    #ax1.clim(-12,12)
#    #ax1.colorbar(label='Ensemble Mean Temperature Bias ($\circ$ C)') 
#    plt.title('Near Surface Cold Season RMSE',fontsize=14,weight='bold') 
#    divider = make_axes_locatable(ax1)
#    cax=divider.append_axes("right",size="5%", pad=0.05)
#    cb = plt.colorbar(GridCells1,cax)
#    cb.set_label(label = 'Soil Temperature RMSE ($\circ$ C)', size='large')
#    cb.ax.tick_params(labelsize='large')
#    plt.tight_layout()
#    plt_nam = ''.join(['/mnt/data/users/herringtont/soil_temp/plots/validation_sites/new_data/CLSM_res/RMSE_top30_'+str(product_i)+'_cold_Sep2021.png'])
#    fig.savefig(plt_nam)
#    plt.close()
#
#    # RMSE Top-30cm Subplot Warm #
#    fig = plt.figure(figsize = (5,5))
#    ax2 = fig.add_subplot(111)
#    map2 = Basemap(projection='npstere',boundinglat=45,lon_0=0,resolution='h',llcrnrlat=45,urcrnrlat=45,llcrnrlon=-45,urcrnrlon=45)
#    map2.drawmapboundary(fill_color='aqua',zorder=0) # fill to edge
#    map2.fillcontinents(color='darkgrey',lake_color='aqua',zorder=1)
#    #map2.drawcoastlines()
#
#    #draw parallels on map
#    map2.drawparallels(parallels,labels=[False,True,True,False])
#    map2.drawmeridians(meridians,labels=[False,False,False,False])
#
#    #plot scatterplot
#    x2,y2 = map2(lon_top_warm,lat_top_warm)
#    GridCells2 = ax2.scatter(x2,y2,alpha=0.7,marker='.',s=120,c=RMSE_top_warm, cmap='YlOrRd', vmin=0,vmax=10,zorder=2)
#    #ax2.clim(-12,12)
#    #ax2.colorbar(label='Ensemble Mean Temperature Bias ($\circ$ C)') 
#    plt.title('Near Surface Warm Season RMSE', fontsize=14,weight='bold')
#    divider = make_axes_locatable(ax2)
#    cax=divider.append_axes("right",size="5%", pad=0.05)
#    cb = plt.colorbar(GridCells2,cax)
#    cb.set_label(label = 'Soil Temperature RMSE ($\circ$ C)', size='large')
#    cb.ax.tick_params(labelsize='large')
#    plt.tight_layout()
#    plt_nam = ''.join(['/mnt/data/users/herringtont/soil_temp/plots/validation_sites/new_data/CLSM_res/RMSE_top30_'+str(product_i)+'_warm_Sep2021.png'])
#    fig.savefig(plt_nam)
#    plt.close()
#
#
#
#    # RMSE Depth Subplot Cold #
#    fig = plt.figure(figsize = (5,5))
#    ax3 = fig.add_subplot(111)
#    map3 = Basemap(projection='npstere',boundinglat=45,lon_0=0,resolution='h',llcrnrlat=45,urcrnrlat=45,llcrnrlon=-45,urcrnrlon=45)
#    map3.drawmapboundary(fill_color='aqua',zorder=0) # fill to edge
#    map3.fillcontinents(color='darkgrey',lake_color='aqua',zorder=1)
#    #map3.drawcoastlines()
#
#    #draw parallels on map
#    map3.drawparallels(parallels,labels=[True,True,True,True])
#    map3.drawmeridians(meridians,labels=[False,False,False,False])
#
#    #plot scatterplot
#    x3,y3 = map3(lon_btm_cold,lat_btm_cold)
#    GridCells3 = ax3.scatter(x3,y3,alpha=0.7,marker='.',s=120,c=RMSE_btm_cold, cmap='YlOrRd', vmin=0,vmax=10,zorder=2)
#    #ax3.clim(-12,12)
#    #ax3.colorbar(label='Ensemble Mean Temperature Bias ($\circ$ C)') 
#    plt.title('Depth Cold Season RMSE', fontsize=14,weight='bold')
#    divider = make_axes_locatable(ax3)
#    cax=divider.append_axes("right",size="5%", pad=0.05)
#    cb = plt.colorbar(GridCells3,cax)
#    cb.set_label(label = 'Soil Temperature RMSE ($\circ$ C)', size='large')
#    cb.ax.tick_params(labelsize='large')
#    plt.tight_layout()
#    plt_nam = ''.join(['/mnt/data/users/herringtont/soil_temp/plots/validation_sites/new_data/CLSM_res/RMSE_depth_'+str(product_i)+'_cold_Sep2021.png'])
#    fig.savefig(plt_nam)
#    plt.close()
#
#
#    # Bias Depth Subplot Warm #
#    fig = plt.figure(figsize = (5,5))
#    ax4 = fig.add_subplot(111)
#    map4 = Basemap(projection='npstere',boundinglat=45,lon_0=0,resolution='h',llcrnrlat=45,urcrnrlat=45,llcrnrlon=-45,urcrnrlon=45)
#    map4.drawmapboundary(fill_color='aqua',zorder=0) # fill to edge
#    map4.fillcontinents(color='darkgrey',lake_color='aqua',zorder=1)
#    #map4.drawcoastlines()
#
#    #draw parallels on map
#    map4.drawparallels(parallels,labels=[True,True,True,True])
#    map4.drawmeridians(meridians,labels=[False,False,False,False])
#
#    #plot scatterplot
#    x4,y4 = map4(lon_btm_warm,lat_btm_warm)
#    GridCells4 = ax4.scatter(x4,y4,alpha=0.7,marker='.',s=120,c=RMSE_btm_warm, cmap='YlOrRd', vmin=0,vmax=10,zorder=2)
#    #ax4.clim(-12,12)
#    plt.title('Depth Warm Season RMSE', fontsize=14,weight='bold')
#    divider = make_axes_locatable(ax4)
#    cax=divider.append_axes("right",size="5%", pad=0.05)
#    cb = plt.colorbar(GridCells4,cax)
#    cb.set_label(label = 'Soil Temperature RMSE ($\circ$ C)', size='large')
#    cb.ax.tick_params(labelsize='large')
#    plt.tight_layout()
#    plt_nam = ''.join(['/mnt/data/users/herringtont/soil_temp/plots/validation_sites/new_data/CLSM_res/RMSE_depth_'+str(product_i)+'_warm_Sep2021.png'])
#    fig.savefig(plt_nam)
#    plt.close()
#
#
#
######## Create Correlation figures ######
#
#    fig = plt.figure(figsize = (5,5))
#
#    #create array for parallels
#    parallels = np.arange(50.,91.,10.)
#    meridians = np.arange(0.,351.,10.)
#    # Bias Top-30cm Subplot Cold #
#    ax1 = fig.add_subplot(111)
#    map1 = Basemap(projection='npstere',boundinglat=45,lon_0=0,resolution='h',llcrnrlat=45,urcrnrlat=45,llcrnrlon=-45,urcrnrlon=45)
#    map1.drawmapboundary(fill_color='aqua',zorder=0) # fill to edge
#    map1.fillcontinents(color='darkgrey',lake_color='aqua',zorder=1)
#    #map1.drawcoastlines()
#
#    #draw parallels on map
#    map1.drawparallels(parallels,labels=[False,True,True,False])
#    map1.drawmeridians(meridians,labels=[False,False,False,False])
#
#    #plot scatterplot
#    x1,y1 = map1(lon_top_cold,lat_top_cold)
#    GridCells1 = ax1.scatter(x1,y1,alpha=0.7,marker='.',s=120,c=corr_top_cold, cmap='Reds', vmin=0,vmax=1.0,zorder=2)
#    #ax1.clim(-12,12)
#    #ax1.colorbar(label='Ensemble Mean Temperature Bias ($\circ$ C)') 
#    plt.title('Near Surface Cold Season Correlation',fontsize=14,weight='bold') 
#    divider = make_axes_locatable(ax1)
#    cax=divider.append_axes("right",size="5%", pad=0.05)
#    cb = plt.colorbar(GridCells1,cax)
#    cb.set_label(label = 'Pearson Correlation', size='large')
#    cb.ax.tick_params(labelsize='large')
#    plt.tight_layout()
#    plt_nam = ''.join(['/mnt/data/users/herringtont/soil_temp/plots/validation_sites/new_data/CLSM_res/Corr_top30_'+str(product_i)+'_cold_Sep2021.png'])
#    fig.savefig(plt_nam)
#    plt.close()
#
#    # RMSE Top-30cm Subplot Warm #
#    fig = plt.figure(figsize = (5,5))
#    ax2 = fig.add_subplot(111)
#    map2 = Basemap(projection='npstere',boundinglat=45,lon_0=0,resolution='h',llcrnrlat=45,urcrnrlat=45,llcrnrlon=-45,urcrnrlon=45)
#    map2.drawmapboundary(fill_color='aqua',zorder=0) # fill to edge
#    map2.fillcontinents(color='darkgrey',lake_color='aqua',zorder=1)
#    #map2.drawcoastlines()
#
#    #draw parallels on map
#    map2.drawparallels(parallels,labels=[False,True,True,False])
#    map2.drawmeridians(meridians,labels=[False,False,False,False])
#
#    #plot scatterplot
#    x2,y2 = map2(lon_top_warm,lat_top_warm)
#    GridCells2 = ax2.scatter(x2,y2,alpha=0.7,marker='.',s=120,c=corr_top_warm, cmap='Reds', vmin=0,vmax=1.0,zorder=2)
#    #ax2.clim(-12,12)
#    #ax2.colorbar(label='Ensemble Mean Temperature Bias ($\circ$ C)') 
#    plt.title('Near Surface Warm Season Correlation', fontsize=14,weight='bold')
#    divider = make_axes_locatable(ax2)
#    cax=divider.append_axes("right",size="5%", pad=0.05)
#    cb = plt.colorbar(GridCells2,cax)
#    cb.set_label(label = 'Pearson Correlation', size='large')
#    cb.ax.tick_params(labelsize='large')
#    plt.tight_layout()
#    plt_nam = ''.join(['/mnt/data/users/herringtont/soil_temp/plots/validation_sites/new_data/CLSM_res/Corr_top30_'+str(product_i)+'_warm_Sep2021.png'])
#    fig.savefig(plt_nam)
#    plt.close()
#
#
#
#    # RMSE Depth Subplot Cold #
#    fig = plt.figure(figsize = (5,5))
#    ax3 = fig.add_subplot(111)
#    map3 = Basemap(projection='npstere',boundinglat=45,lon_0=0,resolution='h',llcrnrlat=45,urcrnrlat=45,llcrnrlon=-45,urcrnrlon=45)
#    map3.drawmapboundary(fill_color='aqua',zorder=0) # fill to edge
#    map3.fillcontinents(color='darkgrey',lake_color='aqua',zorder=1)
#    #map3.drawcoastlines()
#
#    #draw parallels on map
#    map3.drawparallels(parallels,labels=[True,True,True,True])
#    map3.drawmeridians(meridians,labels=[False,False,False,False])
#
#    #plot scatterplot
#    x3,y3 = map3(lon_btm_cold,lat_btm_cold)
#    GridCells3 = ax3.scatter(x3,y3,alpha=0.7,marker='.',s=120,c=corr_btm_cold, cmap='Reds', vmin=0,vmax=1.0,zorder=2)
#    #ax3.clim(-12,12)
#    #ax3.colorbar(label='Ensemble Mean Temperature Bias ($\circ$ C)') 
#    plt.title('Depth Cold Season Correlation', fontsize=14,weight='bold')
#    divider = make_axes_locatable(ax3)
#    cax=divider.append_axes("right",size="5%", pad=0.05)
#    cb = plt.colorbar(GridCells3,cax)
#    cb.set_label(label = 'Pearson Correlation', size='large')
#    cb.ax.tick_params(labelsize='large')
#    plt.tight_layout()
#    plt_nam = ''.join(['/mnt/data/users/herringtont/soil_temp/plots/validation_sites/new_data/CLSM_res/Corr_depth_'+str(product_i)+'_cold_Sep2021.png'])
#    fig.savefig(plt_nam)
#    plt.close()
#
#
#    # Bias Depth Subplot Warm #
#    fig = plt.figure(figsize = (5,5))
#    ax4 = fig.add_subplot(111)
#    map4 = Basemap(projection='npstere',boundinglat=45,lon_0=0,resolution='h',llcrnrlat=45,urcrnrlat=45,llcrnrlon=-45,urcrnrlon=45)
#    map4.drawmapboundary(fill_color='aqua',zorder=0) # fill to edge
#    map4.fillcontinents(color='darkgrey',lake_color='aqua',zorder=1)
#    #map4.drawcoastlines()
#
#    #draw parallels on map
#    map4.drawparallels(parallels,labels=[True,True,True,True])
#    map4.drawmeridians(meridians,labels=[False,False,False,False])
#
#    #plot scatterplot
#    x4,y4 = map4(lon_btm_warm,lat_btm_warm)
#    GridCells4 = ax4.scatter(x4,y4,alpha=0.7,marker='.',s=120,c=corr_btm_warm, cmap='Reds', vmin=0,vmax=1.0,zorder=2)
#    #ax4.clim(-12,12)
#    plt.title('Depth Warm Season Correlation', fontsize=14,weight='bold')
#    divider = make_axes_locatable(ax4)
#    cax=divider.append_axes("right",size="5%", pad=0.05)
#    cb = plt.colorbar(GridCells4,cax)
#    cb.set_label(label = 'Pearson Correlation', size='large')
#    cb.ax.tick_params(labelsize='large')
#    plt.tight_layout()
#    plt_nam = ''.join(['/mnt/data/users/herringtont/soil_temp/plots/validation_sites/new_data/CLSM_res/Corr_depth_'+str(product_i)+'_warm_Sep2021.png'])
#    fig.savefig(plt_nam)
#    plt.close()
#
#
######## Create Std Dev Figures ######
#
#    fig = plt.figure(figsize = (5,5))
#
#    #create array for parallels
#    parallels = np.arange(50.,91.,10.)
#    meridians = np.arange(0.,351.,10.)
#    # Bias Top-30cm Subplot Cold #
#    ax1 = fig.add_subplot(111)
#    map1 = Basemap(projection='npstere',boundinglat=45,lon_0=0,resolution='h',llcrnrlat=45,urcrnrlat=45,llcrnrlon=-45,urcrnrlon=45)
#    map1.drawmapboundary(fill_color='aqua',zorder=0) # fill to edge
#    map1.fillcontinents(color='darkgrey',lake_color='aqua',zorder=1)
#    #map1.drawcoastlines()
#
#    #draw parallels on map
#    map1.drawparallels(parallels,labels=[False,True,True,False])
#    map1.drawmeridians(meridians,labels=[False,False,False,False])
#
#    #plot scatterplot
#    x1,y1 = map1(lon_top_cold,lat_top_cold)
#    GridCells1 = ax1.scatter(x1,y1,alpha=0.7,marker='.',s=120,c=std_top_cold, cmap='bwr', vmin=0,vmax=2,zorder=2)
#    #ax1.clim(-12,12)
#    #ax1.colorbar(label='Ensemble Mean Temperature Bias ($\circ$ C)') 
#    plt.title('Near Surface Cold Season Std Dev',fontsize=14,weight='bold') 
#    divider = make_axes_locatable(ax1)
#    cax=divider.append_axes("right",size="5%", pad=0.05)
#    cb = plt.colorbar(GridCells1,cax)
#    cb.set_label(label = 'Normalized Std Dev', size='large')
#    cb.ax.tick_params(labelsize='large')
#    plt.tight_layout()
#    plt_nam = ''.join(['/mnt/data/users/herringtont/soil_temp/plots/validation_sites/new_data/CLSM_res/Stdev_top30_'+str(product_i)+'_cold_Sep2021.png'])
#    fig.savefig(plt_nam)
#    plt.close()
#
#    # STD DEV Top-30cm Subplot Warm #
#    fig = plt.figure(figsize = (5,5))
#    ax2 = fig.add_subplot(111)
#    map2 = Basemap(projection='npstere',boundinglat=45,lon_0=0,resolution='h',llcrnrlat=45,urcrnrlat=45,llcrnrlon=-45,urcrnrlon=45)
#    map2.drawmapboundary(fill_color='aqua',zorder=0) # fill to edge
#    map2.fillcontinents(color='darkgrey',lake_color='aqua',zorder=1)
#    #map2.drawcoastlines()
#
#    #draw parallels on map
#    map2.drawparallels(parallels,labels=[False,True,True,False])
#    map2.drawmeridians(meridians,labels=[False,False,False,False])
#
#    #plot scatterplot
#    x2,y2 = map2(lon_top_warm,lat_top_warm)
#    GridCells2 = ax2.scatter(x2,y2,alpha=0.7,marker='.',s=120,c=std_top_warm, cmap='bwr', vmin=0,vmax=2,zorder=2)
#    #ax2.clim(-12,12)
#    #ax2.colorbar(label='Ensemble Mean Temperature Bias ($\circ$ C)') 
#    plt.title('Near Surface Warm Season Std Dev', fontsize=14,weight='bold')
#    divider = make_axes_locatable(ax2)
#    cax=divider.append_axes("right",size="5%", pad=0.05)
#    cb = plt.colorbar(GridCells2,cax)
#    cb.set_label(label = 'Normalized Std Dev', size='large')
#    cb.ax.tick_params(labelsize='large')
#    plt.tight_layout()
#    plt_nam = ''.join(['/mnt/data/users/herringtont/soil_temp/plots/validation_sites/new_data/CLSM_res/Stdev_top30_'+str(product_i)+'_warm_Sep2021.png'])
#    fig.savefig(plt_nam)
#    plt.close()
#
#
#
#    # STD DEV Depth Subplot Cold #
#    fig = plt.figure(figsize = (5,5))
#    ax3 = fig.add_subplot(111)
#    map3 = Basemap(projection='npstere',boundinglat=45,lon_0=0,resolution='h',llcrnrlat=45,urcrnrlat=45,llcrnrlon=-45,urcrnrlon=45)
#    map3.drawmapboundary(fill_color='aqua',zorder=0) # fill to edge
#    map3.fillcontinents(color='darkgrey',lake_color='aqua',zorder=1)
#    #map3.drawcoastlines()
#
#    #draw parallels on map
#    map3.drawparallels(parallels,labels=[True,True,True,True])
#    map3.drawmeridians(meridians,labels=[False,False,False,False])
#
#    #plot scatterplot
#    x3,y3 = map3(lon_btm_cold,lat_btm_cold)
#    GridCells3 = ax3.scatter(x3,y3,alpha=0.7,marker='.',s=120,c=std_btm_cold, cmap='bwr', vmin=0,vmax=2,zorder=2)
#    #ax3.clim(-12,12)
#    #ax3.colorbar(label='Ensemble Mean Temperature Bias ($\circ$ C)') 
#    plt.title('Depth Cold Season Std Dev', fontsize=14,weight='bold')
#    divider = make_axes_locatable(ax3)
#    cax=divider.append_axes("right",size="5%", pad=0.05)
#    cb = plt.colorbar(GridCells3,cax)
#    cb.set_label(label = 'Normalized Std Dev', size='large')
#    cb.ax.tick_params(labelsize='large')
#    plt.tight_layout()
#    plt_nam = ''.join(['/mnt/data/users/herringtont/soil_temp/plots/validation_sites/new_data/CLSM_res/Stdev_depth_'+str(product_i)+'_cold_Sep2021.png'])
#    fig.savefig(plt_nam)
#    plt.close()
#
#
#    # STD DEV Depth Subplot Warm #
#    fig = plt.figure(figsize = (5,5))
#    ax4 = fig.add_subplot(111)
#    map4 = Basemap(projection='npstere',boundinglat=45,lon_0=0,resolution='h',llcrnrlat=45,urcrnrlat=45,llcrnrlon=-45,urcrnrlon=45)
#    map4.drawmapboundary(fill_color='aqua',zorder=0) # fill to edge
#    map4.fillcontinents(color='darkgrey',lake_color='aqua',zorder=1)
#    #map4.drawcoastlines()
#
#    #draw parallels on map
#    map4.drawparallels(parallels,labels=[True,True,True,True])
#    map4.drawmeridians(meridians,labels=[False,False,False,False])
#
#    #plot scatterplot
#    x4,y4 = map4(lon_btm_warm,lat_btm_warm)
#    GridCells4 = ax4.scatter(x4,y4,alpha=0.7,marker='.',s=120,c=std_btm_warm, cmap='bwr', vmin=0,vmax=2,zorder=2)
#    #ax4.clim(-12,12)
#    plt.title('Depth Warm Season Std Dev', fontsize=14,weight='bold')
#    divider = make_axes_locatable(ax4)
#    cax=divider.append_axes("right",size="5%", pad=0.05)
#    cb = plt.colorbar(GridCells4,cax)
#    cb.set_label(label = 'Normalized Std Dev', size='large')
#    cb.ax.tick_params(labelsize='large')
#    plt.tight_layout()
#    plt_nam = ''.join(['/mnt/data/users/herringtont/soil_temp/plots/validation_sites/new_data/CLSM_res/Stdev_depth_'+str(product_i)+'_warm_Sep2021.png'])
#    fig.savefig(plt_nam)
#    plt.close()
