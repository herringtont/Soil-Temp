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
    					ensmean_4_bias_cold_master = []
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
    					ensmean_4_var_cold_master = []
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
    					ensmean_4_std_cold_master = []
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
    					ensmean_4_rmse_cold_master = []
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
    					ensmean_4_ubrmse_cold_master = []
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
    					ensmean_4_corr_cold_master = []
    					CFSR_corr_cold_master = []
    					ERAI_corr_cold_master = []
    					ERA5_corr_cold_master = []
    					ERA5_Land_corr_cold_master = []
    					JRA_corr_cold_master = []
    					MERRA2_corr_cold_master = []
    					GLDAS_corr_cold_master = []
    					GLDAS_CLSM_corr_cold_master = []

## Grab Data ## 
    					fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/CLSM_res/subset/'+str(remap_type)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_dframe_scatterplot_CMOS_CLSM_subset_permafrost_cold_warm_BEST_Sep2021_ensmean4_airtemp.csv'])
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
    						naive_temp_cold = dframe_cold_season_gcell['Naive Blend'].values
    						naive_noJRA_temp_cold = dframe_cold_season_gcell['Naive Blend no JRA55'].values
    						naive_noJRAold_temp_cold = dframe_cold_season_gcell['Naive Blend no JRA55 Old'].values
    						naive_all_temp_cold = dframe_cold_season_gcell['Naive Blend All'].values
    						ensmean_4_temp_cold = dframe_cold_season_gcell['Ensemble Mean 4 Model'].values
    						CFSR_temp_cold = dframe_cold_season_gcell['CFSR'].values
    						ERAI_temp_cold = dframe_cold_season_gcell['ERA-Interim'].values
    						ERA5_temp_cold = dframe_cold_season_gcell['ERA5'].values
    						ERA5_Land_temp_cold = dframe_cold_season_gcell['ERA5-Land'].values
    						JRA_temp_cold = dframe_cold_season_gcell['JRA55'].values
    						MERRA2_temp_cold = dframe_cold_season_gcell['MERRA2'].values
    						GLDAS_temp_cold = dframe_cold_season_gcell['GLDAS-Noah'].values
    						GLDAS_CLSM_temp_cold = dframe_cold_season_gcell['GLDAS-CLSM'].values


## Bias ##
    						naive_bias_cold = bias(naive_temp_cold, station_temp_cold)
    						naive_bias_cold_master.append(naive_bias_cold)
    						naive_noJRA_bias_cold = bias(naive_noJRA_temp_cold, station_temp_cold)
    						naive_noJRA_bias_cold_master.append(naive_noJRA_bias_cold)
    						naive_noJRAold_bias_cold = bias(naive_noJRAold_temp_cold, station_temp_cold)
    						naive_noJRAold_bias_cold_master.append(naive_noJRAold_bias_cold)
    						naive_all_bias_cold = bias(naive_all_temp_cold, station_temp_cold)
    						naive_all_bias_cold_master.append(naive_all_bias_cold)
    						ensmean_4_bias_cold = bias(ensmean_4_temp_cold, station_temp_cold)
    						ensmean_4_bias_cold_master.append(ensmean_4_bias_cold)
    						CFSR_bias_cold = bias(CFSR_temp_cold, station_temp_cold)
    						CFSR_bias_cold_master.append(CFSR_bias_cold)
    						ERAI_bias_cold = bias(ERAI_temp_cold, station_temp_cold)
    						ERAI_bias_cold_master.append(ERAI_bias_cold)
    						ERA5_bias_cold = bias(ERA5_temp_cold, station_temp_cold)
    						ERA5_bias_cold_master.append(ERA5_bias_cold)
    						ERA5_Land_bias_cold = bias(ERA5_Land_temp_cold, station_temp_cold)
    						ERA5_Land_bias_cold_master.append(ERA5_Land_bias_cold)
    						JRA_bias_cold = bias(JRA_temp_cold, station_temp_cold)
    						JRA_bias_cold_master.append(JRA_bias_cold)
    						MERRA2_bias_cold = bias(MERRA2_temp_cold, station_temp_cold)
    						MERRA2_bias_cold_master.append(MERRA2_bias_cold)
    						GLDAS_bias_cold = bias(GLDAS_temp_cold, station_temp_cold)
    						GLDAS_bias_cold_master.append(GLDAS_bias_cold)
    						GLDAS_CLSM_bias_cold = bias(GLDAS_CLSM_temp_cold, station_temp_cold)
    						GLDAS_CLSM_bias_cold_master.append(GLDAS_CLSM_bias_cold)

## Variance ##

    						stn_var_cold =  np.var(station_temp_cold)
    						stn_var_cold_master.append(stn_var_cold)
    						naive_var_cold = np.var(naive_temp_cold)
    						naive_var_cold_master.append(naive_var_cold)
    						naive_noJRA_var_cold = np.var(naive_noJRA_temp_cold)
    						naive_noJRA_var_cold_master.append(naive_noJRA_var_cold)
    						naive_noJRAold_var_cold = np.var(naive_noJRAold_temp_cold)
    						naive_noJRAold_var_cold_master.append(naive_noJRAold_var_cold)
    						naive_all_var_cold = np.var(naive_all_temp_cold)
    						naive_all_var_cold_master.append(naive_all_var_cold)
    						ensmean_4_var_cold = np.var(ensmean_4_temp_cold)
    						ensmean_4_var_cold_master.append(ensmean_4_var_cold)					
    						CFSR_var_cold = np.var(CFSR_temp_cold)
    						CFSR_var_cold_master.append(CFSR_var_cold)
    						ERAI_var_cold = np.var(ERAI_temp_cold)
    						ERAI_var_cold_master.append(ERAI_var_cold)    					
    						ERA5_var_cold = np.var(ERA5_temp_cold)
    						ERA5_var_cold_master.append(ERA5_var_cold)
    						ERA5_Land_var_cold = np.var(ERA5_Land_temp_cold)
    						ERA5_Land_var_cold_master.append(ERA5_Land_var_cold)
    						JRA_var_cold = np.var(JRA_temp_cold)
    						JRA_var_cold_master.append(JRA_var_cold)
    						MERRA2_var_cold = np.var(MERRA2_temp_cold)
    						MERRA2_var_cold_master.append(MERRA2_var_cold)
    						GLDAS_var_cold = np.var(GLDAS_temp_cold)
    						GLDAS_var_cold_master.append(GLDAS_var_cold)
    						GLDAS_CLSM_var_cold = np.var(GLDAS_CLSM_temp_cold)
    						GLDAS_CLSM_var_cold_master.append(GLDAS_CLSM_var_cold)


## Standard Deviation ##

    						stn_std_cold =  np.std(station_temp_cold)
    						stn_std_cold_master.append(stn_std_cold)
    						naive_std_cold = np.std(naive_temp_cold)/stn_std_cold
    						naive_std_cold_master.append(naive_std_cold)
    						naive_noJRA_std_cold = np.std(naive_noJRA_temp_cold)/stn_std_cold
    						naive_noJRA_std_cold_master.append(naive_noJRA_std_cold)
    						naive_noJRAold_std_cold = np.std(naive_noJRAold_temp_cold)/stn_std_cold
    						naive_noJRAold_std_cold_master.append(naive_noJRAold_std_cold)
    						naive_all_std_cold = np.std(naive_all_temp_cold)/stn_std_cold
    						naive_all_std_cold_master.append(naive_all_std_cold)
    						ensmean_4_std_cold = np.std(ensmean_4_temp_cold)/stn_std_cold
    						ensmean_4_std_cold_master.append(ensmean_4_std_cold)					
    						CFSR_std_cold = np.std(CFSR_temp_cold)/stn_std_cold
    						CFSR_std_cold_master.append(CFSR_std_cold)
    						ERAI_std_cold = np.std(ERAI_temp_cold)/stn_std_cold
    						ERAI_std_cold_master.append(ERAI_std_cold)    					
    						ERA5_std_cold = np.std(ERA5_temp_cold)/stn_std_cold
    						ERA5_std_cold_master.append(ERA5_std_cold)
    						ERA5_Land_std_cold = np.std(ERA5_Land_temp_cold)/stn_std_cold
    						ERA5_Land_std_cold_master.append(ERA5_Land_std_cold)
    						JRA_std_cold = np.std(JRA_temp_cold)/stn_std_cold
    						JRA_std_cold_master.append(JRA_std_cold)
    						MERRA2_std_cold = np.std(MERRA2_temp_cold)/stn_std_cold
    						MERRA2_std_cold_master.append(MERRA2_std_cold)
    						GLDAS_std_cold = np.std(GLDAS_temp_cold)/stn_std_cold
    						GLDAS_std_cold_master.append(GLDAS_std_cold)
    						GLDAS_CLSM_std_cold = np.std(GLDAS_CLSM_temp_cold)/stn_std_cold
    						GLDAS_CLSM_std_cold_master.append(GLDAS_CLSM_std_cold)


## RMSE and ubRMSE ##
    						naive_rmse_cold = mean_squared_error(station_temp_cold,naive_temp_cold, squared=False)
    						naive_rmse_cold_master.append(naive_rmse_cold)
    						naive_noJRA_rmse_cold = mean_squared_error(station_temp_cold,naive_noJRA_temp_cold, squared=False)
    						naive_noJRA_rmse_cold_master.append(naive_noJRA_rmse_cold)
    						naive_noJRAold_rmse_cold = mean_squared_error(station_temp_cold,naive_noJRAold_temp_cold, squared=False)
    						naive_noJRAold_rmse_cold_master.append(naive_noJRAold_rmse_cold)
    						naive_all_rmse_cold = mean_squared_error(station_temp_cold,naive_all_temp_cold, squared=False)
    						naive_all_rmse_cold_master.append(naive_all_rmse_cold)
    						ensmean_4_rmse_cold = mean_squared_error(station_temp_cold,ensmean_4_temp_cold, squared=False)
    						ensmean_4_rmse_cold_master.append(ensmean_4_rmse_cold)
    						CFSR_rmse_cold = mean_squared_error(station_temp_cold,CFSR_temp_cold, squared=False)
    						CFSR_rmse_cold_master.append(CFSR_rmse_cold)
    						ERAI_rmse_cold = mean_squared_error(station_temp_cold,ERAI_temp_cold, squared=False)
    						ERAI_rmse_cold_master.append(ERAI_rmse_cold)
    						ERA5_rmse_cold = mean_squared_error(station_temp_cold,ERA5_temp_cold, squared=False)
    						ERA5_rmse_cold_master.append(ERA5_rmse_cold)
    						ERA5_Land_rmse_cold = mean_squared_error(station_temp_cold,ERA5_Land_temp_cold, squared=False)
    						ERA5_Land_rmse_cold_master.append(ERA5_Land_rmse_cold)
    						JRA_rmse_cold = mean_squared_error(station_temp_cold,JRA_temp_cold, squared=False)
    						JRA_rmse_cold_master.append(JRA_rmse_cold)
    						MERRA2_rmse_cold = mean_squared_error(station_temp_cold,MERRA2_temp_cold, squared=False)
    						MERRA2_rmse_cold_master.append(MERRA2_rmse_cold)
    						GLDAS_rmse_cold = mean_squared_error(station_temp_cold,GLDAS_temp_cold, squared=False)
    						GLDAS_rmse_cold_master.append(GLDAS_rmse_cold)
    						GLDAS_CLSM_rmse_cold = mean_squared_error(station_temp_cold,GLDAS_CLSM_temp_cold, squared=False)
    						GLDAS_CLSM_rmse_cold_master.append(GLDAS_CLSM_rmse_cold)

    						naive_ubrmse_cold = ubrmsd(station_temp_cold,naive_temp_cold)
    						naive_ubrmse_cold_master.append(naive_ubrmse_cold)
    						naive_noJRA_ubrmse_cold = ubrmsd(station_temp_cold,naive_noJRA_temp_cold)
    						naive_noJRA_ubrmse_cold_master.append(naive_noJRA_ubrmse_cold)
    						naive_noJRAold_ubrmse_cold = ubrmsd(station_temp_cold,naive_noJRAold_temp_cold)
    						naive_noJRAold_ubrmse_cold_master.append(naive_noJRAold_ubrmse_cold)
    						naive_all_ubrmse_cold = ubrmsd(station_temp_cold,naive_all_temp_cold)
    						naive_all_ubrmse_cold_master.append(naive_all_ubrmse_cold)
    						ensmean_4_ubrmse_cold = ubrmsd(station_temp_cold,ensmean_4_temp_cold)
    						ensmean_4_ubrmse_cold_master.append(ensmean_4_ubrmse_cold)
    						CFSR_ubrmse_cold = ubrmsd(station_temp_cold,CFSR_temp_cold)
    						CFSR_ubrmse_cold_master.append(CFSR_ubrmse_cold)
    						ERAI_ubrmse_cold = ubrmsd(station_temp_cold,ERAI_temp_cold)
    						ERAI_ubrmse_cold_master.append(ERAI_ubrmse_cold)
    						ERA5_ubrmse_cold = ubrmsd(station_temp_cold,ERA5_temp_cold)
    						ERA5_ubrmse_cold_master.append(ERA5_ubrmse_cold)
    						ERA5_Land_ubrmse_cold = ubrmsd(station_temp_cold,ERA5_Land_temp_cold)
    						ERA5_Land_ubrmse_cold_master.append(ERA5_Land_ubrmse_cold)
    						JRA_ubrmse_cold = ubrmsd(station_temp_cold,JRA_temp_cold)
    						JRA_ubrmse_cold_master.append(JRA_ubrmse_cold)
    						MERRA2_ubrmse_cold = ubrmsd(station_temp_cold,MERRA2_temp_cold)
    						MERRA2_ubrmse_cold_master.append(MERRA2_ubrmse_cold)
    						GLDAS_ubrmse_cold = ubrmsd(station_temp_cold,GLDAS_temp_cold)
    						GLDAS_ubrmse_cold_master.append(GLDAS_ubrmse_cold)
    						GLDAS_CLSM_ubrmse_cold = ubrmsd(station_temp_cold,GLDAS_CLSM_temp_cold)
    						GLDAS_CLSM_ubrmse_cold_master.append(GLDAS_CLSM_ubrmse_cold)



## Pearson Correlations ##
    						naive_corr_cold,_ = pearsonr(naive_temp_cold, station_temp_cold)
    						naive_corr_cold_master.append(naive_corr_cold)
    						naive_noJRA_corr_cold,_ = pearsonr(naive_noJRA_temp_cold, station_temp_cold)
    						naive_noJRA_corr_cold_master.append(naive_noJRA_corr_cold)
    						naive_noJRAold_corr_cold,_ = pearsonr(naive_noJRAold_temp_cold, station_temp_cold)
    						naive_noJRAold_corr_cold_master.append(naive_noJRAold_corr_cold)
    						naive_all_corr_cold,_ = pearsonr(naive_all_temp_cold, station_temp_cold)
    						naive_all_corr_cold_master.append(naive_all_corr_cold)
    						ensmean_4_corr_cold,_ = pearsonr(ensmean_4_temp_cold, station_temp_cold)
    						ensmean_4_corr_cold_master.append(ensmean_4_corr_cold)
    						CFSR_corr_cold,_ = pearsonr(CFSR_temp_cold, station_temp_cold)
    						CFSR_corr_cold_master.append(CFSR_corr_cold)
    						ERAI_corr_cold,_ = pearsonr(ERAI_temp_cold, station_temp_cold)
    						ERAI_corr_cold_master.append(ERAI_corr_cold)
    						ERA5_corr_cold,_ = pearsonr(ERA5_temp_cold, station_temp_cold)
    						ERA5_corr_cold_master.append(ERA5_corr_cold)
    						ERA5_Land_corr_cold,_ = pearsonr(ERA5_Land_temp_cold, station_temp_cold)
    						ERA5_Land_corr_cold_master.append(ERA5_Land_corr_cold)
    						JRA_corr_cold,_ = pearsonr(JRA_temp_cold, station_temp_cold)
    						JRA_corr_cold_master.append(JRA_corr_cold)
    						MERRA2_corr_cold,_ = pearsonr(MERRA2_temp_cold, station_temp_cold)
    						MERRA2_corr_cold_master.append(MERRA2_corr_cold)
    						GLDAS_corr_cold,_ = pearsonr(GLDAS_temp_cold, station_temp_cold)
    						GLDAS_corr_cold_master.append(GLDAS_corr_cold)
    						GLDAS_CLSM_corr_cold,_ = pearsonr(GLDAS_CLSM_temp_cold, station_temp_cold)
    						GLDAS_CLSM_corr_cold_master.append(GLDAS_CLSM_corr_cold)




#### Warm Season ####


## Master Arrays ##

    					lat_warm_master = []
    					lon_warm_master = []
    					gcell_warm_master = []
    					naive_bias_warm_master = []
    					naive_noJRA_bias_warm_master = []
    					naive_noJRAold_bias_warm_master = []
    					naive_all_bias_warm_master = []
    					ensmean_4_bias_warm_master = []
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
    					ensmean_4_var_warm_master = []
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
    					ensmean_4_std_warm_master = []
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
    					ensmean_4_rmse_warm_master = []
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
    					ensmean_4_ubrmse_warm_master = []
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
    					ensmean_4_corr_warm_master = []
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

    					print(dframe_warm_season_permafrost)

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
    						naive_temp_warm = dframe_warm_season_gcell['Naive Blend'].values
    						naive_noJRA_temp_warm = dframe_warm_season_gcell['Naive Blend no JRA55'].values
    						naive_noJRAold_temp_warm = dframe_warm_season_gcell['Naive Blend no JRA55 Old'].values
    						naive_all_temp_warm = dframe_warm_season_gcell['Naive Blend All'].values
    						ensmean_4_temp_warm = dframe_warm_season_gcell['Ensemble Mean 4 Model'].values
    						CFSR_temp_warm = dframe_warm_season_gcell['CFSR'].values
    						ERAI_temp_warm = dframe_warm_season_gcell['ERA-Interim'].values
    						ERA5_temp_warm = dframe_warm_season_gcell['ERA5'].values
    						ERA5_Land_temp_warm = dframe_warm_season_gcell['ERA5-Land'].values
    						JRA_temp_warm = dframe_warm_season_gcell['JRA55'].values
    						MERRA2_temp_warm = dframe_warm_season_gcell['MERRA2'].values
    						GLDAS_temp_warm = dframe_warm_season_gcell['GLDAS-Noah'].values
    						GLDAS_CLSM_temp_warm = dframe_warm_season_gcell['GLDAS-CLSM'].values


## Bias ##
    						naive_bias_warm = bias(naive_temp_warm, station_temp_warm)
    						naive_bias_warm_master.append(naive_bias_warm)
    						naive_noJRA_bias_warm = bias(naive_noJRA_temp_warm, station_temp_warm)
    						naive_noJRA_bias_warm_master.append(naive_noJRA_bias_warm)
    						naive_noJRAold_bias_warm = bias(naive_noJRAold_temp_warm, station_temp_warm)
    						naive_noJRAold_bias_warm_master.append(naive_noJRAold_bias_warm)
    						naive_all_bias_warm = bias(naive_all_temp_warm, station_temp_warm)
    						naive_all_bias_warm_master.append(naive_all_bias_warm)
    						ensmean_4_bias_warm = bias(ensmean_4_temp_warm, station_temp_warm)
    						ensmean_4_bias_warm_master.append(ensmean_4_bias_warm)
    						CFSR_bias_warm = bias(CFSR_temp_warm, station_temp_warm)
    						CFSR_bias_warm_master.append(CFSR_bias_warm)
    						ERAI_bias_warm = bias(ERAI_temp_warm, station_temp_warm)
    						ERAI_bias_warm_master.append(ERAI_bias_warm)
    						ERA5_bias_warm = bias(ERA5_temp_warm, station_temp_warm)
    						ERA5_bias_warm_master.append(ERA5_bias_warm)
    						ERA5_Land_bias_warm = bias(ERA5_Land_temp_warm, station_temp_warm)
    						ERA5_Land_bias_warm_master.append(ERA5_Land_bias_warm)
    						JRA_bias_warm = bias(JRA_temp_warm, station_temp_warm)
    						JRA_bias_warm_master.append(JRA_bias_warm)
    						MERRA2_bias_warm = bias(MERRA2_temp_warm, station_temp_warm)
    						MERRA2_bias_warm_master.append(MERRA2_bias_warm)
    						GLDAS_bias_warm = bias(GLDAS_temp_warm, station_temp_warm)
    						GLDAS_bias_warm_master.append(GLDAS_bias_warm)
    						GLDAS_CLSM_bias_warm = bias(GLDAS_CLSM_temp_warm, station_temp_warm)
    						GLDAS_CLSM_bias_warm_master.append(GLDAS_CLSM_bias_warm)

## Variance ##

    						stn_var_warm =  np.var(station_temp_warm)
    						stn_var_warm_master.append(stn_var_warm)
    						naive_var_warm = np.var(naive_temp_warm)
    						naive_var_warm_master.append(naive_var_warm)
    						naive_noJRA_var_warm = np.var(naive_noJRA_temp_warm)
    						naive_noJRA_var_warm_master.append(naive_noJRA_var_warm)
    						naive_noJRAold_var_warm = np.var(naive_noJRAold_temp_warm)
    						naive_noJRAold_var_warm_master.append(naive_noJRAold_var_warm)
    						naive_all_var_warm = np.var(naive_all_temp_warm)
    						naive_all_var_warm_master.append(naive_all_var_warm)
    						ensmean_4_var_warm = np.var(ensmean_4_temp_warm)
    						ensmean_4_var_warm_master.append(ensmean_4_var_warm)					
    						CFSR_var_warm = np.var(CFSR_temp_warm)
    						CFSR_var_warm_master.append(CFSR_var_warm)
    						ERAI_var_warm = np.var(ERAI_temp_warm)
    						ERAI_var_warm_master.append(ERAI_var_warm)    					
    						ERA5_var_warm = np.var(ERA5_temp_warm)
    						ERA5_var_warm_master.append(ERA5_var_warm)
    						ERA5_Land_var_warm = np.var(ERA5_Land_temp_warm)
    						ERA5_Land_var_warm_master.append(ERA5_Land_var_warm)
    						JRA_var_warm = np.var(JRA_temp_warm)
    						JRA_var_warm_master.append(JRA_var_warm)
    						MERRA2_var_warm = np.var(MERRA2_temp_warm)
    						MERRA2_var_warm_master.append(MERRA2_var_warm)
    						GLDAS_var_warm = np.var(GLDAS_temp_warm)
    						GLDAS_var_warm_master.append(GLDAS_var_warm)
    						GLDAS_CLSM_var_warm = np.var(GLDAS_CLSM_temp_warm)
    						GLDAS_CLSM_var_warm_master.append(GLDAS_CLSM_var_warm)


## Standard Deviation ##

    						stn_std_warm =  np.std(station_temp_warm)
    						stn_std_warm_master.append(stn_std_warm)
    						naive_std_warm = np.std(naive_temp_warm)/stn_std_warm
    						naive_std_warm_master.append(naive_std_warm)
    						naive_noJRA_std_warm = np.std(naive_noJRA_temp_warm)/stn_std_warm
    						naive_noJRA_std_warm_master.append(naive_noJRA_std_warm)
    						naive_noJRAold_std_warm = np.std(naive_noJRAold_temp_warm)/stn_std_warm
    						naive_noJRAold_std_warm_master.append(naive_noJRAold_std_warm)
    						naive_all_std_warm = np.std(naive_all_temp_warm)/stn_std_warm
    						naive_all_std_warm_master.append(naive_all_std_warm)
    						ensmean_4_std_warm = np.std(ensmean_4_temp_warm)/stn_std_warm
    						ensmean_4_std_warm_master.append(ensmean_4_std_warm)					
    						CFSR_std_warm = np.std(CFSR_temp_warm)/stn_std_warm
    						CFSR_std_warm_master.append(CFSR_std_warm)
    						ERAI_std_warm = np.std(ERAI_temp_warm)/stn_std_warm
    						ERAI_std_warm_master.append(ERAI_std_warm)    					
    						ERA5_std_warm = np.std(ERA5_temp_warm)/stn_std_warm
    						ERA5_std_warm_master.append(ERA5_std_warm)
    						ERA5_Land_std_warm = np.std(ERA5_Land_temp_warm)/stn_std_warm
    						ERA5_Land_std_warm_master.append(ERA5_Land_std_warm)
    						JRA_std_warm = np.std(JRA_temp_warm)/stn_std_warm
    						JRA_std_warm_master.append(JRA_std_warm)
    						MERRA2_std_warm = np.std(MERRA2_temp_warm)/stn_std_warm
    						MERRA2_std_warm_master.append(MERRA2_std_warm)
    						GLDAS_std_warm = np.std(GLDAS_temp_warm)/stn_std_warm
    						GLDAS_std_warm_master.append(GLDAS_std_warm)
    						GLDAS_CLSM_std_warm = np.std(GLDAS_CLSM_temp_warm)/stn_std_warm
    						GLDAS_CLSM_std_warm_master.append(GLDAS_CLSM_std_warm)


## RMSE and ubRMSE ##
    						naive_rmse_warm = mean_squared_error(station_temp_warm,naive_temp_warm, squared=False)
    						naive_rmse_warm_master.append(naive_rmse_warm)
    						naive_noJRA_rmse_warm = mean_squared_error(station_temp_warm,naive_noJRA_temp_warm, squared=False)
    						naive_noJRA_rmse_warm_master.append(naive_noJRA_rmse_warm)
    						naive_noJRAold_rmse_warm = mean_squared_error(station_temp_warm,naive_noJRAold_temp_warm, squared=False)
    						naive_noJRAold_rmse_warm_master.append(naive_noJRAold_rmse_warm)
    						naive_all_rmse_warm = mean_squared_error(station_temp_warm,naive_all_temp_warm, squared=False)
    						naive_all_rmse_warm_master.append(naive_all_rmse_warm)
    						ensmean_4_rmse_warm = mean_squared_error(station_temp_warm,ensmean_4_temp_warm, squared=False)
    						ensmean_4_rmse_warm_master.append(ensmean_4_rmse_warm)
    						CFSR_rmse_warm = mean_squared_error(station_temp_warm,CFSR_temp_warm, squared=False)
    						CFSR_rmse_warm_master.append(CFSR_rmse_warm)
    						ERAI_rmse_warm = mean_squared_error(station_temp_warm,ERAI_temp_warm, squared=False)
    						ERAI_rmse_warm_master.append(ERAI_rmse_warm)
    						ERA5_rmse_warm = mean_squared_error(station_temp_warm,ERA5_temp_warm, squared=False)
    						ERA5_rmse_warm_master.append(ERA5_rmse_warm)
    						ERA5_Land_rmse_warm = mean_squared_error(station_temp_warm,ERA5_Land_temp_warm, squared=False)
    						ERA5_Land_rmse_warm_master.append(ERA5_Land_rmse_warm)
    						JRA_rmse_warm = mean_squared_error(station_temp_warm,JRA_temp_warm, squared=False)
    						JRA_rmse_warm_master.append(JRA_rmse_warm)
    						MERRA2_rmse_warm = mean_squared_error(station_temp_warm,MERRA2_temp_warm, squared=False)
    						MERRA2_rmse_warm_master.append(MERRA2_rmse_warm)
    						GLDAS_rmse_warm = mean_squared_error(station_temp_warm,GLDAS_temp_warm, squared=False)
    						GLDAS_rmse_warm_master.append(GLDAS_rmse_warm)
    						GLDAS_CLSM_rmse_warm = mean_squared_error(station_temp_warm,GLDAS_CLSM_temp_warm, squared=False)
    						GLDAS_CLSM_rmse_warm_master.append(GLDAS_CLSM_rmse_warm)

    						naive_ubrmse_warm = ubrmsd(station_temp_warm,naive_temp_warm)
    						naive_ubrmse_warm_master.append(naive_ubrmse_warm)
    						naive_noJRA_ubrmse_warm = ubrmsd(station_temp_warm,naive_noJRA_temp_warm)
    						naive_noJRA_ubrmse_warm_master.append(naive_noJRA_ubrmse_warm)
    						naive_noJRAold_ubrmse_warm = ubrmsd(station_temp_warm,naive_noJRAold_temp_warm)
    						naive_noJRAold_ubrmse_warm_master.append(naive_noJRAold_ubrmse_warm)
    						naive_all_ubrmse_warm = ubrmsd(station_temp_warm,naive_all_temp_warm)
    						naive_all_ubrmse_warm_master.append(naive_all_ubrmse_warm)
    						ensmean_4_ubrmse_warm = ubrmsd(station_temp_warm,ensmean_4_temp_warm)
    						ensmean_4_ubrmse_warm_master.append(ensmean_4_ubrmse_warm)
    						CFSR_ubrmse_warm = ubrmsd(station_temp_warm,CFSR_temp_warm)
    						CFSR_ubrmse_warm_master.append(CFSR_ubrmse_warm)
    						ERAI_ubrmse_warm = ubrmsd(station_temp_warm,ERAI_temp_warm)
    						ERAI_ubrmse_warm_master.append(ERAI_ubrmse_warm)
    						ERA5_ubrmse_warm = ubrmsd(station_temp_warm,ERA5_temp_warm)
    						ERA5_ubrmse_warm_master.append(ERA5_ubrmse_warm)
    						ERA5_Land_ubrmse_warm = ubrmsd(station_temp_warm,ERA5_Land_temp_warm)
    						ERA5_Land_ubrmse_warm_master.append(ERA5_Land_ubrmse_warm)
    						JRA_ubrmse_warm = ubrmsd(station_temp_warm,JRA_temp_warm)
    						JRA_ubrmse_warm_master.append(JRA_ubrmse_warm)
    						MERRA2_ubrmse_warm = ubrmsd(station_temp_warm,MERRA2_temp_warm)
    						MERRA2_ubrmse_warm_master.append(MERRA2_ubrmse_warm)
    						GLDAS_ubrmse_warm = ubrmsd(station_temp_warm,GLDAS_temp_warm)
    						GLDAS_ubrmse_warm_master.append(GLDAS_ubrmse_warm)
    						GLDAS_CLSM_ubrmse_warm = ubrmsd(station_temp_warm,GLDAS_CLSM_temp_warm)
    						GLDAS_CLSM_ubrmse_warm_master.append(GLDAS_CLSM_ubrmse_warm)



## Pearson Correlations ##
    						naive_corr_warm,_ = pearsonr(naive_temp_warm, station_temp_warm)
    						naive_corr_warm_master.append(naive_corr_warm)
    						naive_noJRA_corr_warm,_ = pearsonr(naive_noJRA_temp_warm, station_temp_warm)
    						naive_noJRA_corr_warm_master.append(naive_noJRA_corr_warm)
    						naive_noJRAold_corr_warm,_ = pearsonr(naive_noJRAold_temp_warm, station_temp_warm)
    						naive_noJRAold_corr_warm_master.append(naive_noJRAold_corr_warm)
    						naive_all_corr_warm,_ = pearsonr(naive_all_temp_warm, station_temp_warm)
    						naive_all_corr_warm_master.append(naive_all_corr_warm)
    						ensmean_4_corr_warm,_ = pearsonr(ensmean_4_temp_warm, station_temp_warm)
    						ensmean_4_corr_warm_master.append(ensmean_4_corr_warm)
    						CFSR_corr_warm,_ = pearsonr(CFSR_temp_warm, station_temp_warm)
    						CFSR_corr_warm_master.append(CFSR_corr_warm)
    						ERAI_corr_warm,_ = pearsonr(ERAI_temp_warm, station_temp_warm)
    						ERAI_corr_warm_master.append(ERAI_corr_warm)
    						ERA5_corr_warm,_ = pearsonr(ERA5_temp_warm, station_temp_warm)
    						ERA5_corr_warm_master.append(ERA5_corr_warm)
    						ERA5_Land_corr_warm,_ = pearsonr(ERA5_Land_temp_warm, station_temp_warm)
    						ERA5_Land_corr_warm_master.append(ERA5_Land_corr_warm)
    						JRA_corr_warm,_ = pearsonr(JRA_temp_warm, station_temp_warm)
    						JRA_corr_warm_master.append(JRA_corr_warm)
    						MERRA2_corr_warm,_ = pearsonr(MERRA2_temp_warm, station_temp_warm)
    						MERRA2_corr_warm_master.append(MERRA2_corr_warm)
    						GLDAS_corr_warm,_ = pearsonr(GLDAS_temp_warm, station_temp_warm)
    						GLDAS_corr_warm_master.append(GLDAS_corr_warm)
    						GLDAS_CLSM_corr_warm,_ = pearsonr(GLDAS_CLSM_temp_warm, station_temp_warm)
    						GLDAS_CLSM_corr_warm_master.append(GLDAS_CLSM_corr_warm)



########## Save CSV Files #########

    					dframe_cold_final = pd.DataFrame(data=gcell_cold_master, columns=['Grid Cell'])
    					dframe_cold_final['Lat'] = lat_cold_master
    					dframe_cold_final['Lon'] = lon_cold_master
    					dframe_cold_final['Bias'] = naive_all_bias_cold_master
    					dframe_cold_final['RMSE'] = naive_all_rmse_cold_master
    					dframe_cold_final['Correlation'] = naive_all_corr_cold_master
    					dframe_cold_final['Norm Std'] = naive_all_std_cold_master

    					metrics_cold_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CMOS_poster/CLSM_res/'+str(remap_type)+'_'+str(lyr_l)+'_naive_metrics_CLSM_EnsMean_by_grid_cell_BEST_cold_season_Sep2021.csv'])  					
    					dframe_cold_final.to_csv(metrics_cold_fil)
    					print(metrics_cold_fil)

    					dframe_cold_final = pd.DataFrame(data=gcell_cold_master, columns=['Grid Cell'])
    					dframe_cold_final['Lat'] = lat_cold_master
    					dframe_cold_final['Lon'] = lon_cold_master
    					dframe_cold_final['Bias'] = ensmean_4_bias_cold_master
    					dframe_cold_final['RMSE'] = ensmean_4_rmse_cold_master
    					dframe_cold_final['Correlation'] = ensmean_4_corr_cold_master
    					dframe_cold_final['Norm Std'] = ensmean_4_std_cold_master

    					metrics_cold_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CMOS_poster/CLSM_res/'+str(remap_type)+'_'+str(lyr_l)+'_naive_metrics_CLSM_EnsMean_4Model_by_grid_cell_BEST_cold_season_Sep2021.csv'])  					
    					dframe_cold_final.to_csv(metrics_cold_fil)
    					print(metrics_cold_fil)

    					dframe_cold_final = pd.DataFrame(data=gcell_cold_master, columns=['Grid Cell'])
    					dframe_cold_final['Lat'] = lat_cold_master
    					dframe_cold_final['Lon'] = lon_cold_master
    					dframe_cold_final['Bias'] = CFSR_bias_cold_master
    					dframe_cold_final['RMSE'] = CFSR_rmse_cold_master
    					dframe_cold_final['Correlation'] = CFSR_corr_cold_master
    					dframe_cold_final['Norm Std'] = CFSR_std_cold_master

    					metrics_cold_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CMOS_poster/CLSM_res/'+str(remap_type)+'_'+str(lyr_l)+'_naive_metrics_CLSM_CFSR_by_grid_cell_BEST_cold_season_Sep2021.csv'])  					
    					dframe_cold_final.to_csv(metrics_cold_fil)
    					print(metrics_cold_fil)

    					dframe_cold_final = pd.DataFrame(data=gcell_cold_master, columns=['Grid Cell'])
    					dframe_cold_final['Lat'] = lat_cold_master
    					dframe_cold_final['Lon'] = lon_cold_master
    					dframe_cold_final['Bias'] = ERAI_bias_cold_master
    					dframe_cold_final['RMSE'] = ERAI_rmse_cold_master
    					dframe_cold_final['Correlation'] = ERAI_corr_cold_master
    					dframe_cold_final['Norm Std'] = ERAI_std_cold_master

    					metrics_cold_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CMOS_poster/CLSM_res/'+str(remap_type)+'_'+str(lyr_l)+'_naive_metrics_CLSM_ERAI_by_grid_cell_BEST_cold_season_Sep2021.csv'])  					
    					dframe_cold_final.to_csv(metrics_cold_fil)
    					print(metrics_cold_fil)


    					dframe_cold_final = pd.DataFrame(data=gcell_cold_master, columns=['Grid Cell'])
    					dframe_cold_final['Lat'] = lat_cold_master
    					dframe_cold_final['Lon'] = lon_cold_master
    					dframe_cold_final['Bias'] = ERA5_bias_cold_master
    					dframe_cold_final['RMSE'] = ERA5_rmse_cold_master
    					dframe_cold_final['Correlation'] = ERA5_corr_cold_master
    					dframe_cold_final['Norm Std'] = ERA5_std_cold_master

    					metrics_cold_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CMOS_poster/CLSM_res/'+str(remap_type)+'_'+str(lyr_l)+'_naive_metrics_CLSM_ERA5_by_grid_cell_BEST_cold_season_Sep2021.csv'])  					
    					dframe_cold_final.to_csv(metrics_cold_fil)
    					print(metrics_cold_fil)


    					dframe_cold_final = pd.DataFrame(data=gcell_cold_master, columns=['Grid Cell'])
    					dframe_cold_final['Lat'] = lat_cold_master
    					dframe_cold_final['Lon'] = lon_cold_master
    					dframe_cold_final['Bias'] = ERA5_Land_bias_cold_master
    					dframe_cold_final['RMSE'] = ERA5_Land_rmse_cold_master
    					dframe_cold_final['Correlation'] = ERA5_Land_corr_cold_master
    					dframe_cold_final['Norm Std'] = ERA5_Land_std_cold_master

    					metrics_cold_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CMOS_poster/CLSM_res/'+str(remap_type)+'_'+str(lyr_l)+'_naive_metrics_CLSM_ERA5_Land_by_grid_cell_BEST_cold_season_Sep2021.csv'])  					
    					dframe_cold_final.to_csv(metrics_cold_fil)
    					print(metrics_cold_fil)

    					dframe_cold_final = pd.DataFrame(data=gcell_cold_master, columns=['Grid Cell'])
    					dframe_cold_final['Lat'] = lat_cold_master
    					dframe_cold_final['Lon'] = lon_cold_master
    					dframe_cold_final['Bias'] = JRA_bias_cold_master
    					dframe_cold_final['RMSE'] = JRA_rmse_cold_master
    					dframe_cold_final['Correlation'] = JRA_corr_cold_master
    					dframe_cold_final['Norm Std'] = JRA_std_cold_master

    					metrics_cold_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CMOS_poster/CLSM_res/'+str(remap_type)+'_'+str(lyr_l)+'_naive_metrics_CLSM_JRA55_by_grid_cell_BEST_cold_season_Sep2021.csv'])  					
    					dframe_cold_final.to_csv(metrics_cold_fil)
    					print(metrics_cold_fil)

    					dframe_cold_final = pd.DataFrame(data=gcell_cold_master, columns=['Grid Cell'])
    					dframe_cold_final['Lat'] = lat_cold_master
    					dframe_cold_final['Lon'] = lon_cold_master
    					dframe_cold_final['Bias'] = MERRA2_bias_cold_master
    					dframe_cold_final['RMSE'] = MERRA2_rmse_cold_master
    					dframe_cold_final['Correlation'] = MERRA2_corr_cold_master
    					dframe_cold_final['Norm Std'] = MERRA2_std_cold_master

    					metrics_cold_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CMOS_poster/CLSM_res/'+str(remap_type)+'_'+str(lyr_l)+'_naive_metrics_CLSM_MERRA2_by_grid_cell_BEST_cold_season_Sep2021.csv'])  					
    					dframe_cold_final.to_csv(metrics_cold_fil)
    					print(metrics_cold_fil)


    					dframe_cold_final = pd.DataFrame(data=gcell_cold_master, columns=['Grid Cell'])
    					dframe_cold_final['Lat'] = lat_cold_master
    					dframe_cold_final['Lon'] = lon_cold_master
    					dframe_cold_final['Bias'] = GLDAS_bias_cold_master
    					dframe_cold_final['RMSE'] = GLDAS_rmse_cold_master
    					dframe_cold_final['Correlation'] = GLDAS_corr_cold_master
    					dframe_cold_final['Norm Std'] = GLDAS_std_cold_master

    					metrics_cold_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CMOS_poster/CLSM_res/'+str(remap_type)+'_'+str(lyr_l)+'_naive_metrics_CLSM_GLDAS_by_grid_cell_BEST_cold_season_Sep2021.csv'])  					
    					dframe_cold_final.to_csv(metrics_cold_fil)
    					print(metrics_cold_fil)


    					dframe_cold_final = pd.DataFrame(data=gcell_cold_master, columns=['Grid Cell'])
    					dframe_cold_final['Lat'] = lat_cold_master
    					dframe_cold_final['Lon'] = lon_cold_master
    					dframe_cold_final['Bias'] = GLDAS_CLSM_bias_cold_master
    					dframe_cold_final['RMSE'] = GLDAS_CLSM_rmse_cold_master
    					dframe_cold_final['Correlation'] = GLDAS_CLSM_corr_cold_master
    					dframe_cold_final['Norm Std'] = GLDAS_CLSM_std_cold_master

    					metrics_cold_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CMOS_poster/CLSM_res/'+str(remap_type)+'_'+str(lyr_l)+'_naive_metrics_CLSM_GLDAS_CLSM_by_grid_cell_BEST_cold_season_Sep2021.csv'])  					
    					dframe_cold_final.to_csv(metrics_cold_fil)
    					print(metrics_cold_fil)


    					dframe_warm_final = pd.DataFrame(data=gcell_warm_master, columns=['Grid Cell'])
    					dframe_warm_final['Lat'] = lat_warm_master
    					dframe_warm_final['Lon'] = lon_warm_master
    					dframe_warm_final['Bias'] = naive_all_bias_warm_master
    					dframe_warm_final['RMSE'] = naive_all_rmse_warm_master
    					dframe_warm_final['Correlation'] = naive_all_corr_warm_master
    					dframe_warm_final['Norm Std'] = naive_all_std_warm_master

    					metrics_warm_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CMOS_poster/CLSM_res/'+str(remap_type)+'_'+str(lyr_l)+'_naive_metrics_CLSM_EnsMean_by_grid_cell_BEST_warm_season_Sep2021.csv'])  					
    					dframe_warm_final.to_csv(metrics_warm_fil)
    					print(metrics_warm_fil)

    					dframe_warm_final = pd.DataFrame(data=gcell_warm_master, columns=['Grid Cell'])
    					dframe_warm_final['Lat'] = lat_warm_master
    					dframe_warm_final['Lon'] = lon_warm_master
    					dframe_warm_final['Bias'] = ensmean_4_bias_warm_master
    					dframe_warm_final['RMSE'] = ensmean_4_rmse_warm_master
    					dframe_warm_final['Correlation'] = ensmean_4_corr_warm_master
    					dframe_warm_final['Norm Std'] = ensmean_4_std_warm_master

    					print(dframe_warm_final)

    					metrics_warm_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CMOS_poster/CLSM_res/'+str(remap_type)+'_'+str(lyr_l)+'_naive_metrics_CLSM_EnsMean_4Model_by_grid_cell_BEST_warm_season_Sep2021.csv'])  					
    					dframe_warm_final.to_csv(metrics_warm_fil)
    					print(metrics_warm_fil)

    					dframe_warm_final = pd.DataFrame(data=gcell_warm_master, columns=['Grid Cell'])
    					dframe_warm_final['Lat'] = lat_warm_master
    					dframe_warm_final['Lon'] = lon_warm_master
    					dframe_warm_final['Bias'] = CFSR_bias_warm_master
    					dframe_warm_final['RMSE'] = CFSR_rmse_warm_master
    					dframe_warm_final['Correlation'] = CFSR_corr_warm_master
    					dframe_warm_final['Norm Std'] = CFSR_std_warm_master

    					metrics_warm_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CMOS_poster/CLSM_res/'+str(remap_type)+'_'+str(lyr_l)+'_naive_metrics_CLSM_CFSR_by_grid_cell_BEST_warm_season_Sep2021.csv'])  					
    					dframe_warm_final.to_csv(metrics_warm_fil)
    					print(metrics_warm_fil)

    					dframe_warm_final = pd.DataFrame(data=gcell_warm_master, columns=['Grid Cell'])
    					dframe_warm_final['Lat'] = lat_warm_master
    					dframe_warm_final['Lon'] = lon_warm_master
    					dframe_warm_final['Bias'] = ERAI_bias_warm_master
    					dframe_warm_final['RMSE'] = ERAI_rmse_warm_master
    					dframe_warm_final['Correlation'] = ERAI_corr_warm_master
    					dframe_warm_final['Norm Std'] = ERAI_std_warm_master

    					metrics_warm_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CMOS_poster/CLSM_res/'+str(remap_type)+'_'+str(lyr_l)+'_naive_metrics_CLSM_ERAI_by_grid_cell_BEST_warm_season_Sep2021.csv'])  					
    					dframe_warm_final.to_csv(metrics_warm_fil)
    					print(metrics_warm_fil)


    					dframe_warm_final = pd.DataFrame(data=gcell_warm_master, columns=['Grid Cell'])
    					dframe_warm_final['Lat'] = lat_warm_master
    					dframe_warm_final['Lon'] = lon_warm_master
    					dframe_warm_final['Bias'] = ERA5_bias_warm_master
    					dframe_warm_final['RMSE'] = ERA5_rmse_warm_master
    					dframe_warm_final['Correlation'] = ERA5_corr_warm_master
    					dframe_warm_final['Norm Std'] = ERA5_std_warm_master

    					metrics_warm_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CMOS_poster/CLSM_res/'+str(remap_type)+'_'+str(lyr_l)+'_naive_metrics_CLSM_ERA5_by_grid_cell_BEST_warm_season_Sep2021.csv'])  					
    					dframe_warm_final.to_csv(metrics_warm_fil)
    					print(metrics_warm_fil)


    					dframe_warm_final = pd.DataFrame(data=gcell_warm_master, columns=['Grid Cell'])
    					dframe_warm_final['Lat'] = lat_warm_master
    					dframe_warm_final['Lon'] = lon_warm_master
    					dframe_warm_final['Bias'] = ERA5_Land_bias_warm_master
    					dframe_warm_final['RMSE'] = ERA5_Land_rmse_warm_master
    					dframe_warm_final['Correlation'] = ERA5_Land_corr_warm_master
    					dframe_warm_final['Norm Std'] = ERA5_Land_std_warm_master

    					metrics_warm_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CMOS_poster/CLSM_res/'+str(remap_type)+'_'+str(lyr_l)+'_naive_metrics_CLSM_ERA5_Land_by_grid_cell_BEST_warm_season_Sep2021.csv'])  					
    					dframe_warm_final.to_csv(metrics_warm_fil)
    					print(metrics_warm_fil)

    					dframe_warm_final = pd.DataFrame(data=gcell_warm_master, columns=['Grid Cell'])
    					dframe_warm_final['Lat'] = lat_warm_master
    					dframe_warm_final['Lon'] = lon_warm_master
    					dframe_warm_final['Bias'] = JRA_bias_warm_master
    					dframe_warm_final['RMSE'] = JRA_rmse_warm_master
    					dframe_warm_final['Correlation'] = JRA_corr_warm_master
    					dframe_warm_final['Norm Std'] = JRA_std_warm_master

    					metrics_warm_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CMOS_poster/CLSM_res/'+str(remap_type)+'_'+str(lyr_l)+'_naive_metrics_CLSM_JRA55_by_grid_cell_BEST_warm_season_Sep2021.csv'])  					
    					dframe_warm_final.to_csv(metrics_warm_fil)
    					print(metrics_warm_fil)

    					dframe_warm_final = pd.DataFrame(data=gcell_warm_master, columns=['Grid Cell'])
    					dframe_warm_final['Lat'] = lat_warm_master
    					dframe_warm_final['Lon'] = lon_warm_master
    					dframe_warm_final['Bias'] = MERRA2_bias_warm_master
    					dframe_warm_final['RMSE'] = MERRA2_rmse_warm_master
    					dframe_warm_final['Correlation'] = MERRA2_corr_warm_master
    					dframe_warm_final['Norm Std'] = MERRA2_std_warm_master

    					metrics_warm_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CMOS_poster/CLSM_res/'+str(remap_type)+'_'+str(lyr_l)+'_naive_metrics_CLSM_MERRA2_by_grid_cell_BEST_warm_season_Sep2021.csv'])  					
    					dframe_warm_final.to_csv(metrics_warm_fil)
    					print(metrics_warm_fil)


    					dframe_warm_final = pd.DataFrame(data=gcell_warm_master, columns=['Grid Cell'])
    					dframe_warm_final['Lat'] = lat_warm_master
    					dframe_warm_final['Lon'] = lon_warm_master
    					dframe_warm_final['Bias'] = GLDAS_bias_warm_master
    					dframe_warm_final['RMSE'] = GLDAS_rmse_warm_master
    					dframe_warm_final['Correlation'] = GLDAS_corr_warm_master
    					dframe_warm_final['Norm Std'] = GLDAS_std_warm_master

    					metrics_warm_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CMOS_poster/CLSM_res/'+str(remap_type)+'_'+str(lyr_l)+'_naive_metrics_CLSM_GLDAS_by_grid_cell_BEST_warm_season_Sep2021.csv'])  					
    					dframe_warm_final.to_csv(metrics_warm_fil)
    					print(metrics_warm_fil)


    					dframe_warm_final = pd.DataFrame(data=gcell_warm_master, columns=['Grid Cell'])
    					dframe_warm_final['Lat'] = lat_warm_master
    					dframe_warm_final['Lon'] = lon_warm_master
    					dframe_warm_final['Bias'] = GLDAS_CLSM_bias_warm_master
    					dframe_warm_final['RMSE'] = GLDAS_CLSM_rmse_warm_master
    					dframe_warm_final['Correlation'] = GLDAS_CLSM_corr_warm_master
    					dframe_warm_final['Norm Std'] = GLDAS_CLSM_std_warm_master

    					metrics_warm_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CMOS_poster/CLSM_res/'+str(remap_type)+'_'+str(lyr_l)+'_naive_metrics_CLSM_GLDAS_CLSM_by_grid_cell_BEST_warm_season_Sep2021.csv'])  					
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


product = ['EnsMean','EnsMean_4Model','CFSR','ERAI','ERA5','ERA5_Land','JRA55','MERRA2','GLDAS','GLDAS_CLSM']

########## Create Plots ###########

for i in product:
    product_i = i
    print('Product:',product_i)    	
    top_cold_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CMOS_poster/CLSM_res/remapcon_top_30cm_naive_metrics_CLSM_'+str(product_i)+'_by_grid_cell_BEST_cold_season_Sep2021.csv'])
    top_warm_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CMOS_poster/CLSM_res/remapcon_top_30cm_naive_metrics_CLSM_'+str(product_i)+'_by_grid_cell_BEST_warm_season_Sep2021.csv'])
    btm_cold_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CMOS_poster/CLSM_res/remapcon_30cm_300cm_naive_metrics_CLSM_'+str(product_i)+'_by_grid_cell_BEST_cold_season_Sep2021.csv'])
    btm_warm_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CMOS_poster/CLSM_res/remapcon_30cm_300cm_naive_metrics_CLSM_'+str(product_i)+'_by_grid_cell_BEST_warm_season_Sep2021.csv'])

    dframe_top_cold = pd.read_csv(top_cold_fil)
    len_top_cold = len(dframe_top_cold)
    grid_top_cold = dframe_top_cold['Grid Cell'].values
    lat_top_cold = dframe_top_cold['Lat'].values
    lon_top_cold = dframe_top_cold['Lon'].values
    bias_top_cold = dframe_top_cold['Bias'].values
    RMSE_top_cold = dframe_top_cold['RMSE'].values
    corr_top_cold = dframe_top_cold['Correlation'].values
    std_top_cold = dframe_top_cold['Norm Std'].values

    dframe_btm_cold = pd.read_csv(btm_cold_fil)
    len_btm_cold = len(dframe_btm_cold)
    grid_btm_cold = dframe_btm_cold['Grid Cell'].values
    lat_btm_cold = dframe_btm_cold['Lat'].values
    lon_btm_cold = dframe_btm_cold['Lon'].values
    bias_btm_cold = dframe_btm_cold['Bias'].values
    RMSE_btm_cold = dframe_btm_cold['RMSE'].values
    corr_btm_cold = dframe_btm_cold['Correlation'].values
    std_btm_cold = dframe_btm_cold['Norm Std'].values

    dframe_top_warm = pd.read_csv(top_warm_fil)
    len_top_warm = len(dframe_top_warm)
    grid_top_warm = dframe_top_warm['Grid Cell'].values
    lat_top_warm = dframe_top_warm['Lat'].values
    lon_top_warm = dframe_top_warm['Lon'].values
    bias_top_warm = dframe_top_warm['Bias'].values
    RMSE_top_warm = dframe_top_warm['RMSE'].values 
    corr_top_warm = dframe_top_warm['Correlation'].values
    std_top_warm = dframe_top_warm['Norm Std'].values

    dframe_btm_warm = pd.read_csv(btm_warm_fil)
    len_btm_warm = len(dframe_btm_warm)
    grid_btm_warm = dframe_btm_warm['Grid Cell'].values
    lat_btm_warm = dframe_btm_warm['Lat'].values
    lon_btm_warm = dframe_btm_warm['Lon'].values
    bias_btm_warm = dframe_btm_warm['Bias'].values
    RMSE_btm_warm = dframe_btm_warm['RMSE'].values
    corr_btm_warm = dframe_btm_warm['Correlation'].values
    std_btm_warm = dframe_btm_warm['Norm Std'].values

    #print(grid_btm_cold)


    top_grid_dir = "/mnt/data/users/herringtont/soil_temp/In-Situ/All/spatial_average_anom/remapcon/no_outliers/zscore/top_30cm/thr_100/CLSM"
    btm_grid_dir = "/mnt/data/users/herringtont/soil_temp/In-Situ/All/spatial_average_anom/remapcon/no_outliers/zscore/30_299.9/thr_100/CLSM"


    top_lyr_dir = "/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/zscore/top_30cm"
    btm_lyr_dir = "/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/zscore/30_299.9"

#### loop through grid cells ####


    top_cold_grid_master = []
    top_cold_continent_master = []
    top_cold_lat_master = []
    top_cold_lon_master = []
    top_cold_bias_master = []
    top_cold_RMSE_master = []
    top_cold_corr_master = []
    top_cold_std_master = []    
    top_cold_avg_min_depth_master = []    
    for j in range(0,len_top_cold):
    	grid_i = grid_top_cold[j]
    	lat_i = lat_top_cold[j]
    	lon_i = lon_top_cold[j]
    	if (-179.5 <= lon_i <= -15):
    		continent = "North_America"

    	elif (-15 < lon_i < 179.5):
    		continent = "Eurasia"
    	bias_i = bias_top_cold[j]
    	RMSE_i = RMSE_top_cold[j]
    	corr_i = corr_top_cold[j]
    	std_i = std_top_cold[j]		    		
    	grid_fil = ''.join([top_grid_dir,'/grid_'+str(grid_i)+'_anom.csv'])

    	dframe_top_cold = pd.read_csv(grid_fil)
    	dframe_top_cold = dframe_top_cold.drop(['Date','Grid Cell','Central Lat','Central Lon','Spatial Avg Anom','Spatial Avg Temp','Sites Incl'],axis=1)
    	#print(dframe_top_cold)

    	sites_top_cold = dframe_top_cold.columns.values.tolist()



    	min_depth_master = []
    	## loop through sites ##
    	for k in sites_top_cold:
    		site_j = k   	
    		site_fil = ''.join([top_lyr_dir+'/site_'+str(site_j)+'.csv'])

    		dframe_top_cold_lyr = pd.read_csv(site_fil)
    		dframe_top_cold_lyr = dframe_top_cold_lyr.drop(['Date','Dataset','Lat','Long','Layer_Avg','Depths_Incl'],axis=1)
    		depths_top_cold = dframe_top_cold_lyr.columns.values
    		min_depth = max(depths_top_cold)

    		#print(depths_top_cold)
    		#print(min_depth)
    		min_depth_master.append(float(min_depth))

    	#min_depth_master = [i for sub in min_depth_master for i in sub]
    	len_depth = len(min_depth_master)
    	#print(min_depth_master)
    	if (len_depth > 1):
    		avg_min_depth = mean(min_depth_master)
    	else:
    		avg_min_depth = str(min_depth_master)[1:-1]
    		avg_min_depth = float(avg_min_depth)

    	top_cold_avg_min_depth_master.append(avg_min_depth)
    	top_cold_grid_master.append(grid_i)
    	top_cold_continent_master.append(continent)
    	top_cold_lat_master.append(lat_i)
    	top_cold_lon_master.append(lon_i)
    	top_cold_bias_master.append(bias_i)
    	top_cold_RMSE_master.append(RMSE_i)
    	top_cold_corr_master.append(corr_i)
    	top_cold_std_master.append(std_i)
    #print(top_cold_grid_master)
    #print(top_cold_avg_min_depth_master)
    #top_cold_grid_master = [i for sub in top_cold_grid_master for i in sub]
    #top_cold_avg_min_depth_master = [i for sub in top_cold_avg_min_depth_master for i in sub]

    dframe_top_cold_master = pd.DataFrame(data = top_cold_grid_master, columns=['Grid Cell'])
    dframe_top_cold_master['Lat'] = top_cold_lat_master
    dframe_top_cold_master['Lon'] = top_cold_lon_master
    dframe_top_cold_master['Continent'] = top_cold_continent_master
    dframe_top_cold_master['Bias'] = top_cold_bias_master
    dframe_top_cold_master['RMSE'] = top_cold_RMSE_master
    dframe_top_cold_master['Corr'] = top_cold_corr_master
    dframe_top_cold_master['Std Dev'] = top_cold_std_master
    dframe_top_cold_master['Avg Min Depth'] = top_cold_avg_min_depth_master

    #print(dframe_top_cold_master)

    top_cold_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CMOS_poster/CLSM_res/depth_corr/remapcon_top_30cm_naive_metrics_CLSM_'+str(product_i)+'_by_grid_cell_BEST_cold_season_Sep2021_depth_corr.csv'])
    dframe_top_cold_master.to_csv(top_cold_fil,index=False) 

    top_warm_grid_master = []
    top_warm_continent_master = []
    top_warm_lat_master = []
    top_warm_lon_master = []
    top_warm_bias_master = []
    top_warm_RMSE_master = []
    top_warm_corr_master = []
    top_warm_std_master = []    
    top_warm_avg_min_depth_master = []    
    for j in range(0,len_top_warm):
    	grid_i = grid_top_warm[j]
    	lat_i = lat_top_warm[j]
    	lon_i = lon_top_warm[j]
    	if (-179.5 <= lon_i <= -15):
    		continent = "North_America"

    	elif (-15 < lon_i < 179.5):
    		continent = "Eurasia"
    	bias_i = bias_top_warm[j]
    	RMSE_i = RMSE_top_warm[j]
    	corr_i = corr_top_warm[j]
    	std_i = std_top_warm[j]		    		
    	grid_fil = ''.join([top_grid_dir,'/grid_'+str(grid_i)+'_anom.csv'])

    	dframe_top_warm = pd.read_csv(grid_fil)
    	dframe_top_warm = dframe_top_warm.drop(['Date','Grid Cell','Central Lat','Central Lon','Spatial Avg Anom','Spatial Avg Temp','Sites Incl'],axis=1)
    	#print(dframe_top_warm)

    	sites_top_warm = dframe_top_warm.columns.values.tolist()



    	min_depth_master = []
    	## loop through sites ##
    	for k in sites_top_warm:
    		site_j = k   	
    		site_fil = ''.join([top_lyr_dir+'/site_'+str(site_j)+'.csv'])

    		dframe_top_warm_lyr = pd.read_csv(site_fil)
    		dframe_top_warm_lyr = dframe_top_warm_lyr.drop(['Date','Dataset','Lat','Long','Layer_Avg','Depths_Incl'],axis=1)
    		depths_top_warm = dframe_top_warm_lyr.columns.values
    		min_depth = max(depths_top_warm)

    		#print(depths_top_warm)
    		print(min_depth)
    		min_depth_master.append(float(min_depth))

    	#min_depth_master = [i for sub in min_depth_master for i in sub]
    	len_depth = len(min_depth_master)
    	#print(min_depth_master)
    	if (len_depth > 1):
    		avg_min_depth = mean(min_depth_master)
    	else:
    		avg_min_depth = str(min_depth_master)[1:-1]
    		avg_min_depth = float(avg_min_depth)

    	top_warm_avg_min_depth_master.append(avg_min_depth)
    	top_warm_grid_master.append(grid_i)
    	top_warm_continent_master.append(continent)
    	top_warm_lat_master.append(lat_i)
    	top_warm_lon_master.append(lon_i)
    	top_warm_bias_master.append(bias_i)
    	top_warm_RMSE_master.append(RMSE_i)
    	top_warm_corr_master.append(corr_i)
    	top_warm_std_master.append(std_i)
    #print(top_warm_grid_master)
    #print(top_warm_avg_min_depth_master)
    #top_warm_grid_master = [i for sub in top_warm_grid_master for i in sub]
    #top_warm_avg_min_depth_master = [i for sub in top_warm_avg_min_depth_master for i in sub]

    dframe_top_warm_master = pd.DataFrame(data = top_warm_grid_master, columns=['Grid Cell'])
    dframe_top_warm_master['Lat'] = top_warm_lat_master
    dframe_top_warm_master['Lon'] = top_warm_lon_master
    dframe_top_warm_master['Continent'] = top_warm_continent_master
    dframe_top_warm_master['Bias'] = top_warm_bias_master
    dframe_top_warm_master['RMSE'] = top_warm_RMSE_master
    dframe_top_warm_master['Corr'] = top_warm_corr_master
    dframe_top_warm_master['Std Dev'] = top_warm_std_master
    dframe_top_warm_master['Avg Min Depth'] = top_warm_avg_min_depth_master

    #print(dframe_top_warm_master)

    top_warm_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CMOS_poster/CLSM_res/depth_corr/remapcon_top_30cm_naive_metrics_CLSM_'+str(product_i)+'_by_grid_cell_BEST_warm_season_Sep2021_depth_corr.csv'])
    dframe_top_warm_master.to_csv(top_warm_fil,index=False)


    btm_cold_grid_master = []
    btm_cold_continent_master = []
    btm_cold_lat_master = []
    btm_cold_lon_master = []
    btm_cold_bias_master = []
    btm_cold_RMSE_master = []
    btm_cold_corr_master = []
    btm_cold_std_master = []    
    btm_cold_avg_min_depth_master = []    
    for j in range(0,len_btm_cold):
    	grid_i = grid_btm_cold[j]
    	if(grid_i == 2211):
    		continue
    	lat_i = lat_btm_cold[j]
    	lon_i = lon_btm_cold[j]
    	if (-179.5 <= lon_i <= -15):
    		continent = "North_America"

    	elif (-15 < lon_i < 179.5):
    		continent = "Eurasia"
    	bias_i = bias_btm_cold[j]
    	RMSE_i = RMSE_btm_cold[j]
    	corr_i = corr_btm_cold[j]
    	std_i = std_btm_cold[j]		    		
    	grid_fil = ''.join([btm_grid_dir,'/grid_'+str(grid_i)+'_anom.csv'])

    	dframe_btm_cold = pd.read_csv(grid_fil)
    	dframe_btm_cold = dframe_btm_cold.drop(['Date','Grid Cell','Central Lat','Central Lon','Spatial Avg Anom','Spatial Avg Temp','Sites Incl'],axis=1)
    	#print(dframe_btm_cold)

    	sites_btm_cold = dframe_btm_cold.columns.values.tolist()



    	min_depth_master = []
    	## loop through sites ##
    	for k in sites_btm_cold:
    		site_j = k   	
    		site_fil = ''.join([btm_lyr_dir+'/site_'+str(site_j)+'.csv'])

    		dframe_btm_cold_lyr = pd.read_csv(site_fil)
    		dframe_btm_cold_lyr = dframe_btm_cold_lyr.drop(['Date','Dataset','Lat','Long','Layer_Avg','Depths_Incl'],axis=1)
    		depths_btm_cold = dframe_btm_cold_lyr.columns.values
    		min_depth = max(depths_btm_cold)

    		#print(depths_btm_cold)
    		#print(min_depth)
    		min_depth_master.append(float(min_depth))

    	#min_depth_master = [i for sub in min_depth_master for i in sub]
    	len_depth = len(min_depth_master)
    	#print(min_depth_master)
    	if (len_depth > 1):
    		avg_min_depth = mean(min_depth_master)
    	else:
    		avg_min_depth = str(min_depth_master)[1:-1]
    		avg_min_depth = float(avg_min_depth)

    	btm_cold_avg_min_depth_master.append(avg_min_depth)
    	btm_cold_grid_master.append(grid_i)
    	btm_cold_continent_master.append(continent)
    	btm_cold_lat_master.append(lat_i)
    	btm_cold_lon_master.append(lon_i)
    	btm_cold_bias_master.append(bias_i)
    	btm_cold_RMSE_master.append(RMSE_i)
    	btm_cold_corr_master.append(corr_i)
    	btm_cold_std_master.append(std_i)
    #print(btm_cold_grid_master)
    #print(btm_cold_avg_min_depth_master)
    #btm_cold_grid_master = [i for sub in btm_cold_grid_master for i in sub]
    #btm_cold_avg_min_depth_master = [i for sub in btm_cold_avg_min_depth_master for i in sub]

    dframe_btm_cold_master = pd.DataFrame(data = btm_cold_grid_master, columns=['Grid Cell'])
    dframe_btm_cold_master['Lat'] = btm_cold_lat_master
    dframe_btm_cold_master['Lon'] = btm_cold_lon_master
    dframe_btm_cold_master['Continent'] = btm_cold_continent_master
    dframe_btm_cold_master['Bias'] = btm_cold_bias_master
    dframe_btm_cold_master['RMSE'] = btm_cold_RMSE_master
    dframe_btm_cold_master['Corr'] = btm_cold_corr_master
    dframe_btm_cold_master['Std Dev'] = btm_cold_std_master
    dframe_btm_cold_master['Avg Min Depth'] = btm_cold_avg_min_depth_master

    #print(dframe_btm_cold_master)

    btm_cold_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CMOS_poster/CLSM_res/depth_corr/remapcon_30cm_300cm_naive_metrics_CLSM_'+str(product_i)+'_by_grid_cell_BEST_cold_season_Sep2021_depth_corr.csv'])
    dframe_btm_cold_master.to_csv(btm_cold_fil,index=False) 


    btm_warm_grid_master = []
    btm_warm_continent_master = []
    btm_warm_lat_master = []
    btm_warm_lon_master = []
    btm_warm_bias_master = []
    btm_warm_RMSE_master = []
    btm_warm_corr_master = []
    btm_warm_std_master = []    
    btm_warm_avg_min_depth_master = []    
    for j in range(0,len_btm_warm):
    	grid_i = grid_btm_warm[j]
    	if(grid_i == 2211):
    		continue
    	lat_i = lat_btm_warm[j]
    	lon_i = lon_btm_warm[j]
    	if (-179.5 <= lon_i <= -15):
    		continent = "North_America"

    	elif (-15 < lon_i < 179.5):
    		continent = "Eurasia"
    	bias_i = bias_btm_warm[j]
    	RMSE_i = RMSE_btm_warm[j]
    	corr_i = corr_btm_warm[j]
    	std_i = std_btm_warm[j]		    		
    	grid_fil = ''.join([btm_grid_dir,'/grid_'+str(grid_i)+'_anom.csv'])

    	dframe_btm_warm = pd.read_csv(grid_fil)
    	dframe_btm_warm = dframe_btm_warm.drop(['Date','Grid Cell','Central Lat','Central Lon','Spatial Avg Anom','Spatial Avg Temp','Sites Incl'],axis=1)
    	#print(dframe_btm_warm)

    	sites_btm_warm = dframe_btm_warm.columns.values.tolist()



    	min_depth_master = []
    	## loop through sites ##
    	for k in sites_btm_warm:
    		site_j = k   	
    		site_fil = ''.join([btm_lyr_dir+'/site_'+str(site_j)+'.csv'])

    		dframe_btm_warm_lyr = pd.read_csv(site_fil)
    		dframe_btm_warm_lyr = dframe_btm_warm_lyr.drop(['Date','Dataset','Lat','Long','Layer_Avg','Depths_Incl'],axis=1)
    		depths_btm_warm = dframe_btm_warm_lyr.columns.values
    		min_depth = max(depths_btm_warm)

    		#print(depths_btm_warm)
    		#print(min_depth)
    		min_depth_master.append(float(min_depth))

    	#min_depth_master = [i for sub in min_depth_master for i in sub]
    	len_depth = len(min_depth_master)
    	#print(min_depth_master)
    	if (len_depth > 1):
    		avg_min_depth = mean(min_depth_master)
    	else:
    		avg_min_depth = str(min_depth_master)[1:-1]
    		avg_min_depth = float(avg_min_depth)

    	btm_warm_avg_min_depth_master.append(avg_min_depth)
    	btm_warm_grid_master.append(grid_i)
    	btm_warm_continent_master.append(continent)
    	btm_warm_lat_master.append(lat_i)
    	btm_warm_lon_master.append(lon_i)
    	btm_warm_bias_master.append(bias_i)
    	btm_warm_RMSE_master.append(RMSE_i)
    	btm_warm_corr_master.append(corr_i)
    	btm_warm_std_master.append(std_i)
    #print(btm_warm_grid_master)
    #print(btm_warm_avg_min_depth_master)
    #btm_warm_grid_master = [i for sub in btm_warm_grid_master for i in sub]
    #btm_warm_avg_min_depth_master = [i for sub in btm_warm_avg_min_depth_master for i in sub]

    dframe_btm_warm_master = pd.DataFrame(data = btm_warm_grid_master, columns=['Grid Cell'])
    dframe_btm_warm_master['Lat'] = btm_warm_lat_master
    dframe_btm_warm_master['Lon'] = btm_warm_lon_master
    dframe_btm_warm_master['Continent'] = btm_warm_continent_master
    dframe_btm_warm_master['Bias'] = btm_warm_bias_master
    dframe_btm_warm_master['RMSE'] = btm_warm_RMSE_master
    dframe_btm_warm_master['Corr'] = btm_warm_corr_master
    dframe_btm_warm_master['Std Dev'] = btm_warm_std_master
    dframe_btm_warm_master['Avg Min Depth'] = btm_warm_avg_min_depth_master

    #print(dframe_btm_warm_master)

    btm_warm_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CMOS_poster/CLSM_res/depth_corr/remapcon_30cm_300cm_naive_metrics_CLSM_'+str(product_i)+'_by_grid_cell_BEST_warm_season_Sep2021_depth_corr.csv'])
    dframe_btm_warm_master.to_csv(btm_warm_fil,index=False) 




