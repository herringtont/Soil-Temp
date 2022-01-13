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
permafrost_type = ['RS_2002_permafrost', 'RS_2002_none', 'RS_2002_all']

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

    					fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/CLSM_res/subset/sites_1yr/'+str(remap_type)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_dframe_scatterplot_CMOS_CLSM_subset_permafrost_cold_warm_BEST_Sep2021_NWT_Yukon_sites_1yr.csv'])
    					dframe = pd.read_csv(fil)

#### Cold Season ####


## Master Arrays ##

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


    					dframe_cold_season = dframe[dframe['Season'] == 'Cold']

    					gcell_cold = dframe_cold_season['Grid Cell'].values
    					gcell_cold_uq = np.unique(gcell_cold)

    					if (permafrost_type_o == 'RS_2002_permafrost'):
    						dframe_cold_season_permafrost = dframe_cold_season[(dframe_cold_season['RS 2002 Permafrost'] == 'continuous') | (dframe_cold_season['RS 2002 Permafrost'] == 'discontinuous')]

    					elif (permafrost_type_o == 'RS_2002_none'):
    						dframe_cold_season_permafrost = dframe_cold_season[dframe_cold_season['RS 2002 Permafrost'] == 'none']


    					elif (permafrost_type_o == 'RS_2002_all'):
    						dframe_cold_season_permafrost = dframe_cold_season

    					gcell_cold = dframe_cold_season_permafrost['Grid Cell'].values
    					gcell_cold_uq = np.unique(gcell_cold)

    					for p in gcell_cold_uq: # loop through grid cells
    						gcell_p = p
    						if (gcell_p == 33777):
    							continue
    						dframe_cold_season_gcell = dframe_cold_season_permafrost[dframe_cold_season_permafrost['Grid Cell'] == gcell_p]
    						if (len(dframe_cold_season_gcell) < 2):
    							continue    						
    						station_temp_cold = dframe_cold_season_gcell['Station'].values
    						naive_temp_cold = dframe_cold_season_gcell['Naive Blend'].values
    						naive_noJRA_temp_cold = dframe_cold_season_gcell['Naive Blend no JRA55'].values
    						naive_noJRAold_temp_cold = dframe_cold_season_gcell['Naive Blend no JRA55 Old'].values
    						naive_all_temp_cold = dframe_cold_season_gcell['Naive Blend All'].values
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



## RMSE and ubRMSE ##
    						naive_rmse_cold = mean_squared_error(station_temp_cold,naive_temp_cold, squared=False)
    						naive_rmse_cold_master.append(naive_rmse_cold)
    						naive_noJRA_rmse_cold = mean_squared_error(station_temp_cold,naive_noJRA_temp_cold, squared=False)
    						naive_noJRA_rmse_cold_master.append(naive_noJRA_rmse_cold)
    						naive_noJRAold_rmse_cold = mean_squared_error(station_temp_cold,naive_noJRAold_temp_cold, squared=False)
    						naive_noJRAold_rmse_cold_master.append(naive_noJRAold_rmse_cold)
    						naive_all_rmse_cold = mean_squared_error(station_temp_cold,naive_all_temp_cold, squared=False)
    						naive_all_rmse_cold_master.append(naive_all_rmse_cold)
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


## Calculate Average Metrics Across Grid Cells ##

    					naive_bias_cold_avg = mean(naive_bias_cold_master)
    					naive_noJRA_bias_cold_avg = mean(naive_noJRA_bias_cold_master)
    					naive_noJRAold_bias_cold_avg = mean(naive_noJRAold_bias_cold_master)
    					naive_all_bias_cold_avg = mean(naive_all_bias_cold_master)
    					CFSR_bias_cold_avg = mean(CFSR_bias_cold_master)
    					ERAI_bias_cold_avg = mean(ERAI_bias_cold_master)
    					ERA5_bias_cold_avg = mean(ERA5_bias_cold_master)
    					ERA5_Land_bias_cold_avg = mean(ERA5_Land_bias_cold_master)
    					JRA_bias_cold_avg = mean(JRA_bias_cold_master)
    					MERRA2_bias_cold_avg = mean(MERRA2_bias_cold_master)
    					GLDAS_bias_cold_avg = mean(GLDAS_bias_cold_master)
    					GLDAS_CLSM_bias_cold_avg = mean(GLDAS_CLSM_bias_cold_master)

    					naive_bias_cold_stddev = stdev(naive_bias_cold_master)
    					naive_noJRA_bias_cold_stddev = stdev(naive_noJRA_bias_cold_master)
    					naive_noJRAold_bias_cold_stddev = stdev(naive_noJRAold_bias_cold_master)
    					naive_all_bias_cold_stddev = stdev(naive_all_bias_cold_master)
    					CFSR_bias_cold_stddev = stdev(CFSR_bias_cold_master)
    					ERAI_bias_cold_stddev = stdev(ERAI_bias_cold_master)
    					ERA5_bias_cold_stddev = stdev(ERA5_bias_cold_master)
    					ERA5_Land_bias_cold_stddev = stdev(ERA5_Land_bias_cold_master)
    					JRA_bias_cold_stddev = stdev(JRA_bias_cold_master)
    					MERRA2_bias_cold_stddev = stdev(MERRA2_bias_cold_master)
    					GLDAS_bias_cold_stddev = stdev(GLDAS_bias_cold_master)
    					GLDAS_CLSM_bias_cold_stddev = stdev(GLDAS_CLSM_bias_cold_master)

    					stn_var_cold_avg = mean(stn_var_cold_master)
    					naive_var_cold_avg = mean(naive_var_cold_master)
    					naive_noJRA_var_cold_avg = mean(naive_noJRA_var_cold_master)
    					naive_noJRAold_var_cold_avg = mean(naive_noJRAold_var_cold_master)
    					naive_all_var_cold_avg = mean(naive_all_var_cold_master)
    					CFSR_var_cold_avg = mean(CFSR_var_cold_master)
    					ERAI_var_cold_avg = mean(ERAI_var_cold_master)
    					ERA5_var_cold_avg = mean(ERA5_var_cold_master)
    					ERA5_Land_var_cold_avg = mean(ERA5_Land_var_cold_master)
    					JRA_var_cold_avg = mean(JRA_var_cold_master)
    					MERRA2_var_cold_avg = mean(MERRA2_var_cold_master)
    					GLDAS_var_cold_avg = mean(GLDAS_var_cold_master)
    					GLDAS_CLSM_var_cold_avg = mean(GLDAS_CLSM_var_cold_master)

    					stn_sdev_cold_avg = math.sqrt(stn_var_cold_avg)
    					naive_sdev_cold_avg = math.sqrt(naive_var_cold_avg)
    					naive_noJRA_sdev_cold_avg = math.sqrt(naive_noJRA_var_cold_avg)
    					naive_noJRAold_sdev_cold_avg = math.sqrt(naive_noJRAold_var_cold_avg)
    					naive_all_sdev_cold_avg = math.sqrt(naive_all_var_cold_avg)
    					CFSR_sdev_cold_avg = math.sqrt(CFSR_var_cold_avg)
    					ERAI_sdev_cold_avg = math.sqrt(ERAI_var_cold_avg)
    					ERA5_sdev_cold_avg = math.sqrt(ERA5_var_cold_avg)
    					ERA5_Land_sdev_cold_avg = math.sqrt(ERA5_Land_var_cold_avg)
    					JRA_sdev_cold_avg = math.sqrt(JRA_var_cold_avg)
    					MERRA2_sdev_cold_avg = math.sqrt(MERRA2_var_cold_avg)
    					GLDAS_sdev_cold_avg = math.sqrt(GLDAS_var_cold_avg)
    					GLDAS_CLSM_sdev_cold_avg = math.sqrt(GLDAS_CLSM_var_cold_avg)

    					naive_SDV_cold_avg = naive_sdev_cold_avg/stn_sdev_cold_avg
    					naive_noJRA_SDV_cold_avg = naive_noJRA_sdev_cold_avg/stn_sdev_cold_avg
    					naive_noJRAold_SDV_cold_avg = naive_noJRAold_sdev_cold_avg/stn_sdev_cold_avg
    					naive_all_SDV_cold_avg = naive_all_sdev_cold_avg/stn_sdev_cold_avg
    					CFSR_SDV_cold_avg = CFSR_sdev_cold_avg/stn_sdev_cold_avg
    					ERAI_SDV_cold_avg = ERAI_sdev_cold_avg/stn_sdev_cold_avg
    					ERA5_SDV_cold_avg = ERA5_sdev_cold_avg/stn_sdev_cold_avg
    					ERA5_Land_SDV_cold_avg = ERA5_Land_sdev_cold_avg/stn_sdev_cold_avg
    					JRA_SDV_cold_avg = JRA_sdev_cold_avg/stn_sdev_cold_avg
    					MERRA2_SDV_cold_avg = MERRA2_sdev_cold_avg/stn_sdev_cold_avg
    					GLDAS_SDV_cold_avg = GLDAS_sdev_cold_avg/stn_sdev_cold_avg
    					GLDAS_CLSM_SDV_cold_avg = GLDAS_CLSM_sdev_cold_avg/stn_sdev_cold_avg

    					naive_rmse_cold_avg = mean(naive_rmse_cold_master)
    					naive_noJRA_rmse_cold_avg = mean(naive_noJRA_rmse_cold_master)
    					naive_noJRAold_rmse_cold_avg = mean(naive_noJRAold_rmse_cold_master)
    					naive_all_rmse_cold_avg = mean(naive_all_rmse_cold_master)
    					CFSR_rmse_cold_avg = mean(CFSR_rmse_cold_master)
    					ERAI_rmse_cold_avg = mean(ERAI_rmse_cold_master)
    					ERA5_rmse_cold_avg = mean(ERA5_rmse_cold_master)
    					ERA5_Land_rmse_cold_avg = mean(ERA5_Land_rmse_cold_master)
    					JRA_rmse_cold_avg = mean(JRA_rmse_cold_master)
    					MERRA2_rmse_cold_avg = mean(MERRA2_rmse_cold_master)
    					GLDAS_rmse_cold_avg = mean(GLDAS_rmse_cold_master)
    					GLDAS_CLSM_rmse_cold_avg = mean(GLDAS_CLSM_rmse_cold_master)

    					naive_rmse_cold_stddev = stdev(naive_rmse_cold_master)
    					naive_noJRA_rmse_cold_stddev = stdev(naive_noJRA_rmse_cold_master)
    					naive_noJRAold_rmse_cold_stddev = stdev(naive_noJRAold_rmse_cold_master)
    					naive_all_rmse_cold_stddev = stdev(naive_all_rmse_cold_master)
    					CFSR_rmse_cold_stddev = stdev(CFSR_rmse_cold_master)
    					ERAI_rmse_cold_stddev = stdev(ERAI_rmse_cold_master)
    					ERA5_rmse_cold_stddev = stdev(ERA5_rmse_cold_master)
    					ERA5_Land_rmse_cold_stddev = stdev(ERA5_Land_rmse_cold_master)
    					JRA_rmse_cold_stddev = stdev(JRA_rmse_cold_master)
    					MERRA2_rmse_cold_stddev = stdev(MERRA2_rmse_cold_master)
    					GLDAS_rmse_cold_stddev = stdev(GLDAS_rmse_cold_master)
    					GLDAS_CLSM_rmse_cold_stddev = stdev(GLDAS_CLSM_rmse_cold_master)

    					naive_ubrmse_cold_avg = mean(naive_ubrmse_cold_master)
    					naive_noJRA_ubrmse_cold_avg = mean(naive_noJRA_ubrmse_cold_master)
    					naive_noJRAold_ubrmse_cold_avg = mean(naive_noJRAold_ubrmse_cold_master)
    					naive_all_ubrmse_cold_avg = mean(naive_all_ubrmse_cold_master)
    					CFSR_ubrmse_cold_avg = mean(CFSR_ubrmse_cold_master)
    					ERAI_ubrmse_cold_avg = mean(ERAI_ubrmse_cold_master)
    					ERA5_ubrmse_cold_avg = mean(ERA5_ubrmse_cold_master)
    					ERA5_Land_ubrmse_cold_avg = mean(ERA5_Land_ubrmse_cold_master)
    					JRA_ubrmse_cold_avg = mean(JRA_ubrmse_cold_master)
    					MERRA2_ubrmse_cold_avg = mean(MERRA2_ubrmse_cold_master)
    					GLDAS_ubrmse_cold_avg = mean(GLDAS_ubrmse_cold_master)
    					GLDAS_CLSM_ubrmse_cold_avg = mean(GLDAS_CLSM_ubrmse_cold_master)

    					naive_ubrmse_cold_stddev = stdev(naive_ubrmse_cold_master)
    					naive_noJRA_ubrmse_cold_stddev = stdev(naive_noJRA_ubrmse_cold_master)
    					naive_noJRAold_ubrmse_cold_stddev = stdev(naive_noJRAold_ubrmse_cold_master)
    					naive_all_ubrmse_cold_stddev = stdev(naive_all_ubrmse_cold_master)
    					CFSR_ubrmse_cold_stddev = stdev(CFSR_ubrmse_cold_master)
    					ERAI_ubrmse_cold_stddev = stdev(ERAI_ubrmse_cold_master)
    					ERA5_ubrmse_cold_stddev = stdev(ERA5_ubrmse_cold_master)
    					ERA5_Land_ubrmse_cold_stddev = stdev(ERA5_Land_ubrmse_cold_master)
    					JRA_ubrmse_cold_stddev = stdev(JRA_ubrmse_cold_master)
    					MERRA2_ubrmse_cold_stddev = stdev(MERRA2_ubrmse_cold_master)
    					GLDAS_ubrmse_cold_stddev = stdev(GLDAS_ubrmse_cold_master)
    					GLDAS_CLSM_ubrmse_cold_stddev = stdev(GLDAS_CLSM_ubrmse_cold_master)

    					naive_corr_cold_avg = mean(naive_corr_cold_master)
    					naive_noJRA_corr_cold_avg = mean(naive_noJRA_corr_cold_master)
    					naive_noJRAold_corr_cold_avg = mean(naive_noJRAold_corr_cold_master)
    					naive_all_corr_cold_avg = mean(naive_all_corr_cold_master)
    					CFSR_corr_cold_avg = mean(CFSR_corr_cold_master)
    					ERAI_corr_cold_avg = mean(ERAI_corr_cold_master)
    					ERA5_corr_cold_avg = mean(ERA5_corr_cold_master)
    					ERA5_Land_corr_cold_avg = mean(ERA5_Land_corr_cold_master)
    					JRA_corr_cold_avg = mean(JRA_corr_cold_master)
    					MERRA2_corr_cold_avg = mean(MERRA2_corr_cold_master)
    					GLDAS_corr_cold_avg = mean(GLDAS_corr_cold_master)
    					GLDAS_CLSM_corr_cold_avg = mean(GLDAS_CLSM_corr_cold_master)

    					naive_corr_cold_stddev = stdev(naive_corr_cold_master)
    					naive_noJRA_corr_cold_stddev = stdev(naive_noJRA_corr_cold_master)
    					naive_noJRAold_corr_cold_stddev = stdev(naive_noJRAold_corr_cold_master)
    					naive_all_corr_cold_stddev = stdev(naive_all_corr_cold_master)
    					CFSR_corr_cold_stddev = stdev(CFSR_corr_cold_master)
    					ERAI_corr_cold_stddev = stdev(ERAI_corr_cold_master)
    					ERA5_corr_cold_stddev = stdev(ERA5_corr_cold_master)
    					ERA5_Land_corr_cold_stddev = stdev(ERA5_Land_corr_cold_master)
    					JRA_corr_cold_stddev = stdev(JRA_corr_cold_master)
    					MERRA2_corr_cold_stddev = stdev(MERRA2_corr_cold_master)
    					GLDAS_corr_cold_stddev = stdev(GLDAS_corr_cold_master)
    					GLDAS_CLSM_corr_cold_stddev = stdev(GLDAS_CLSM_corr_cold_master)					



#### Warm Season ####


## Master Arrays ##

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


    					dframe_warm_season = dframe[dframe['Season'] == 'Warm']
    					gcell_warm = dframe_warm_season['Grid Cell'].values
    					gcell_warm_uq = np.unique(gcell_warm)

    					if (permafrost_type_o == 'RS_2002_permafrost'):
    						dframe_warm_season_permafrost = dframe_warm_season[(dframe_warm_season['RS 2002 Permafrost'] == 'continuous') | (dframe_warm_season['RS 2002 Permafrost'] == 'discontinuous')]

    					elif (permafrost_type_o == 'RS_2002_none'):
    						dframe_warm_season_permafrost = dframe_warm_season[dframe_warm_season['RS 2002 Permafrost'] == 'none']


    					elif (permafrost_type_o == 'RS_2002_all'):
    						dframe_warm_season_permafrost = dframe_warm_season

    					gcell_warm = dframe_warm_season_permafrost['Grid Cell'].values
    					gcell_warm_uq = np.unique(gcell_warm)

    					for q in gcell_warm_uq: # loop through grid cells
    						gcell_p = q
    						#print("Grid Cell:",q)
    						if (gcell_p == 33777):
    							continue
    						dframe_warm_season_gcell = dframe_warm_season_permafrost[dframe_warm_season_permafrost['Grid Cell'] == gcell_p]
    						print(dframe_warm_season_gcell)
    						if (len(dframe_warm_season_gcell) < 2):
    							continue    						
    						station_temp_warm = dframe_warm_season_gcell['Station'].values
    						naive_temp_warm = dframe_warm_season_gcell['Naive Blend'].values
    						naive_noJRA_temp_warm = dframe_warm_season_gcell['Naive Blend no JRA55'].values
    						naive_noJRAold_temp_warm = dframe_warm_season_gcell['Naive Blend no JRA55 Old'].values
    						naive_all_temp_warm = dframe_warm_season_gcell['Naive Blend All'].values
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

    						#print(ERA5_Land_var_warm)


## RMSE and ubRMSE ##
    						naive_rmse_warm = mean_squared_error(station_temp_warm,naive_temp_warm, squared=False)
    						naive_rmse_warm_master.append(naive_rmse_warm)
    						naive_noJRA_rmse_warm = mean_squared_error(station_temp_warm,naive_noJRA_temp_warm, squared=False)
    						naive_noJRA_rmse_warm_master.append(naive_noJRA_rmse_warm)
    						naive_noJRAold_rmse_warm = mean_squared_error(station_temp_warm,naive_noJRAold_temp_warm, squared=False)
    						naive_noJRAold_rmse_warm_master.append(naive_noJRAold_rmse_warm)
    						naive_all_rmse_warm = mean_squared_error(station_temp_warm,naive_all_temp_warm, squared=False)
    						naive_all_rmse_warm_master.append(naive_all_rmse_warm)
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


## Calculate Average Metrics Across Grid Cells ##

    					naive_bias_warm_avg = mean(naive_bias_warm_master)
    					naive_noJRA_bias_warm_avg = mean(naive_noJRA_bias_warm_master)
    					naive_noJRAold_bias_warm_avg = mean(naive_noJRAold_bias_warm_master)
    					naive_all_bias_warm_avg = mean(naive_all_bias_warm_master)
    					CFSR_bias_warm_avg = mean(CFSR_bias_warm_master)
    					ERAI_bias_warm_avg = mean(ERAI_bias_warm_master)
    					ERA5_bias_warm_avg = mean(ERA5_bias_warm_master)
    					ERA5_Land_bias_warm_avg = mean(ERA5_Land_bias_warm_master)
    					JRA_bias_warm_avg = mean(JRA_bias_warm_master)
    					MERRA2_bias_warm_avg = mean(MERRA2_bias_warm_master)
    					GLDAS_bias_warm_avg = mean(GLDAS_bias_warm_master)
    					GLDAS_CLSM_bias_warm_avg = mean(GLDAS_CLSM_bias_warm_master)

    					naive_bias_warm_stddev = stdev(naive_bias_warm_master)
    					naive_noJRA_bias_warm_stddev = stdev(naive_noJRA_bias_warm_master)
    					naive_noJRAold_bias_warm_stddev = stdev(naive_noJRAold_bias_warm_master)
    					naive_all_bias_warm_stddev = stdev(naive_all_bias_warm_master)
    					CFSR_bias_warm_stddev = stdev(CFSR_bias_warm_master)
    					ERAI_bias_warm_stddev = stdev(ERAI_bias_warm_master)
    					ERA5_bias_warm_stddev = stdev(ERA5_bias_warm_master)
    					ERA5_Land_bias_warm_stddev = stdev(ERA5_Land_bias_warm_master)
    					JRA_bias_warm_stddev = stdev(JRA_bias_warm_master)
    					MERRA2_bias_warm_stddev = stdev(MERRA2_bias_warm_master)
    					GLDAS_bias_warm_stddev = stdev(GLDAS_bias_warm_master)
    					GLDAS_CLSM_bias_warm_stddev = stdev(GLDAS_CLSM_bias_warm_master)

    					stn_var_warm_avg = mean(stn_var_warm_master)
    					naive_var_warm_avg = mean(naive_var_warm_master)
    					naive_noJRA_var_warm_avg = mean(naive_noJRA_var_warm_master)
    					naive_noJRAold_var_warm_avg = mean(naive_noJRAold_var_warm_master)
    					naive_all_var_warm_avg = mean(naive_all_var_warm_master)
    					CFSR_var_warm_avg = mean(CFSR_var_warm_master)
    					ERAI_var_warm_avg = mean(ERAI_var_warm_master)
    					ERA5_var_warm_avg = mean(ERA5_var_warm_master)
    					ERA5_Land_var_warm_avg = mean(ERA5_Land_var_warm_master)
    					JRA_var_warm_avg = mean(JRA_var_warm_master)
    					MERRA2_var_warm_avg = mean(MERRA2_var_warm_master)
    					GLDAS_var_warm_avg = mean(GLDAS_var_warm_master)
    					GLDAS_CLSM_var_warm_avg = mean(GLDAS_CLSM_var_warm_master)

    					print("Naive Var:",naive_all_var_warm_avg)
    					print("CFSR Var:",CFSR_var_warm_avg)
    					print("ERA-Interim Var:",ERAI_var_warm_avg)
    					print("ERA5 Var:",ERA5_var_warm_avg)
    					print("ERA5-Land Var:",ERA5_Land_var_warm_avg)
    					print("JRA-55 Var:",JRA_var_warm_avg)
    					print("MERRA2 Var:",MERRA2_var_warm_avg)
    					print("GLDAS Var:",GLDAS_var_warm_avg)
    					print("GDLAS-CLSM Var:",GLDAS_CLSM_var_warm_avg)

    					stn_sdev_warm_avg = math.sqrt(stn_var_warm_avg)
    					naive_sdev_warm_avg = math.sqrt(naive_var_warm_avg)
    					naive_noJRA_sdev_warm_avg = math.sqrt(naive_noJRA_var_warm_avg)
    					naive_noJRAold_sdev_warm_avg = math.sqrt(naive_noJRAold_var_warm_avg)
    					naive_all_sdev_warm_avg = math.sqrt(naive_all_var_warm_avg)
    					CFSR_sdev_warm_avg = math.sqrt(CFSR_var_warm_avg)
    					ERAI_sdev_warm_avg = math.sqrt(ERAI_var_warm_avg)
    					ERA5_sdev_warm_avg = math.sqrt(ERA5_var_warm_avg)
    					ERA5_Land_sdev_warm_avg = math.sqrt(ERA5_Land_var_warm_avg)
    					JRA_sdev_warm_avg = math.sqrt(JRA_var_warm_avg)
    					MERRA2_sdev_warm_avg = math.sqrt(MERRA2_var_warm_avg)
    					GLDAS_sdev_warm_avg = math.sqrt(GLDAS_var_warm_avg)
    					GLDAS_CLSM_sdev_warm_avg = math.sqrt(GLDAS_CLSM_var_warm_avg)

    					naive_SDV_warm_avg = naive_sdev_warm_avg/stn_sdev_warm_avg
    					naive_noJRA_SDV_warm_avg = naive_noJRA_sdev_warm_avg/stn_sdev_warm_avg
    					naive_noJRAold_SDV_warm_avg = naive_noJRAold_sdev_warm_avg/stn_sdev_warm_avg
    					naive_all_SDV_warm_avg = naive_all_sdev_warm_avg/stn_sdev_warm_avg
    					CFSR_SDV_warm_avg = CFSR_sdev_warm_avg/stn_sdev_warm_avg
    					ERAI_SDV_warm_avg = ERAI_sdev_warm_avg/stn_sdev_warm_avg
    					ERA5_SDV_warm_avg = ERA5_sdev_warm_avg/stn_sdev_warm_avg
    					ERA5_Land_SDV_warm_avg = ERA5_Land_sdev_warm_avg/stn_sdev_warm_avg
    					JRA_SDV_warm_avg = JRA_sdev_warm_avg/stn_sdev_warm_avg
    					MERRA2_SDV_warm_avg = MERRA2_sdev_warm_avg/stn_sdev_warm_avg
    					GLDAS_SDV_warm_avg = GLDAS_sdev_warm_avg/stn_sdev_warm_avg
    					GLDAS_CLSM_SDV_warm_avg = GLDAS_CLSM_sdev_warm_avg/stn_sdev_warm_avg

    					naive_rmse_warm_avg = mean(naive_rmse_warm_master)
    					naive_noJRA_rmse_warm_avg = mean(naive_noJRA_rmse_warm_master)
    					naive_noJRAold_rmse_warm_avg = mean(naive_noJRAold_rmse_warm_master)
    					naive_all_rmse_warm_avg = mean(naive_all_rmse_warm_master)
    					CFSR_rmse_warm_avg = mean(CFSR_rmse_warm_master)
    					ERAI_rmse_warm_avg = mean(ERAI_rmse_warm_master)
    					ERA5_rmse_warm_avg = mean(ERA5_rmse_warm_master)
    					ERA5_Land_rmse_warm_avg = mean(ERA5_Land_rmse_warm_master)
    					JRA_rmse_warm_avg = mean(JRA_rmse_warm_master)
    					MERRA2_rmse_warm_avg = mean(MERRA2_rmse_warm_master)
    					GLDAS_rmse_warm_avg = mean(GLDAS_rmse_warm_master)
    					GLDAS_CLSM_rmse_warm_avg = mean(GLDAS_CLSM_rmse_warm_master)

    					naive_rmse_warm_stddev = stdev(naive_rmse_warm_master)
    					naive_noJRA_rmse_warm_stddev = stdev(naive_noJRA_rmse_warm_master)
    					naive_noJRAold_rmse_warm_stddev = stdev(naive_noJRAold_rmse_warm_master)
    					naive_all_rmse_warm_stddev = stdev(naive_all_rmse_warm_master)
    					CFSR_rmse_warm_stddev = stdev(CFSR_rmse_warm_master)
    					ERAI_rmse_warm_stddev = stdev(ERAI_rmse_warm_master)
    					ERA5_rmse_warm_stddev = stdev(ERA5_rmse_warm_master)
    					ERA5_Land_rmse_warm_stddev = stdev(ERA5_Land_rmse_warm_master)
    					JRA_rmse_warm_stddev = stdev(JRA_rmse_warm_master)
    					MERRA2_rmse_warm_stddev = stdev(MERRA2_rmse_warm_master)
    					GLDAS_rmse_warm_stddev = stdev(GLDAS_rmse_warm_master)
    					GLDAS_CLSM_rmse_warm_stddev = stdev(GLDAS_CLSM_rmse_warm_master)

    					naive_ubrmse_warm_avg = mean(naive_ubrmse_warm_master)
    					naive_noJRA_ubrmse_warm_avg = mean(naive_noJRA_ubrmse_warm_master)
    					naive_noJRAold_ubrmse_warm_avg = mean(naive_noJRAold_ubrmse_warm_master)
    					naive_all_ubrmse_warm_avg = mean(naive_all_ubrmse_warm_master)
    					CFSR_ubrmse_warm_avg = mean(CFSR_ubrmse_warm_master)
    					ERAI_ubrmse_warm_avg = mean(ERAI_ubrmse_warm_master)
    					ERA5_ubrmse_warm_avg = mean(ERA5_ubrmse_warm_master)
    					ERA5_Land_ubrmse_warm_avg = mean(ERA5_Land_ubrmse_warm_master)
    					JRA_ubrmse_warm_avg = mean(JRA_ubrmse_warm_master)
    					MERRA2_ubrmse_warm_avg = mean(MERRA2_ubrmse_warm_master)
    					GLDAS_ubrmse_warm_avg = mean(GLDAS_ubrmse_warm_master)
    					GLDAS_CLSM_ubrmse_warm_avg = mean(GLDAS_CLSM_ubrmse_warm_master)

    					naive_ubrmse_warm_stddev = stdev(naive_ubrmse_warm_master)
    					naive_noJRA_ubrmse_warm_stddev = stdev(naive_noJRA_ubrmse_warm_master)
    					naive_noJRAold_ubrmse_warm_stddev = stdev(naive_noJRAold_ubrmse_warm_master)
    					naive_all_ubrmse_warm_stddev = stdev(naive_all_ubrmse_warm_master)
    					CFSR_ubrmse_warm_stddev = stdev(CFSR_ubrmse_warm_master)
    					ERAI_ubrmse_warm_stddev = stdev(ERAI_ubrmse_warm_master)
    					ERA5_ubrmse_warm_stddev = stdev(ERA5_ubrmse_warm_master)
    					ERA5_Land_ubrmse_warm_stddev = stdev(ERA5_Land_ubrmse_warm_master)
    					JRA_ubrmse_warm_stddev = stdev(JRA_ubrmse_warm_master)
    					MERRA2_ubrmse_warm_stddev = stdev(MERRA2_ubrmse_warm_master)
    					GLDAS_ubrmse_warm_stddev = stdev(GLDAS_ubrmse_warm_master)
    					GLDAS_CLSM_ubrmse_warm_stddev = stdev(GLDAS_CLSM_ubrmse_warm_master)

    					naive_corr_warm_avg = mean(naive_corr_warm_master)
    					naive_noJRA_corr_warm_avg = mean(naive_noJRA_corr_warm_master)
    					naive_noJRAold_corr_warm_avg = mean(naive_noJRAold_corr_warm_master)
    					naive_all_corr_warm_avg = mean(naive_all_corr_warm_master)
    					CFSR_corr_warm_avg = mean(CFSR_corr_warm_master)
    					ERAI_corr_warm_avg = mean(ERAI_corr_warm_master)
    					ERA5_corr_warm_avg = mean(ERA5_corr_warm_master)
    					ERA5_Land_corr_warm_avg = mean(ERA5_Land_corr_warm_master)
    					JRA_corr_warm_avg = mean(JRA_corr_warm_master)
    					MERRA2_corr_warm_avg = mean(MERRA2_corr_warm_master)
    					GLDAS_corr_warm_avg = mean(GLDAS_corr_warm_master)
    					GLDAS_CLSM_corr_warm_avg = mean(GLDAS_CLSM_corr_warm_master)

    					naive_corr_warm_stddev = stdev(naive_corr_warm_master)
    					naive_noJRA_corr_warm_stddev = stdev(naive_noJRA_corr_warm_master)
    					naive_noJRAold_corr_warm_stddev = stdev(naive_noJRAold_corr_warm_master)
    					naive_all_corr_warm_stddev = stdev(naive_all_corr_warm_master)
    					CFSR_corr_warm_stddev = stdev(CFSR_corr_warm_master)
    					ERAI_corr_warm_stddev = stdev(ERAI_corr_warm_master)
    					ERA5_corr_warm_stddev = stdev(ERA5_corr_warm_master)
    					ERA5_Land_corr_warm_stddev = stdev(ERA5_Land_corr_warm_master)
    					JRA_corr_warm_stddev = stdev(JRA_corr_warm_master)
    					MERRA2_corr_warm_stddev = stdev(MERRA2_corr_warm_master)
    					GLDAS_corr_warm_stddev = stdev(GLDAS_corr_warm_master)
    					GLDAS_CLSM_corr_warm_stddev = stdev(GLDAS_CLSM_corr_warm_master)







## Create Dataframe ##

    					dict_all = {"Bias Cold Season": pd.Series([naive_bias_cold_avg,naive_noJRAold_bias_cold_avg,naive_all_bias_cold_avg,naive_noJRA_bias_cold_avg,CFSR_bias_cold_avg,ERAI_bias_cold_avg,ERA5_bias_cold_avg,ERA5_Land_bias_cold_avg,JRA_bias_cold_avg,MERRA2_bias_cold_avg,GLDAS_bias_cold_avg,GLDAS_CLSM_bias_cold_avg], 
					index=["Naive Blend","Naive Blend no JRA55 Old","Naive Blend All","Naive Blend no JRA55","CFSR","ERA-Interim","ERA5","ERA5-Land","JRA55","MERRA2","GLDAS-Noah","GLDAS-CLSM"]),
					"Bias Warm Season": pd.Series([naive_bias_warm_avg,naive_noJRAold_bias_warm_avg,naive_all_bias_warm_avg,naive_noJRA_bias_warm_avg, CFSR_bias_warm_avg,ERAI_bias_warm_avg,ERA5_bias_warm_avg,ERA5_Land_bias_warm_avg,JRA_bias_warm_avg,MERRA2_bias_warm_avg,GLDAS_bias_warm_avg,GLDAS_CLSM_bias_warm_avg], 
					index=["Naive Blend","Naive Blend no JRA55 Old","Naive Blend All","Naive Blend no JRA55","CFSR","ERA-Interim","ERA5","ERA5-Land","JRA55","MERRA2","GLDAS-Noah","GLDAS-CLSM"]),
					"SDEV Cold Season":pd.Series([naive_sdev_cold_avg,naive_noJRAold_sdev_cold_avg,naive_all_sdev_cold_avg,naive_noJRA_sdev_cold_avg,CFSR_sdev_cold_avg,ERAI_sdev_cold_avg,ERA5_sdev_cold_avg,ERA5_Land_sdev_cold_avg,JRA_sdev_cold_avg,MERRA2_sdev_cold_avg,GLDAS_sdev_cold_avg,GLDAS_CLSM_sdev_cold_avg], 
					index=["Naive Blend","Naive Blend no JRA55 Old","Naive Blend All","Naive Blend no JRA55","CFSR","ERA-Interim","ERA5","ERA5-Land","JRA55","MERRA2","GLDAS-Noah","GLDAS-CLSM"]),
					"Norm SDV Cold Season": pd.Series([naive_SDV_cold_avg,naive_noJRAold_SDV_cold_avg,naive_all_SDV_cold_avg,naive_noJRA_SDV_cold_avg,CFSR_SDV_cold_avg,ERAI_SDV_cold_avg,ERA5_SDV_cold_avg,ERA5_Land_SDV_cold_avg,JRA_SDV_cold_avg,MERRA2_SDV_cold_avg,GLDAS_SDV_cold_avg,GLDAS_CLSM_SDV_cold_avg], 
					index=["Naive Blend","Naive Blend no JRA55 Old","Naive Blend All","Naive Blend no JRA55","CFSR","ERA-Interim","ERA5","ERA5-Land","JRA55","MERRA2","GLDAS-Noah","GLDAS-CLSM"]),
					"SDEV Warm Season":pd.Series([naive_sdev_warm_avg,naive_noJRAold_sdev_warm_avg,naive_all_sdev_warm_avg,naive_noJRA_sdev_warm_avg,CFSR_sdev_warm_avg,ERAI_sdev_warm_avg,ERA5_sdev_warm_avg,ERA5_Land_sdev_warm_avg,JRA_sdev_warm_avg,MERRA2_sdev_warm_avg,GLDAS_sdev_warm_avg,GLDAS_CLSM_sdev_warm_avg], 
					index=["Naive Blend","Naive Blend no JRA55 Old","Naive Blend All","Naive Blend no JRA55","CFSR","ERA-Interim","ERA5","ERA5-Land","JRA55","MERRA2","GLDAS-Noah","GLDAS-CLSM"]),
					"Norm SDV Warm Season": pd.Series([naive_SDV_warm_avg,naive_noJRAold_SDV_warm_avg,naive_all_SDV_warm_avg,naive_noJRA_SDV_warm_avg,CFSR_SDV_warm_avg,ERAI_SDV_warm_avg,ERA5_SDV_warm_avg,ERA5_Land_SDV_warm_avg,JRA_SDV_warm_avg,MERRA2_SDV_warm_avg,GLDAS_SDV_warm_avg,GLDAS_CLSM_SDV_warm_avg], 
					index=["Naive Blend","Naive Blend no JRA55 Old","Naive Blend All","Naive Blend no JRA55","CFSR","ERA-Interim","ERA5","ERA5-Land","JRA55","MERRA2","GLDAS-Noah","GLDAS-CLSM"]),
					"RMSE Cold Season": pd.Series([naive_rmse_cold_avg,naive_noJRAold_rmse_cold_avg,naive_all_rmse_cold_avg,naive_noJRA_rmse_cold_avg,CFSR_rmse_cold_avg,ERAI_rmse_cold_avg,ERA5_rmse_cold_avg,ERA5_Land_rmse_cold_avg,JRA_rmse_cold_avg,MERRA2_rmse_cold_avg,GLDAS_rmse_cold_avg,GLDAS_CLSM_rmse_cold_avg], 
					index=["Naive Blend","Naive Blend no JRA55 Old","Naive Blend All","Naive Blend no JRA55","CFSR","ERA-Interim","ERA5","ERA5-Land","JRA55","MERRA2","GLDAS-Noah","GLDAS-CLSM"]),
					"RMSE Warm Season": pd.Series([naive_rmse_warm_avg,naive_noJRAold_rmse_warm_avg,naive_all_rmse_warm_avg,naive_noJRA_rmse_warm_avg,CFSR_rmse_warm_avg,ERAI_rmse_warm_avg,ERA5_rmse_warm_avg,ERA5_Land_rmse_warm_avg,JRA_rmse_warm_avg,MERRA2_rmse_warm_avg,GLDAS_rmse_warm_avg,GLDAS_CLSM_rmse_warm_avg], 
					index=["Naive Blend","Naive Blend no JRA55 Old","Naive Blend All","Naive Blend no JRA55","CFSR","ERA-Interim","ERA5","ERA5-Land","JRA55","MERRA2","GLDAS-Noah","GLDAS-CLSM"]),
					"ubRMSE Cold Season": pd.Series([naive_ubrmse_cold_avg,naive_noJRAold_ubrmse_cold_avg,naive_all_ubrmse_cold_avg,naive_noJRA_ubrmse_cold_avg,CFSR_ubrmse_cold_avg,ERAI_ubrmse_cold_avg,ERA5_ubrmse_cold_avg,ERA5_Land_ubrmse_cold_avg,JRA_ubrmse_cold_avg,MERRA2_ubrmse_cold_avg,GLDAS_ubrmse_cold_avg,GLDAS_CLSM_ubrmse_cold_avg], 
					index=["Naive Blend","Naive Blend no JRA55 Old","Naive Blend All","Naive Blend no JRA55","CFSR","ERA-Interim","ERA5","ERA5-Land","JRA55","MERRA2","GLDAS-Noah","GLDAS-CLSM"]),
					"ubRMSE Warm Season": pd.Series([naive_ubrmse_warm_avg,naive_noJRAold_ubrmse_warm_avg,naive_all_ubrmse_warm_avg,naive_noJRA_ubrmse_warm_avg,CFSR_ubrmse_warm_avg,ERAI_ubrmse_warm_avg,ERA5_ubrmse_warm_avg,ERA5_Land_ubrmse_warm_avg,JRA_ubrmse_warm_avg,MERRA2_ubrmse_warm_avg,GLDAS_ubrmse_warm_avg,GLDAS_CLSM_ubrmse_warm_avg], 
					index=["Naive Blend","Naive Blend no JRA55 Old","Naive Blend All","Naive Blend no JRA55","CFSR","ERA-Interim","ERA5","ERA5-Land","JRA55","MERRA2","GLDAS-Noah","GLDAS-CLSM"]),
					"Pearson Correlation Cold Season": pd.Series([naive_corr_cold_avg,naive_noJRAold_corr_cold_avg,naive_all_corr_cold_avg,naive_noJRA_corr_cold_avg,CFSR_corr_cold_avg,ERAI_corr_cold_avg,ERA5_corr_cold_avg,ERA5_Land_corr_cold_avg,JRA_corr_cold_avg,MERRA2_corr_cold_avg,GLDAS_corr_cold_avg,GLDAS_CLSM_corr_cold_avg],
					index=["Naive Blend","Naive Blend no JRA55 Old","Naive Blend All","Naive Blend no JRA55","CFSR","ERA-Interim","ERA5","ERA5-Land","JRA55","MERRA2","GLDAS-Noah","GLDAS-CLSM"]),
					"Pearson Correlation Warm Season": pd.Series([naive_corr_warm_avg,naive_noJRAold_corr_warm_avg,naive_all_corr_warm_avg,naive_noJRA_corr_warm_avg,CFSR_corr_warm_avg,ERAI_corr_warm_avg,ERA5_corr_warm_avg,ERA5_Land_corr_warm_avg,JRA_corr_warm_avg,MERRA2_corr_warm_avg,GLDAS_corr_warm_avg,GLDAS_CLSM_corr_warm_avg],
					index=["Naive Blend","Naive Blend no JRA55 Old","Naive Blend All","Naive Blend no JRA55","CFSR","ERA-Interim","ERA5","ERA5-Land","JRA55","MERRA2","GLDAS-Noah","GLDAS-CLSM"]),
					"Bias Cold Season Standard Deviation": pd.Series([naive_bias_cold_stddev,naive_noJRAold_bias_cold_stddev,naive_all_bias_cold_stddev,naive_noJRA_bias_cold_stddev,CFSR_bias_cold_stddev,ERAI_bias_cold_stddev,ERA5_bias_cold_stddev,ERA5_Land_bias_cold_stddev,JRA_bias_cold_stddev,MERRA2_bias_cold_stddev,GLDAS_bias_cold_stddev,GLDAS_CLSM_bias_cold_stddev], 
					index=["Naive Blend","Naive Blend no JRA55 Old","Naive Blend All","Naive Blend no JRA55","CFSR","ERA-Interim","ERA5","ERA5-Land","JRA55","MERRA2","GLDAS-Noah","GLDAS-CLSM"]),
					"Bias Warm Season Standard Deviation": pd.Series([naive_bias_warm_stddev,naive_noJRAold_bias_warm_stddev,naive_all_bias_warm_stddev,naive_noJRA_bias_warm_stddev, CFSR_bias_warm_stddev,ERAI_bias_warm_stddev,ERA5_bias_warm_stddev,ERA5_Land_bias_warm_stddev,JRA_bias_warm_stddev,MERRA2_bias_warm_stddev,GLDAS_bias_warm_stddev,GLDAS_CLSM_bias_warm_stddev], 
					index=["Naive Blend","Naive Blend no JRA55 Old","Naive Blend All","Naive Blend no JRA55","CFSR","ERA-Interim","ERA5","ERA5-Land","JRA55","MERRA2","GLDAS-Noah","GLDAS-CLSM"]),
					"RMSE Cold Season Standard Deviation": pd.Series([naive_rmse_cold_stddev,naive_noJRAold_rmse_cold_stddev,naive_all_rmse_cold_stddev,naive_noJRA_rmse_cold_stddev,CFSR_rmse_cold_stddev,ERAI_rmse_cold_stddev,ERA5_rmse_cold_stddev,ERA5_Land_rmse_cold_stddev,JRA_rmse_cold_stddev,MERRA2_rmse_cold_stddev,GLDAS_rmse_cold_stddev,GLDAS_CLSM_rmse_cold_stddev], 
					index=["Naive Blend","Naive Blend no JRA55 Old","Naive Blend All","Naive Blend no JRA55","CFSR","ERA-Interim","ERA5","ERA5-Land","JRA55","MERRA2","GLDAS-Noah","GLDAS-CLSM"]),
					"RMSE Warm Season Standard Deviation": pd.Series([naive_rmse_warm_stddev,naive_noJRAold_rmse_warm_stddev,naive_all_rmse_warm_stddev,naive_noJRA_rmse_warm_stddev,CFSR_rmse_warm_stddev,ERAI_rmse_warm_stddev,ERA5_rmse_warm_stddev,ERA5_Land_rmse_warm_stddev,JRA_rmse_warm_stddev,MERRA2_rmse_warm_stddev,GLDAS_rmse_warm_stddev,GLDAS_CLSM_rmse_warm_stddev], 
					index=["Naive Blend","Naive Blend no JRA55 Old","Naive Blend All","Naive Blend no JRA55","CFSR","ERA-Interim","ERA5","ERA5-Land","JRA55","MERRA2","GLDAS-Noah","GLDAS-CLSM"]),
					"ubRMSE Cold Season Standard Deviation": pd.Series([naive_ubrmse_cold_stddev,naive_noJRAold_ubrmse_cold_stddev,naive_all_ubrmse_cold_stddev,naive_noJRA_ubrmse_cold_stddev,CFSR_ubrmse_cold_stddev,ERAI_ubrmse_cold_stddev,ERA5_ubrmse_cold_stddev,ERA5_Land_ubrmse_cold_stddev,JRA_ubrmse_cold_stddev,MERRA2_ubrmse_cold_stddev,GLDAS_ubrmse_cold_stddev,GLDAS_CLSM_ubrmse_cold_stddev], 
					index=["Naive Blend","Naive Blend no JRA55 Old","Naive Blend All","Naive Blend no JRA55","CFSR","ERA-Interim","ERA5","ERA5-Land","JRA55","MERRA2","GLDAS-Noah","GLDAS-CLSM"]),
					"ubRMSE Warm Season Standard Deviation": pd.Series([naive_ubrmse_warm_stddev,naive_noJRAold_ubrmse_warm_stddev,naive_all_ubrmse_warm_stddev,naive_noJRA_ubrmse_warm_stddev,CFSR_ubrmse_warm_stddev,ERAI_ubrmse_warm_stddev,ERA5_ubrmse_warm_stddev,ERA5_Land_ubrmse_warm_stddev,JRA_ubrmse_warm_stddev,MERRA2_ubrmse_warm_stddev,GLDAS_ubrmse_warm_stddev,GLDAS_CLSM_ubrmse_warm_stddev], 
					index=["Naive Blend","Naive Blend no JRA55 Old","Naive Blend All","Naive Blend no JRA55","CFSR","ERA-Interim","ERA5","ERA5-Land","JRA55","MERRA2","GLDAS-Noah","GLDAS-CLSM"]),
					"Pearson Correlation Cold Season Standard Deviation": pd.Series([naive_corr_cold_stddev,naive_noJRAold_corr_cold_stddev,naive_all_corr_cold_stddev,naive_noJRA_corr_cold_stddev,CFSR_corr_cold_stddev,ERAI_corr_cold_stddev,ERA5_corr_cold_stddev,ERA5_Land_corr_cold_stddev,JRA_corr_cold_stddev,MERRA2_corr_cold_stddev,GLDAS_corr_cold_stddev,GLDAS_CLSM_corr_cold_stddev],
					index=["Naive Blend","Naive Blend no JRA55 Old","Naive Blend All","Naive Blend no JRA55","CFSR","ERA-Interim","ERA5","ERA5-Land","JRA55","MERRA2","GLDAS-Noah","GLDAS-CLSM"]),
					"Pearson Correlation Warm Season Standard Deviation": pd.Series([naive_corr_warm_stddev,naive_noJRAold_corr_warm_stddev,naive_all_corr_warm_stddev,naive_noJRA_corr_warm_stddev,CFSR_corr_warm_stddev,ERAI_corr_warm_stddev,ERA5_corr_warm_stddev,ERA5_Land_corr_warm_stddev,JRA_corr_warm_stddev,MERRA2_corr_warm_stddev,GLDAS_corr_warm_stddev,GLDAS_CLSM_corr_warm_stddev],
					index=["Naive Blend","Naive Blend no JRA55 Old","Naive Blend All","Naive Blend no JRA55","CFSR","ERA-Interim","ERA5","ERA5-Land","JRA55","MERRA2","GLDAS-Noah","GLDAS-CLSM"])}
    					df_summary_all=pd.DataFrame(dict_all)

    					#print(df_summary_all)
    					metrics_all_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CMOS_poster/CLSM_res/'+str(remap_type)+'_'+str(lyr_l)+'_naive_metrics_CLSM_'+str(permafrost_type_o)+'_grid_avg_BEST_Sep2021_NWT_Yukon_sites_1yr.csv'])  					
    					df_summary_all.to_csv(metrics_all_fil)
    					print(metrics_all_fil)

