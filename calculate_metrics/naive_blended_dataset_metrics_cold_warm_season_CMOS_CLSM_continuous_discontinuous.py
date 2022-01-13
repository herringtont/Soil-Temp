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
permafrost_type = ['RS_2002_continuous','Brown_1970_continuous','RS_2002_discontinuous','Brown_1970_discontinuous','RS_2002_none','Brown_1970_none']

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

    					cold_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/CLSM_res/subset/'+str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_dframe_cold_season_temp_master_ERA5_'+str(temp_thr_o)+'_CMOS_CLSM_subset_permafrost.csv'])
    					dframe_cold_season = pd.read_csv(cold_fil)
    					gcell_cold = dframe_cold_season['Grid Cell'].values
    					gcell_cold_uq = np.unique(gcell_cold)

    					if (permafrost_type_o == 'RS_2002_continuous'):
    						dframe_cold_season_permafrost = dframe_cold_season[dframe_cold_season['RS 2002 Permafrost'] == 'continuous']

    					elif (permafrost_type_o == 'Brown_1970_continuous'):
    						dframe_cold_season_permafrost = dframe_cold_season[dframe_cold_season['Brown 1970 Permafrost'] == 'continuous']

    					elif (permafrost_type_o == 'RS_2002_discontinuous'):
    						dframe_cold_season_permafrost = dframe_cold_season[dframe_cold_season['RS 2002 Permafrost'] == 'discontinuous']

    					elif (permafrost_type_o == 'Brown_1970_discontinuous'):
    						dframe_cold_season_permafrost = dframe_cold_season[dframe_cold_season['Brown 1970 Permafrost'] == 'discontinuous']

    					elif (permafrost_type_o == 'Brown_1970_none'):
    						dframe_cold_season_permafrost = dframe_cold_season[dframe_cold_season['Brown 1970 Permafrost'] == 'none']

    					elif (permafrost_type_o == 'RS_2002_none'):
    						dframe_cold_season_permafrost = dframe_cold_season[dframe_cold_season['RS 2002 Permafrost'] == 'none']
    					station_temp_cold = dframe_cold_season_permafrost['Station'].values
    					naive_temp_cold = dframe_cold_season_permafrost['Naive Blend'].values
    					naive_noJRA_temp_cold = dframe_cold_season_permafrost['Naive Blend no JRA55'].values
    					naive_noJRAold_temp_cold = dframe_cold_season_permafrost['Naive Blend no JRA55 Old'].values
    					naive_all_temp_cold = dframe_cold_season_permafrost['Naive Blend All'].values
    					CFSR_temp_cold = dframe_cold_season_permafrost['CFSR'].values
    					ERAI_temp_cold = dframe_cold_season_permafrost['ERA-Interim'].values
    					ERA5_temp_cold = dframe_cold_season_permafrost['ERA5'].values
    					ERA5_Land_temp_cold = dframe_cold_season_permafrost['ERA5-Land'].values
    					JRA_temp_cold = dframe_cold_season_permafrost['JRA55'].values
    					MERRA2_temp_cold = dframe_cold_season_permafrost['MERRA2'].values
    					GLDAS_temp_cold = dframe_cold_season_permafrost['GLDAS-Noah'].values
    					GLDAS_CLSM_temp_cold = dframe_cold_season_permafrost['GLDAS-CLSM'].values

## Bias ##
    					naive_bias_cold = bias(naive_temp_cold, station_temp_cold)
    					naive_noJRA_bias_cold = bias(naive_noJRA_temp_cold, station_temp_cold)
    					naive_noJRAold_bias_cold = bias(naive_noJRAold_temp_cold, station_temp_cold)
    					naive_all_bias_cold = bias(naive_all_temp_cold, station_temp_cold)
    					CFSR_bias_cold = bias(CFSR_temp_cold, station_temp_cold)
    					ERAI_bias_cold = bias(ERAI_temp_cold, station_temp_cold)
    					ERA5_bias_cold = bias(ERA5_temp_cold, station_temp_cold)
    					ERA5_Land_bias_cold = bias(ERA5_Land_temp_cold, station_temp_cold)
    					JRA_bias_cold = bias(JRA_temp_cold, station_temp_cold)
    					MERRA2_bias_cold = bias(MERRA2_temp_cold, station_temp_cold)
    					GLDAS_bias_cold = bias(GLDAS_temp_cold, station_temp_cold)
    					GLDAS_CLSM_bias_cold = bias(GLDAS_CLSM_temp_cold, station_temp_cold)

## STD DEV ##

    					stn_sdev_cold =  np.std(station_temp_cold)
    					naive_sdev_cold = np.std(naive_temp_cold)
    					naive_noJRA_sdev_cold = np.std(naive_noJRA_temp_cold)
    					naive_noJRAold_sdev_cold = np.std(naive_noJRAold_temp_cold)
    					naive_all_sdev_cold = np.std(naive_all_temp_cold)					
    					CFSR_sdev_cold = np.std(CFSR_temp_cold)
    					ERAI_sdev_cold = np.std(ERAI_temp_cold)    					
    					ERA5_sdev_cold = np.std(ERA5_temp_cold)
    					ERA5_Land_sdev_cold = np.std(ERA5_Land_temp_cold)
    					JRA_sdev_cold = np.std(JRA_temp_cold)
    					MERRA2_sdev_cold = np.std(MERRA2_temp_cold)
    					GLDAS_sdev_cold = np.std(GLDAS_temp_cold)
    					GLDAS_CLSM_sdev_cold = np.std(GLDAS_CLSM_temp_cold)



## Normalized Standard Deviations ##
    					naive_SDV_cold = SDVnorm(naive_temp_cold, station_temp_cold)
    					naive_noJRA_SDV_cold = SDVnorm(naive_noJRA_temp_cold, station_temp_cold)
    					naive_noJRAold_SDV_cold = SDVnorm(naive_noJRAold_temp_cold, station_temp_cold)
    					naive_all_SDV_cold = SDVnorm(naive_all_temp_cold, station_temp_cold)
    					CFSR_SDV_cold = SDVnorm(CFSR_temp_cold, station_temp_cold)
    					ERAI_SDV_cold = SDVnorm(ERAI_temp_cold, station_temp_cold)
    					ERA5_SDV_cold = SDVnorm(ERA5_temp_cold, station_temp_cold)
    					ERA5_Land_SDV_cold = SDVnorm(ERA5_Land_temp_cold, station_temp_cold)
    					JRA_SDV_cold = SDVnorm(JRA_temp_cold, station_temp_cold)
    					MERRA2_SDV_cold = SDVnorm(MERRA2_temp_cold, station_temp_cold)
    					GLDAS_SDV_cold = SDVnorm(GLDAS_temp_cold, station_temp_cold)
    					GLDAS_CLSM_SDV_cold = SDVnorm(GLDAS_CLSM_temp_cold, station_temp_cold)


## RMSE and ubRMSE ##
    					naive_rmse_cold = mean_squared_error(station_temp_cold,naive_temp_cold, squared=False)
    					naive_noJRA_rmse_cold = mean_squared_error(station_temp_cold,naive_noJRA_temp_cold, squared=False)
    					naive_noJRAold_rmse_cold = mean_squared_error(station_temp_cold,naive_noJRAold_temp_cold, squared=False)
    					naive_all_rmse_cold = mean_squared_error(station_temp_cold,naive_all_temp_cold, squared=False)
    					CFSR_rmse_cold = mean_squared_error(station_temp_cold,CFSR_temp_cold, squared=False)
    					ERAI_rmse_cold = mean_squared_error(station_temp_cold,ERAI_temp_cold, squared=False)
    					ERA5_rmse_cold = mean_squared_error(station_temp_cold,ERA5_temp_cold, squared=False)
    					ERA5_Land_rmse_cold = mean_squared_error(station_temp_cold,ERA5_Land_temp_cold, squared=False)
    					JRA_rmse_cold = mean_squared_error(station_temp_cold,JRA_temp_cold, squared=False)
    					MERRA2_rmse_cold = mean_squared_error(station_temp_cold,MERRA2_temp_cold, squared=False)
    					GLDAS_rmse_cold = mean_squared_error(station_temp_cold,GLDAS_temp_cold, squared=False)
    					GLDAS_CLSM_rmse_cold = mean_squared_error(station_temp_cold,GLDAS_CLSM_temp_cold, squared=False)

    					naive_ubrmse_cold = ubrmsd(station_temp_cold,naive_temp_cold)
    					naive_noJRA_ubrmse_cold = ubrmsd(station_temp_cold,naive_noJRA_temp_cold)
    					naive_noJRAold_ubrmse_cold = ubrmsd(station_temp_cold,naive_noJRAold_temp_cold)
    					naive_all_ubrmse_cold = ubrmsd(station_temp_cold,naive_all_temp_cold)
    					CFSR_ubrmse_cold = ubrmsd(station_temp_cold,CFSR_temp_cold)
    					ERAI_ubrmse_cold = ubrmsd(station_temp_cold,ERAI_temp_cold)
    					ERA5_ubrmse_cold = ubrmsd(station_temp_cold,ERA5_temp_cold)
    					ERA5_Land_ubrmse_cold = ubrmsd(station_temp_cold,ERA5_Land_temp_cold)
    					JRA_ubrmse_cold = ubrmsd(station_temp_cold,JRA_temp_cold)
    					MERRA2_ubrmse_cold = ubrmsd(station_temp_cold,MERRA2_temp_cold)
    					GLDAS_ubrmse_cold = ubrmsd(station_temp_cold,GLDAS_temp_cold)
    					GLDAS_CLSM_ubrmse_cold = ubrmsd(station_temp_cold,GLDAS_CLSM_temp_cold)


## Pearson Correlations ##
    					naive_corr_cold,_ = pearsonr(naive_temp_cold, station_temp_cold)
    					naive_noJRA_corr_cold,_ = pearsonr(naive_noJRA_temp_cold, station_temp_cold)
    					naive_noJRAold_corr_cold,_ = pearsonr(naive_noJRAold_temp_cold, station_temp_cold)
    					naive_all_corr_cold,_ = pearsonr(naive_all_temp_cold, station_temp_cold)
    					CFSR_corr_cold,_ = pearsonr(CFSR_temp_cold, station_temp_cold)
    					ERAI_corr_cold,_ = pearsonr(ERAI_temp_cold, station_temp_cold)
    					ERA5_corr_cold,_ = pearsonr(ERA5_temp_cold, station_temp_cold)
    					ERA5_Land_corr_cold,_ = pearsonr(ERA5_Land_temp_cold, station_temp_cold)
    					JRA_corr_cold,_ = pearsonr(JRA_temp_cold, station_temp_cold)
    					MERRA2_corr_cold,_ = pearsonr(MERRA2_temp_cold, station_temp_cold)
    					GLDAS_corr_cold,_ = pearsonr(GLDAS_temp_cold, station_temp_cold)
    					GLDAS_CLSM_corr_cold,_ = pearsonr(GLDAS_CLSM_temp_cold, station_temp_cold)


#### Warm Season ####

    					warm_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/CLSM_res/subset/'+str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_dframe_warm_season_temp_master_ERA5_'+str(temp_thr_o)+'_CMOS_CLSM_subset_permafrost.csv'])
    					dframe_warm_season = pd.read_csv(warm_fil)
    					gcell_warm = dframe_warm_season['Grid Cell'].values
    					gcell_warm_uq = np.unique(gcell_warm)

    					if (permafrost_type_o == 'RS_2002_continuous'):
    						dframe_warm_season_permafrost = dframe_warm_season[dframe_warm_season['RS 2002 Permafrost'] == 'continuous']

    					elif (permafrost_type_o == 'Brown_1970_continuous'):
    						dframe_warm_season_permafrost = dframe_warm_season[dframe_warm_season['Brown 1970 Permafrost'] == 'continuous']

    					elif (permafrost_type_o == 'RS_2002_discontinuous'):
    						dframe_warm_season_permafrost = dframe_warm_season[dframe_warm_season['RS 2002 Permafrost'] == 'discontinuous']

    					elif (permafrost_type_o == 'Brown_1970_discontinuous'):
    						dframe_warm_season_permafrost = dframe_warm_season[dframe_warm_season['Brown 1970 Permafrost'] == 'discontinuous']
    					elif (permafrost_type_o == 'Brown_1970_none'):
    						dframe_warm_season_permafrost = dframe_warm_season[dframe_warm_season['Brown 1970 Permafrost'] == 'none']

    					elif (permafrost_type_o == 'RS_2002_none'):
    						dframe_warm_season_permafrost = dframe_warm_season[dframe_warm_season['RS 2002 Permafrost'] == 'none']
    					station_temp_warm = dframe_warm_season_permafrost['Station'].values
    					naive_temp_warm = dframe_warm_season_permafrost['Naive Blend'].values
    					naive_noJRA_temp_warm = dframe_warm_season_permafrost['Naive Blend no JRA55'].values
    					naive_noJRAold_temp_warm = dframe_warm_season_permafrost['Naive Blend no JRA55 Old'].values
    					naive_all_temp_warm = dframe_warm_season_permafrost['Naive Blend All'].values
    					CFSR_temp_warm = dframe_warm_season_permafrost['CFSR'].values
    					ERAI_temp_warm = dframe_warm_season_permafrost['ERA-Interim'].values
    					ERA5_temp_warm = dframe_warm_season_permafrost['ERA5'].values
    					ERA5_Land_temp_warm = dframe_warm_season_permafrost['ERA5-Land'].values
    					JRA_temp_warm = dframe_warm_season_permafrost['JRA55'].values
    					MERRA2_temp_warm = dframe_warm_season_permafrost['MERRA2'].values
    					GLDAS_temp_warm = dframe_warm_season_permafrost['GLDAS-Noah'].values
    					GLDAS_CLSM_temp_warm = dframe_warm_season_permafrost['GLDAS-CLSM'].values

## Bias ##
    					naive_bias_warm = bias(naive_temp_warm, station_temp_warm)
    					naive_noJRA_bias_warm = bias(naive_noJRA_temp_warm, station_temp_warm)
    					naive_noJRAold_bias_warm = bias(naive_noJRAold_temp_warm, station_temp_warm)
    					naive_all_bias_warm = bias(naive_all_temp_warm, station_temp_warm)
    					CFSR_bias_warm = bias(CFSR_temp_warm, station_temp_warm)
    					ERAI_bias_warm = bias(ERAI_temp_warm, station_temp_warm)
    					ERA5_bias_warm = bias(ERA5_temp_warm, station_temp_warm)
    					ERA5_Land_bias_warm = bias(ERA5_Land_temp_warm, station_temp_warm)
    					JRA_bias_warm = bias(JRA_temp_warm, station_temp_warm)
    					MERRA2_bias_warm = bias(MERRA2_temp_warm, station_temp_warm)
    					GLDAS_bias_warm = bias(GLDAS_temp_warm, station_temp_warm)
    					GLDAS_CLSM_bias_warm = bias(GLDAS_CLSM_temp_warm, station_temp_warm)

## STD DEV ##

    					stn_sdev_warm =  np.std(station_temp_warm)
    					naive_sdev_warm = np.std(naive_temp_warm)
    					naive_noJRA_sdev_warm = np.std(naive_noJRA_temp_warm)
    					naive_noJRAold_sdev_warm = np.std(naive_noJRAold_temp_warm)
    					naive_all_sdev_warm = np.std(naive_all_temp_warm)					
    					CFSR_sdev_warm = np.std(CFSR_temp_warm)
    					ERAI_sdev_warm = np.std(ERAI_temp_warm)    					
    					ERA5_sdev_warm = np.std(ERA5_temp_warm)
    					ERA5_Land_sdev_warm = np.std(ERA5_Land_temp_warm)
    					JRA_sdev_warm = np.std(JRA_temp_warm)
    					MERRA2_sdev_warm = np.std(MERRA2_temp_warm)
    					GLDAS_sdev_warm = np.std(GLDAS_temp_warm)
    					GLDAS_CLSM_sdev_warm = np.std(GLDAS_CLSM_temp_warm)



## Normalized Standard Deviations ##
    					naive_SDV_warm = SDVnorm(naive_temp_warm, station_temp_warm)
    					naive_noJRA_SDV_warm = SDVnorm(naive_noJRA_temp_warm, station_temp_warm)
    					naive_noJRAold_SDV_warm = SDVnorm(naive_noJRAold_temp_warm, station_temp_warm)
    					naive_all_SDV_warm = SDVnorm(naive_all_temp_warm, station_temp_warm)
    					CFSR_SDV_warm = SDVnorm(CFSR_temp_warm, station_temp_warm)
    					ERAI_SDV_warm = SDVnorm(ERAI_temp_warm, station_temp_warm)
    					ERA5_SDV_warm = SDVnorm(ERA5_temp_warm, station_temp_warm)
    					ERA5_Land_SDV_warm = SDVnorm(ERA5_Land_temp_warm, station_temp_warm)
    					JRA_SDV_warm = SDVnorm(JRA_temp_warm, station_temp_warm)
    					MERRA2_SDV_warm = SDVnorm(MERRA2_temp_warm, station_temp_warm)
    					GLDAS_SDV_warm = SDVnorm(GLDAS_temp_warm, station_temp_warm)
    					GLDAS_CLSM_SDV_warm = SDVnorm(GLDAS_CLSM_temp_warm, station_temp_warm)


## RMSE and ubRMSE ##
    					naive_rmse_warm = mean_squared_error(station_temp_warm,naive_temp_warm, squared=False)
    					naive_noJRA_rmse_warm = mean_squared_error(station_temp_warm,naive_noJRA_temp_warm, squared=False)
    					naive_noJRAold_rmse_warm = mean_squared_error(station_temp_warm,naive_noJRAold_temp_warm, squared=False)
    					naive_all_rmse_warm = mean_squared_error(station_temp_warm,naive_all_temp_warm, squared=False)
    					CFSR_rmse_warm = mean_squared_error(station_temp_warm,CFSR_temp_warm, squared=False)
    					ERAI_rmse_warm = mean_squared_error(station_temp_warm,ERAI_temp_warm, squared=False)
    					ERA5_rmse_warm = mean_squared_error(station_temp_warm,ERA5_temp_warm, squared=False)
    					ERA5_Land_rmse_warm = mean_squared_error(station_temp_warm,ERA5_Land_temp_warm, squared=False)
    					JRA_rmse_warm = mean_squared_error(station_temp_warm,JRA_temp_warm, squared=False)
    					MERRA2_rmse_warm = mean_squared_error(station_temp_warm,MERRA2_temp_warm, squared=False)
    					GLDAS_rmse_warm = mean_squared_error(station_temp_warm,GLDAS_temp_warm, squared=False)
    					GLDAS_CLSM_rmse_warm = mean_squared_error(station_temp_warm,GLDAS_CLSM_temp_warm, squared=False)

    					naive_ubrmse_warm = ubrmsd(station_temp_warm,naive_temp_warm)
    					naive_noJRA_ubrmse_warm = ubrmsd(station_temp_warm,naive_noJRA_temp_warm)
    					naive_noJRAold_ubrmse_warm = ubrmsd(station_temp_warm,naive_noJRAold_temp_warm)
    					naive_all_ubrmse_warm = ubrmsd(station_temp_warm,naive_all_temp_warm)
    					CFSR_ubrmse_warm = ubrmsd(station_temp_warm,CFSR_temp_warm)
    					ERAI_ubrmse_warm = ubrmsd(station_temp_warm,ERAI_temp_warm)
    					ERA5_ubrmse_warm = ubrmsd(station_temp_warm,ERA5_temp_warm)
    					ERA5_Land_ubrmse_warm = ubrmsd(station_temp_warm,ERA5_Land_temp_warm)
    					JRA_ubrmse_warm = ubrmsd(station_temp_warm,JRA_temp_warm)
    					MERRA2_ubrmse_warm = ubrmsd(station_temp_warm,MERRA2_temp_warm)
    					GLDAS_ubrmse_warm = ubrmsd(station_temp_warm,GLDAS_temp_warm)
    					GLDAS_CLSM_ubrmse_warm = ubrmsd(station_temp_warm,GLDAS_CLSM_temp_warm)


## Pearson Correlations ##
    					naive_corr_warm,_ = pearsonr(naive_temp_warm, station_temp_warm)
    					naive_noJRA_corr_warm,_ = pearsonr(naive_noJRA_temp_warm, station_temp_warm)
    					naive_noJRAold_corr_warm,_ = pearsonr(naive_noJRAold_temp_warm, station_temp_warm)
    					naive_all_corr_warm,_ = pearsonr(naive_all_temp_warm, station_temp_warm)
    					CFSR_corr_warm,_ = pearsonr(CFSR_temp_warm, station_temp_warm)
    					ERAI_corr_warm,_ = pearsonr(ERAI_temp_warm, station_temp_warm)
    					ERA5_corr_warm,_ = pearsonr(ERA5_temp_warm, station_temp_warm)
    					ERA5_Land_corr_warm,_ = pearsonr(ERA5_Land_temp_warm, station_temp_warm)
    					JRA_corr_warm,_ = pearsonr(JRA_temp_warm, station_temp_warm)
    					MERRA2_corr_warm,_ = pearsonr(MERRA2_temp_warm, station_temp_warm)
    					GLDAS_corr_warm,_ = pearsonr(GLDAS_temp_warm, station_temp_warm)
    					GLDAS_CLSM_corr_warm,_ = pearsonr(GLDAS_CLSM_temp_warm, station_temp_warm)







## Create Dataframe ##

    					dict_all = {"Bias Cold Season": pd.Series([naive_bias_cold,naive_noJRAold_bias_cold,naive_all_bias_cold,naive_noJRA_bias_cold,CFSR_bias_cold,ERAI_bias_cold,ERA5_bias_cold,ERA5_Land_bias_cold,JRA_bias_cold,MERRA2_bias_cold,GLDAS_bias_cold,GLDAS_CLSM_bias_cold], 
					index=["Naive Blend","Naive Blend no JRA55 Old","Naive Blend All","Naive Blend no JRA55","CFSR","ERA-Interim","ERA5","ERA5-Land","JRA55","MERRA2","GLDAS-Noah","GLDAS-CLSM"]),
					"Bias Warm Season": pd.Series([naive_bias_warm,naive_noJRAold_bias_warm,naive_all_bias_warm,naive_noJRA_bias_warm, CFSR_bias_warm,ERAI_bias_warm,ERA5_bias_warm,ERA5_Land_bias_warm,JRA_bias_warm,MERRA2_bias_warm,GLDAS_bias_warm,GLDAS_CLSM_bias_warm], 
					index=["Naive Blend","Naive Blend no JRA55 Old","Naive Blend All","Naive Blend no JRA55","CFSR","ERA-Interim","ERA5","ERA5-Land","JRA55","MERRA2","GLDAS-Noah","GLDAS-CLSM"]),
					"SDEV Cold Season":pd.Series([naive_sdev_cold,naive_noJRAold_sdev_cold,naive_all_sdev_cold,naive_noJRA_sdev_cold,CFSR_sdev_cold,ERAI_sdev_cold,ERA5_sdev_cold,ERA5_Land_sdev_cold,JRA_sdev_cold,MERRA2_sdev_cold,GLDAS_sdev_cold,GLDAS_CLSM_sdev_cold], 
					index=["Naive Blend","Naive Blend no JRA55 Old","Naive Blend All","Naive Blend no JRA55","CFSR","ERA-Interim","ERA5","ERA5-Land","JRA55","MERRA2","GLDAS-Noah","GLDAS-CLSM"]),
					"Norm SDV Cold Season": pd.Series([naive_SDV_cold,naive_noJRAold_SDV_cold,naive_all_SDV_cold,naive_noJRA_SDV_cold,CFSR_SDV_cold,ERAI_SDV_cold,ERA5_SDV_cold,ERA5_Land_SDV_cold,JRA_SDV_cold,MERRA2_SDV_cold,GLDAS_SDV_cold,GLDAS_CLSM_SDV_cold], 
					index=["Naive Blend","Naive Blend no JRA55 Old","Naive Blend All","Naive Blend no JRA55","CFSR","ERA-Interim","ERA5","ERA5-Land","JRA55","MERRA2","GLDAS-Noah","GLDAS-CLSM"]),
					"SDEV Warm Season":pd.Series([naive_sdev_warm,naive_noJRAold_sdev_warm,naive_all_sdev_warm,naive_noJRA_sdev_warm,CFSR_sdev_warm,ERAI_sdev_warm,ERA5_sdev_warm,ERA5_Land_sdev_warm,JRA_sdev_warm,MERRA2_sdev_warm,GLDAS_sdev_warm,GLDAS_CLSM_sdev_warm], 
					index=["Naive Blend","Naive Blend no JRA55 Old","Naive Blend All","Naive Blend no JRA55","CFSR","ERA-Interim","ERA5","ERA5-Land","JRA55","MERRA2","GLDAS-Noah","GLDAS-CLSM"]),
					"Norm SDV Warm Season": pd.Series([naive_SDV_warm,naive_noJRAold_SDV_warm,naive_all_SDV_warm,naive_noJRA_SDV_warm,CFSR_SDV_warm,ERAI_SDV_warm,ERA5_SDV_warm,ERA5_Land_SDV_warm,JRA_SDV_warm,MERRA2_SDV_warm,GLDAS_SDV_warm,GLDAS_CLSM_SDV_warm], 
					index=["Naive Blend","Naive Blend no JRA55 Old","Naive Blend All","Naive Blend no JRA55","CFSR","ERA-Interim","ERA5","ERA5-Land","JRA55","MERRA2","GLDAS-Noah","GLDAS-CLSM"]),
					"RMSE Cold Season": pd.Series([naive_rmse_cold,naive_noJRAold_rmse_cold,naive_all_rmse_cold,naive_noJRA_rmse_cold,CFSR_rmse_cold,ERAI_rmse_cold,ERA5_rmse_cold,ERA5_Land_rmse_cold,JRA_rmse_cold,MERRA2_rmse_cold,GLDAS_rmse_cold,GLDAS_CLSM_rmse_cold], 
					index=["Naive Blend","Naive Blend no JRA55 Old","Naive Blend All","Naive Blend no JRA55","CFSR","ERA-Interim","ERA5","ERA5-Land","JRA55","MERRA2","GLDAS-Noah","GLDAS-CLSM"]),
					"RMSE Warm Season": pd.Series([naive_rmse_warm,naive_noJRAold_rmse_warm,naive_all_rmse_warm,naive_noJRA_rmse_warm,CFSR_rmse_warm,ERAI_rmse_warm,ERA5_rmse_warm,ERA5_Land_rmse_warm,JRA_rmse_warm,MERRA2_rmse_warm,GLDAS_rmse_warm,GLDAS_CLSM_rmse_warm], 
					index=["Naive Blend","Naive Blend no JRA55 Old","Naive Blend All","Naive Blend no JRA55","CFSR","ERA-Interim","ERA5","ERA5-Land","JRA55","MERRA2","GLDAS-Noah","GLDAS-CLSM"]),
					"ubRMSE Cold Season": pd.Series([naive_ubrmse_cold,naive_noJRAold_ubrmse_cold,naive_all_ubrmse_cold,naive_noJRA_ubrmse_cold,CFSR_ubrmse_cold,ERAI_ubrmse_cold,ERA5_ubrmse_cold,ERA5_Land_ubrmse_cold,JRA_ubrmse_cold,MERRA2_ubrmse_cold,GLDAS_ubrmse_cold,GLDAS_CLSM_ubrmse_cold], 
					index=["Naive Blend","Naive Blend no JRA55 Old","Naive Blend All","Naive Blend no JRA55","CFSR","ERA-Interim","ERA5","ERA5-Land","JRA55","MERRA2","GLDAS-Noah","GLDAS-CLSM"]),
					"ubRMSE Warm Season": pd.Series([naive_ubrmse_warm,naive_noJRAold_ubrmse_warm,naive_all_ubrmse_warm,naive_noJRA_ubrmse_warm,CFSR_ubrmse_warm,ERAI_ubrmse_warm,ERA5_ubrmse_warm,ERA5_Land_ubrmse_warm,JRA_ubrmse_warm,MERRA2_ubrmse_warm,GLDAS_ubrmse_warm,GLDAS_CLSM_ubrmse_warm], 
					index=["Naive Blend","Naive Blend no JRA55 Old","Naive Blend All","Naive Blend no JRA55","CFSR","ERA-Interim","ERA5","ERA5-Land","JRA55","MERRA2","GLDAS-Noah","GLDAS-CLSM"]),
					"Pearson Correlation Cold Season": pd.Series([naive_corr_cold,naive_noJRAold_corr_cold,naive_all_corr_cold,naive_noJRA_corr_cold,CFSR_corr_cold,ERAI_corr_cold,ERA5_corr_cold,ERA5_Land_corr_cold,JRA_corr_cold,MERRA2_corr_cold,GLDAS_corr_cold,GLDAS_CLSM_corr_cold],
					index=["Naive Blend","Naive Blend no JRA55 Old","Naive Blend All","Naive Blend no JRA55","CFSR","ERA-Interim","ERA5","ERA5-Land","JRA55","MERRA2","GLDAS-Noah","GLDAS-CLSM"]),
					"Pearson Correlation Warm Season": pd.Series([naive_corr_warm,naive_noJRAold_corr_warm,naive_all_corr_warm,naive_noJRA_corr_warm,CFSR_corr_warm,ERAI_corr_warm,ERA5_corr_warm,ERA5_Land_corr_warm,JRA_corr_warm,MERRA2_corr_warm,GLDAS_corr_warm,GLDAS_CLSM_corr_warm],
					index=["Naive Blend","Naive Blend no JRA55 Old","Naive Blend All","Naive Blend no JRA55","CFSR","ERA-Interim","ERA5","ERA5-Land","JRA55","MERRA2","GLDAS-Noah","GLDAS-CLSM"])}
    					df_summary_all=pd.DataFrame(dict_all)

    					#print(df_summary_all)
    					metrics_all_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CMOS_poster/CLSM_res/'+str(remap_type)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_naive_blended_metrics_CLSM_'+str(permafrost_type_o)+'.csv'])  					
    					df_summary_all.to_csv(metrics_all_fil)
    					print(metrics_all_fil)

