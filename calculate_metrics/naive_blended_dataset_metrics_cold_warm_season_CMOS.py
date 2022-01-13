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
temp_thr = ['-2C']#['0C','-2C','-5C','-10C']


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


    				for o in temp_thr:
    					temp_thr_o = o

    					if (temp_thr_o == "0C"):
    						tmp_val = 0

    					if (temp_thr_o == "-2C"):
    						tmp_val = -2

    					if (temp_thr_o == "-5C"):
    						tmp_val = -5

    					if (temp_thr_o == "-10C"):
    						tmp_val = -10

    					print("Remap Type:",remap_type)
    					print("Layer:",lyr_l)
    					print("Temp Threshold:", temp_thr_o)


###### Overall (across all validation grid cells) ######

#### Cold Season ####

    					cold_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/new_depth/'+str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_dframe_cold_season_temp_master_ERA5_'+str(temp_thr_o)+'_CMOS_newdepth.csv'])
    					dframe_cold_season = pd.read_csv(cold_fil)
    					gcell_cold = dframe_cold_season['Grid Cell'].values
    					gcell_cold_uq = np.unique(gcell_cold)





    					station_temp_cold = dframe_cold_season['Station'].values
    					naive_temp_cold = dframe_cold_season['Naive Blend'].values
    					naive_noJRA_temp_cold = dframe_cold_season['Naive Blend no JRA55'].values
    					naive_noJRAold_temp_cold = dframe_cold_season['Naive Blend no JRA55 Old'].values
    					naive_all_temp_cold = dframe_cold_season['Naive Blend All'].values
    					CFSR_temp_cold = dframe_cold_season['CFSR'].values
    					ERAI_temp_cold = dframe_cold_season['ERA-Interim'].values
    					ERA5_temp_cold = dframe_cold_season['ERA5'].values
    					ERA5_Land_temp_cold = dframe_cold_season['ERA5-Land'].values
    					JRA_temp_cold = dframe_cold_season['JRA55'].values
    					MERRA2_temp_cold = dframe_cold_season['MERRA2'].values
    					GLDAS_temp_cold = dframe_cold_season['GLDAS-Noah'].values
    					GLDAS_CLSM_temp_cold = dframe_cold_season['GLDAS-CLSM'].values

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

    					warm_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/new_depth/'+str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_dframe_warm_season_temp_master_ERA5_'+str(temp_thr_o)+'_CMOS_newdepth.csv'])
    					dframe_warm_season = pd.read_csv(warm_fil)
    					gcell_warm = dframe_warm_season['Grid Cell'].values
    					gcell_warm_uq = np.unique(gcell_warm)


    					station_temp_warm = dframe_warm_season['Station'].values
    					naive_temp_warm = dframe_warm_season['Naive Blend'].values
    					naive_noJRA_temp_warm = dframe_warm_season['Naive Blend no JRA55'].values
    					naive_noJRAold_temp_warm = dframe_warm_season['Naive Blend no JRA55 Old'].values
    					naive_all_temp_warm = dframe_warm_season['Naive Blend All'].values
    					CFSR_temp_warm = dframe_warm_season['CFSR'].values
    					ERAI_temp_warm = dframe_warm_season['ERA-Interim'].values
    					ERA5_temp_warm = dframe_warm_season['ERA5'].values
    					ERA5_Land_temp_warm = dframe_warm_season['ERA5-Land'].values
    					JRA_temp_warm = dframe_warm_season['JRA55'].values
    					MERRA2_temp_warm = dframe_warm_season['MERRA2'].values
    					GLDAS_temp_warm = dframe_warm_season['GLDAS-Noah'].values
    					GLDAS_CLSM_temp_warm = dframe_warm_season['GLDAS-CLSM'].values

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
    					metrics_all_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CMOS_poster/new_depth/'+str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_naive_blended_metrics_cold_warm_all_ERA5_'+str(temp_thr_o)+'_new_data_CMOS_newdepth.csv'])  					
    					df_summary_all.to_csv(metrics_all_fil)
    					print(metrics_all_fil)

####### Grid Cell Level ######
#
##### Cold Season ####
#    					gcell_master_cold = []
#    					sample_master_cold = []
#    					station_master_cold = []
#    					naive_master_cold = []
#    					CFSR_master_cold = []
#    					ERAI_master_cold = []
#    					ERA5_master_cold = []
#    					ERA5_Land_master_cold = []
#    					JRA_master_cold = []
#    					MERRA2_master_cold = []
#    					GLDAS_master_cold = []
#    					GLDAS_CLSM_master_cold = []
#
#    					naive_temp_master_gcell_cold = []
#    					CFSR_temp_master_gcell_cold = []
#    					ERAI_temp_master_gcell_cold = []
#    					ERA5_temp_master_gcell_cold = []
#    					ERA5_Land_temp_master_gcell_cold = []
#    					JRA_temp_master_gcell_cold = []
#    					MERRA2_temp_master_gcell_cold = []
#    					GLDAS_temp_master_gcell_cold = []
#    					GLDAS_CLSM_temp_master_gcell_cold = []
#
#    					naive_bias_master_gcell_cold = []
#    					CFSR_bias_master_gcell_cold = []
#    					ERAI_bias_master_gcell_cold = []
#    					ERA5_bias_master_gcell_cold = []
#    					ERA5_Land_bias_master_gcell_cold = []
#    					JRA_bias_master_gcell_cold = []
#    					MERRA2_bias_master_gcell_cold = []
#    					GLDAS_bias_master_gcell_cold = []
#    					GLDAS_CLSM_bias_master_gcell_cold = []
#
#    					naive_SDV_master_gcell_cold = []
#    					CFSR_SDV_master_gcell_cold = []
#    					ERAI_SDV_master_gcell_cold = []
#    					ERA5_SDV_master_gcell_cold = []
#    					ERA5_Land_SDV_master_gcell_cold = []
#    					JRA_SDV_master_gcell_cold = []
#    					MERRA2_SDV_master_gcell_cold = []
#    					GLDAS_SDV_master_gcell_cold = []
#    					GLDAS_CLSM_SDV_master_gcell_cold = []
#
#    					naive_rmse_master_gcell_cold = []
#    					CFSR_rmse_master_gcell_cold = []
#    					ERAI_rmse_master_gcell_cold = []
#    					ERA5_rmse_master_gcell_cold = []
#    					ERA5_Land_rmse_master_gcell_cold = []
#    					JRA_rmse_master_gcell_cold = []
#    					MERRA2_rmse_master_gcell_cold = []
#    					GLDAS_rmse_master_gcell_cold = []
#    					GLDAS_CLSM_rmse_master_gcell_cold = []
#
#    					naive_ubrmse_master_gcell_cold = []
#    					CFSR_ubrmse_master_gcell_cold = []
#    					ERAI_ubrmse_master_gcell_cold = []
#    					ERA5_ubrmse_master_gcell_cold = []
#    					ERA5_Land_ubrmse_master_gcell_cold = []
#    					JRA_ubrmse_master_gcell_cold = []
#    					MERRA2_ubrmse_master_gcell_cold = []
#    					GLDAS_ubrmse_master_gcell_cold = []
#    					GLDAS_CLSM_ubrmse_master_gcell_cold = []
#
#    					naive_corr_master_gcell_cold = []
#    					CFSR_corr_master_gcell_cold = []
#    					ERAI_corr_master_gcell_cold = []
#    					ERA5_corr_master_gcell_cold = []
#    					ERA5_Land_corr_master_gcell_cold = []
#    					JRA_corr_master_gcell_cold = []
#    					MERRA2_corr_master_gcell_cold = []
#    					GLDAS_CLSM_corr_master_gcell_cold = []
#
#										
#    					for p in gcell_cold_uq:
#    						gcell_p = p
#    						dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == gcell_p]
#    						gcell_p2 = dframe_cold_season_gcell['Grid Cell']  						
#    						sample_size_cold = dframe_cold_season_gcell['N'].iloc[0]
#    						sample_cold = dframe_cold_season_gcell['N']
#    						if (sample_size_cold < 30):
#    							continue
#    						gcell_master_cold.append(gcell_p2)
#    						sample_master_cold.append(sample_cold)   						
#    						station_cold = dframe_cold_season_gcell['Station'].values.tolist()
#    						len_station = len(station_cold)
#    						naive_cold = dframe_cold_season_gcell['Naive Blend'].values.tolist()
#    						CFSR_cold = dframe_cold_season_gcell['CFSR'].values.tolist()
#    						ERAI_cold = dframe_cold_season_gcell['ERA-Interim'].values.tolist()
#    						ERA5_cold = dframe_cold_season_gcell['ERA5'].values.tolist()					
#    						ERA5_Land_cold = dframe_cold_season_gcell['ERA5-Land'].values.tolist()					
#    						JRA_cold = dframe_cold_season_gcell['JRA55'].values.tolist()
#    						MERRA2_cold = dframe_cold_season_gcell['MERRA2'].values.tolist()
#    						GLDAS_cold = dframe_cold_season_gcell['GLDAS-Noah'].values.tolist()
#    						GLDAS_CLSM_cold = dframe_cold_season_gcell['GLDAS-CLSM'].values.tolist()
#
#
##    						station_master_cold.append(station_cold) 
##    						naive_master_cold.append(naive_cold) 
##    						CFSR_master_cold.append(station_cold)
##    						ERAI_master_cold.append(ERAI_cold)
##    						ERA5_master_cold.append(ERA5_cold)
##    						ERA5_Land_master_cold.append(ERA5_Land_cold)
##    						JRA_master_cold.append(JRA_cold)
##    						MERRA2_master_cold.append(MERRA2_cold)
##    						GLDAS_master_cold.append(GLDAS_cold)
##    						GLDAS_CLSM_master_cold.append(GLDAS_CLSM_cold)     						   
##
##
##    					gcell_master_cold = [i for sub in gcell_master_cold for i in sub]
##    					sample_master_cold = [i for sub in sample_master_cold for i in sub]
##    					station_master_cold = [i for sub in station_master_cold for i in sub]
##    					naive_master_cold = [i for sub in naive_master_cold for i in sub]    					
##    					CFSR_master_cold = [i for sub in CFSR_master_cold for i in sub]
##    					ERA5_master_cold = [i for sub in ERA5_master_cold for i in sub]
##    					ERA5_Land_master_cold = [i for sub in ERA5_Land_master_cold for i in sub]
##    					JRA_master_cold = [i for sub in JRA_master_cold for i in sub]
##    					MERRA2_master_cold = [i for sub in MERRA2_master_cold for i in sub]
##    					GLDAS_master_cold = [i for sub in GLDAS_master_cold for i in sub]
##    					GLDAS_CLSM_master_cold = [i for sub in GLDAS_CLSM_master_cold for i in sub]
##
##
##    					dframe_master_cold = pd.DataFrame(data=gcell_master_cold,columns=['Grid Cell'])
##    					dframe_master_cold['Sample Size'] = sample_master_cold
##    					dframe_master_cold['Station'] = station_master_cold
##    					dframe_master_cold['Naive Blend'] = naive_master_cold
##    					dframe_master_cold['CFSR'] = CFSR_master_cold
##    					dframe_master_cold['ERA5'] = ERA5_master_cold
##    					dframe_master_cold['ERA5-Land'] = ERA5_Land_master_cold
##    					dframe_master_cold['JRA55'] = JRA_master_cold										
##    					dframe_master_cold['MERRA2'] = MERRA2_master_cold
##    					dframe_master_cold['GLDAS'] = GLDAS_master_cold
##    					dframe_master_cold['GLDAS-CLSM'] = GLDAS_CLSM_master_cold
#
#
####### Warm Season ######
#
#    					naive_temp_master_gcell_warm = []
#    					CFSR_temp_master_gcell_warm = []
#    					ERAI_temp_master_gcell_warm = []
#    					ERA5_temp_master_gcell_warm = []
#    					ERA5_Land_temp_master_gcell_warm = []
#    					JRA_temp_master_gcell_warm = []
#    					MERRA2_temp_master_gcell_warm = []
#    					GLDAS_temp_master_gcell_warm = []
#    					GLDAS_CLSM_temp_master_gcell_warm = []
#
#    					naive_bias_master_gcell_warm = []
#    					CFSR_bias_master_gcell_warm = []
#    					ERAI_bias_master_gcell_warm = []
#    					ERA5_bias_master_gcell_warm = []
#    					ERA5_Land_bias_master_gcell_warm = []
#    					JRA_bias_master_gcell_warm = []
#    					MERRA2_bias_master_gcell_warm = []
#    					GLDAS_bias_master_gcell_warm = []
#    					GLDAS_CLSM_bias_master_gcell_warm = []
#
#    					naive_SDV_master_gcell_warm = []
#    					CFSR_SDV_master_gcell_warm = []
#    					ERAI_SDV_master_gcell_warm = []
#    					ERA5_SDV_master_gcell_warm = []
#    					ERA5_Land_SDV_master_gcell_warm = []
#    					JRA_SDV_master_gcell_warm = []
#    					MERRA2_SDV_master_gcell_warm = []
#    					GLDAS_SDV_master_gcell_warm = []
#    					GLDAS_CLSM_SDV_master_gcell_warm = []
#
#    					naive_rmse_master_gcell_warm = []
#    					CFSR_rmse_master_gcell_warm = []
#    					ERAI_rmse_master_gcell_warm = []
#    					ERA5_rmse_master_gcell_warm = []
#    					ERA5_Land_rmse_master_gcell_warm = []
#    					JRA_rmse_master_gcell_warm = []
#    					MERRA2_rmse_master_gcell_warm = []
#    					GLDAS_rmse_master_gcell_warm = []
#    					GLDAS_CLSM_rmse_master_gcell_warm = []
#
#    					naive_ubrmse_master_gcell_warm = []
#    					CFSR_ubrmse_master_gcell_warm = []
#    					ERAI_ubrmse_master_gcell_warm = []
#    					ERA5_ubrmse_master_gcell_warm = []
#    					ERA5_Land_ubrmse_master_gcell_warm = []
#    					JRA_ubrmse_master_gcell_warm = []
#    					MERRA2_ubrmse_master_gcell_warm = []
#    					GLDAS_ubrmse_master_gcell_warm = []
#    					GLDAS_CLSM_ubrmse_master_gcell_warm = []
#
#    					naive_corr_master_gcell_warm = []
#    					CFSR_corr_master_gcell_warm = []
#    					ERAI_corr_master_gcell_warm = []
#    					ERA5_corr_master_gcell_warm = []
#    					ERA5_Land_corr_master_gcell_warm = []
#    					JRA_corr_master_gcell_warm = []
#    					MERRA2_corr_master_gcell_warm = []
#    					GLDAS_CLSM_corr_master_gcell_warm = []
