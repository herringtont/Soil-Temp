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
from scipy.stats import mannwhitneyu
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
#permafrost_type = ['RS_2002_permafrost', 'RS_2002_none', 'RS_2002_all']
permafrost_type = ['RS_2002_permafrost']
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

    					fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/CLSM_res/subset/'+str(remap_type)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_dframe_scatterplot_CMOS_CLSM_subset_permafrost_cold_warm_BEST_Sep2021_airtemp.csv'])
    					dframe = pd.read_csv(fil)

    					dframe_reg_NAm = dframe[dframe['Continent'] == 'North_America']
    					dframe_reg_Eur = dframe[dframe['Continent'] == 'Eurasia']



#### Cold Season ####

### North America ###


## Master Arrays ##

    					naive_bias_cold_master_NAm = []
    					naive_noJRA_bias_cold_master_NAm = []
    					naive_noJRAold_bias_cold_master_NAm = []
    					naive_all_bias_cold_master_NAm = []
    					CFSR_bias_cold_master_NAm = []
    					ERAI_bias_cold_master_NAm = []
    					ERA5_bias_cold_master_NAm = []
    					ERA5_Land_bias_cold_master_NAm = []
    					JRA_bias_cold_master_NAm = []
    					MERRA2_bias_cold_master_NAm = []
    					GLDAS_bias_cold_master_NAm = []
    					GLDAS_CLSM_bias_cold_master_NAm = []

    					stn_var_cold_master_NAm = []
    					naive_var_cold_master_NAm = []
    					naive_noJRA_var_cold_master_NAm = []
    					naive_noJRAold_var_cold_master_NAm = []
    					naive_all_var_cold_master_NAm = []
    					CFSR_var_cold_master_NAm = []
    					ERAI_var_cold_master_NAm = []
    					ERA5_var_cold_master_NAm = []
    					ERA5_Land_var_cold_master_NAm = []
    					JRA_var_cold_master_NAm = []
    					MERRA2_var_cold_master_NAm = []
    					GLDAS_var_cold_master_NAm = []
    					GLDAS_CLSM_var_cold_master_NAm = []

    					naive_rmse_cold_master_NAm = []
    					naive_noJRA_rmse_cold_master_NAm = []
    					naive_noJRAold_rmse_cold_master_NAm = []
    					naive_all_rmse_cold_master_NAm = []
    					CFSR_rmse_cold_master_NAm = []
    					ERAI_rmse_cold_master_NAm = []
    					ERA5_rmse_cold_master_NAm = []
    					ERA5_Land_rmse_cold_master_NAm = []
    					JRA_rmse_cold_master_NAm = []
    					MERRA2_rmse_cold_master_NAm = []
    					GLDAS_rmse_cold_master_NAm = []
    					GLDAS_CLSM_rmse_cold_master_NAm = []

    					naive_ubrmse_cold_master_NAm = []
    					naive_noJRA_ubrmse_cold_master_NAm = []
    					naive_noJRAold_ubrmse_cold_master_NAm = []
    					naive_all_ubrmse_cold_master_NAm = []
    					CFSR_ubrmse_cold_master_NAm = []
    					ERAI_ubrmse_cold_master_NAm = []
    					ERA5_ubrmse_cold_master_NAm = []
    					ERA5_Land_ubrmse_cold_master_NAm = []
    					JRA_ubrmse_cold_master_NAm = []
    					MERRA2_ubrmse_cold_master_NAm = []
    					GLDAS_ubrmse_cold_master_NAm = []
    					GLDAS_CLSM_ubrmse_cold_master_NAm = []

    					naive_corr_cold_master_NAm = []
    					naive_noJRA_corr_cold_master_NAm = []
    					naive_noJRAold_corr_cold_master_NAm = []
    					naive_all_corr_cold_master_NAm = []
    					CFSR_corr_cold_master_NAm = []
    					ERAI_corr_cold_master_NAm = []
    					ERA5_corr_cold_master_NAm = []
    					ERA5_Land_corr_cold_master_NAm = []
    					JRA_corr_cold_master_NAm = []
    					MERRA2_corr_cold_master_NAm = []
    					GLDAS_corr_cold_master_NAm = []
    					GLDAS_CLSM_corr_cold_master_NAm = []

## Grab Data ## 


    					dframe_cold_season = dframe_reg_NAm[dframe_reg_NAm['Season'] == 'Cold']

    					#print(dframe_cold_season)

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
    						naive_bias_cold_master_NAm.append(naive_bias_cold)
    						naive_noJRA_bias_cold = bias(naive_noJRA_temp_cold, station_temp_cold)
    						naive_noJRA_bias_cold_master_NAm.append(naive_noJRA_bias_cold)
    						naive_noJRAold_bias_cold = bias(naive_noJRAold_temp_cold, station_temp_cold)
    						naive_noJRAold_bias_cold_master_NAm.append(naive_noJRAold_bias_cold)
    						naive_all_bias_cold = bias(naive_all_temp_cold, station_temp_cold)
    						naive_all_bias_cold_master_NAm.append(naive_all_bias_cold)
    						CFSR_bias_cold = bias(CFSR_temp_cold, station_temp_cold)
    						CFSR_bias_cold_master_NAm.append(CFSR_bias_cold)
    						ERAI_bias_cold = bias(ERAI_temp_cold, station_temp_cold)
    						ERAI_bias_cold_master_NAm.append(ERAI_bias_cold)
    						ERA5_bias_cold = bias(ERA5_temp_cold, station_temp_cold)
    						ERA5_bias_cold_master_NAm.append(ERA5_bias_cold)
    						ERA5_Land_bias_cold = bias(ERA5_Land_temp_cold, station_temp_cold)
    						ERA5_Land_bias_cold_master_NAm.append(ERA5_Land_bias_cold)
    						JRA_bias_cold = bias(JRA_temp_cold, station_temp_cold)
    						JRA_bias_cold_master_NAm.append(JRA_bias_cold)
    						MERRA2_bias_cold = bias(MERRA2_temp_cold, station_temp_cold)
    						MERRA2_bias_cold_master_NAm.append(MERRA2_bias_cold)
    						GLDAS_bias_cold = bias(GLDAS_temp_cold, station_temp_cold)
    						GLDAS_bias_cold_master_NAm.append(GLDAS_bias_cold)
    						GLDAS_CLSM_bias_cold = bias(GLDAS_CLSM_temp_cold, station_temp_cold)
    						GLDAS_CLSM_bias_cold_master_NAm.append(GLDAS_CLSM_bias_cold)

## Variance ##

    						stn_var_cold =  np.var(station_temp_cold)
    						stn_var_cold_master_NAm.append(stn_var_cold)
    						naive_var_cold = np.var(naive_temp_cold)
    						naive_var_cold_master_NAm.append(naive_var_cold)
    						naive_noJRA_var_cold = np.var(naive_noJRA_temp_cold)
    						naive_noJRA_var_cold_master_NAm.append(naive_noJRA_var_cold)
    						naive_noJRAold_var_cold = np.var(naive_noJRAold_temp_cold)
    						naive_noJRAold_var_cold_master_NAm.append(naive_noJRAold_var_cold)
    						naive_all_var_cold = np.var(naive_all_temp_cold)
    						naive_all_var_cold_master_NAm.append(naive_all_var_cold)					
    						CFSR_var_cold = np.var(CFSR_temp_cold)
    						CFSR_var_cold_master_NAm.append(CFSR_var_cold)
    						ERAI_var_cold = np.var(ERAI_temp_cold)
    						ERAI_var_cold_master_NAm.append(ERAI_var_cold)    					
    						ERA5_var_cold = np.var(ERA5_temp_cold)
    						ERA5_var_cold_master_NAm.append(ERA5_var_cold)
    						ERA5_Land_var_cold = np.var(ERA5_Land_temp_cold)
    						ERA5_Land_var_cold_master_NAm.append(ERA5_Land_var_cold)
    						JRA_var_cold = np.var(JRA_temp_cold)
    						JRA_var_cold_master_NAm.append(JRA_var_cold)
    						MERRA2_var_cold = np.var(MERRA2_temp_cold)
    						MERRA2_var_cold_master_NAm.append(MERRA2_var_cold)
    						GLDAS_var_cold = np.var(GLDAS_temp_cold)
    						GLDAS_var_cold_master_NAm.append(GLDAS_var_cold)
    						GLDAS_CLSM_var_cold = np.var(GLDAS_CLSM_temp_cold)
    						GLDAS_CLSM_var_cold_master_NAm.append(GLDAS_CLSM_var_cold)



## RMSE and ubRMSE ##
    						naive_rmse_cold = mean_squared_error(station_temp_cold,naive_temp_cold, squared=False)
    						naive_rmse_cold_master_NAm.append(naive_rmse_cold)
    						naive_noJRA_rmse_cold = mean_squared_error(station_temp_cold,naive_noJRA_temp_cold, squared=False)
    						naive_noJRA_rmse_cold_master_NAm.append(naive_noJRA_rmse_cold)
    						naive_noJRAold_rmse_cold = mean_squared_error(station_temp_cold,naive_noJRAold_temp_cold, squared=False)
    						naive_noJRAold_rmse_cold_master_NAm.append(naive_noJRAold_rmse_cold)
    						naive_all_rmse_cold = mean_squared_error(station_temp_cold,naive_all_temp_cold, squared=False)
    						naive_all_rmse_cold_master_NAm.append(naive_all_rmse_cold)
    						CFSR_rmse_cold = mean_squared_error(station_temp_cold,CFSR_temp_cold, squared=False)
    						CFSR_rmse_cold_master_NAm.append(CFSR_rmse_cold)
    						ERAI_rmse_cold = mean_squared_error(station_temp_cold,ERAI_temp_cold, squared=False)
    						ERAI_rmse_cold_master_NAm.append(ERAI_rmse_cold)
    						ERA5_rmse_cold = mean_squared_error(station_temp_cold,ERA5_temp_cold, squared=False)
    						ERA5_rmse_cold_master_NAm.append(ERA5_rmse_cold)
    						ERA5_Land_rmse_cold = mean_squared_error(station_temp_cold,ERA5_Land_temp_cold, squared=False)
    						ERA5_Land_rmse_cold_master_NAm.append(ERA5_Land_rmse_cold)
    						JRA_rmse_cold = mean_squared_error(station_temp_cold,JRA_temp_cold, squared=False)
    						JRA_rmse_cold_master_NAm.append(JRA_rmse_cold)
    						MERRA2_rmse_cold = mean_squared_error(station_temp_cold,MERRA2_temp_cold, squared=False)
    						MERRA2_rmse_cold_master_NAm.append(MERRA2_rmse_cold)
    						GLDAS_rmse_cold = mean_squared_error(station_temp_cold,GLDAS_temp_cold, squared=False)
    						GLDAS_rmse_cold_master_NAm.append(GLDAS_rmse_cold)
    						GLDAS_CLSM_rmse_cold = mean_squared_error(station_temp_cold,GLDAS_CLSM_temp_cold, squared=False)
    						GLDAS_CLSM_rmse_cold_master_NAm.append(GLDAS_CLSM_rmse_cold)

    						naive_ubrmse_cold = ubrmsd(station_temp_cold,naive_temp_cold)
    						naive_ubrmse_cold_master_NAm.append(naive_ubrmse_cold)
    						naive_noJRA_ubrmse_cold = ubrmsd(station_temp_cold,naive_noJRA_temp_cold)
    						naive_noJRA_ubrmse_cold_master_NAm.append(naive_noJRA_ubrmse_cold)
    						naive_noJRAold_ubrmse_cold = ubrmsd(station_temp_cold,naive_noJRAold_temp_cold)
    						naive_noJRAold_ubrmse_cold_master_NAm.append(naive_noJRAold_ubrmse_cold)
    						naive_all_ubrmse_cold = ubrmsd(station_temp_cold,naive_all_temp_cold)
    						naive_all_ubrmse_cold_master_NAm.append(naive_all_ubrmse_cold)
    						CFSR_ubrmse_cold = ubrmsd(station_temp_cold,CFSR_temp_cold)
    						CFSR_ubrmse_cold_master_NAm.append(CFSR_ubrmse_cold)
    						ERAI_ubrmse_cold = ubrmsd(station_temp_cold,ERAI_temp_cold)
    						ERAI_ubrmse_cold_master_NAm.append(ERAI_ubrmse_cold)
    						ERA5_ubrmse_cold = ubrmsd(station_temp_cold,ERA5_temp_cold)
    						ERA5_ubrmse_cold_master_NAm.append(ERA5_ubrmse_cold)
    						ERA5_Land_ubrmse_cold = ubrmsd(station_temp_cold,ERA5_Land_temp_cold)
    						ERA5_Land_ubrmse_cold_master_NAm.append(ERA5_Land_ubrmse_cold)
    						JRA_ubrmse_cold = ubrmsd(station_temp_cold,JRA_temp_cold)
    						JRA_ubrmse_cold_master_NAm.append(JRA_ubrmse_cold)
    						MERRA2_ubrmse_cold = ubrmsd(station_temp_cold,MERRA2_temp_cold)
    						MERRA2_ubrmse_cold_master_NAm.append(MERRA2_ubrmse_cold)
    						GLDAS_ubrmse_cold = ubrmsd(station_temp_cold,GLDAS_temp_cold)
    						GLDAS_ubrmse_cold_master_NAm.append(GLDAS_ubrmse_cold)
    						GLDAS_CLSM_ubrmse_cold = ubrmsd(station_temp_cold,GLDAS_CLSM_temp_cold)
    						GLDAS_CLSM_ubrmse_cold_master_NAm.append(GLDAS_CLSM_ubrmse_cold)



## Pearson Correlations ##
    						naive_corr_cold,_ = pearsonr(naive_temp_cold, station_temp_cold)
    						naive_corr_cold_master_NAm.append(naive_corr_cold)
    						naive_noJRA_corr_cold,_ = pearsonr(naive_noJRA_temp_cold, station_temp_cold)
    						naive_noJRA_corr_cold_master_NAm.append(naive_noJRA_corr_cold)
    						naive_noJRAold_corr_cold,_ = pearsonr(naive_noJRAold_temp_cold, station_temp_cold)
    						naive_noJRAold_corr_cold_master_NAm.append(naive_noJRAold_corr_cold)
    						naive_all_corr_cold,_ = pearsonr(naive_all_temp_cold, station_temp_cold)
    						naive_all_corr_cold_master_NAm.append(naive_all_corr_cold)
    						CFSR_corr_cold,_ = pearsonr(CFSR_temp_cold, station_temp_cold)
    						CFSR_corr_cold_master_NAm.append(CFSR_corr_cold)
    						ERAI_corr_cold,_ = pearsonr(ERAI_temp_cold, station_temp_cold)
    						ERAI_corr_cold_master_NAm.append(ERAI_corr_cold)
    						ERA5_corr_cold,_ = pearsonr(ERA5_temp_cold, station_temp_cold)
    						ERA5_corr_cold_master_NAm.append(ERA5_corr_cold)
    						ERA5_Land_corr_cold,_ = pearsonr(ERA5_Land_temp_cold, station_temp_cold)
    						ERA5_Land_corr_cold_master_NAm.append(ERA5_Land_corr_cold)
    						JRA_corr_cold,_ = pearsonr(JRA_temp_cold, station_temp_cold)
    						JRA_corr_cold_master_NAm.append(JRA_corr_cold)
    						MERRA2_corr_cold,_ = pearsonr(MERRA2_temp_cold, station_temp_cold)
    						MERRA2_corr_cold_master_NAm.append(MERRA2_corr_cold)
    						GLDAS_corr_cold,_ = pearsonr(GLDAS_temp_cold, station_temp_cold)
    						GLDAS_corr_cold_master_NAm.append(GLDAS_corr_cold)
    						GLDAS_CLSM_corr_cold,_ = pearsonr(GLDAS_CLSM_temp_cold, station_temp_cold)
    						GLDAS_CLSM_corr_cold_master_NAm.append(GLDAS_CLSM_corr_cold)




### Eurasia ###



## Master Arrays ##

    					naive_bias_cold_master_Eur = []
    					naive_noJRA_bias_cold_master_Eur = []
    					naive_noJRAold_bias_cold_master_Eur = []
    					naive_all_bias_cold_master_Eur = []
    					CFSR_bias_cold_master_Eur = []
    					ERAI_bias_cold_master_Eur = []
    					ERA5_bias_cold_master_Eur = []
    					ERA5_Land_bias_cold_master_Eur = []
    					JRA_bias_cold_master_Eur = []
    					MERRA2_bias_cold_master_Eur = []
    					GLDAS_bias_cold_master_Eur = []
    					GLDAS_CLSM_bias_cold_master_Eur = []

    					stn_var_cold_master_Eur = []
    					naive_var_cold_master_Eur = []
    					naive_noJRA_var_cold_master_Eur = []
    					naive_noJRAold_var_cold_master_Eur = []
    					naive_all_var_cold_master_Eur = []
    					CFSR_var_cold_master_Eur = []
    					ERAI_var_cold_master_Eur = []
    					ERA5_var_cold_master_Eur = []
    					ERA5_Land_var_cold_master_Eur = []
    					JRA_var_cold_master_Eur = []
    					MERRA2_var_cold_master_Eur = []
    					GLDAS_var_cold_master_Eur = []
    					GLDAS_CLSM_var_cold_master_Eur = []

    					naive_rmse_cold_master_Eur = []
    					naive_noJRA_rmse_cold_master_Eur = []
    					naive_noJRAold_rmse_cold_master_Eur = []
    					naive_all_rmse_cold_master_Eur = []
    					CFSR_rmse_cold_master_Eur = []
    					ERAI_rmse_cold_master_Eur = []
    					ERA5_rmse_cold_master_Eur = []
    					ERA5_Land_rmse_cold_master_Eur = []
    					JRA_rmse_cold_master_Eur = []
    					MERRA2_rmse_cold_master_Eur = []
    					GLDAS_rmse_cold_master_Eur = []
    					GLDAS_CLSM_rmse_cold_master_Eur = []

    					naive_ubrmse_cold_master_Eur = []
    					naive_noJRA_ubrmse_cold_master_Eur = []
    					naive_noJRAold_ubrmse_cold_master_Eur = []
    					naive_all_ubrmse_cold_master_Eur = []
    					CFSR_ubrmse_cold_master_Eur = []
    					ERAI_ubrmse_cold_master_Eur = []
    					ERA5_ubrmse_cold_master_Eur = []
    					ERA5_Land_ubrmse_cold_master_Eur = []
    					JRA_ubrmse_cold_master_Eur = []
    					MERRA2_ubrmse_cold_master_Eur = []
    					GLDAS_ubrmse_cold_master_Eur = []
    					GLDAS_CLSM_ubrmse_cold_master_Eur = []

    					naive_corr_cold_master_Eur = []
    					naive_noJRA_corr_cold_master_Eur = []
    					naive_noJRAold_corr_cold_master_Eur = []
    					naive_all_corr_cold_master_Eur = []
    					CFSR_corr_cold_master_Eur = []
    					ERAI_corr_cold_master_Eur = []
    					ERA5_corr_cold_master_Eur = []
    					ERA5_Land_corr_cold_master_Eur = []
    					JRA_corr_cold_master_Eur = []
    					MERRA2_corr_cold_master_Eur = []
    					GLDAS_corr_cold_master_Eur = []
    					GLDAS_CLSM_corr_cold_master_Eur = []

## Grab Data ## 


    					dframe_cold_season = dframe_reg_Eur[dframe_reg_Eur['Season'] == 'Cold']

    					#print(dframe_cold_season)

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
    						naive_bias_cold_master_Eur.append(naive_bias_cold)
    						naive_noJRA_bias_cold = bias(naive_noJRA_temp_cold, station_temp_cold)
    						naive_noJRA_bias_cold_master_Eur.append(naive_noJRA_bias_cold)
    						naive_noJRAold_bias_cold = bias(naive_noJRAold_temp_cold, station_temp_cold)
    						naive_noJRAold_bias_cold_master_Eur.append(naive_noJRAold_bias_cold)
    						naive_all_bias_cold = bias(naive_all_temp_cold, station_temp_cold)
    						naive_all_bias_cold_master_Eur.append(naive_all_bias_cold)
    						CFSR_bias_cold = bias(CFSR_temp_cold, station_temp_cold)
    						CFSR_bias_cold_master_Eur.append(CFSR_bias_cold)
    						ERAI_bias_cold = bias(ERAI_temp_cold, station_temp_cold)
    						ERAI_bias_cold_master_Eur.append(ERAI_bias_cold)
    						ERA5_bias_cold = bias(ERA5_temp_cold, station_temp_cold)
    						ERA5_bias_cold_master_Eur.append(ERA5_bias_cold)
    						ERA5_Land_bias_cold = bias(ERA5_Land_temp_cold, station_temp_cold)
    						ERA5_Land_bias_cold_master_Eur.append(ERA5_Land_bias_cold)
    						JRA_bias_cold = bias(JRA_temp_cold, station_temp_cold)
    						JRA_bias_cold_master_Eur.append(JRA_bias_cold)
    						MERRA2_bias_cold = bias(MERRA2_temp_cold, station_temp_cold)
    						MERRA2_bias_cold_master_Eur.append(MERRA2_bias_cold)
    						GLDAS_bias_cold = bias(GLDAS_temp_cold, station_temp_cold)
    						GLDAS_bias_cold_master_Eur.append(GLDAS_bias_cold)
    						GLDAS_CLSM_bias_cold = bias(GLDAS_CLSM_temp_cold, station_temp_cold)
    						GLDAS_CLSM_bias_cold_master_Eur.append(GLDAS_CLSM_bias_cold)

## Variance ##

    						stn_var_cold =  np.var(station_temp_cold)
    						stn_var_cold_master_Eur.append(stn_var_cold)
    						naive_var_cold = np.var(naive_temp_cold)
    						naive_var_cold_master_Eur.append(naive_var_cold)
    						naive_noJRA_var_cold = np.var(naive_noJRA_temp_cold)
    						naive_noJRA_var_cold_master_Eur.append(naive_noJRA_var_cold)
    						naive_noJRAold_var_cold = np.var(naive_noJRAold_temp_cold)
    						naive_noJRAold_var_cold_master_Eur.append(naive_noJRAold_var_cold)
    						naive_all_var_cold = np.var(naive_all_temp_cold)
    						naive_all_var_cold_master_Eur.append(naive_all_var_cold)					
    						CFSR_var_cold = np.var(CFSR_temp_cold)
    						CFSR_var_cold_master_Eur.append(CFSR_var_cold)
    						ERAI_var_cold = np.var(ERAI_temp_cold)
    						ERAI_var_cold_master_Eur.append(ERAI_var_cold)    					
    						ERA5_var_cold = np.var(ERA5_temp_cold)
    						ERA5_var_cold_master_Eur.append(ERA5_var_cold)
    						ERA5_Land_var_cold = np.var(ERA5_Land_temp_cold)
    						ERA5_Land_var_cold_master_Eur.append(ERA5_Land_var_cold)
    						JRA_var_cold = np.var(JRA_temp_cold)
    						JRA_var_cold_master_Eur.append(JRA_var_cold)
    						MERRA2_var_cold = np.var(MERRA2_temp_cold)
    						MERRA2_var_cold_master_Eur.append(MERRA2_var_cold)
    						GLDAS_var_cold = np.var(GLDAS_temp_cold)
    						GLDAS_var_cold_master_Eur.append(GLDAS_var_cold)
    						GLDAS_CLSM_var_cold = np.var(GLDAS_CLSM_temp_cold)
    						GLDAS_CLSM_var_cold_master_Eur.append(GLDAS_CLSM_var_cold)



## RMSE and ubRMSE ##
    						naive_rmse_cold = mean_squared_error(station_temp_cold,naive_temp_cold, squared=False)
    						naive_rmse_cold_master_Eur.append(naive_rmse_cold)
    						naive_noJRA_rmse_cold = mean_squared_error(station_temp_cold,naive_noJRA_temp_cold, squared=False)
    						naive_noJRA_rmse_cold_master_Eur.append(naive_noJRA_rmse_cold)
    						naive_noJRAold_rmse_cold = mean_squared_error(station_temp_cold,naive_noJRAold_temp_cold, squared=False)
    						naive_noJRAold_rmse_cold_master_Eur.append(naive_noJRAold_rmse_cold)
    						naive_all_rmse_cold = mean_squared_error(station_temp_cold,naive_all_temp_cold, squared=False)
    						naive_all_rmse_cold_master_Eur.append(naive_all_rmse_cold)
    						CFSR_rmse_cold = mean_squared_error(station_temp_cold,CFSR_temp_cold, squared=False)
    						CFSR_rmse_cold_master_Eur.append(CFSR_rmse_cold)
    						ERAI_rmse_cold = mean_squared_error(station_temp_cold,ERAI_temp_cold, squared=False)
    						ERAI_rmse_cold_master_Eur.append(ERAI_rmse_cold)
    						ERA5_rmse_cold = mean_squared_error(station_temp_cold,ERA5_temp_cold, squared=False)
    						ERA5_rmse_cold_master_Eur.append(ERA5_rmse_cold)
    						ERA5_Land_rmse_cold = mean_squared_error(station_temp_cold,ERA5_Land_temp_cold, squared=False)
    						ERA5_Land_rmse_cold_master_Eur.append(ERA5_Land_rmse_cold)
    						JRA_rmse_cold = mean_squared_error(station_temp_cold,JRA_temp_cold, squared=False)
    						JRA_rmse_cold_master_Eur.append(JRA_rmse_cold)
    						MERRA2_rmse_cold = mean_squared_error(station_temp_cold,MERRA2_temp_cold, squared=False)
    						MERRA2_rmse_cold_master_Eur.append(MERRA2_rmse_cold)
    						GLDAS_rmse_cold = mean_squared_error(station_temp_cold,GLDAS_temp_cold, squared=False)
    						GLDAS_rmse_cold_master_Eur.append(GLDAS_rmse_cold)
    						GLDAS_CLSM_rmse_cold = mean_squared_error(station_temp_cold,GLDAS_CLSM_temp_cold, squared=False)
    						GLDAS_CLSM_rmse_cold_master_Eur.append(GLDAS_CLSM_rmse_cold)

    						naive_ubrmse_cold = ubrmsd(station_temp_cold,naive_temp_cold)
    						naive_ubrmse_cold_master_Eur.append(naive_ubrmse_cold)
    						naive_noJRA_ubrmse_cold = ubrmsd(station_temp_cold,naive_noJRA_temp_cold)
    						naive_noJRA_ubrmse_cold_master_Eur.append(naive_noJRA_ubrmse_cold)
    						naive_noJRAold_ubrmse_cold = ubrmsd(station_temp_cold,naive_noJRAold_temp_cold)
    						naive_noJRAold_ubrmse_cold_master_Eur.append(naive_noJRAold_ubrmse_cold)
    						naive_all_ubrmse_cold = ubrmsd(station_temp_cold,naive_all_temp_cold)
    						naive_all_ubrmse_cold_master_Eur.append(naive_all_ubrmse_cold)
    						CFSR_ubrmse_cold = ubrmsd(station_temp_cold,CFSR_temp_cold)
    						CFSR_ubrmse_cold_master_Eur.append(CFSR_ubrmse_cold)
    						ERAI_ubrmse_cold = ubrmsd(station_temp_cold,ERAI_temp_cold)
    						ERAI_ubrmse_cold_master_Eur.append(ERAI_ubrmse_cold)
    						ERA5_ubrmse_cold = ubrmsd(station_temp_cold,ERA5_temp_cold)
    						ERA5_ubrmse_cold_master_Eur.append(ERA5_ubrmse_cold)
    						ERA5_Land_ubrmse_cold = ubrmsd(station_temp_cold,ERA5_Land_temp_cold)
    						ERA5_Land_ubrmse_cold_master_Eur.append(ERA5_Land_ubrmse_cold)
    						JRA_ubrmse_cold = ubrmsd(station_temp_cold,JRA_temp_cold)
    						JRA_ubrmse_cold_master_Eur.append(JRA_ubrmse_cold)
    						MERRA2_ubrmse_cold = ubrmsd(station_temp_cold,MERRA2_temp_cold)
    						MERRA2_ubrmse_cold_master_Eur.append(MERRA2_ubrmse_cold)
    						GLDAS_ubrmse_cold = ubrmsd(station_temp_cold,GLDAS_temp_cold)
    						GLDAS_ubrmse_cold_master_Eur.append(GLDAS_ubrmse_cold)
    						GLDAS_CLSM_ubrmse_cold = ubrmsd(station_temp_cold,GLDAS_CLSM_temp_cold)
    						GLDAS_CLSM_ubrmse_cold_master_Eur.append(GLDAS_CLSM_ubrmse_cold)



## Pearson Correlations ##
    						naive_corr_cold,_ = pearsonr(naive_temp_cold, station_temp_cold)
    						naive_corr_cold_master_Eur.append(naive_corr_cold)
    						naive_noJRA_corr_cold,_ = pearsonr(naive_noJRA_temp_cold, station_temp_cold)
    						naive_noJRA_corr_cold_master_Eur.append(naive_noJRA_corr_cold)
    						naive_noJRAold_corr_cold,_ = pearsonr(naive_noJRAold_temp_cold, station_temp_cold)
    						naive_noJRAold_corr_cold_master_Eur.append(naive_noJRAold_corr_cold)
    						naive_all_corr_cold,_ = pearsonr(naive_all_temp_cold, station_temp_cold)
    						naive_all_corr_cold_master_Eur.append(naive_all_corr_cold)
    						CFSR_corr_cold,_ = pearsonr(CFSR_temp_cold, station_temp_cold)
    						CFSR_corr_cold_master_Eur.append(CFSR_corr_cold)
    						ERAI_corr_cold,_ = pearsonr(ERAI_temp_cold, station_temp_cold)
    						ERAI_corr_cold_master_Eur.append(ERAI_corr_cold)
    						ERA5_corr_cold,_ = pearsonr(ERA5_temp_cold, station_temp_cold)
    						ERA5_corr_cold_master_Eur.append(ERA5_corr_cold)
    						ERA5_Land_corr_cold,_ = pearsonr(ERA5_Land_temp_cold, station_temp_cold)
    						ERA5_Land_corr_cold_master_Eur.append(ERA5_Land_corr_cold)
    						JRA_corr_cold,_ = pearsonr(JRA_temp_cold, station_temp_cold)
    						JRA_corr_cold_master_Eur.append(JRA_corr_cold)
    						MERRA2_corr_cold,_ = pearsonr(MERRA2_temp_cold, station_temp_cold)
    						MERRA2_corr_cold_master_Eur.append(MERRA2_corr_cold)
    						GLDAS_corr_cold,_ = pearsonr(GLDAS_temp_cold, station_temp_cold)
    						GLDAS_corr_cold_master_Eur.append(GLDAS_corr_cold)
    						GLDAS_CLSM_corr_cold,_ = pearsonr(GLDAS_CLSM_temp_cold, station_temp_cold)
    						GLDAS_CLSM_corr_cold_master_Eur.append(GLDAS_CLSM_corr_cold)




#### Wilcoxon Rank Sum Tests (NAm vs Eur) ####


    					print("Cold Season:")

## Bias ##


# Ensemble Mean #
    					sample_NAm_ensmean = naive_all_bias_cold_master_NAm
    					sample_Eur_ensmean = naive_all_bias_cold_master_Eur

    					result = mannwhitneyu(sample_NAm_ensmean,sample_Eur_ensmean)
    					print('EnsMean result: ',result)
					
# CFSR #
    					sample_NAm_CFSR = CFSR_bias_cold_master_NAm
    					sample_Eur_CFSR = CFSR_bias_cold_master_Eur

    					result = mannwhitneyu(sample_NAm_CFSR,sample_Eur_CFSR)
    					print('CFSR result: ',result)


# ERAI #
    					sample_NAm_ERAI = ERAI_bias_cold_master_NAm
    					sample_Eur_ERAI = ERAI_bias_cold_master_Eur

    					result = mannwhitneyu(sample_NAm_ERAI,sample_Eur_ERAI)
    					print('ERAI result: ',result)


# ERA5 #
    					sample_NAm_ERA5 = ERA5_bias_cold_master_NAm
    					sample_Eur_ERA5 = ERA5_bias_cold_master_Eur

    					result = mannwhitneyu(sample_NAm_ERA5,sample_Eur_ERA5)
    					print('ERA5 result: ',result)

# ERA5-Land #
    					sample_NAm_ERA5_L = ERA5_Land_bias_cold_master_NAm
    					sample_Eur_ERA5_L = ERA5_Land_bias_cold_master_Eur

    					result = mannwhitneyu(sample_NAm_ERA5_L,sample_Eur_ERA5_L)
    					print('ERA5-Land result: ',result)

# JRA #
    					sample_NAm_JRA = JRA_bias_cold_master_NAm
    					sample_Eur_JRA = JRA_bias_cold_master_Eur

    					result = mannwhitneyu(sample_NAm_JRA,sample_Eur_JRA)
    					print('JRA result: ',result)


# MERRA2 #
    					sample_NAm_MERRA2 = MERRA2_bias_cold_master_NAm
    					sample_Eur_MERRA2 = MERRA2_bias_cold_master_Eur

    					result = mannwhitneyu(sample_NAm_MERRA2,sample_Eur_MERRA2)
    					print('MERRA2 result: ',result)



# GLDAS #
    					sample_NAm_GLDAS = GLDAS_bias_cold_master_NAm
    					sample_Eur_GLDAS = GLDAS_bias_cold_master_Eur

    					result = mannwhitneyu(sample_NAm_GLDAS,sample_Eur_GLDAS)
    					print('GLDAS result: ',result)


# GLDAS_CLSM #
    					sample_NAm_GLDAS_CLSM = GLDAS_CLSM_bias_cold_master_NAm
    					sample_Eur_GLDAS_CLSM = GLDAS_CLSM_bias_cold_master_Eur

    					result = mannwhitneyu(sample_NAm_GLDAS_CLSM,sample_Eur_GLDAS_CLSM)
    					print('GLDAS_CLSM result: ',result)


## RMSE ##


    					print('RMSE:')
# Ensemble Mean #
    					sample_NAm_ensmean = naive_all_rmse_cold_master_NAm
    					sample_Eur_ensmean = naive_all_rmse_cold_master_Eur

    					result = mannwhitneyu(sample_NAm_ensmean,sample_Eur_ensmean)
    					print('EnsMean result: ',result)
					
# CFSR #
    					sample_NAm_CFSR = CFSR_rmse_cold_master_NAm
    					sample_Eur_CFSR = CFSR_rmse_cold_master_Eur

    					result = mannwhitneyu(sample_NAm_CFSR,sample_Eur_CFSR)
    					print('CFSR result: ',result)


# ERAI #
    					sample_NAm_ERAI = ERAI_rmse_cold_master_NAm
    					sample_Eur_ERAI = ERAI_rmse_cold_master_Eur

    					result = mannwhitneyu(sample_NAm_ERAI,sample_Eur_ERAI)
    					print('ERAI result: ',result)


# ERA5 #
    					sample_NAm_ERA5 = ERA5_rmse_cold_master_NAm
    					sample_Eur_ERA5 = ERA5_rmse_cold_master_Eur

    					result = mannwhitneyu(sample_NAm_ERA5,sample_Eur_ERA5)
    					print('ERA5 result: ',result)

# ERA5-Land #
    					sample_NAm_ERA5_L = ERA5_Land_rmse_cold_master_NAm
    					sample_Eur_ERA5_L = ERA5_Land_rmse_cold_master_Eur

    					result = mannwhitneyu(sample_NAm_ERA5_L,sample_Eur_ERA5_L)
    					print('ERA5-Land result: ',result)


# JRA #
    					sample_NAm_JRA = JRA_rmse_cold_master_NAm
    					sample_Eur_JRA = JRA_rmse_cold_master_Eur

    					result = mannwhitneyu(sample_NAm_JRA,sample_Eur_JRA)
    					print('JRA result: ',result)


# MERRA2 #
    					sample_NAm_MERRA2 = MERRA2_rmse_cold_master_NAm
    					sample_Eur_MERRA2 = MERRA2_rmse_cold_master_Eur

    					result = mannwhitneyu(sample_NAm_MERRA2,sample_Eur_MERRA2)
    					print('MERRA2 result: ',result)


# GLDAS #
    					sample_NAm_GLDAS = GLDAS_rmse_cold_master_NAm
    					sample_Eur_GLDAS = GLDAS_rmse_cold_master_Eur

    					result = mannwhitneyu(sample_NAm_GLDAS,sample_Eur_GLDAS)
    					print('GLDAS result: ',result)


# GLDAS_CLSM #
    					sample_NAm_GLDAS_CLSM = GLDAS_CLSM_rmse_cold_master_NAm
    					sample_Eur_GLDAS_CLSM = GLDAS_CLSM_rmse_cold_master_Eur

    					result = mannwhitneyu(sample_NAm_GLDAS_CLSM,sample_Eur_GLDAS_CLSM)
    					print('GLDAS_CLSM result: ',result)


#### Warm Season ####

### North America ###


## Master Arrays ##

    					naive_bias_warm_master_NAm = []
    					naive_noJRA_bias_warm_master_NAm = []
    					naive_noJRAold_bias_warm_master_NAm = []
    					naive_all_bias_warm_master_NAm = []
    					CFSR_bias_warm_master_NAm = []
    					ERAI_bias_warm_master_NAm = []
    					ERA5_bias_warm_master_NAm = []
    					ERA5_Land_bias_warm_master_NAm = []
    					JRA_bias_warm_master_NAm = []
    					MERRA2_bias_warm_master_NAm = []
    					GLDAS_bias_warm_master_NAm = []
    					GLDAS_CLSM_bias_warm_master_NAm = []

    					stn_var_warm_master_NAm = []
    					naive_var_warm_master_NAm = []
    					naive_noJRA_var_warm_master_NAm = []
    					naive_noJRAold_var_warm_master_NAm = []
    					naive_all_var_warm_master_NAm = []
    					CFSR_var_warm_master_NAm = []
    					ERAI_var_warm_master_NAm = []
    					ERA5_var_warm_master_NAm = []
    					ERA5_Land_var_warm_master_NAm = []
    					JRA_var_warm_master_NAm = []
    					MERRA2_var_warm_master_NAm = []
    					GLDAS_var_warm_master_NAm = []
    					GLDAS_CLSM_var_warm_master_NAm = []

    					naive_rmse_warm_master_NAm = []
    					naive_noJRA_rmse_warm_master_NAm = []
    					naive_noJRAold_rmse_warm_master_NAm = []
    					naive_all_rmse_warm_master_NAm = []
    					CFSR_rmse_warm_master_NAm = []
    					ERAI_rmse_warm_master_NAm = []
    					ERA5_rmse_warm_master_NAm = []
    					ERA5_Land_rmse_warm_master_NAm = []
    					JRA_rmse_warm_master_NAm = []
    					MERRA2_rmse_warm_master_NAm = []
    					GLDAS_rmse_warm_master_NAm = []
    					GLDAS_CLSM_rmse_warm_master_NAm = []

    					naive_ubrmse_warm_master_NAm = []
    					naive_noJRA_ubrmse_warm_master_NAm = []
    					naive_noJRAold_ubrmse_warm_master_NAm = []
    					naive_all_ubrmse_warm_master_NAm = []
    					CFSR_ubrmse_warm_master_NAm = []
    					ERAI_ubrmse_warm_master_NAm = []
    					ERA5_ubrmse_warm_master_NAm = []
    					ERA5_Land_ubrmse_warm_master_NAm = []
    					JRA_ubrmse_warm_master_NAm = []
    					MERRA2_ubrmse_warm_master_NAm = []
    					GLDAS_ubrmse_warm_master_NAm = []
    					GLDAS_CLSM_ubrmse_warm_master_NAm = []

    					naive_corr_warm_master_NAm = []
    					naive_noJRA_corr_warm_master_NAm = []
    					naive_noJRAold_corr_warm_master_NAm = []
    					naive_all_corr_warm_master_NAm = []
    					CFSR_corr_warm_master_NAm = []
    					ERAI_corr_warm_master_NAm = []
    					ERA5_corr_warm_master_NAm = []
    					ERA5_Land_corr_warm_master_NAm = []
    					JRA_corr_warm_master_NAm = []
    					MERRA2_corr_warm_master_NAm = []
    					GLDAS_corr_warm_master_NAm = []
    					GLDAS_CLSM_corr_warm_master_NAm = []

## Grab Data ## 


    					dframe_warm_season = dframe_reg_NAm[dframe_reg_NAm['Season'] == 'Cold']

    					#print(dframe_warm_season)

    					if (permafrost_type_o == 'RS_2002_permafrost'):
    						dframe_warm_season_permafrost = dframe_warm_season[(dframe_warm_season['RS 2002 Permafrost'] == 'continuous') | (dframe_warm_season['RS 2002 Permafrost'] == 'discontinuous')]

    					elif (permafrost_type_o == 'RS_2002_none'):
    						dframe_warm_season_permafrost = dframe_warm_season[dframe_warm_season['RS 2002 Permafrost'] == 'none']


    					elif (permafrost_type_o == 'RS_2002_all'):
    						dframe_warm_season_permafrost = dframe_warm_season

    					gcell_warm = dframe_warm_season_permafrost['Grid Cell'].values
    					gcell_warm_uq = np.unique(gcell_warm)

    					for p in gcell_warm_uq: # loop through grid cells
    						gcell_p = p
    						if (gcell_p == 33777):
    							continue
    						dframe_warm_season_gcell = dframe_warm_season_permafrost[dframe_warm_season_permafrost['Grid Cell'] == gcell_p]
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
    						naive_bias_warm_master_NAm.append(naive_bias_warm)
    						naive_noJRA_bias_warm = bias(naive_noJRA_temp_warm, station_temp_warm)
    						naive_noJRA_bias_warm_master_NAm.append(naive_noJRA_bias_warm)
    						naive_noJRAold_bias_warm = bias(naive_noJRAold_temp_warm, station_temp_warm)
    						naive_noJRAold_bias_warm_master_NAm.append(naive_noJRAold_bias_warm)
    						naive_all_bias_warm = bias(naive_all_temp_warm, station_temp_warm)
    						naive_all_bias_warm_master_NAm.append(naive_all_bias_warm)
    						CFSR_bias_warm = bias(CFSR_temp_warm, station_temp_warm)
    						CFSR_bias_warm_master_NAm.append(CFSR_bias_warm)
    						ERAI_bias_warm = bias(ERAI_temp_warm, station_temp_warm)
    						ERAI_bias_warm_master_NAm.append(ERAI_bias_warm)
    						ERA5_bias_warm = bias(ERA5_temp_warm, station_temp_warm)
    						ERA5_bias_warm_master_NAm.append(ERA5_bias_warm)
    						ERA5_Land_bias_warm = bias(ERA5_Land_temp_warm, station_temp_warm)
    						ERA5_Land_bias_warm_master_NAm.append(ERA5_Land_bias_warm)
    						JRA_bias_warm = bias(JRA_temp_warm, station_temp_warm)
    						JRA_bias_warm_master_NAm.append(JRA_bias_warm)
    						MERRA2_bias_warm = bias(MERRA2_temp_warm, station_temp_warm)
    						MERRA2_bias_warm_master_NAm.append(MERRA2_bias_warm)
    						GLDAS_bias_warm = bias(GLDAS_temp_warm, station_temp_warm)
    						GLDAS_bias_warm_master_NAm.append(GLDAS_bias_warm)
    						GLDAS_CLSM_bias_warm = bias(GLDAS_CLSM_temp_warm, station_temp_warm)
    						GLDAS_CLSM_bias_warm_master_NAm.append(GLDAS_CLSM_bias_warm)

## Variance ##

    						stn_var_warm =  np.var(station_temp_warm)
    						stn_var_warm_master_NAm.append(stn_var_warm)
    						naive_var_warm = np.var(naive_temp_warm)
    						naive_var_warm_master_NAm.append(naive_var_warm)
    						naive_noJRA_var_warm = np.var(naive_noJRA_temp_warm)
    						naive_noJRA_var_warm_master_NAm.append(naive_noJRA_var_warm)
    						naive_noJRAold_var_warm = np.var(naive_noJRAold_temp_warm)
    						naive_noJRAold_var_warm_master_NAm.append(naive_noJRAold_var_warm)
    						naive_all_var_warm = np.var(naive_all_temp_warm)
    						naive_all_var_warm_master_NAm.append(naive_all_var_warm)					
    						CFSR_var_warm = np.var(CFSR_temp_warm)
    						CFSR_var_warm_master_NAm.append(CFSR_var_warm)
    						ERAI_var_warm = np.var(ERAI_temp_warm)
    						ERAI_var_warm_master_NAm.append(ERAI_var_warm)    					
    						ERA5_var_warm = np.var(ERA5_temp_warm)
    						ERA5_var_warm_master_NAm.append(ERA5_var_warm)
    						ERA5_Land_var_warm = np.var(ERA5_Land_temp_warm)
    						ERA5_Land_var_warm_master_NAm.append(ERA5_Land_var_warm)
    						JRA_var_warm = np.var(JRA_temp_warm)
    						JRA_var_warm_master_NAm.append(JRA_var_warm)
    						MERRA2_var_warm = np.var(MERRA2_temp_warm)
    						MERRA2_var_warm_master_NAm.append(MERRA2_var_warm)
    						GLDAS_var_warm = np.var(GLDAS_temp_warm)
    						GLDAS_var_warm_master_NAm.append(GLDAS_var_warm)
    						GLDAS_CLSM_var_warm = np.var(GLDAS_CLSM_temp_warm)
    						GLDAS_CLSM_var_warm_master_NAm.append(GLDAS_CLSM_var_warm)



## RMSE and ubRMSE ##
    						naive_rmse_warm = mean_squared_error(station_temp_warm,naive_temp_warm, squared=False)
    						naive_rmse_warm_master_NAm.append(naive_rmse_warm)
    						naive_noJRA_rmse_warm = mean_squared_error(station_temp_warm,naive_noJRA_temp_warm, squared=False)
    						naive_noJRA_rmse_warm_master_NAm.append(naive_noJRA_rmse_warm)
    						naive_noJRAold_rmse_warm = mean_squared_error(station_temp_warm,naive_noJRAold_temp_warm, squared=False)
    						naive_noJRAold_rmse_warm_master_NAm.append(naive_noJRAold_rmse_warm)
    						naive_all_rmse_warm = mean_squared_error(station_temp_warm,naive_all_temp_warm, squared=False)
    						naive_all_rmse_warm_master_NAm.append(naive_all_rmse_warm)
    						CFSR_rmse_warm = mean_squared_error(station_temp_warm,CFSR_temp_warm, squared=False)
    						CFSR_rmse_warm_master_NAm.append(CFSR_rmse_warm)
    						ERAI_rmse_warm = mean_squared_error(station_temp_warm,ERAI_temp_warm, squared=False)
    						ERAI_rmse_warm_master_NAm.append(ERAI_rmse_warm)
    						ERA5_rmse_warm = mean_squared_error(station_temp_warm,ERA5_temp_warm, squared=False)
    						ERA5_rmse_warm_master_NAm.append(ERA5_rmse_warm)
    						ERA5_Land_rmse_warm = mean_squared_error(station_temp_warm,ERA5_Land_temp_warm, squared=False)
    						ERA5_Land_rmse_warm_master_NAm.append(ERA5_Land_rmse_warm)
    						JRA_rmse_warm = mean_squared_error(station_temp_warm,JRA_temp_warm, squared=False)
    						JRA_rmse_warm_master_NAm.append(JRA_rmse_warm)
    						MERRA2_rmse_warm = mean_squared_error(station_temp_warm,MERRA2_temp_warm, squared=False)
    						MERRA2_rmse_warm_master_NAm.append(MERRA2_rmse_warm)
    						GLDAS_rmse_warm = mean_squared_error(station_temp_warm,GLDAS_temp_warm, squared=False)
    						GLDAS_rmse_warm_master_NAm.append(GLDAS_rmse_warm)
    						GLDAS_CLSM_rmse_warm = mean_squared_error(station_temp_warm,GLDAS_CLSM_temp_warm, squared=False)
    						GLDAS_CLSM_rmse_warm_master_NAm.append(GLDAS_CLSM_rmse_warm)

    						naive_ubrmse_warm = ubrmsd(station_temp_warm,naive_temp_warm)
    						naive_ubrmse_warm_master_NAm.append(naive_ubrmse_warm)
    						naive_noJRA_ubrmse_warm = ubrmsd(station_temp_warm,naive_noJRA_temp_warm)
    						naive_noJRA_ubrmse_warm_master_NAm.append(naive_noJRA_ubrmse_warm)
    						naive_noJRAold_ubrmse_warm = ubrmsd(station_temp_warm,naive_noJRAold_temp_warm)
    						naive_noJRAold_ubrmse_warm_master_NAm.append(naive_noJRAold_ubrmse_warm)
    						naive_all_ubrmse_warm = ubrmsd(station_temp_warm,naive_all_temp_warm)
    						naive_all_ubrmse_warm_master_NAm.append(naive_all_ubrmse_warm)
    						CFSR_ubrmse_warm = ubrmsd(station_temp_warm,CFSR_temp_warm)
    						CFSR_ubrmse_warm_master_NAm.append(CFSR_ubrmse_warm)
    						ERAI_ubrmse_warm = ubrmsd(station_temp_warm,ERAI_temp_warm)
    						ERAI_ubrmse_warm_master_NAm.append(ERAI_ubrmse_warm)
    						ERA5_ubrmse_warm = ubrmsd(station_temp_warm,ERA5_temp_warm)
    						ERA5_ubrmse_warm_master_NAm.append(ERA5_ubrmse_warm)
    						ERA5_Land_ubrmse_warm = ubrmsd(station_temp_warm,ERA5_Land_temp_warm)
    						ERA5_Land_ubrmse_warm_master_NAm.append(ERA5_Land_ubrmse_warm)
    						JRA_ubrmse_warm = ubrmsd(station_temp_warm,JRA_temp_warm)
    						JRA_ubrmse_warm_master_NAm.append(JRA_ubrmse_warm)
    						MERRA2_ubrmse_warm = ubrmsd(station_temp_warm,MERRA2_temp_warm)
    						MERRA2_ubrmse_warm_master_NAm.append(MERRA2_ubrmse_warm)
    						GLDAS_ubrmse_warm = ubrmsd(station_temp_warm,GLDAS_temp_warm)
    						GLDAS_ubrmse_warm_master_NAm.append(GLDAS_ubrmse_warm)
    						GLDAS_CLSM_ubrmse_warm = ubrmsd(station_temp_warm,GLDAS_CLSM_temp_warm)
    						GLDAS_CLSM_ubrmse_warm_master_NAm.append(GLDAS_CLSM_ubrmse_warm)



## Pearson Correlations ##
    						naive_corr_warm,_ = pearsonr(naive_temp_warm, station_temp_warm)
    						naive_corr_warm_master_NAm.append(naive_corr_warm)
    						naive_noJRA_corr_warm,_ = pearsonr(naive_noJRA_temp_warm, station_temp_warm)
    						naive_noJRA_corr_warm_master_NAm.append(naive_noJRA_corr_warm)
    						naive_noJRAold_corr_warm,_ = pearsonr(naive_noJRAold_temp_warm, station_temp_warm)
    						naive_noJRAold_corr_warm_master_NAm.append(naive_noJRAold_corr_warm)
    						naive_all_corr_warm,_ = pearsonr(naive_all_temp_warm, station_temp_warm)
    						naive_all_corr_warm_master_NAm.append(naive_all_corr_warm)
    						CFSR_corr_warm,_ = pearsonr(CFSR_temp_warm, station_temp_warm)
    						CFSR_corr_warm_master_NAm.append(CFSR_corr_warm)
    						ERAI_corr_warm,_ = pearsonr(ERAI_temp_warm, station_temp_warm)
    						ERAI_corr_warm_master_NAm.append(ERAI_corr_warm)
    						ERA5_corr_warm,_ = pearsonr(ERA5_temp_warm, station_temp_warm)
    						ERA5_corr_warm_master_NAm.append(ERA5_corr_warm)
    						ERA5_Land_corr_warm,_ = pearsonr(ERA5_Land_temp_warm, station_temp_warm)
    						ERA5_Land_corr_warm_master_NAm.append(ERA5_Land_corr_warm)
    						JRA_corr_warm,_ = pearsonr(JRA_temp_warm, station_temp_warm)
    						JRA_corr_warm_master_NAm.append(JRA_corr_warm)
    						MERRA2_corr_warm,_ = pearsonr(MERRA2_temp_warm, station_temp_warm)
    						MERRA2_corr_warm_master_NAm.append(MERRA2_corr_warm)
    						GLDAS_corr_warm,_ = pearsonr(GLDAS_temp_warm, station_temp_warm)
    						GLDAS_corr_warm_master_NAm.append(GLDAS_corr_warm)
    						GLDAS_CLSM_corr_warm,_ = pearsonr(GLDAS_CLSM_temp_warm, station_temp_warm)
    						GLDAS_CLSM_corr_warm_master_NAm.append(GLDAS_CLSM_corr_warm)




### Eurasia ###



## Master Arrays ##

    					naive_bias_warm_master_Eur = []
    					naive_noJRA_bias_warm_master_Eur = []
    					naive_noJRAold_bias_warm_master_Eur = []
    					naive_all_bias_warm_master_Eur = []
    					CFSR_bias_warm_master_Eur = []
    					ERAI_bias_warm_master_Eur = []
    					ERA5_bias_warm_master_Eur = []
    					ERA5_Land_bias_warm_master_Eur = []
    					JRA_bias_warm_master_Eur = []
    					MERRA2_bias_warm_master_Eur = []
    					GLDAS_bias_warm_master_Eur = []
    					GLDAS_CLSM_bias_warm_master_Eur = []

    					stn_var_warm_master_Eur = []
    					naive_var_warm_master_Eur = []
    					naive_noJRA_var_warm_master_Eur = []
    					naive_noJRAold_var_warm_master_Eur = []
    					naive_all_var_warm_master_Eur = []
    					CFSR_var_warm_master_Eur = []
    					ERAI_var_warm_master_Eur = []
    					ERA5_var_warm_master_Eur = []
    					ERA5_Land_var_warm_master_Eur = []
    					JRA_var_warm_master_Eur = []
    					MERRA2_var_warm_master_Eur = []
    					GLDAS_var_warm_master_Eur = []
    					GLDAS_CLSM_var_warm_master_Eur = []

    					naive_rmse_warm_master_Eur = []
    					naive_noJRA_rmse_warm_master_Eur = []
    					naive_noJRAold_rmse_warm_master_Eur = []
    					naive_all_rmse_warm_master_Eur = []
    					CFSR_rmse_warm_master_Eur = []
    					ERAI_rmse_warm_master_Eur = []
    					ERA5_rmse_warm_master_Eur = []
    					ERA5_Land_rmse_warm_master_Eur = []
    					JRA_rmse_warm_master_Eur = []
    					MERRA2_rmse_warm_master_Eur = []
    					GLDAS_rmse_warm_master_Eur = []
    					GLDAS_CLSM_rmse_warm_master_Eur = []

    					naive_ubrmse_warm_master_Eur = []
    					naive_noJRA_ubrmse_warm_master_Eur = []
    					naive_noJRAold_ubrmse_warm_master_Eur = []
    					naive_all_ubrmse_warm_master_Eur = []
    					CFSR_ubrmse_warm_master_Eur = []
    					ERAI_ubrmse_warm_master_Eur = []
    					ERA5_ubrmse_warm_master_Eur = []
    					ERA5_Land_ubrmse_warm_master_Eur = []
    					JRA_ubrmse_warm_master_Eur = []
    					MERRA2_ubrmse_warm_master_Eur = []
    					GLDAS_ubrmse_warm_master_Eur = []
    					GLDAS_CLSM_ubrmse_warm_master_Eur = []

    					naive_corr_warm_master_Eur = []
    					naive_noJRA_corr_warm_master_Eur = []
    					naive_noJRAold_corr_warm_master_Eur = []
    					naive_all_corr_warm_master_Eur = []
    					CFSR_corr_warm_master_Eur = []
    					ERAI_corr_warm_master_Eur = []
    					ERA5_corr_warm_master_Eur = []
    					ERA5_Land_corr_warm_master_Eur = []
    					JRA_corr_warm_master_Eur = []
    					MERRA2_corr_warm_master_Eur = []
    					GLDAS_corr_warm_master_Eur = []
    					GLDAS_CLSM_corr_warm_master_Eur = []

## Grab Data ## 


    					dframe_warm_season = dframe_reg_Eur[dframe_reg_Eur['Season'] == 'Warm']

    					#print(dframe_warm_season)

    					if (permafrost_type_o == 'RS_2002_permafrost'):
    						dframe_warm_season_permafrost = dframe_warm_season[(dframe_warm_season['RS 2002 Permafrost'] == 'continuous') | (dframe_warm_season['RS 2002 Permafrost'] == 'discontinuous')]

    					elif (permafrost_type_o == 'RS_2002_none'):
    						dframe_warm_season_permafrost = dframe_warm_season[dframe_warm_season['RS 2002 Permafrost'] == 'none']


    					elif (permafrost_type_o == 'RS_2002_all'):
    						dframe_warm_season_permafrost = dframe_warm_season

    					gcell_warm = dframe_warm_season_permafrost['Grid Cell'].values
    					gcell_warm_uq = np.unique(gcell_warm)

    					for p in gcell_warm_uq: # loop through grid cells
    						gcell_p = p
    						if (gcell_p == 33777):
    							continue
    						dframe_warm_season_gcell = dframe_warm_season_permafrost[dframe_warm_season_permafrost['Grid Cell'] == gcell_p]
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
    						naive_bias_warm_master_Eur.append(naive_bias_warm)
    						naive_noJRA_bias_warm = bias(naive_noJRA_temp_warm, station_temp_warm)
    						naive_noJRA_bias_warm_master_Eur.append(naive_noJRA_bias_warm)
    						naive_noJRAold_bias_warm = bias(naive_noJRAold_temp_warm, station_temp_warm)
    						naive_noJRAold_bias_warm_master_Eur.append(naive_noJRAold_bias_warm)
    						naive_all_bias_warm = bias(naive_all_temp_warm, station_temp_warm)
    						naive_all_bias_warm_master_Eur.append(naive_all_bias_warm)
    						CFSR_bias_warm = bias(CFSR_temp_warm, station_temp_warm)
    						CFSR_bias_warm_master_Eur.append(CFSR_bias_warm)
    						ERAI_bias_warm = bias(ERAI_temp_warm, station_temp_warm)
    						ERAI_bias_warm_master_Eur.append(ERAI_bias_warm)
    						ERA5_bias_warm = bias(ERA5_temp_warm, station_temp_warm)
    						ERA5_bias_warm_master_Eur.append(ERA5_bias_warm)
    						ERA5_Land_bias_warm = bias(ERA5_Land_temp_warm, station_temp_warm)
    						ERA5_Land_bias_warm_master_Eur.append(ERA5_Land_bias_warm)
    						JRA_bias_warm = bias(JRA_temp_warm, station_temp_warm)
    						JRA_bias_warm_master_Eur.append(JRA_bias_warm)
    						MERRA2_bias_warm = bias(MERRA2_temp_warm, station_temp_warm)
    						MERRA2_bias_warm_master_Eur.append(MERRA2_bias_warm)
    						GLDAS_bias_warm = bias(GLDAS_temp_warm, station_temp_warm)
    						GLDAS_bias_warm_master_Eur.append(GLDAS_bias_warm)
    						GLDAS_CLSM_bias_warm = bias(GLDAS_CLSM_temp_warm, station_temp_warm)
    						GLDAS_CLSM_bias_warm_master_Eur.append(GLDAS_CLSM_bias_warm)

## Variance ##

    						stn_var_warm =  np.var(station_temp_warm)
    						stn_var_warm_master_Eur.append(stn_var_warm)
    						naive_var_warm = np.var(naive_temp_warm)
    						naive_var_warm_master_Eur.append(naive_var_warm)
    						naive_noJRA_var_warm = np.var(naive_noJRA_temp_warm)
    						naive_noJRA_var_warm_master_Eur.append(naive_noJRA_var_warm)
    						naive_noJRAold_var_warm = np.var(naive_noJRAold_temp_warm)
    						naive_noJRAold_var_warm_master_Eur.append(naive_noJRAold_var_warm)
    						naive_all_var_warm = np.var(naive_all_temp_warm)
    						naive_all_var_warm_master_Eur.append(naive_all_var_warm)					
    						CFSR_var_warm = np.var(CFSR_temp_warm)
    						CFSR_var_warm_master_Eur.append(CFSR_var_warm)
    						ERAI_var_warm = np.var(ERAI_temp_warm)
    						ERAI_var_warm_master_Eur.append(ERAI_var_warm)    					
    						ERA5_var_warm = np.var(ERA5_temp_warm)
    						ERA5_var_warm_master_Eur.append(ERA5_var_warm)
    						ERA5_Land_var_warm = np.var(ERA5_Land_temp_warm)
    						ERA5_Land_var_warm_master_Eur.append(ERA5_Land_var_warm)
    						JRA_var_warm = np.var(JRA_temp_warm)
    						JRA_var_warm_master_Eur.append(JRA_var_warm)
    						MERRA2_var_warm = np.var(MERRA2_temp_warm)
    						MERRA2_var_warm_master_Eur.append(MERRA2_var_warm)
    						GLDAS_var_warm = np.var(GLDAS_temp_warm)
    						GLDAS_var_warm_master_Eur.append(GLDAS_var_warm)
    						GLDAS_CLSM_var_warm = np.var(GLDAS_CLSM_temp_warm)
    						GLDAS_CLSM_var_warm_master_Eur.append(GLDAS_CLSM_var_warm)



## RMSE and ubRMSE ##
    						naive_rmse_warm = mean_squared_error(station_temp_warm,naive_temp_warm, squared=False)
    						naive_rmse_warm_master_Eur.append(naive_rmse_warm)
    						naive_noJRA_rmse_warm = mean_squared_error(station_temp_warm,naive_noJRA_temp_warm, squared=False)
    						naive_noJRA_rmse_warm_master_Eur.append(naive_noJRA_rmse_warm)
    						naive_noJRAold_rmse_warm = mean_squared_error(station_temp_warm,naive_noJRAold_temp_warm, squared=False)
    						naive_noJRAold_rmse_warm_master_Eur.append(naive_noJRAold_rmse_warm)
    						naive_all_rmse_warm = mean_squared_error(station_temp_warm,naive_all_temp_warm, squared=False)
    						naive_all_rmse_warm_master_Eur.append(naive_all_rmse_warm)
    						CFSR_rmse_warm = mean_squared_error(station_temp_warm,CFSR_temp_warm, squared=False)
    						CFSR_rmse_warm_master_Eur.append(CFSR_rmse_warm)
    						ERAI_rmse_warm = mean_squared_error(station_temp_warm,ERAI_temp_warm, squared=False)
    						ERAI_rmse_warm_master_Eur.append(ERAI_rmse_warm)
    						ERA5_rmse_warm = mean_squared_error(station_temp_warm,ERA5_temp_warm, squared=False)
    						ERA5_rmse_warm_master_Eur.append(ERA5_rmse_warm)
    						ERA5_Land_rmse_warm = mean_squared_error(station_temp_warm,ERA5_Land_temp_warm, squared=False)
    						ERA5_Land_rmse_warm_master_Eur.append(ERA5_Land_rmse_warm)
    						JRA_rmse_warm = mean_squared_error(station_temp_warm,JRA_temp_warm, squared=False)
    						JRA_rmse_warm_master_Eur.append(JRA_rmse_warm)
    						MERRA2_rmse_warm = mean_squared_error(station_temp_warm,MERRA2_temp_warm, squared=False)
    						MERRA2_rmse_warm_master_Eur.append(MERRA2_rmse_warm)
    						GLDAS_rmse_warm = mean_squared_error(station_temp_warm,GLDAS_temp_warm, squared=False)
    						GLDAS_rmse_warm_master_Eur.append(GLDAS_rmse_warm)
    						GLDAS_CLSM_rmse_warm = mean_squared_error(station_temp_warm,GLDAS_CLSM_temp_warm, squared=False)
    						GLDAS_CLSM_rmse_warm_master_Eur.append(GLDAS_CLSM_rmse_warm)

    						naive_ubrmse_warm = ubrmsd(station_temp_warm,naive_temp_warm)
    						naive_ubrmse_warm_master_Eur.append(naive_ubrmse_warm)
    						naive_noJRA_ubrmse_warm = ubrmsd(station_temp_warm,naive_noJRA_temp_warm)
    						naive_noJRA_ubrmse_warm_master_Eur.append(naive_noJRA_ubrmse_warm)
    						naive_noJRAold_ubrmse_warm = ubrmsd(station_temp_warm,naive_noJRAold_temp_warm)
    						naive_noJRAold_ubrmse_warm_master_Eur.append(naive_noJRAold_ubrmse_warm)
    						naive_all_ubrmse_warm = ubrmsd(station_temp_warm,naive_all_temp_warm)
    						naive_all_ubrmse_warm_master_Eur.append(naive_all_ubrmse_warm)
    						CFSR_ubrmse_warm = ubrmsd(station_temp_warm,CFSR_temp_warm)
    						CFSR_ubrmse_warm_master_Eur.append(CFSR_ubrmse_warm)
    						ERAI_ubrmse_warm = ubrmsd(station_temp_warm,ERAI_temp_warm)
    						ERAI_ubrmse_warm_master_Eur.append(ERAI_ubrmse_warm)
    						ERA5_ubrmse_warm = ubrmsd(station_temp_warm,ERA5_temp_warm)
    						ERA5_ubrmse_warm_master_Eur.append(ERA5_ubrmse_warm)
    						ERA5_Land_ubrmse_warm = ubrmsd(station_temp_warm,ERA5_Land_temp_warm)
    						ERA5_Land_ubrmse_warm_master_Eur.append(ERA5_Land_ubrmse_warm)
    						JRA_ubrmse_warm = ubrmsd(station_temp_warm,JRA_temp_warm)
    						JRA_ubrmse_warm_master_Eur.append(JRA_ubrmse_warm)
    						MERRA2_ubrmse_warm = ubrmsd(station_temp_warm,MERRA2_temp_warm)
    						MERRA2_ubrmse_warm_master_Eur.append(MERRA2_ubrmse_warm)
    						GLDAS_ubrmse_warm = ubrmsd(station_temp_warm,GLDAS_temp_warm)
    						GLDAS_ubrmse_warm_master_Eur.append(GLDAS_ubrmse_warm)
    						GLDAS_CLSM_ubrmse_warm = ubrmsd(station_temp_warm,GLDAS_CLSM_temp_warm)
    						GLDAS_CLSM_ubrmse_warm_master_Eur.append(GLDAS_CLSM_ubrmse_warm)



## Pearson Correlations ##
    						naive_corr_warm,_ = pearsonr(naive_temp_warm, station_temp_warm)
    						naive_corr_warm_master_Eur.append(naive_corr_warm)
    						naive_noJRA_corr_warm,_ = pearsonr(naive_noJRA_temp_warm, station_temp_warm)
    						naive_noJRA_corr_warm_master_Eur.append(naive_noJRA_corr_warm)
    						naive_noJRAold_corr_warm,_ = pearsonr(naive_noJRAold_temp_warm, station_temp_warm)
    						naive_noJRAold_corr_warm_master_Eur.append(naive_noJRAold_corr_warm)
    						naive_all_corr_warm,_ = pearsonr(naive_all_temp_warm, station_temp_warm)
    						naive_all_corr_warm_master_Eur.append(naive_all_corr_warm)
    						CFSR_corr_warm,_ = pearsonr(CFSR_temp_warm, station_temp_warm)
    						CFSR_corr_warm_master_Eur.append(CFSR_corr_warm)
    						ERAI_corr_warm,_ = pearsonr(ERAI_temp_warm, station_temp_warm)
    						ERAI_corr_warm_master_Eur.append(ERAI_corr_warm)
    						ERA5_corr_warm,_ = pearsonr(ERA5_temp_warm, station_temp_warm)
    						ERA5_corr_warm_master_Eur.append(ERA5_corr_warm)
    						ERA5_Land_corr_warm,_ = pearsonr(ERA5_Land_temp_warm, station_temp_warm)
    						ERA5_Land_corr_warm_master_Eur.append(ERA5_Land_corr_warm)
    						JRA_corr_warm,_ = pearsonr(JRA_temp_warm, station_temp_warm)
    						JRA_corr_warm_master_Eur.append(JRA_corr_warm)
    						MERRA2_corr_warm,_ = pearsonr(MERRA2_temp_warm, station_temp_warm)
    						MERRA2_corr_warm_master_Eur.append(MERRA2_corr_warm)
    						GLDAS_corr_warm,_ = pearsonr(GLDAS_temp_warm, station_temp_warm)
    						GLDAS_corr_warm_master_Eur.append(GLDAS_corr_warm)
    						GLDAS_CLSM_corr_warm,_ = pearsonr(GLDAS_CLSM_temp_warm, station_temp_warm)
    						GLDAS_CLSM_corr_warm_master_Eur.append(GLDAS_CLSM_corr_warm)




#### Wilcoxon Rank Sum Tests (NAm vs Eur) ####


    					print("Warm Season:")

# Bias ##

    					print('Bias:')

# Ensemble Mean #
    					sample_NAm_ensmean = naive_all_bias_warm_master_NAm
    					sample_Eur_ensmean = naive_all_bias_warm_master_Eur

    					result = mannwhitneyu(sample_NAm_ensmean,sample_Eur_ensmean)
    					print('EnsMean result: ',result)
					
# CFSR #
    					sample_NAm_CFSR = CFSR_bias_warm_master_NAm
    					sample_Eur_CFSR = CFSR_bias_warm_master_Eur

    					result = mannwhitneyu(sample_NAm_CFSR,sample_Eur_CFSR)
    					print('CFSR result: ',result)


# ERAI #
    					sample_NAm_ERAI = ERAI_bias_warm_master_NAm
    					sample_Eur_ERAI = ERAI_bias_warm_master_Eur

    					result = mannwhitneyu(sample_NAm_ERAI,sample_Eur_ERAI)
    					print('ERAI result: ',result)


# ERA5 #
    					sample_NAm_ERA5 = ERA5_bias_warm_master_NAm
    					sample_Eur_ERA5 = ERA5_bias_warm_master_Eur

    					result = mannwhitneyu(sample_NAm_ERA5,sample_Eur_ERA5)
    					print('ERA5 result: ',result)
# ERA5-Land #
    					sample_NAm_ERA5_L = ERA5_Land_bias_warm_master_NAm
    					sample_Eur_ERA5_L = ERA5_Land_bias_warm_master_Eur

    					result = mannwhitneyu(sample_NAm_ERA5_L,sample_Eur_ERA5_L)
    					print('ERA5-Land result: ',result)


# JRA #
    					sample_NAm_JRA = JRA_bias_warm_master_NAm
    					sample_Eur_JRA = JRA_bias_warm_master_Eur

    					result = mannwhitneyu(sample_NAm_JRA,sample_Eur_JRA)
    					print('JRA result: ',result)


# MERRA2 #
    					sample_NAm_MERRA2 = MERRA2_bias_warm_master_NAm
    					sample_Eur_MERRA2 = MERRA2_bias_warm_master_Eur

    					result = mannwhitneyu(sample_NAm_MERRA2,sample_Eur_MERRA2)
    					print('MERRA2 result: ',result)


# GLDAS #
    					sample_NAm_GLDAS = GLDAS_bias_warm_master_NAm
    					sample_Eur_GLDAS = GLDAS_bias_warm_master_Eur

    					result = mannwhitneyu(sample_NAm_GLDAS,sample_Eur_GLDAS)
    					print('GLDAS result: ',result)


# GLDAS_CLSM #
    					sample_NAm_GLDAS_CLSM = GLDAS_CLSM_bias_warm_master_NAm
    					sample_Eur_GLDAS_CLSM = GLDAS_CLSM_bias_warm_master_Eur

    					result = mannwhitneyu(sample_NAm_GLDAS_CLSM,sample_Eur_GLDAS_CLSM)
    					print('GLDAS_CLSM result: ',result)


## RMSE ##


    					print('RMSE:')
# Ensemble Mean #
    					sample_NAm_ensmean = naive_all_rmse_warm_master_NAm
    					sample_Eur_ensmean = naive_all_rmse_warm_master_Eur

    					result = mannwhitneyu(sample_NAm_ensmean,sample_Eur_ensmean)
    					print('EnsMean result: ',result)
					
# CFSR #
    					sample_NAm_CFSR = CFSR_rmse_warm_master_NAm
    					sample_Eur_CFSR = CFSR_rmse_warm_master_Eur

    					result = mannwhitneyu(sample_NAm_CFSR,sample_Eur_CFSR)
    					print('CFSR result: ',result)


# ERAI #
    					sample_NAm_ERAI = ERAI_rmse_warm_master_NAm
    					sample_Eur_ERAI = ERAI_rmse_warm_master_Eur

    					result = mannwhitneyu(sample_NAm_ERAI,sample_Eur_ERAI)
    					print('ERAI result: ',result)


# ERA5 #
    					sample_NAm_ERA5 = ERA5_rmse_warm_master_NAm
    					sample_Eur_ERA5 = ERA5_rmse_warm_master_Eur

    					result = mannwhitneyu(sample_NAm_ERA5,sample_Eur_ERA5)
    					print('ERA5 result: ',result)

# ERA5-Land #
    					sample_NAm_ERA5_L = ERA5_Land_rmse_warm_master_NAm
    					sample_Eur_ERA5_L = ERA5_Land_rmse_warm_master_Eur

    					result = mannwhitneyu(sample_NAm_ERA5_L,sample_Eur_ERA5_L)
    					print('ERA5-Land result: ',result)


# JRA #
    					sample_NAm_JRA = JRA_rmse_warm_master_NAm
    					sample_Eur_JRA = JRA_rmse_warm_master_Eur

    					result = mannwhitneyu(sample_NAm_JRA,sample_Eur_JRA)
    					print('JRA result: ',result)


# MERRA2 #
    					sample_NAm_MERRA2 = MERRA2_rmse_warm_master_NAm
    					sample_Eur_MERRA2 = MERRA2_rmse_warm_master_Eur

    					result = mannwhitneyu(sample_NAm_MERRA2,sample_Eur_MERRA2)
    					print('MERRA2 result: ',result)


# GLDAS #
    					sample_NAm_GLDAS = GLDAS_rmse_warm_master_NAm
    					sample_Eur_GLDAS = GLDAS_rmse_warm_master_Eur

    					result = mannwhitneyu(sample_NAm_GLDAS,sample_Eur_GLDAS)
    					print('GLDAS result: ',result)


# GLDAS_CLSM #
    					sample_NAm_GLDAS_CLSM = GLDAS_CLSM_rmse_warm_master_NAm
    					sample_Eur_GLDAS_CLSM = GLDAS_CLSM_rmse_warm_master_Eur

    					result = mannwhitneyu(sample_NAm_GLDAS_CLSM,sample_Eur_GLDAS_CLSM)
    					print('GLDAS_CLSM result: ',result)


















