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


#/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/by_date/remapnn_simple_average_top_30cm_thr_75_warm_season_date_summary.csv
#/home/herringtont/anaconda3/envs/SoilTemp/lib/python3.8/site-packages/numpy/core/_methods.py:233: RuntimeWarning: Degrees of freedom <= 0 for slice
#  ret = _var(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
#/home/herringtont/anaconda3/envs/SoilTemp/lib/python3.8/site-packages/numpy/core/_methods.py:226: RuntimeWarning: invalid value encountered in double_scalars
#  ret = ret.dtype.type(ret / rcount)
#Traceback (most recent call last):
#  File "naive_blended_taylor_diagram.py", line 803, in <module>
#    naive_corr_cold_temp, _ = scipy.stats.pearsonr(station_cold_temp,naive_cold_temp)
#  File "/home/herringtont/anaconda3/envs/SoilTemp/lib/python3.8/site-packages/scipy/stats/stats.py", line 3501, in pearsonr
#    raise ValueError('x and y must have length at least 2.')
#ValueError: x and y must have length at least 2.






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
temp_thr_n = '-2C'#['0C','-2C','-5C','-10C']
permafrost_type = ['RS_2002_permafrost','RS_2002_none']


############ Grab ERA5 MAAT data #############

ERA5_MAAT_fi = '/mnt/data/users/herringtont/soil_temp/reanalysis/2m_AirTemp/rename/land_only/common_grid_CLSM/remapcon/MAAT/ERA5_2m_air_MAAT_1981_2010_clim.nc'
ERA5_MAAT_fil = xr.open_dataset(ERA5_MAAT_fi)
ERA5_MAAT = ERA5_MAAT_fil['Air_Temp'] - 273.15

############# Grab Data ##############

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
    
    				for n in permafrost_type:
    					permafrost_type_o = n
    					cold_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/CLSM_res/subset/'+str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_dframe_cold_season_temp_master_ERA5_'+str(temp_thr_n)+'_CMOS_CLSM_subset_permafrost.csv'])
    					warm_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/CLSM_res/subset/'+str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_dframe_warm_season_temp_master_ERA5_'+str(temp_thr_n)+'_CMOS_CLSM_subset_permafrost.csv'])    			
    					scatter_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/CLSM_res/subset/'+str(remap_type)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_dframe_scatterplot_CMOS_CLSM_subset_permafrost_reorder.csv'])

##### Cold Season ######

    					dframe_cold = pd.read_csv(cold_fil)
    					if (permafrost_type_o == 'RS_2002_permafrost'):
    						dframe_cold_permafrost = dframe_cold[(dframe_cold['RS 2002 Permafrost'] == 'continuous') | (dframe_cold['RS 2002 Permafrost'] == 'discontinuous')]

    					elif (permafrost_type_o == 'RS_2002_none'):
    						dframe_cold_permafrost = dframe_cold[dframe_cold['RS 2002 Permafrost'] == 'none']

    					station_cold = dframe_cold_permafrost['Station'].values
    					naive_cold = dframe_cold_permafrost['Naive Blend'].values
    					naive_noJRA_cold = dframe_cold_permafrost['Naive Blend no JRA55'].values
    					naive_noJRAold_cold = dframe_cold_permafrost['Naive Blend no JRA55 Old'].values
    					naive_all_cold = dframe_cold_permafrost['Naive Blend All'].values
    					CFSR_cold = dframe_cold_permafrost['CFSR'].values
    					ERAI_cold = dframe_cold_permafrost['ERA-Interim'].values
    					ERA5_cold = dframe_cold_permafrost['ERA5'].values
    					ERA5_Land_cold = dframe_cold_permafrost['ERA5-Land'].values
    					JRA_cold = dframe_cold_permafrost['JRA55'].values
    					MERRA2_cold = dframe_cold_permafrost['MERRA2'].values
    					GLDAS_cold = dframe_cold_permafrost['GLDAS-Noah'].values
    					GLDAS_CLSM_cold = dframe_cold_permafrost['GLDAS-CLSM'].values



# Master Arrays #

    					gcell_cold_master = []
    					MAAT_cold_master = []

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

# Calculate grid cell level statistics #

    					GLDAS_CLSM_cold = dframe_cold_permafrost['GLDAS-CLSM'].values

    					gcell_cold = dframe_cold_permafrost['Grid Cell'].values
    					gcell_cold_uq = np.unique(gcell_cold)

    					for p in gcell_cold_uq: #loop through grid cells
    						gcell_p = p
    						if (gcell_p == 33777):
    							continue
    						gcell_cold_master.append(gcell_p)
    						dframe_cold_season_gcell = dframe_cold_permafrost[dframe_cold_permafrost['Grid Cell'] == gcell_p]

    						central_lat_cold = dframe_cold_season_gcell['Central Lat'].iloc[0]
    						central_lon_cold = dframe_cold_season_gcell['Central Lon'].iloc[0]
    						ERA5_MAAT_cold = ERA5_MAAT.sel(lat=central_lat_cold,lon=central_lon_cold,method='nearest',drop = True).values
    						MAAT_cold_master.append(ERA5_MAAT_cold)    						
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


## Create Final Dataframe ##

    					MAAT_cold_master = [i for sub in MAAT_cold_master for i in sub]

    					dframe_cold_final = pd.DataFrame(data=gcell_cold_master, columns=['Grid Cell'])
    					dframe_cold_final['ERA5 MAAT'] = MAAT_cold_master
    					dframe_cold_final['Ensemble Mean Corr'] = naive_all_corr_cold_master
    					dframe_cold_final['CFSR Corr'] = CFSR_corr_cold_master
    					dframe_cold_final['ERA-Interim Corr'] = ERAI_corr_cold_master
    					dframe_cold_final['ERA5 Corr'] = ERA5_corr_cold_master
    					dframe_cold_final['ERA5-Land Corr'] = ERA5_Land_corr_cold_master
    					dframe_cold_final['JRA55 Corr'] = JRA_corr_cold_master
    					dframe_cold_final['MERRA2 Corr'] = MERRA2_corr_cold_master
    					dframe_cold_final['GLDAS-Noah Corr'] = GLDAS_corr_cold_master
    					dframe_cold_final['GLDAS-CLSM Corr'] = GLDAS_CLSM_corr_cold_master
					

    					dframe_cold_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CMOS_poster/CLSM_res/'+str(remap_type)+'_'+str(lyr_l)+'_corr_cold_season_MAAT_CLSM_'+str(permafrost_type_o)+'_grid_averages.csv'])
    					dframe_cold_final.to_csv(dframe_cold_fil)


##### Warm Season ######

    					dframe_warm = pd.read_csv(warm_fil)
    					if (permafrost_type_o == 'RS_2002_permafrost'):
    						dframe_warm_permafrost = dframe_warm[(dframe_warm['RS 2002 Permafrost'] == 'continuous') | (dframe_warm['RS 2002 Permafrost'] == 'discontinuous')]

    					elif (permafrost_type_o == 'RS_2002_none'):
    						dframe_warm_permafrost = dframe_warm[dframe_warm['RS 2002 Permafrost'] == 'none']

    					station_warm = dframe_warm_permafrost['Station'].values
    					naive_warm = dframe_warm_permafrost['Naive Blend'].values
    					naive_noJRA_warm = dframe_warm_permafrost['Naive Blend no JRA55'].values
    					naive_noJRAold_warm = dframe_warm_permafrost['Naive Blend no JRA55 Old'].values
    					naive_all_warm = dframe_warm_permafrost['Naive Blend All'].values
    					CFSR_warm = dframe_warm_permafrost['CFSR'].values
    					ERAI_warm = dframe_warm_permafrost['ERA-Interim'].values
    					ERA5_warm = dframe_warm_permafrost['ERA5'].values
    					ERA5_Land_warm = dframe_warm_permafrost['ERA5-Land'].values
    					JRA_warm = dframe_warm_permafrost['JRA55'].values
    					MERRA2_warm = dframe_warm_permafrost['MERRA2'].values
    					GLDAS_warm = dframe_warm_permafrost['GLDAS-Noah'].values
    					GLDAS_CLSM_warm = dframe_warm_permafrost['GLDAS-CLSM'].values



# Master Arrays #

    					gcell_warm_master = []
    					MAAT_warm_master = []

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

# Calculate grid cell level statistics #

    					GLDAS_CLSM_warm = dframe_warm_permafrost['GLDAS-CLSM'].values

    					gcell_warm = dframe_warm_permafrost['Grid Cell'].values
    					gcell_warm_uq = np.unique(gcell_warm)

    					for p in gcell_warm_uq: #loop through grid cells
    						gcell_p = p
    						if (gcell_p == 33777):
    							continue
    						gcell_warm_master.append(gcell_p)
    						dframe_warm_season_gcell = dframe_warm_permafrost[dframe_warm_permafrost['Grid Cell'] == gcell_p]

    						central_lat_warm = dframe_warm_season_gcell['Central Lat'].iloc[0]
    						central_lon_warm = dframe_warm_season_gcell['Central Lon'].iloc[0]
    						ERA5_MAAT_warm = ERA5_MAAT.sel(lat=central_lat_warm,lon=central_lon_warm,method='nearest',drop = True).values
    						MAAT_warm_master.append(ERA5_MAAT_warm)    						
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


## Create Final Dataframe ##

    					dframe_warm_final = pd.DataFrame(data=gcell_warm_master, columns=['Grid Cell'])
    					dframe_warm_final['ERA5 MAAT'] = MAAT_warm_master
    					dframe_warm_final['Ensemble Mean Corr'] = naive_all_corr_warm_master
    					dframe_warm_final['CFSR Corr'] = CFSR_corr_warm_master
    					dframe_warm_final['ERA-Interim Corr'] = ERAI_corr_warm_master
    					dframe_warm_final['ERA5 Corr'] = ERA5_corr_warm_master
    					dframe_warm_final['ERA5-Land Corr'] = ERA5_Land_corr_warm_master
    					dframe_warm_final['JRA55 Corr'] = JRA_corr_warm_master
    					dframe_warm_final['MERRA2 Corr'] = MERRA2_corr_warm_master
    					dframe_warm_final['GLDAS-Noah Corr'] = GLDAS_corr_warm_master
    					dframe_warm_final['GLDAS-CLSM Corr'] = GLDAS_CLSM_corr_warm_master


    					dframe_warm_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CMOS_poster/CLSM_res/'+str(remap_type)+'_'+str(lyr_l)+'_corr_warm_season_MAAT_CLSM_'+str(permafrost_type_o)+'_grid_averages.csv'])
    					dframe_warm_final.to_csv(dframe_warm_fil)
