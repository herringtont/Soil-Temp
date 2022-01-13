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
permafrost_type = ['RS_2002_all']

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

    					cold_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/CLSM_res/subset/'+str(remap_type)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_dframe_scatterplot_CMOS_CLSM_subset_permafrost_cold_warm_BEST.csv'])
    					warm_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/CLSM_res/subset/'+str(remap_type)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_dframe_scatterplot_CMOS_CLSM_subset_permafrost_cold_warm_BEST.csv'])    			
    					scatter_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/CLSM_res/subset/'+str(remap_type)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_dframe_scatterplot_CMOS_CLSM_subset_permafrost_cold_warm_BEST.csv'])


    					dframe = pd.read_csv(scatter_fil)
    					dframe_cold = dframe[dframe['Season'] == 'Cold']
    					dframe_warm = dframe[dframe['Season'] == 'Warm']
    					if (permafrost_type_o == 'RS_2002_all'):
    						dframe_cold_permafrost = dframe_cold

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

##### Cold Season ######

# Master Arrays #

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

    						dframe_cold_season_gcell = dframe_cold_permafrost[dframe_cold_permafrost['Grid Cell'] == gcell_p]
    						
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

    					stn_corr_cold_avg = 1.0
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


    					stn_crmsd_cold_avg_square = (stn_var_cold_avg**2 + stn_var_cold_avg**2) - 2*(stn_var_cold_avg*stn_var_cold_avg*stn_corr_cold_avg)
    					stn_crmsd_cold_avg = math.sqrt(stn_crmsd_cold_avg_square)    					
    					naive_crmsd_cold_avg_square = (naive_var_cold_avg**2 + stn_var_cold_avg**2) - 2*(naive_var_cold_avg*stn_var_cold_avg*naive_corr_cold_avg)
    					naive_crmsd_cold_avg = math.sqrt(naive_crmsd_cold_avg_square)
    					naive_noJRA_crmsd_cold_avg_square = (naive_noJRA_var_cold_avg**2 + stn_var_cold_avg**2) - 2*(naive_noJRA_var_cold_avg*stn_var_cold_avg*naive_noJRA_corr_cold_avg)
    					naive_noJRA_crmsd_cold_avg = math.sqrt(naive_noJRA_crmsd_cold_avg_square)
    					naive_noJRAold_crmsd_cold_avg_square = (naive_noJRAold_var_cold_avg**2 + stn_var_cold_avg**2) - 2*(naive_noJRAold_var_cold_avg*stn_var_cold_avg*naive_noJRAold_corr_cold_avg)
    					naive_noJRAold_crmsd_cold_avg = math.sqrt(naive_noJRAold_crmsd_cold_avg_square)
    					naive_all_crmsd_cold_avg_square = (naive_all_var_cold_avg**2 + stn_var_cold_avg**2) - 2*(naive_all_var_cold_avg*stn_var_cold_avg*naive_all_corr_cold_avg)
    					naive_all_crmsd_cold_avg = math.sqrt(naive_all_crmsd_cold_avg_square)
    					CFSR_crmsd_cold_avg_square = (CFSR_var_cold_avg**2 + stn_var_cold_avg**2) - 2*(CFSR_var_cold_avg*stn_var_cold_avg*CFSR_corr_cold_avg)
    					CFSR_crmsd_cold_avg = math.sqrt(CFSR_crmsd_cold_avg_square)
    					ERAI_crmsd_cold_avg_square = (ERAI_var_cold_avg**2 + stn_var_cold_avg**2) - 2*(ERAI_var_cold_avg*stn_var_cold_avg*ERAI_corr_cold_avg)
    					ERAI_crmsd_cold_avg = math.sqrt(ERAI_crmsd_cold_avg_square)
    					ERA5_crmsd_cold_avg_square = (ERA5_var_cold_avg**2 + stn_var_cold_avg**2) - 2*(ERA5_var_cold_avg*stn_var_cold_avg*ERA5_corr_cold_avg)
    					ERA5_crmsd_cold_avg = math.sqrt(ERA5_crmsd_cold_avg_square)
    					ERA5_Land_crmsd_cold_avg_square = (ERA5_Land_var_cold_avg**2 + stn_var_cold_avg**2) - 2*(ERA5_Land_var_cold_avg*stn_var_cold_avg*ERA5_Land_corr_cold_avg)
    					ERA5_Land_crmsd_cold_avg = math.sqrt(ERA5_Land_crmsd_cold_avg_square)
    					JRA_crmsd_cold_avg_square = (JRA_var_cold_avg**2 + JRA_var_cold_avg**2) - 2*(JRA_var_cold_avg*JRA_var_cold_avg*JRA_corr_cold_avg)
    					JRA_crmsd_cold_avg = math.sqrt(JRA_crmsd_cold_avg_square)
    					MERRA2_crmsd_cold_avg_square = (MERRA2_var_cold_avg**2 + MERRA2_var_cold_avg**2) - 2*(MERRA2_var_cold_avg*MERRA2_var_cold_avg*MERRA2_corr_cold_avg)
    					MERRA2_crmsd_cold_avg = math.sqrt(MERRA2_crmsd_cold_avg_square)
    					GLDAS_crmsd_cold_avg_square = (GLDAS_var_cold_avg**2 + GLDAS_var_cold_avg**2) - 2*(GLDAS_var_cold_avg*GLDAS_var_cold_avg*GLDAS_corr_cold_avg)
    					GLDAS_crmsd_cold_avg = math.sqrt(GLDAS_crmsd_cold_avg_square)
    					GLDAS_CLSM_crmsd_cold_avg_square = (GLDAS_CLSM_var_cold_avg**2 + GLDAS_CLSM_var_cold_avg**2) - 2*(GLDAS_CLSM_var_cold_avg*GLDAS_CLSM_var_cold_avg*GLDAS_CLSM_corr_cold_avg)
    					GLDAS_CLSM_crmsd_cold_avg = math.sqrt(GLDAS_CLSM_crmsd_cold_avg_square)


##################### Create Taylor Diagrams ####################

    					matplotlib.rc('font',size=14)
    					print('Remap Style:',remap_type)
    					print('Layer:',lyr_l)
    					print('Temp Threshold:',temp_thr_n)


    					taylor_dir = '/mnt/data/users/herringtont/soil_temp/plots/taylor_diagrams/new_data/CMOS_poster/CLSM_res/'
    					taylor_plt_fil_cold = ''.join([taylor_dir,str(remap_type)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_taylor_diagram_cold_new_data_CMOS_CLSM_grid_cell_BEST.png'])
		
    					stn_norm_sdev_cold_avg = stn_sdev_cold_avg/stn_sdev_cold_avg
    					naive_norm_sdev_cold_avg = naive_sdev_cold_avg/stn_sdev_cold_avg
    					naive_noJRA_norm_sdev_cold_avg = naive_noJRA_sdev_cold_avg/stn_sdev_cold_avg
    					naive_noJRAold_norm_sdev_cold_avg = naive_noJRAold_sdev_cold_avg/stn_sdev_cold_avg
    					naive_all_norm_sdev_cold_avg = naive_all_sdev_cold_avg/stn_sdev_cold_avg					
    					CFSR_norm_sdev_cold_avg = CFSR_sdev_cold_avg/stn_sdev_cold_avg
    					ERAI_norm_sdev_cold_avg = ERAI_sdev_cold_avg/stn_sdev_cold_avg
    					ERA5_norm_sdev_cold_avg = ERA5_sdev_cold_avg/stn_sdev_cold_avg
    					ERA5_Land_norm_sdev_cold_avg = ERA5_Land_sdev_cold_avg/stn_sdev_cold_avg
    					JRA_norm_sdev_cold_avg = JRA_sdev_cold_avg/stn_sdev_cold_avg
    					MERRA2_norm_sdev_cold_avg = MERRA2_sdev_cold_avg/stn_sdev_cold_avg
    					GLDAS_norm_sdev_cold_avg = GLDAS_sdev_cold_avg/stn_sdev_cold_avg
    					GLDAS_CLSM_norm_sdev_cold_avg = GLDAS_CLSM_sdev_cold_avg/stn_sdev_cold_avg

    					sdev_naive = np.array([stn_norm_sdev_cold_avg,naive_norm_sdev_cold_avg])
    					crmsd_naive = np.array([stn_crmsd_cold_avg,naive_crmsd_cold_avg])
    					ccoef_naive = np.array([stn_corr_cold_avg,naive_corr_cold_avg])

    					print('Naive std dev:',sdev_naive)
    					print('Naive CRMSD:',crmsd_naive)
    					print('Naive corr:',ccoef_naive)

    					sdev_naive_noJRA = np.array([stn_norm_sdev_cold_avg,naive_noJRA_norm_sdev_cold_avg])
    					crmsd_naive_noJRA = np.array([stn_crmsd_cold_avg,naive_noJRA_crmsd_cold_avg])
    					ccoef_naive_noJRA = np.array([stn_corr_cold_avg,naive_noJRA_corr_cold_avg])

    					sdev_naive_noJRAold = np.array([stn_norm_sdev_cold_avg,naive_noJRAold_norm_sdev_cold_avg])
    					crmsd_naive_noJRAold = np.array([stn_crmsd_cold_avg,naive_noJRAold_crmsd_cold_avg])
    					ccoef_naive_noJRAold = np.array([stn_corr_cold_avg,naive_noJRAold_corr_cold_avg])

    					sdev_naive_all = np.array([stn_norm_sdev_cold_avg,naive_all_norm_sdev_cold_avg])
    					crmsd_naive_all = np.array([stn_crmsd_cold_avg,naive_all_crmsd_cold_avg])
    					ccoef_naive_all = np.array([stn_corr_cold_avg,naive_all_corr_cold_avg])

    					sdev_CFSR = np.array([stn_norm_sdev_cold_avg,CFSR_norm_sdev_cold_avg])
    					crmsd_CFSR = np.array([stn_crmsd_cold_avg,CFSR_crmsd_cold_avg])
    					ccoef_CFSR = np.array([stn_corr_cold_avg,CFSR_corr_cold_avg])

    					sdev_ERAI = np.array([stn_norm_sdev_cold_avg,ERAI_norm_sdev_cold_avg])
    					crmsd_ERAI = np.array([stn_crmsd_cold_avg,ERAI_crmsd_cold_avg])
    					ccoef_ERAI = np.array([stn_corr_cold_avg,ERAI_corr_cold_avg])

    					sdev_ERA5 = np.array([stn_norm_sdev_cold_avg,ERA5_norm_sdev_cold_avg])
    					crmsd_ERA5 = np.array([stn_crmsd_cold_avg,ERA5_crmsd_cold_avg])
    					ccoef_ERA5 = np.array([stn_corr_cold_avg,ERA5_corr_cold_avg])

    					sdev_ERA5_Land = np.array([stn_norm_sdev_cold_avg,ERA5_Land_norm_sdev_cold_avg])
    					crmsd_ERA5_Land = np.array([stn_crmsd_cold_avg,ERA5_Land_crmsd_cold_avg])
    					ccoef_ERA5_Land = np.array([stn_corr_cold_avg,ERA5_Land_corr_cold_avg])

    					sdev_JRA = np.array([stn_norm_sdev_cold_avg,JRA_norm_sdev_cold_avg])
    					crmsd_JRA = np.array([stn_crmsd_cold_avg,JRA_crmsd_cold_avg])
    					ccoef_JRA = np.array([stn_corr_cold_avg,JRA_corr_cold_avg])

    					sdev_MERRA2 = np.array([stn_norm_sdev_cold_avg,MERRA2_norm_sdev_cold_avg])
    					crmsd_MERRA2 = np.array([stn_crmsd_cold_avg,MERRA2_crmsd_cold_avg])
    					ccoef_MERRA2 = np.array([stn_corr_cold_avg,MERRA2_corr_cold_avg])

    					sdev_GLDAS = np.array([stn_norm_sdev_cold_avg,GLDAS_norm_sdev_cold_avg])
    					crmsd_GLDAS = np.array([stn_crmsd_cold_avg,GLDAS_crmsd_cold_avg])
    					ccoef_GLDAS = np.array([stn_corr_cold_avg,GLDAS_corr_cold_avg])

    					sdev_GLDAS_CLSM = np.array([stn_norm_sdev_cold_avg,GLDAS_CLSM_norm_sdev_cold_avg])
    					crmsd_GLDAS_CLSM = np.array([stn_crmsd_cold_avg,GLDAS_CLSM_crmsd_cold_avg])
    					ccoef_GLDAS_CLSM = np.array([stn_corr_cold_avg,GLDAS_CLSM_corr_cold_avg])

    					label = {'Station':'dimgrey','Ensemble Mean':'fuchsia','CFSR': 'm','ERA-Interim': 'limegreen', 'ERA5': 'cyan','ERA5-Land':'skyblue','JRA55': 'red', 'MERRA2': 'goldenrod', 'GLDAS-Noah': 'black','GLDAS-CLSM':'darkgrey'}
								
    					#sm.taylor_diagram(sdev_naive,crmsd_naive,ccoef_naive,tickSTD=[0,0.25,0.5,0.75,1,1.25,1.5,1.75,2],axismax=2.0,styleSTD = '-.', widthSTD = 0.25,styleOBS = '-',tickRMS=[0.0],titleRMS='off',tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5,colOBS = 'dimgrey',markerobs = 'o',titleOBS = 'Station', markercolor='dodgerblue', markerLabelcolor='dodgerblue')
    					sm.taylor_diagram(sdev_naive_all,crmsd_naive_all,ccoef_naive_all,tickSTD=[0,0.25,0.5,0.75,1,1.25,1.5,1.75,2],axismax=2.0,styleSTD = '-.', widthSTD = 0.25,styleOBS = '-',tickRMS=[0.0],titleRMS='off',tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5,colOBS = 'dimgrey',markerobs = 'o',titleOBS = 'Station', markercolor='hotpink', markerLabelcolor='hotpink', markerSize=15,showlabelsRMS = 'off',alpha=0.6)
    					#sm.taylor_diagram(sdev_naive_noJRAold,crmsd_naive_noJRAold,ccoef_naive_noJRAold,tickSTD=[0,0.25,0.5,0.75,1,1.25,1.5,1.75,2],axismax=2.0,styleSTD = '-.', widthSTD = 0.25,styleOBS = '-',tickRMS=[0.0],titleRMS='off',tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='mediumslateblue', markerLabelcolor='mediumslateblue',overlay='on')
    					#sm.taylor_diagram(sdev_naive_all,crmsd_naive_all,ccoef_naive_all,tickSTD=[0,0.25,0.5,0.75,1,1.25,1.5,1.75,2],axismax=2.0,styleSTD = '-.', widthSTD = 0.25,styleOBS = '-',tickRMS=[0.0],titleRMS='off',tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='fuchsia', markerLabelcolor='fuchsia',overlay='on')
    					#sm.taylor_diagram(sdev_naive_noJRA,crmsd_naive_noJRA,ccoef_naive_noJRA,tickSTD=[0,0.25,0.5,0.75,1,1.25,1.5,1.75,2],axismax=2.0,styleSTD = '-.', widthSTD = 0.25,styleOBS = '-',tickRMS=[0.0],titleRMS='off',tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='coral', markerLabelcolor='coral',overlay='on')
    					sm.taylor_diagram(sdev_CFSR,crmsd_CFSR,ccoef_CFSR,tickSTD=[0,0.25,0.5,0.75,1,1.25,1.5,1.75,2],axismax=2.0,styleSTD = '-.', widthSTD = 0.25,styleOBS = '-',tickRMS=[0.0],titleRMS='off',tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='darkviolet', markerLabelcolor='darkviolet',markerSize=15,showlabelsRMS = 'off',alpha=0.6, overlay='on')
    					sm.taylor_diagram(sdev_ERAI,crmsd_ERAI,ccoef_ERAI,tickSTD=[0,0.25,0.5,0.75,1,1.25,1.5,1.75,2],axismax=2.0,styleSTD = '-.', widthSTD = 0.25,styleOBS = '-',tickRMS=[0.0],titleRMS='off',tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='springgreen', markerLabelcolor='springgreen',markerSize=15,showlabelsRMS = 'off',alpha=0.6,overlay='on')
    					sm.taylor_diagram(sdev_ERA5,crmsd_ERA5,ccoef_ERA5,tickSTD=[0,0.25,0.5,0.75,1,1.25,1.5,1.75,2],axismax=2.0,styleSTD = '-.', widthSTD = 0.25,styleOBS = '-',tickRMS=[0.0],titleRMS='off',tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='greenyellow', markerLabelcolor='greenyellow',markerSize=15,showlabelsRMS = 'off',alpha=0.6,overlay='on')
    					sm.taylor_diagram(sdev_ERA5_Land,crmsd_ERA5_Land,ccoef_ERA5_Land,tickSTD=[0,0.25,0.5,0.75,1,1.25,1.5,1.75,2],axismax=2.0,styleSTD = '-.', widthSTD = 0.25,styleOBS = '-',tickRMS=[0.0],titleRMS='off',tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='dodgerblue',markerLabelcolor='dodgerblue',markerSize=15,showlabelsRMS = 'off',alpha=0.6,overlay='on')
    					sm.taylor_diagram(sdev_JRA,crmsd_JRA,ccoef_JRA,tickSTD=[0,0.25,0.5,0.75,1,1.25,1.5,1.75,2],axismax=2.0,styleSTD = '-.', widthSTD = 0.25,styleOBS = '-',tickRMS=[0.0],titleRMS='off',tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='red',markerLabelcolor='red',markerSize=15,showlabelsRMS = 'off',alpha=0.6,overlay='on')
    					sm.taylor_diagram(sdev_MERRA2,crmsd_MERRA2,ccoef_MERRA2,tickSTD=[0,0.25,0.5,0.75,1,1.25,1.5,1.75,2],axismax=2.0,styleSTD = '-.', widthSTD = 0.25,styleOBS = '-',tickRMS=[0.0],titleRMS='off',tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='goldenrod',markerLabelcolor='goldenrod',markerSize=15,showlabelsRMS = 'off',alpha=0.6,overlay='on')
    					sm.taylor_diagram(sdev_GLDAS,crmsd_GLDAS,ccoef_GLDAS,tickSTD=[0,0.25,0.5,0.75,1,1.25,1.5,1.75,2],axismax=2.0,styleSTD = '-.', widthSTD = 0.25,styleOBS = '-',tickRMS=[0.0],titleRMS='off',tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='black',markerLabelcolor='black',markerSize=15,showlabelsRMS = 'off',alpha=0.6,overlay='on')
    					#sm.taylor_diagram(sdev_GLDAS_CLSM,crmsd_GLDAS_CLSM,ccoef_GLDAS_CLSM,tickSTD=[0,0.25,0.5,0.75,1,1.25,1.5,1.75,2],axismax=2.0,styleSTD = '-.', widthSTD = 0.25,styleOBS = '-',tickRMS=[0.0],titleRMS='off',tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='darkgrey',markerLabelcolor='darkgrey',markerSize=15,alpha=0.6,overlay='on',markerLabel=label)
    					sm.taylor_diagram(sdev_GLDAS_CLSM,crmsd_GLDAS_CLSM,ccoef_GLDAS_CLSM,tickSTD=[0,0.25,0.5,0.75,1,1.25,1.5,1.75,2],axismax=2.0,styleSTD = '-.', widthSTD = 0.25,styleOBS = '-',tickRMS=[0.0],titleRMS='off',tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='darkgrey',markerLabelcolor='darkgrey',markerSize=15,showlabelsRMS = 'off',alpha=0.6,overlay='on')

    					plt.savefig(taylor_plt_fil_cold)
    					plt.close()
    					print(taylor_plt_fil_cold)


##### Warm Season ######

# Master Arrays #

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

    					if (permafrost_type_o == 'RS_2002_all'):
    						dframe_warm_permafrost = dframe_warm

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

    					gcell_warm = dframe_warm_permafrost['Grid Cell'].values
    					gcell_warm_uq = np.unique(gcell_warm)

    					for p in gcell_warm_uq: #loop through grid cells
    						gcell_p = p

    						dframe_warm_season_gcell = dframe_warm_permafrost[dframe_warm_permafrost['Grid Cell'] == gcell_p]
    						
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

    					stn_corr_warm_avg = 1.0
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


    					stn_crmsd_warm_avg_square = (stn_var_warm_avg**2 + stn_var_warm_avg**2) - 2*(stn_var_warm_avg*stn_var_warm_avg*stn_corr_warm_avg)
    					stn_crmsd_warm_avg = math.sqrt(stn_crmsd_warm_avg_square)    					
    					naive_crmsd_warm_avg_square = (naive_var_warm_avg**2 + stn_var_warm_avg**2) - 2*(naive_var_warm_avg*stn_var_warm_avg*naive_corr_warm_avg)
    					naive_crmsd_warm_avg = math.sqrt(naive_crmsd_warm_avg_square)
    					naive_noJRA_crmsd_warm_avg_square = (naive_noJRA_var_warm_avg**2 + stn_var_warm_avg**2) - 2*(naive_noJRA_var_warm_avg*stn_var_warm_avg*naive_noJRA_corr_warm_avg)
    					naive_noJRA_crmsd_warm_avg = math.sqrt(naive_noJRA_crmsd_warm_avg_square)
    					naive_noJRAold_crmsd_warm_avg_square = (naive_noJRAold_var_warm_avg**2 + stn_var_warm_avg**2) - 2*(naive_noJRAold_var_warm_avg*stn_var_warm_avg*naive_noJRAold_corr_warm_avg)
    					naive_noJRAold_crmsd_warm_avg = math.sqrt(naive_noJRAold_crmsd_warm_avg_square)
    					naive_all_crmsd_warm_avg_square = (naive_all_var_warm_avg**2 + stn_var_warm_avg**2) - 2*(naive_all_var_warm_avg*stn_var_warm_avg*naive_all_corr_warm_avg)
    					naive_all_crmsd_warm_avg = math.sqrt(naive_all_crmsd_warm_avg_square)
    					CFSR_crmsd_warm_avg_square = (CFSR_var_warm_avg**2 + stn_var_warm_avg**2) - 2*(CFSR_var_warm_avg*stn_var_warm_avg*CFSR_corr_warm_avg)
    					CFSR_crmsd_warm_avg = math.sqrt(CFSR_crmsd_warm_avg_square)
    					ERAI_crmsd_warm_avg_square = (ERAI_var_warm_avg**2 + stn_var_warm_avg**2) - 2*(ERAI_var_warm_avg*stn_var_warm_avg*ERAI_corr_warm_avg)
    					ERAI_crmsd_warm_avg = math.sqrt(ERAI_crmsd_warm_avg_square)
    					ERA5_crmsd_warm_avg_square = (ERA5_var_warm_avg**2 + stn_var_warm_avg**2) - 2*(ERA5_var_warm_avg*stn_var_warm_avg*ERA5_corr_warm_avg)
    					ERA5_crmsd_warm_avg = math.sqrt(ERA5_crmsd_warm_avg_square)
    					ERA5_Land_crmsd_warm_avg_square = (ERA5_Land_var_warm_avg**2 + stn_var_warm_avg**2) - 2*(ERA5_Land_var_warm_avg*stn_var_warm_avg*ERA5_Land_corr_warm_avg)
    					ERA5_Land_crmsd_warm_avg = math.sqrt(ERA5_Land_crmsd_warm_avg_square)
    					JRA_crmsd_warm_avg_square = (JRA_var_warm_avg**2 + JRA_var_warm_avg**2) - 2*(JRA_var_warm_avg*JRA_var_warm_avg*JRA_corr_warm_avg)
    					JRA_crmsd_warm_avg = math.sqrt(JRA_crmsd_warm_avg_square)
    					MERRA2_crmsd_warm_avg_square = (MERRA2_var_warm_avg**2 + MERRA2_var_warm_avg**2) - 2*(MERRA2_var_warm_avg*MERRA2_var_warm_avg*MERRA2_corr_warm_avg)
    					MERRA2_crmsd_warm_avg = math.sqrt(MERRA2_crmsd_warm_avg_square)
    					GLDAS_crmsd_warm_avg_square = (GLDAS_var_warm_avg**2 + GLDAS_var_warm_avg**2) - 2*(GLDAS_var_warm_avg*GLDAS_var_warm_avg*GLDAS_corr_warm_avg)
    					GLDAS_crmsd_warm_avg = math.sqrt(GLDAS_crmsd_warm_avg_square)
    					GLDAS_CLSM_crmsd_warm_avg_square = (GLDAS_CLSM_var_warm_avg**2 + GLDAS_CLSM_var_warm_avg**2) - 2*(GLDAS_CLSM_var_warm_avg*GLDAS_CLSM_var_warm_avg*GLDAS_CLSM_corr_warm_avg)
    					GLDAS_CLSM_crmsd_warm_avg = math.sqrt(GLDAS_CLSM_crmsd_warm_avg_square)


##################### Create Taylor Diagrams ####################

    					print('Remap Style:',remap_type)
    					print('Layer:',lyr_l)
    					print('Temp Threshold:',temp_thr_n)


    					taylor_dir = '/mnt/data/users/herringtont/soil_temp/plots/taylor_diagrams/new_data/CMOS_poster/CLSM_res/'
    					taylor_plt_fil_warm = ''.join([taylor_dir,str(remap_type)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_taylor_diagram_warm_new_data_CMOS_CLSM_grid_cell_BEST.png'])
		

    					stn_norm_sdev_warm_avg = stn_sdev_warm_avg/stn_sdev_warm_avg
    					naive_norm_sdev_warm_avg = naive_sdev_warm_avg/stn_sdev_warm_avg
    					naive_noJRA_norm_sdev_warm_avg = naive_noJRA_sdev_warm_avg/stn_sdev_warm_avg
    					naive_noJRAold_norm_sdev_warm_avg = naive_noJRAold_sdev_warm_avg/stn_sdev_warm_avg
    					naive_all_norm_sdev_warm_avg = naive_all_sdev_warm_avg/stn_sdev_warm_avg					
    					CFSR_norm_sdev_warm_avg = CFSR_sdev_warm_avg/stn_sdev_warm_avg
    					ERAI_norm_sdev_warm_avg = ERAI_sdev_warm_avg/stn_sdev_warm_avg
    					ERA5_norm_sdev_warm_avg = ERA5_sdev_warm_avg/stn_sdev_warm_avg
    					ERA5_Land_norm_sdev_warm_avg = ERA5_Land_sdev_warm_avg/stn_sdev_warm_avg
    					JRA_norm_sdev_warm_avg = JRA_sdev_warm_avg/stn_sdev_warm_avg
    					MERRA2_norm_sdev_warm_avg = MERRA2_sdev_warm_avg/stn_sdev_warm_avg
    					GLDAS_norm_sdev_warm_avg = GLDAS_sdev_warm_avg/stn_sdev_warm_avg
    					GLDAS_CLSM_norm_sdev_warm_avg = GLDAS_CLSM_sdev_warm_avg/stn_sdev_warm_avg

    					sdev_naive = np.array([stn_norm_sdev_warm_avg,naive_norm_sdev_warm_avg])
    					crmsd_naive = np.array([stn_crmsd_warm_avg,naive_crmsd_warm_avg])
    					ccoef_naive = np.array([stn_corr_warm_avg,naive_corr_warm_avg])

    					print('Naive std dev:',sdev_naive)
    					print('Naive CRMSD:',crmsd_naive)
    					print('Naive corr:',ccoef_naive)

    					sdev_naive_noJRA = np.array([stn_norm_sdev_warm_avg,naive_noJRA_norm_sdev_warm_avg])
    					crmsd_naive_noJRA = np.array([stn_crmsd_warm_avg,naive_noJRA_crmsd_warm_avg])
    					ccoef_naive_noJRA = np.array([stn_corr_warm_avg,naive_noJRA_corr_warm_avg])

    					sdev_naive_noJRAold = np.array([stn_norm_sdev_warm_avg,naive_noJRAold_norm_sdev_warm_avg])
    					crmsd_naive_noJRAold = np.array([stn_crmsd_warm_avg,naive_noJRAold_crmsd_warm_avg])
    					ccoef_naive_noJRAold = np.array([stn_corr_warm_avg,naive_noJRAold_corr_warm_avg])

    					sdev_naive_all = np.array([stn_norm_sdev_warm_avg,naive_all_norm_sdev_warm_avg])
    					crmsd_naive_all = np.array([stn_crmsd_warm_avg,naive_all_crmsd_warm_avg])
    					ccoef_naive_all = np.array([stn_corr_warm_avg,naive_all_corr_warm_avg])

    					sdev_CFSR = np.array([stn_norm_sdev_warm_avg,CFSR_norm_sdev_warm_avg])
    					crmsd_CFSR = np.array([stn_crmsd_warm_avg,CFSR_crmsd_warm_avg])
    					ccoef_CFSR = np.array([stn_corr_warm_avg,CFSR_corr_warm_avg])

    					sdev_ERAI = np.array([stn_norm_sdev_warm_avg,ERAI_norm_sdev_warm_avg])
    					crmsd_ERAI = np.array([stn_crmsd_warm_avg,ERAI_crmsd_warm_avg])
    					ccoef_ERAI = np.array([stn_corr_warm_avg,ERAI_corr_warm_avg])

    					sdev_ERA5 = np.array([stn_norm_sdev_warm_avg,ERA5_norm_sdev_warm_avg])
    					crmsd_ERA5 = np.array([stn_crmsd_warm_avg,ERA5_crmsd_warm_avg])
    					ccoef_ERA5 = np.array([stn_corr_warm_avg,ERA5_corr_warm_avg])

    					sdev_ERA5_Land = np.array([stn_norm_sdev_warm_avg,ERA5_Land_norm_sdev_warm_avg])
    					crmsd_ERA5_Land = np.array([stn_crmsd_warm_avg,ERA5_Land_crmsd_warm_avg])
    					ccoef_ERA5_Land = np.array([stn_corr_warm_avg,ERA5_Land_corr_warm_avg])

    					sdev_JRA = np.array([stn_norm_sdev_warm_avg,JRA_norm_sdev_warm_avg])
    					crmsd_JRA = np.array([stn_crmsd_warm_avg,JRA_crmsd_warm_avg])
    					ccoef_JRA = np.array([stn_corr_warm_avg,JRA_corr_warm_avg])

    					sdev_MERRA2 = np.array([stn_norm_sdev_warm_avg,MERRA2_norm_sdev_warm_avg])
    					crmsd_MERRA2 = np.array([stn_crmsd_warm_avg,MERRA2_crmsd_warm_avg])
    					ccoef_MERRA2 = np.array([stn_corr_warm_avg,MERRA2_corr_warm_avg])

    					sdev_GLDAS = np.array([stn_norm_sdev_warm_avg,GLDAS_norm_sdev_warm_avg])
    					crmsd_GLDAS = np.array([stn_crmsd_warm_avg,GLDAS_crmsd_warm_avg])
    					ccoef_GLDAS = np.array([stn_corr_warm_avg,GLDAS_corr_warm_avg])

    					sdev_GLDAS_CLSM = np.array([stn_norm_sdev_warm_avg,GLDAS_CLSM_norm_sdev_warm_avg])
    					crmsd_GLDAS_CLSM = np.array([stn_crmsd_warm_avg,GLDAS_CLSM_crmsd_warm_avg])
    					ccoef_GLDAS_CLSM = np.array([stn_corr_warm_avg,GLDAS_CLSM_corr_warm_avg])

    					label = {'Station':'dimgrey','Ensemble Mean':'fuchsia','CFSR': 'm','ERA-Interim': 'limegreen', 'ERA5': 'cyan','ERA5-Land':'skyblue','JRA55': 'red', 'MERRA2': 'goldenrod', 'GLDAS-Noah': 'black','GLDAS-CLSM':'darkgrey'}
								
    					#sm.taylor_diagram(sdev_naive,crmsd_naive,ccoef_naive,tickSTD=[0,0.25,0.5,0.75,1,1.25,1.5,1.75,2],axismax=2.0,styleSTD = '-.', widthSTD = 0.25,styleOBS = '-',tickRMS=[0.0],titleRMS='off',tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5,colOBS = 'dimgrey',markerobs = 'o',titleOBS = 'Station', markercolor='dodgerblue', markerLabelcolor='dodgerblue')
    					sm.taylor_diagram(sdev_naive_all,crmsd_naive_all,ccoef_naive_all,tickSTD=[0,0.25,0.5,0.75,1,1.25,1.5,1.75,2],axismax=2.0,styleSTD = '-.', widthSTD = 0.25,styleOBS = '-',tickRMS=[0.0],titleRMS='off',tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5,colOBS = 'dimgrey',markerobs = 'o',titleOBS = 'Station', markercolor='hotpink', markerLabelcolor='hotpink', markerSize=15,showlabelsRMS = 'off',alpha=0.6)
    					#sm.taylor_diagram(sdev_naive_noJRAold,crmsd_naive_noJRAold,ccoef_naive_noJRAold,tickSTD=[0,0.25,0.5,0.75,1,1.25,1.5,1.75,2],axismax=2.0,styleSTD = '-.', widthSTD = 0.25,styleOBS = '-',tickRMS=[0.0],titleRMS='off',tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='mediumslateblue', markerLabelcolor='mediumslateblue',overlay='on')
    					#sm.taylor_diagram(sdev_naive_all,crmsd_naive_all,ccoef_naive_all,tickSTD=[0,0.25,0.5,0.75,1,1.25,1.5,1.75,2],axismax=2.0,styleSTD = '-.', widthSTD = 0.25,styleOBS = '-',tickRMS=[0.0],titleRMS='off',tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='fuchsia', markerLabelcolor='fuchsia',overlay='on')
    					#sm.taylor_diagram(sdev_naive_noJRA,crmsd_naive_noJRA,ccoef_naive_noJRA,tickSTD=[0,0.25,0.5,0.75,1,1.25,1.5,1.75,2],axismax=2.0,styleSTD = '-.', widthSTD = 0.25,styleOBS = '-',tickRMS=[0.0],titleRMS='off',tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='coral', markerLabelcolor='coral',overlay='on')
    					sm.taylor_diagram(sdev_CFSR,crmsd_CFSR,ccoef_CFSR,tickSTD=[0,0.25,0.5,0.75,1,1.25,1.5,1.75,2],axismax=2.0,styleSTD = '-.', widthSTD = 0.25,styleOBS = '-',tickRMS=[0.0],titleRMS='off',tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='darkviolet', markerLabelcolor='darkviolet',markerSize=15,showlabelsRMS = 'off',alpha=0.6, overlay='on')
    					sm.taylor_diagram(sdev_ERAI,crmsd_ERAI,ccoef_ERAI,tickSTD=[0,0.25,0.5,0.75,1,1.25,1.5,1.75,2],axismax=2.0,styleSTD = '-.', widthSTD = 0.25,styleOBS = '-',tickRMS=[0.0],titleRMS='off',tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='springgreen', markerLabelcolor='springgreen',markerSize=15,showlabelsRMS = 'off',alpha=0.6,overlay='on')
    					sm.taylor_diagram(sdev_ERA5,crmsd_ERA5,ccoef_ERA5,tickSTD=[0,0.25,0.5,0.75,1,1.25,1.5,1.75,2],axismax=2.0,styleSTD = '-.', widthSTD = 0.25,styleOBS = '-',tickRMS=[0.0],titleRMS='off',tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='greenyellow', markerLabelcolor='greenyellow',markerSize=15,showlabelsRMS = 'off',alpha=0.6,overlay='on')
    					sm.taylor_diagram(sdev_ERA5_Land,crmsd_ERA5_Land,ccoef_ERA5_Land,tickSTD=[0,0.25,0.5,0.75,1,1.25,1.5,1.75,2],axismax=2.0,styleSTD = '-.', widthSTD = 0.25,styleOBS = '-',tickRMS=[0.0],titleRMS='off',tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='dodgerblue',markerLabelcolor='dodgerblue',markerSize=15,showlabelsRMS = 'off',alpha=0.6,overlay='on')
    					sm.taylor_diagram(sdev_JRA,crmsd_JRA,ccoef_JRA,tickSTD=[0,0.25,0.5,0.75,1,1.25,1.5,1.75,2],axismax=2.0,styleSTD = '-.', widthSTD = 0.25,styleOBS = '-',tickRMS=[0.0],titleRMS='off',tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='red',markerLabelcolor='red',markerSize=15,showlabelsRMS = 'off',alpha=0.6,overlay='on')
    					sm.taylor_diagram(sdev_MERRA2,crmsd_MERRA2,ccoef_MERRA2,tickSTD=[0,0.25,0.5,0.75,1,1.25,1.5,1.75,2],axismax=2.0,styleSTD = '-.', widthSTD = 0.25,styleOBS = '-',tickRMS=[0.0],titleRMS='off',tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='goldenrod',markerLabelcolor='goldenrod',markerSize=15,showlabelsRMS = 'off',alpha=0.6,overlay='on')
    					sm.taylor_diagram(sdev_GLDAS,crmsd_GLDAS,ccoef_GLDAS,tickSTD=[0,0.25,0.5,0.75,1,1.25,1.5,1.75,2],axismax=2.0,styleSTD = '-.', widthSTD = 0.25,styleOBS = '-',tickRMS=[0.0],titleRMS='off',tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='black',markerLabelcolor='black',markerSize=15,showlabelsRMS = 'off',alpha=0.6,overlay='on')
    					#sm.taylor_diagram(sdev_GLDAS_CLSM,crmsd_GLDAS_CLSM,ccoef_GLDAS_CLSM,tickSTD=[0,0.25,0.5,0.75,1,1.25,1.5,1.75,2],axismax=2.0,styleSTD = '-.', widthSTD = 0.25,styleOBS = '-',tickRMS=[0.0],titleRMS='off',tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='darkgrey',markerLabelcolor='darkgrey',markerSize=15,alpha=0.6,overlay='on',markerLabel=label)
    					sm.taylor_diagram(sdev_GLDAS_CLSM,crmsd_GLDAS_CLSM,ccoef_GLDAS_CLSM,tickSTD=[0,0.25,0.5,0.75,1,1.25,1.5,1.75,2],axismax=2.0,styleSTD = '-.', widthSTD = 0.25,styleOBS = '-',tickRMS=[0.0],titleRMS='off',tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='darkgrey',markerLabelcolor='darkgrey',markerSize=15,showlabelsRMS = 'off',alpha=0.6,overlay='on')


    					plt.savefig(taylor_plt_fil_warm)
    					plt.close()
    					print(taylor_plt_fil_warm)

































