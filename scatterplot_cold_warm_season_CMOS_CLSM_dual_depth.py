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

naive_type = 'simple_average'
olr = 'zscore'#['outliers','zscore','IQR']
lyr = ['top_30cm','30cm_300cm']
thr = '100'#['0','25','50','75','100']
remap_type = 'remapcon'#['nn','bil','con']
temp_thr = '-2C'#['0C','-2C','-5C','-10C']

scatter_fil_top = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/CLSM_res/subset/'+str(remap_type)+'_'+str(naive_type)+'_'+str(olr)+'_top_30cm_thr_'+str(thr)+'_dframe_scatterplot_ERA5_'+str(temp_thr)+'_CMOS_CLSM_subset.csv'])
scatter_fil_btm = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/CLSM_res/subset/'+str(remap_type)+'_'+str(naive_type)+'_'+str(olr)+'_30cm_300cm_thr_'+str(thr)+'_dframe_scatterplot_ERA5_'+str(temp_thr)+'_CMOS_CLSM_subset.csv'])

dframe_scatter_top = pd.read_csv(scatter_fil_top)
dframe_scatter_btm = pd.read_csv(scatter_fil_btm)

naive_all_scatter_top = dframe_scatter_top['Naive Blend All'].values
station_scatter_top = dframe_scatter_top['Station'].values
CFSR_scatter_top = dframe_scatter_top['CFSR'].values
ERAI_scatter_top = dframe_scatter_top['ERA-Interim'].values
ERA5_scatter_top = dframe_scatter_top['ERA5'].values
ERA5_Land_scatter_top = dframe_scatter_top['ERA5-Land'].values
JRA_scatter_top = dframe_scatter_top['JRA55'].values
MERRA2_scatter_top = dframe_scatter_top['MERRA2'].values
GLDAS_scatter_top = dframe_scatter_top['GLDAS-Noah'].values
GLDAS_CLSM_scatter_top = dframe_scatter_top['GLDAS-CLSM'].values
season_scatter_top = dframe_scatter_top['Season'].values

naive_all_scatter_btm = dframe_scatter_btm['Naive Blend All'].values
station_scatter_btm = dframe_scatter_btm['Station'].values
CFSR_scatter_btm = dframe_scatter_btm['CFSR'].values
ERAI_scatter_btm = dframe_scatter_btm['ERA-Interim'].values
ERA5_scatter_btm = dframe_scatter_btm['ERA5'].values
ERA5_Land_scatter_btm = dframe_scatter_btm['ERA5-Land'].values
JRA_scatter_btm = dframe_scatter_btm['JRA55'].values
MERRA2_scatter_btm = dframe_scatter_btm['MERRA2'].values
GLDAS_scatter_btm = dframe_scatter_btm['GLDAS-Noah'].values
GLDAS_CLSM_scatter_btm = dframe_scatter_btm['GLDAS-CLSM'].values
season_scatter_btm = dframe_scatter_btm['Season'].values

station_combined = []
station_combined.append(station_scatter_top)
station_combined.append(station_scatter_btm)
station_combined = [i for sub in station_combined for i in sub]
naive_all_combined = []
naive_all_combined.append(naive_all_scatter_top)
naive_all_combined.append(naive_all_scatter_btm)
naive_all_combined = [i for sub in naive_all_combined for i in sub]
CFSR_combined = []
CFSR_combined.append(CFSR_scatter_top)
CFSR_combined.append(CFSR_scatter_btm)
CFSR_combined = [i for sub in CFSR_combined for i in sub]
ERAI_combined = []
ERAI_combined.append(ERAI_scatter_top)
ERAI_combined.append(ERAI_scatter_btm)
ERAI_combined = [i for sub in ERAI_combined for i in sub]
ERA5_combined = []
ERA5_combined.append(ERA5_scatter_top)
ERA5_combined.append(ERA5_scatter_btm)
ERA5_combined = [i for sub in ERA5_combined for i in sub]
ERA5_Land_combined = []
ERA5_Land_combined.append(ERA5_Land_scatter_top)
ERA5_Land_combined.append(ERA5_Land_scatter_btm)
ERA5_Land_combined = [i for sub in ERA5_Land_combined for i in sub]
JRA_combined = []
JRA_combined.append(JRA_scatter_top)
JRA_combined.append(JRA_scatter_btm)
JRA_combined = [i for sub in JRA_combined for i in sub]
MERRA2_combined = []
MERRA2_combined.append(MERRA2_scatter_top)
MERRA2_combined.append(MERRA2_scatter_btm)
MERRA2_combined = [i for sub in MERRA2_combined for i in sub]
GLDAS_combined = []
GLDAS_combined.append(GLDAS_scatter_top)
GLDAS_combined.append(GLDAS_scatter_btm)
GLDAS_combined = [i for sub in GLDAS_combined for i in sub]
GLDAS_CLSM_combined = []
GLDAS_CLSM_combined.append(GLDAS_CLSM_scatter_top)
GLDAS_CLSM_combined.append(GLDAS_CLSM_scatter_btm)
GLDAS_CLSM_combined = [i for sub in GLDAS_CLSM_combined for i in sub]
season_combined = []
season_combined.append(season_scatter_top)
season_combined.append(season_scatter_btm)
season_combined = [i for sub in season_combined for i in sub]


depth_combined = []
len_top = len(station_scatter_top)
len_btm = len(station_scatter_btm)

for i in range(0,len_top):
    depth_combined.append('top-30cm')

for j in range(0,len_btm):
    depth_combined.append('30cm - 300cm')
    
dframe_combined = pd.DataFrame(data = station_combined, columns=['Station'])
dframe_combined['CFSR'] = CFSR_combined
dframe_combined['ERA-Interim'] = ERAI_combined
dframe_combined['ERA5'] = ERA5_combined
dframe_combined['ERA5-Land'] = ERA5_Land_combined
dframe_combined['JRA55'] = JRA_combined
dframe_combined['MERRA2'] = MERRA2_combined
dframe_combined['GLDAS'] = GLDAS_combined
dframe_combined['GLDAS-CLSM'] = GLDAS_CLSM_combined
dframe_combined['Season'] = season_combined
dframe_combined['Depth'] = depth_combined

print(dframe_combined)

scatter1 = sns.PairGrid(dframe_combined)
scatter1.map_upper(sn.pairplot(dframe_combined[dframe_combined['Depth'] == 'top-30cm'],hue='Season'))
scatter1.map_lower(sn.pairplot(dframe_combined[dframe_combined['Depth'] == '30cm - 300cm'],hue='Season'))
plt.show()
#dframe_scatter_top2 = pd.DataFrame({'Station':station_scatter_top,'Season':season_scatter_top,'Ensemble Mean':naive_all_scatter_top,'CFSR':CFSR_scatter_top,'ERA-Interim':ERAI_scatter_top,'ERA5':ERA5_scatter_top,'ERA5-Land':ERA5_Land_scatter_top,'JRA55':JRA_scatter_top,'MERRA2':MERRA2_scatter_top,'GLDAS-Noah':GLDAS_scatter_top,'GLDAS-CLSM':GLDAS_CLSM_scatter_top})
#dframe_scatter_btm2 = pd.DataFrame({'Station':station_scatter_btm,'Season':season_scatter_btm,'Ensemble Mean':naive_all_scatter_btm,'CFSR':CFSR_scatter_btm,'ERA-Interim':ERAI_scatter_btm,'ERA5':ERA5_scatter_btm,'ERA5-Land':ERA5_Land_scatter_btm,'JRA55':JRA_scatter_btm,'MERRA2':MERRA2_scatter_btm,'GLDAS-Noah':GLDAS_scatter_btm,'GLDAS-CLSM':GLDAS_CLSM_scatter_btm})
#
#
############# Create Scatterplot Matrix ##############
#
#concatenated = pd.concat([(dframe_scatter_top2.assign(dataset='set1'), dframe_scatter_btm2.assign(dataset='set2')])
#scatter1.map_upper(sn.pairplot(dframe_scatter_top2,hue='Season'))
#scatter1.map_lower(sn.pairplot(dframe_scatter_btm2,hue='Season'))
#
#plt.show()
