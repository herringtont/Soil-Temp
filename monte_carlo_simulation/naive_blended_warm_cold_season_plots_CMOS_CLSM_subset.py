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
from random import sample


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
temp_thr = ['-2C']#['0C','-2C','-5C','-10C']

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
    
    				for n in temp_thr:
    					temp_thr_n = n
    					cold_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/CLSM_res/subset/'+str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_dframe_cold_season_temp_master_ERA5_'+str(temp_thr_n)+'_CMOS_CLSM_subset.csv'])
    					warm_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/CLSM_res/subset/'+str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_dframe_warm_season_temp_master_ERA5_'+str(temp_thr_n)+'_CMOS_CLSM_subset.csv'])    			
    					scatter_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/CLSM_res/subset/'+str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_dframe_scatterplot_ERA5_'+str(temp_thr_n)+'_CMOS_CLSM_subset.csv'])

##### Calculate Metrics Over Entire Dataset ######

    					dframe_cold = pd.read_csv(cold_fil)
    					dframe_warm = pd.read_csv(warm_fil)
					


#### Cold Season ####


    					station_temp_cold_all = dframe_cold['Station'].values
    					naive_all_temp_cold_all = dframe_cold['Naive Blend All'].values
    					CFSR_temp_cold_all = dframe_cold['CFSR'].values
    					ERAI_temp_cold_all = dframe_cold['ERA-Interim'].values
    					ERA5_temp_cold_all = dframe_cold['ERA5'].values
    					ERA5_Land_temp_cold_all = dframe_cold['ERA5-Land'].values
    					JRA_temp_cold_all = dframe_cold['JRA55'].values
    					MERRA2_temp_cold_all = dframe_cold['MERRA2'].values
    					GLDAS_temp_cold_all = dframe_cold['GLDAS-Noah'].values
    					GLDAS_CLSM_temp_cold_all = dframe_cold['GLDAS-CLSM'].values

## Bias ##

    					naive_all_bias_cold_all = bias(naive_all_temp_cold_all, station_temp_cold_all)
    					CFSR_bias_cold_all = bias(CFSR_temp_cold_all, station_temp_cold_all)
    					ERAI_bias_cold_all = bias(ERAI_temp_cold_all, station_temp_cold_all)
    					ERA5_bias_cold_all = bias(ERA5_temp_cold_all, station_temp_cold_all)
    					ERA5_Land_bias_cold_all = bias(ERA5_Land_temp_cold_all, station_temp_cold_all)
    					JRA_bias_cold_all = bias(JRA_temp_cold_all, station_temp_cold_all)
    					MERRA2_bias_cold_all = bias(MERRA2_temp_cold_all, station_temp_cold_all)
    					GLDAS_bias_cold_all = bias(GLDAS_temp_cold_all, station_temp_cold_all)
    					GLDAS_CLSM_bias_cold_all = bias(GLDAS_CLSM_temp_cold_all, station_temp_cold_all)

## STD DEV ##

    					stn_sdev_cold_all =  np.std(station_temp_cold_all)
    					naive_all_sdev_cold_all = np.std(naive_all_temp_cold_all)					
    					CFSR_sdev_cold_all = np.std(CFSR_temp_cold_all)
    					ERAI_sdev_cold_all = np.std(ERAI_temp_cold_all)    					
    					ERA5_sdev_cold_all = np.std(ERA5_temp_cold_all)
    					ERA5_Land_sdev_cold_all = np.std(ERA5_Land_temp_cold_all)
    					JRA_sdev_cold_all = np.std(JRA_temp_cold_all)
    					MERRA2_sdev_cold_all = np.std(MERRA2_temp_cold_all)
    					GLDAS_sdev_cold_all = np.std(GLDAS_temp_cold_all)
    					GLDAS_CLSM_sdev_cold_all = np.std(GLDAS_CLSM_temp_cold_all)



## RMSE and ubRMSE ##

    					naive_all_rmse_cold_all = mean_squared_error(station_temp_cold_all,naive_all_temp_cold_all, squared=False)
    					CFSR_rmse_cold_all = mean_squared_error(station_temp_cold_all,CFSR_temp_cold_all, squared=False)
    					ERAI_rmse_cold_all = mean_squared_error(station_temp_cold_all,ERAI_temp_cold_all, squared=False)
    					ERA5_rmse_cold_all = mean_squared_error(station_temp_cold_all,ERA5_temp_cold_all, squared=False)
    					ERA5_Land_rmse_cold_all = mean_squared_error(station_temp_cold_all,ERA5_Land_temp_cold_all, squared=False)
    					JRA_rmse_cold_all = mean_squared_error(station_temp_cold_all,JRA_temp_cold_all, squared=False)
    					MERRA2_rmse_cold_all = mean_squared_error(station_temp_cold_all,MERRA2_temp_cold_all, squared=False)
    					GLDAS_rmse_cold_all = mean_squared_error(station_temp_cold_all,GLDAS_temp_cold_all, squared=False)
    					GLDAS_CLSM_rmse_cold_all = mean_squared_error(station_temp_cold_all,GLDAS_CLSM_temp_cold_all, squared=False)


## Pearson Correlations ##
    					naive_all_corr_cold_all,_ = pearsonr(naive_all_temp_cold_all, station_temp_cold_all)
    					CFSR_corr_cold_all,_ = pearsonr(CFSR_temp_cold_all, station_temp_cold_all)
    					ERAI_corr_cold_all,_ = pearsonr(ERAI_temp_cold_all, station_temp_cold_all)
    					ERA5_corr_cold_all,_ = pearsonr(ERA5_temp_cold_all, station_temp_cold_all)
    					ERA5_Land_corr_cold_all,_ = pearsonr(ERA5_Land_temp_cold_all, station_temp_cold_all)
    					JRA_corr_cold_all,_ = pearsonr(JRA_temp_cold_all, station_temp_cold_all)
    					MERRA2_corr_cold_all,_ = pearsonr(MERRA2_temp_cold_all, station_temp_cold_all)
    					GLDAS_corr_cold_all,_ = pearsonr(GLDAS_temp_cold_all, station_temp_cold_all)
    					GLDAS_CLSM_corr_cold_all,_ = pearsonr(GLDAS_CLSM_temp_cold_all, station_temp_cold_all)


#### Warm Season ####


    					station_temp_warm_all = dframe_warm['Station'].values
    					naive_all_temp_warm_all = dframe_warm['Naive Blend All'].values
    					CFSR_temp_warm_all = dframe_warm['CFSR'].values
    					ERAI_temp_warm_all = dframe_warm['ERA-Interim'].values
    					ERA5_temp_warm_all = dframe_warm['ERA5'].values
    					ERA5_Land_temp_warm_all = dframe_warm['ERA5-Land'].values
    					JRA_temp_warm_all = dframe_warm['JRA55'].values
    					MERRA2_temp_warm_all = dframe_warm['MERRA2'].values
    					GLDAS_temp_warm_all = dframe_warm['GLDAS-Noah'].values
    					GLDAS_CLSM_temp_warm_all = dframe_warm['GLDAS-CLSM'].values

## Bias ##

    					naive_all_bias_warm_all = bias(naive_all_temp_warm_all, station_temp_warm_all)
    					CFSR_bias_warm_all = bias(CFSR_temp_warm_all, station_temp_warm_all)
    					ERAI_bias_warm_all = bias(ERAI_temp_warm_all, station_temp_warm_all)
    					ERA5_bias_warm_all = bias(ERA5_temp_warm_all, station_temp_warm_all)
    					ERA5_Land_bias_warm_all = bias(ERA5_Land_temp_warm_all, station_temp_warm_all)
    					JRA_bias_warm_all = bias(JRA_temp_warm_all, station_temp_warm_all)
    					MERRA2_bias_warm_all = bias(MERRA2_temp_warm_all, station_temp_warm_all)
    					GLDAS_bias_warm_all = bias(GLDAS_temp_warm_all, station_temp_warm_all)
    					GLDAS_CLSM_bias_warm_all = bias(GLDAS_CLSM_temp_warm_all, station_temp_warm_all)

## STD DEV ##

    					stn_sdev_warm_all =  np.std(station_temp_warm_all)
    					naive_all_sdev_warm_all = np.std(naive_all_temp_warm_all)					
    					CFSR_sdev_warm_all = np.std(CFSR_temp_warm_all)
    					ERAI_sdev_warm_all = np.std(ERAI_temp_warm_all)    					
    					ERA5_sdev_warm_all = np.std(ERA5_temp_warm_all)
    					ERA5_Land_sdev_warm_all = np.std(ERA5_Land_temp_warm_all)
    					JRA_sdev_warm_all = np.std(JRA_temp_warm_all)
    					MERRA2_sdev_warm_all = np.std(MERRA2_temp_warm_all)
    					GLDAS_sdev_warm_all = np.std(GLDAS_temp_warm_all)
    					GLDAS_CLSM_sdev_warm_all = np.std(GLDAS_CLSM_temp_warm_all)



## RMSE and ubRMSE ##

    					naive_all_rmse_warm_all = mean_squared_error(station_temp_warm_all,naive_all_temp_warm_all, squared=False)
    					CFSR_rmse_warm_all = mean_squared_error(station_temp_warm_all,CFSR_temp_warm_all, squared=False)
    					ERAI_rmse_warm_all = mean_squared_error(station_temp_warm_all,ERAI_temp_warm_all, squared=False)
    					ERA5_rmse_warm_all = mean_squared_error(station_temp_warm_all,ERA5_temp_warm_all, squared=False)
    					ERA5_Land_rmse_warm_all = mean_squared_error(station_temp_warm_all,ERA5_Land_temp_warm_all, squared=False)
    					JRA_rmse_warm_all = mean_squared_error(station_temp_warm_all,JRA_temp_warm_all, squared=False)
    					MERRA2_rmse_warm_all = mean_squared_error(station_temp_warm_all,MERRA2_temp_warm_all, squared=False)
    					GLDAS_rmse_warm_all = mean_squared_error(station_temp_warm_all,GLDAS_temp_warm_all, squared=False)
    					GLDAS_CLSM_rmse_warm_all = mean_squared_error(station_temp_warm_all,GLDAS_CLSM_temp_warm_all, squared=False)


## Pearson Correlations ##
    					naive_all_corr_warm_all,_ = pearsonr(naive_all_temp_warm_all, station_temp_warm_all)
    					CFSR_corr_warm_all,_ = pearsonr(CFSR_temp_warm_all, station_temp_warm_all)
    					ERAI_corr_warm_all,_ = pearsonr(ERAI_temp_warm_all, station_temp_warm_all)
    					ERA5_corr_warm_all,_ = pearsonr(ERA5_temp_warm_all, station_temp_warm_all)
    					ERA5_Land_corr_warm_all,_ = pearsonr(ERA5_Land_temp_warm_all, station_temp_warm_all)
    					JRA_corr_warm_all,_ = pearsonr(JRA_temp_warm_all, station_temp_warm_all)
    					MERRA2_corr_warm_all,_ = pearsonr(MERRA2_temp_warm_all, station_temp_warm_all)
    					GLDAS_corr_warm_all,_ = pearsonr(GLDAS_temp_warm_all, station_temp_warm_all)
    					GLDAS_CLSM_corr_warm_all,_ = pearsonr(GLDAS_CLSM_temp_warm_all, station_temp_warm_all)





######### Extract Eurasian and North American/Greenland Grid Cells ##########

    					dframe_cold_NAG = dframe_cold[dframe_cold['Continent'] == "North_America"]
    					grid_cell_NAG_cold = dframe_cold_NAG['Grid Cell'].values
    					grid_cell_NAG_uq_cold = np.unique(grid_cell_NAG_cold)
    					grid_cell_NAG_uq_cold = grid_cell_NAG_uq_cold.tolist()
    					len_gc_NAG_cold = len(grid_cell_NAG_uq_cold)
    					print("Number of N American Grid Cells:",len_gc_NAG_cold)
    					#print(grid_cell_NAG_uq_cold)


    					dframe_cold_Eur = dframe_cold[dframe_cold['Continent'] == 'Eurasia']
    					grid_cell_Eur_cold = dframe_cold_Eur['Grid Cell'].values
    					grid_cell_Eur_uq_cold = np.unique(grid_cell_Eur_cold)
    					grid_cell_Eur_uq_cold = grid_cell_Eur_uq_cold.tolist()
    					len_gc_Eur_cold = len(grid_cell_Eur_uq_cold)
    					print("Number of Eurasian Grid Cells:",len_gc_Eur_cold)
    					#print(grid_cell_Eur_uq_cold)


    					dframe_warm_NAG = dframe_warm[dframe_warm['Continent'] == "North_America"]
    					grid_cell_NAG_warm = dframe_warm_NAG['Grid Cell'].values
    					grid_cell_NAG_uq_warm = np.unique(grid_cell_NAG_warm)
    					grid_cell_NAG_uq_warm = grid_cell_NAG_uq_warm.tolist()
    					len_gc_NAG_warm = len(grid_cell_NAG_uq_warm)
    					print("Number of N American Grid Cells:",len_gc_NAG_warm)

    					dframe_warm_Eur = dframe_warm[dframe_warm['Continent'] == 'Eurasia']
    					grid_cell_Eur_warm = dframe_warm_Eur['Grid Cell'].values
    					grid_cell_Eur_uq_warm = np.unique(grid_cell_Eur_warm)
    					grid_cell_Eur_uq_warm = grid_cell_Eur_uq_warm.tolist()
    					len_gc_Eur_warm = len(grid_cell_Eur_uq_warm)
    					print("Number of Eurasian Grid Cells:",len_gc_Eur_warm)


############## Perform Monte-Carlo Simulation on Data (extracting subsample of Eurasian Sites) ####################

    					sample_size_cold = len_gc_NAG_cold
    					sample_size_warm = len_gc_NAG_warm

    					naive_all_bias_cold_master = []
    					CFSR_bias_cold_master = []
    					ERAI_bias_cold_master = []
    					ERA5_bias_cold_master = []
    					ERA5_Land_bias_cold_master = []
    					JRA_bias_cold_master = []
    					MERRA2_bias_cold_master = []
    					GLDAS_bias_cold_master = []
    					GLDAS_CLSM_bias_cold_master = []

    					naive_all_bias_warm_master = []
    					CFSR_bias_warm_master = []
    					ERAI_bias_warm_master = []
    					ERA5_bias_warm_master = []
    					ERA5_Land_bias_warm_master = []
    					JRA_bias_warm_master = []
    					MERRA2_bias_warm_master = []
    					GLDAS_bias_warm_master = []
    					GLDAS_CLSM_bias_warm_master = []

    					stn_sdev_cold_master = []
    					naive_all_sdev_cold_master = []
    					CFSR_sdev_cold_master = []
    					ERAI_sdev_cold_master = []
    					ERA5_sdev_cold_master = []
    					ERA5_Land_sdev_cold_master = []
    					JRA_sdev_cold_master = []
    					MERRA2_sdev_cold_master = []
    					GLDAS_sdev_cold_master = []
    					GLDAS_CLSM_sdev_cold_master = []

    					stn_sdev_warm_master = []
    					naive_all_sdev_warm_master = []
    					CFSR_sdev_warm_master = []
    					ERAI_sdev_warm_master = []
    					ERA5_sdev_warm_master = []
    					ERA5_Land_sdev_warm_master = []
    					JRA_sdev_warm_master = []
    					MERRA2_sdev_warm_master = []
    					GLDAS_sdev_warm_master = []
    					GLDAS_CLSM_sdev_warm_master = []

    					naive_all_rmse_cold_master = []
    					CFSR_rmse_cold_master = []
    					ERAI_rmse_cold_master = []
    					ERA5_rmse_cold_master = []
    					ERA5_Land_rmse_cold_master = []
    					JRA_rmse_cold_master = []
    					MERRA2_rmse_cold_master = []
    					GLDAS_rmse_cold_master = []
    					GLDAS_CLSM_rmse_cold_master = []

    					naive_all_rmse_warm_master = []
    					CFSR_rmse_warm_master = []
    					ERAI_rmse_warm_master = []
    					ERA5_rmse_warm_master = []
    					ERA5_Land_rmse_warm_master = []
    					JRA_rmse_warm_master = []
    					MERRA2_rmse_warm_master = []
    					GLDAS_rmse_warm_master = []
    					GLDAS_CLSM_rmse_warm_master = []

    					naive_all_corr_cold_master = []
    					CFSR_corr_cold_master = []
    					ERAI_corr_cold_master = []
    					ERA5_corr_cold_master = []
    					ERA5_Land_corr_cold_master = []
    					JRA_corr_cold_master = []
    					MERRA2_corr_cold_master = []
    					GLDAS_corr_cold_master = []
    					GLDAS_CLSM_corr_cold_master = []

    					naive_all_corr_warm_master = []
    					CFSR_corr_warm_master = []
    					ERAI_corr_warm_master = []
    					ERA5_corr_warm_master = []
    					ERA5_Land_corr_warm_master = []
    					JRA_corr_warm_master = []
    					MERRA2_corr_warm_master = []
    					GLDAS_corr_warm_master = []
    					GLDAS_CLSM_corr_warm_master = []

    					season_master = []
    					dataset_master = []

    					for a in range(0,90000):
    						season_master.append("Cold")
    					for b in range(0,90000):
    						season_master.append("Warm")

    					#season_master = [i for sub in season_master for i in sub]
    					#print(season_master)

    					season_master = np.array(season_master)

    					dataset_master = []
    					for aa in range(0,10000):
    						dataset_master.append("Naive-Blend")
    					for bb in range(0,10000):
    						dataset_master.append("CFSR")
    					for cc in range(0,10000):
    						dataset_master.append("ERA-Interim")
    					for dd in range(0,10000):
    						dataset_master.append("ERA5")
    					for ee in range(0,10000):
    						dataset_master.append("ERA5-Land")
    					for ff in range(0,10000):
    						dataset_master.append("JRA55")
    					for gg in range(0,10000):
    						dataset_master.append("MERRA2")
    					for hh in range(0,10000):
    						dataset_master.append("GLDAS-Noah")
    					for ii in range(0,10000):
    						dataset_master.append("GLDAS-CLSM")
    					for jj in range(0,10000):
    						dataset_master.append("Naive-Blend")
    					for kk in range(0,10000):
    						dataset_master.append("CFSR")
    					for ll in range(0,10000):
    						dataset_master.append("ERA-Interim")
    					for mm in range(0,10000):
    						dataset_master.append("ERA5")
    					for nn in range(0,10000):
    						dataset_master.append("ERA5-Land")
    					for oo in range(0,10000):
    						dataset_master.append("JRA55")
    					for pp in range(0,10000):
    						dataset_master.append("MERRA2")
    					for qq in range(0,10000):
    						dataset_master.append("GLDAS-Noah")
    					for rr in range(0,10000):
    						dataset_master.append("GLDAS-CLSM")

    					dataset_master = np.array(dataset_master)
    					#dataset_master = [i for sub in dataset_master for i in sub]
    					#print(dataset_master)
    					    					
    					for x in range(0,10000): #perform 1000 iterations
    						subset_Eur_gcell_cold = sample(grid_cell_Eur_uq_cold,len_gc_NAG_cold) #grab random sample of Eurasian grid cells with size N = number of North American/Greenland grid cells
    						subset_NAG_gcell_cold = grid_cell_NAG_uq_cold
    						subset_Eur_gcell_warm = subset_Eur_gcell_cold #use same subset of sites for warm season
    						subset_NAG_gcell_warm = grid_cell_NAG_uq_cold


    						station_temp_cold_master = []
    						naive_all_temp_cold_master = []
    						CFSR_temp_cold_master = []    						
    						ERAI_temp_cold_master = []
    						ERA5_temp_cold_master = []
    						ERA5_Land_temp_cold_master = []
    						JRA_temp_cold_master = []
    						MERRA2_temp_cold_master = []
    						GLDAS_temp_cold_master = []
    						GLDAS_CLSM_temp_cold_master = []

    						for y in range(0,len_gc_NAG_cold): #loop through data subsets (cold season) and store temperature information for only those grid cells
    							grid_cell_NAG_cold_y = subset_NAG_gcell_cold[y]
    							grid_cell_Eur_cold_y = subset_Eur_gcell_cold[y]

    							dframe_cold_gcell_NAG_y = dframe_cold[dframe_cold['Grid Cell'] == grid_cell_NAG_cold_y]
    							dframe_cold_gcell_Eur_y = dframe_cold[dframe_cold['Grid Cell'] == grid_cell_Eur_cold_y]

    							station_cold_NAG = dframe_cold_gcell_NAG_y['Station'].values.tolist()
    							naive_all_cold_NAG = dframe_cold_gcell_NAG_y['Naive Blend All'].values.tolist()
    							CFSR_cold_NAG = dframe_cold_gcell_NAG_y['CFSR'].values.tolist()
    							ERAI_cold_NAG = dframe_cold_gcell_NAG_y['ERA-Interim'].values.tolist()
    							ERA5_cold_NAG = dframe_cold_gcell_NAG_y['ERA5'].values.tolist()
    							ERA5_Land_cold_NAG = dframe_cold_gcell_NAG_y['ERA5-Land'].values.tolist()
    							JRA_cold_NAG = dframe_cold_gcell_NAG_y['JRA55'].values.tolist()
    							MERRA2_cold_NAG = dframe_cold_gcell_NAG_y['MERRA2'].values.tolist()
    							GLDAS_cold_NAG = dframe_cold_gcell_NAG_y['GLDAS-Noah'].values.tolist()
    							GLDAS_CLSM_cold_NAG = dframe_cold_gcell_NAG_y['GLDAS-CLSM'].values.tolist()

    							station_cold_Eur = dframe_cold_gcell_Eur_y['Station'].values.tolist()
    							naive_all_cold_Eur = dframe_cold_gcell_Eur_y['Naive Blend All'].values.tolist()
    							CFSR_cold_Eur = dframe_cold_gcell_Eur_y['CFSR'].values.tolist()
    							ERAI_cold_Eur = dframe_cold_gcell_Eur_y['ERA-Interim'].values.tolist()
    							ERA5_cold_Eur = dframe_cold_gcell_Eur_y['ERA5'].values.tolist()
    							ERA5_Land_cold_Eur = dframe_cold_gcell_Eur_y['ERA5-Land'].values.tolist()
    							JRA_cold_Eur = dframe_cold_gcell_Eur_y['JRA55'].values.tolist()
    							MERRA2_cold_Eur = dframe_cold_gcell_Eur_y['MERRA2'].values.tolist()
    							GLDAS_cold_Eur = dframe_cold_gcell_Eur_y['GLDAS-Noah'].values.tolist()
    							GLDAS_CLSM_cold_Eur = dframe_cold_gcell_Eur_y['GLDAS-CLSM'].values.tolist()

    							station_temp_cold_master.append(station_cold_NAG)
    							station_temp_cold_master.append(station_cold_Eur)
    							naive_all_temp_cold_master.append(naive_all_cold_NAG)
    							naive_all_temp_cold_master.append(naive_all_cold_Eur)
    							CFSR_temp_cold_master.append(CFSR_cold_NAG)
    							CFSR_temp_cold_master.append(CFSR_cold_Eur)
    							ERAI_temp_cold_master.append(ERAI_cold_NAG)
    							ERAI_temp_cold_master.append(ERAI_cold_Eur)
    							ERA5_temp_cold_master.append(ERA5_cold_NAG)
    							ERA5_temp_cold_master.append(ERA5_cold_Eur)
    							ERA5_Land_temp_cold_master.append(ERA5_Land_cold_NAG)
    							ERA5_Land_temp_cold_master.append(ERA5_Land_cold_Eur)
    							JRA_temp_cold_master.append(JRA_cold_NAG)
    							JRA_temp_cold_master.append(JRA_cold_Eur)
    							MERRA2_temp_cold_master.append(MERRA2_cold_NAG)
    							MERRA2_temp_cold_master.append(MERRA2_cold_Eur)
    							GLDAS_temp_cold_master.append(GLDAS_cold_NAG)
    							GLDAS_temp_cold_master.append(GLDAS_cold_Eur)
    							GLDAS_CLSM_temp_cold_master.append(GLDAS_CLSM_cold_NAG)
    							GLDAS_CLSM_temp_cold_master.append(GLDAS_CLSM_cold_Eur)

    						station_temp_warm_master = []
    						naive_all_temp_warm_master = []
    						CFSR_temp_warm_master = []    						
    						ERAI_temp_warm_master = []
    						ERA5_temp_warm_master = []
    						ERA5_Land_temp_warm_master = []
    						JRA_temp_warm_master = []
    						MERRA2_temp_warm_master = []
    						GLDAS_temp_warm_master = []
    						GLDAS_CLSM_temp_warm_master = []

    						for z in range(0,len_gc_NAG_warm): #loop through data subsets (warm season) and store temperature information for only those grid cells
    							grid_cell_NAG_warm_z = subset_NAG_gcell_warm[z]
    							grid_cell_Eur_warm_z = subset_Eur_gcell_warm[z]

    							dframe_warm_gcell_NAG_z = dframe_warm[dframe_warm['Grid Cell'] == grid_cell_NAG_warm_z]
    							dframe_warm_gcell_Eur_z = dframe_warm[dframe_warm['Grid Cell'] == grid_cell_Eur_warm_z]

    							station_warm_NAG = dframe_warm_gcell_NAG_z['Station'].values.tolist()
    							naive_all_warm_NAG = dframe_warm_gcell_NAG_z['Naive Blend All'].values.tolist()
    							CFSR_warm_NAG = dframe_warm_gcell_NAG_z['CFSR'].values.tolist()
    							ERAI_warm_NAG = dframe_warm_gcell_NAG_z['ERA-Interim'].values.tolist()
    							ERA5_warm_NAG = dframe_warm_gcell_NAG_z['ERA5'].values.tolist()
    							ERA5_Land_warm_NAG = dframe_warm_gcell_NAG_z['ERA5-Land'].values.tolist()
    							JRA_warm_NAG = dframe_warm_gcell_NAG_z['JRA55'].values.tolist()
    							MERRA2_warm_NAG = dframe_warm_gcell_NAG_z['MERRA2'].values.tolist()
    							GLDAS_warm_NAG = dframe_warm_gcell_NAG_z['GLDAS-Noah'].values.tolist()
    							GLDAS_CLSM_warm_NAG = dframe_warm_gcell_NAG_z['GLDAS-CLSM'].values.tolist()

    							station_warm_Eur = dframe_warm_gcell_Eur_z['Station'].values.tolist()
    							naive_all_warm_Eur = dframe_warm_gcell_Eur_z['Naive Blend All'].values.tolist()
    							CFSR_warm_Eur = dframe_warm_gcell_Eur_z['CFSR'].values.tolist()
    							ERAI_warm_Eur = dframe_warm_gcell_Eur_z['ERA-Interim'].values.tolist()
    							ERA5_warm_Eur = dframe_warm_gcell_Eur_z['ERA5'].values.tolist()
    							ERA5_Land_warm_Eur = dframe_warm_gcell_Eur_z['ERA5-Land'].values.tolist()
    							JRA_warm_Eur = dframe_warm_gcell_Eur_z['JRA55'].values.tolist()
    							MERRA2_warm_Eur = dframe_warm_gcell_Eur_z['MERRA2'].values.tolist()
    							GLDAS_warm_Eur = dframe_warm_gcell_Eur_z['GLDAS-Noah'].values.tolist()
    							GLDAS_CLSM_warm_Eur = dframe_warm_gcell_Eur_z['GLDAS-CLSM'].values.tolist()

    							station_temp_warm_master.append(station_warm_NAG)
    							station_temp_warm_master.append(station_warm_Eur)
    							naive_all_temp_warm_master.append(naive_all_warm_NAG)
    							naive_all_temp_warm_master.append(naive_all_warm_Eur)
    							CFSR_temp_warm_master.append(CFSR_warm_NAG)
    							CFSR_temp_warm_master.append(CFSR_warm_Eur)
    							ERAI_temp_warm_master.append(ERAI_warm_NAG)
    							ERAI_temp_warm_master.append(ERAI_warm_Eur)
    							ERA5_temp_warm_master.append(ERA5_warm_NAG)
    							ERA5_temp_warm_master.append(ERA5_warm_Eur)
    							ERA5_Land_temp_warm_master.append(ERA5_Land_warm_NAG)
    							ERA5_Land_temp_warm_master.append(ERA5_Land_warm_Eur)
    							JRA_temp_warm_master.append(JRA_warm_NAG)
    							JRA_temp_warm_master.append(JRA_warm_Eur)
    							MERRA2_temp_warm_master.append(MERRA2_warm_NAG)
    							MERRA2_temp_warm_master.append(MERRA2_warm_Eur)
    							GLDAS_temp_warm_master.append(GLDAS_warm_NAG)
    							GLDAS_temp_warm_master.append(GLDAS_warm_Eur)
    							GLDAS_CLSM_temp_warm_master.append(GLDAS_CLSM_warm_NAG)
    							GLDAS_CLSM_temp_warm_master.append(GLDAS_CLSM_warm_Eur)	


    						station_temp_cold_master = [i for sub in station_temp_cold_master for i in sub]
    						naive_all_temp_cold_master = [i for sub in naive_all_temp_cold_master for i in sub]
    						CFSR_temp_cold_master = [i for sub in CFSR_temp_cold_master for i in sub]    						
    						ERAI_temp_cold_master = [i for sub in ERAI_temp_cold_master for i in sub]
    						ERA5_temp_cold_master = [i for sub in ERA5_temp_cold_master for i in sub]
    						ERA5_Land_temp_cold_master = [i for sub in ERA5_Land_temp_cold_master for i in sub]
    						JRA_temp_cold_master = [i for sub in JRA_temp_cold_master for i in sub]
    						MERRA2_temp_cold_master = [i for sub in MERRA2_temp_cold_master for i in sub]
    						GLDAS_temp_cold_master = [i for sub in GLDAS_temp_cold_master for i in sub]
    						GLDAS_CLSM_temp_cold_master = [i for sub in GLDAS_CLSM_temp_cold_master for i in sub]

    						station_temp_warm_master = [i for sub in station_temp_warm_master for i in sub]
    						naive_all_temp_warm_master = [i for sub in naive_all_temp_warm_master for i in sub]
    						CFSR_temp_warm_master = [i for sub in CFSR_temp_warm_master for i in sub]    						
    						ERAI_temp_warm_master = [i for sub in ERAI_temp_warm_master for i in sub]
    						ERA5_temp_warm_master = [i for sub in ERA5_temp_warm_master for i in sub]
    						ERA5_Land_temp_warm_master = [i for sub in ERA5_Land_temp_warm_master for i in sub]
    						JRA_temp_warm_master = [i for sub in JRA_temp_warm_master for i in sub]
    						MERRA2_temp_warm_master = [i for sub in MERRA2_temp_warm_master for i in sub]
    						GLDAS_temp_warm_master = [i for sub in GLDAS_temp_warm_master for i in sub]
    						GLDAS_CLSM_temp_warm_master = [i for sub in GLDAS_CLSM_temp_warm_master for i in sub]

###### Cold Season #######
## Bias ##

    						naive_all_bias_cold = bias(naive_all_temp_cold_master, station_temp_cold_master)
    						naive_all_bias_cold_master.append(naive_all_bias_cold)
    						CFSR_bias_cold = bias(CFSR_temp_cold_master, station_temp_cold_master)
    						CFSR_bias_cold_master.append(CFSR_bias_cold)
    						ERAI_bias_cold = bias(ERAI_temp_cold_master, station_temp_cold_master)
    						ERAI_bias_cold_master.append(ERAI_bias_cold)
    						ERA5_bias_cold = bias(ERA5_temp_cold_master, station_temp_cold_master)
    						ERA5_bias_cold_master.append(ERA5_bias_cold)
    						ERA5_Land_bias_cold = bias(ERA5_Land_temp_cold_master, station_temp_cold_master)
    						ERA5_Land_bias_cold_master.append(ERA5_Land_bias_cold)
    						JRA_bias_cold = bias(JRA_temp_cold_master, station_temp_cold_master)
    						JRA_bias_cold_master.append(JRA_bias_cold)
    						MERRA2_bias_cold = bias(MERRA2_temp_cold_master, station_temp_cold_master)
    						MERRA2_bias_cold_master.append(MERRA2_bias_cold)
    						GLDAS_bias_cold = bias(GLDAS_temp_cold_master, station_temp_cold_master)
    						GLDAS_bias_cold_master.append(GLDAS_bias_cold)
    						GLDAS_CLSM_bias_cold = bias(GLDAS_CLSM_temp_cold_master, station_temp_cold_master)
    						GLDAS_CLSM_bias_cold_master.append(GLDAS_CLSM_bias_cold)

## STD DEV ##

    						stn_sdev_cold =  np.std(station_temp_cold_master)
    						naive_all_sdev_cold = np.std(naive_all_temp_cold_master)					
    						CFSR_sdev_cold = np.std(CFSR_temp_cold_master)
    						ERAI_sdev_cold = np.std(ERAI_temp_cold_master)    					
    						ERA5_sdev_cold = np.std(ERA5_temp_cold_master)
    						ERA5_Land_sdev_cold = np.std(ERA5_Land_temp_cold_master)
    						JRA_sdev_cold = np.std(JRA_temp_cold_master)
    						MERRA2_sdev_cold = np.std(MERRA2_temp_cold_master)
    						GLDAS_sdev_cold = np.std(GLDAS_temp_cold_master)
    						GLDAS_CLSM_sdev_cold = np.std(GLDAS_CLSM_temp_cold_master)

    						stn_sdev_cold_master.append(stn_sdev_cold)
    						naive_all_sdev_cold_master.append(naive_all_sdev_cold)
    						CFSR_sdev_cold_master.append(CFSR_sdev_cold)
    						ERAI_sdev_cold_master.append(ERAI_sdev_cold)
    						ERA5_sdev_cold_master.append(ERA5_sdev_cold)
    						ERA5_Land_sdev_cold_master.append(ERA5_Land_sdev_cold)
    						JRA_sdev_cold_master.append(JRA_sdev_cold)
    						MERRA2_sdev_cold_master.append(MERRA2_sdev_cold)
    						GLDAS_sdev_cold_master.append(GLDAS_sdev_cold)
    						GLDAS_CLSM_sdev_cold_master.append(GLDAS_CLSM_sdev_cold)						    						

## RMSE ##

    						naive_all_rmse_cold = mean_squared_error(station_temp_cold_master,naive_all_temp_cold_master, squared=False)
    						CFSR_rmse_cold = mean_squared_error(station_temp_cold_master,CFSR_temp_cold_master, squared=False)
    						ERAI_rmse_cold = mean_squared_error(station_temp_cold_master,ERAI_temp_cold_master, squared=False)
    						ERA5_rmse_cold = mean_squared_error(station_temp_cold_master,ERA5_temp_cold_master, squared=False)
    						ERA5_Land_rmse_cold = mean_squared_error(station_temp_cold_master,ERA5_Land_temp_cold_master, squared=False)
    						JRA_rmse_cold = mean_squared_error(station_temp_cold_master,JRA_temp_cold_master, squared=False)
    						MERRA2_rmse_cold = mean_squared_error(station_temp_cold_master,MERRA2_temp_cold_master, squared=False)
    						GLDAS_rmse_cold = mean_squared_error(station_temp_cold_master,GLDAS_temp_cold_master, squared=False)
    						GLDAS_CLSM_rmse_cold = mean_squared_error(station_temp_cold_master,GLDAS_CLSM_temp_cold_master, squared=False)
						
    						naive_all_rmse_cold_master.append(naive_all_rmse_cold)
    						CFSR_rmse_cold_master.append(CFSR_rmse_cold)
    						ERAI_rmse_cold_master.append(ERAI_rmse_cold)
    						ERA5_rmse_cold_master.append(ERA5_rmse_cold)
    						ERA5_Land_rmse_cold_master.append(ERA5_Land_rmse_cold)
    						JRA_rmse_cold_master.append(JRA_rmse_cold)
    						MERRA2_rmse_cold_master.append(MERRA2_rmse_cold)
    						GLDAS_rmse_cold_master.append(GLDAS_rmse_cold)
    						GLDAS_CLSM_rmse_cold_master.append(GLDAS_CLSM_rmse_cold)


## Pearson Correlations ##

    						naive_all_corr_cold,_ = pearsonr(naive_all_temp_cold_master, station_temp_cold_master)
    						CFSR_corr_cold,_ = pearsonr(CFSR_temp_cold_master, station_temp_cold_master)
    						ERAI_corr_cold,_ = pearsonr(ERAI_temp_cold_master, station_temp_cold_master)
    						ERA5_corr_cold,_ = pearsonr(ERA5_temp_cold_master, station_temp_cold_master)
    						ERA5_Land_corr_cold,_ = pearsonr(ERA5_Land_temp_cold_master, station_temp_cold_master)
    						JRA_corr_cold,_ = pearsonr(JRA_temp_cold_master, station_temp_cold_master)
    						MERRA2_corr_cold,_ = pearsonr(MERRA2_temp_cold_master, station_temp_cold_master)
    						GLDAS_corr_cold,_ = pearsonr(GLDAS_temp_cold_master, station_temp_cold_master)
    						GLDAS_CLSM_corr_cold,_ = pearsonr(GLDAS_CLSM_temp_cold_master, station_temp_cold_master)

    						naive_all_corr_cold_master.append(naive_all_corr_cold)
    						CFSR_corr_cold_master.append(CFSR_corr_cold)
    						ERAI_corr_cold_master.append(ERAI_corr_cold)
    						ERA5_corr_cold_master.append(ERA5_corr_cold)
    						ERA5_Land_corr_cold_master.append(ERA5_Land_corr_cold)
    						JRA_corr_cold_master.append(JRA_corr_cold)
    						MERRA2_corr_cold_master.append(MERRA2_corr_cold)
    						GLDAS_corr_cold_master.append(GLDAS_corr_cold)
    						GLDAS_CLSM_corr_cold_master.append(GLDAS_CLSM_corr_cold)
						

######### Warm Season ###########

## Bias ##

    						naive_all_bias_warm = bias(naive_all_temp_warm_master, station_temp_warm_master)
    						naive_all_bias_warm_master.append(naive_all_bias_warm)
    						CFSR_bias_warm = bias(CFSR_temp_warm_master, station_temp_warm_master)
    						CFSR_bias_warm_master.append(CFSR_bias_warm)
    						ERAI_bias_warm = bias(ERAI_temp_warm_master, station_temp_warm_master)
    						ERAI_bias_warm_master.append(ERAI_bias_warm)
    						ERA5_bias_warm = bias(ERA5_temp_warm_master, station_temp_warm_master)
    						ERA5_bias_warm_master.append(ERA5_bias_warm)
    						ERA5_Land_bias_warm = bias(ERA5_Land_temp_warm_master, station_temp_warm_master)
    						ERA5_Land_bias_warm_master.append(ERA5_Land_bias_warm)
    						JRA_bias_warm = bias(JRA_temp_warm_master, station_temp_warm_master)
    						JRA_bias_warm_master.append(JRA_bias_warm)
    						MERRA2_bias_warm = bias(MERRA2_temp_warm_master, station_temp_warm_master)
    						MERRA2_bias_warm_master.append(MERRA2_bias_warm)
    						GLDAS_bias_warm = bias(GLDAS_temp_warm_master, station_temp_warm_master)
    						GLDAS_bias_warm_master.append(GLDAS_bias_warm)
    						GLDAS_CLSM_bias_warm = bias(GLDAS_CLSM_temp_warm_master, station_temp_warm_master)
    						GLDAS_CLSM_bias_warm_master.append(GLDAS_CLSM_bias_warm)

## STD DEV ##

    						stn_sdev_warm =  np.std(station_temp_warm_master)
    						naive_all_sdev_warm = np.std(naive_all_temp_warm_master)					
    						CFSR_sdev_warm = np.std(CFSR_temp_warm_master)
    						ERAI_sdev_warm = np.std(ERAI_temp_warm_master)    					
    						ERA5_sdev_warm = np.std(ERA5_temp_warm_master)
    						ERA5_Land_sdev_warm = np.std(ERA5_Land_temp_warm_master)
    						JRA_sdev_warm = np.std(JRA_temp_warm_master)
    						MERRA2_sdev_warm = np.std(MERRA2_temp_warm_master)
    						GLDAS_sdev_warm = np.std(GLDAS_temp_warm_master)
    						GLDAS_CLSM_sdev_warm = np.std(GLDAS_CLSM_temp_warm_master)

    						stn_sdev_warm_master.append(stn_sdev_warm)
    						naive_all_sdev_warm_master.append(naive_all_sdev_warm)
    						CFSR_sdev_warm_master.append(CFSR_sdev_warm)
    						ERAI_sdev_warm_master.append(ERAI_sdev_warm)
    						ERA5_sdev_warm_master.append(ERA5_sdev_warm)
    						ERA5_Land_sdev_warm_master.append(ERA5_Land_sdev_warm)
    						JRA_sdev_warm_master.append(JRA_sdev_warm)
    						MERRA2_sdev_warm_master.append(MERRA2_sdev_warm)
    						GLDAS_sdev_warm_master.append(GLDAS_sdev_warm)
    						GLDAS_CLSM_sdev_warm_master.append(GLDAS_CLSM_sdev_warm)						    						

## RMSE ##

    						naive_all_rmse_warm = mean_squared_error(station_temp_warm_master,naive_all_temp_warm_master, squared=False)
    						CFSR_rmse_warm = mean_squared_error(station_temp_warm_master,CFSR_temp_warm_master, squared=False)
    						ERAI_rmse_warm = mean_squared_error(station_temp_warm_master,ERAI_temp_warm_master, squared=False)
    						ERA5_rmse_warm = mean_squared_error(station_temp_warm_master,ERA5_temp_warm_master, squared=False)
    						ERA5_Land_rmse_warm = mean_squared_error(station_temp_warm_master,ERA5_Land_temp_warm_master, squared=False)
    						JRA_rmse_warm = mean_squared_error(station_temp_warm_master,JRA_temp_warm_master, squared=False)
    						MERRA2_rmse_warm = mean_squared_error(station_temp_warm_master,MERRA2_temp_warm_master, squared=False)
    						GLDAS_rmse_warm = mean_squared_error(station_temp_warm_master,GLDAS_temp_warm_master, squared=False)
    						GLDAS_CLSM_rmse_warm = mean_squared_error(station_temp_warm_master,GLDAS_CLSM_temp_warm_master, squared=False)
						
    						naive_all_rmse_warm_master.append(naive_all_rmse_warm)
    						CFSR_rmse_warm_master.append(CFSR_rmse_warm)
    						ERAI_rmse_warm_master.append(ERAI_rmse_warm)
    						ERA5_rmse_warm_master.append(ERA5_rmse_warm)
    						ERA5_Land_rmse_warm_master.append(ERA5_Land_rmse_warm)
    						JRA_rmse_warm_master.append(JRA_rmse_warm)
    						MERRA2_rmse_warm_master.append(MERRA2_rmse_warm)
    						GLDAS_rmse_warm_master.append(GLDAS_rmse_warm)
    						GLDAS_CLSM_rmse_warm_master.append(GLDAS_CLSM_rmse_warm)


## Pearson Correlations ##

    						naive_all_corr_warm,_ = pearsonr(naive_all_temp_warm_master, station_temp_warm_master)
    						CFSR_corr_warm,_ = pearsonr(CFSR_temp_warm_master, station_temp_warm_master)
    						ERAI_corr_warm,_ = pearsonr(ERAI_temp_warm_master, station_temp_warm_master)
    						ERA5_corr_warm,_ = pearsonr(ERA5_temp_warm_master, station_temp_warm_master)
    						ERA5_Land_corr_warm,_ = pearsonr(ERA5_Land_temp_warm_master, station_temp_warm_master)
    						JRA_corr_warm,_ = pearsonr(JRA_temp_warm_master, station_temp_warm_master)
    						MERRA2_corr_warm,_ = pearsonr(MERRA2_temp_warm_master, station_temp_warm_master)
    						GLDAS_corr_warm,_ = pearsonr(GLDAS_temp_warm_master, station_temp_warm_master)
    						GLDAS_CLSM_corr_warm,_ = pearsonr(GLDAS_CLSM_temp_warm_master, station_temp_warm_master)

    						naive_all_corr_warm_master.append(naive_all_corr_warm)
    						CFSR_corr_warm_master.append(CFSR_corr_warm)
    						ERAI_corr_warm_master.append(ERAI_corr_warm)
    						ERA5_corr_warm_master.append(ERA5_corr_warm)
    						ERA5_Land_corr_warm_master.append(ERA5_Land_corr_warm)
    						JRA_corr_warm_master.append(JRA_corr_warm)
    						MERRA2_corr_warm_master.append(MERRA2_corr_warm)
    						GLDAS_corr_warm_master.append(GLDAS_corr_warm)
    						GLDAS_CLSM_corr_warm_master.append(GLDAS_CLSM_corr_warm)


####### Create Master Dataframes #######

 

#### Bias ####

    					overall_bias_master = []
    					overall_bias_master.append(naive_all_bias_cold_master)
    					overall_bias_master.append(CFSR_bias_cold_master)
    					overall_bias_master.append(ERAI_bias_cold_master)
    					overall_bias_master.append(ERA5_bias_cold_master)
    					overall_bias_master.append(ERA5_Land_bias_cold_master)
    					overall_bias_master.append(JRA_bias_cold_master)
    					overall_bias_master.append(MERRA2_bias_cold_master)
    					overall_bias_master.append(GLDAS_bias_cold_master)
    					overall_bias_master.append(GLDAS_CLSM_bias_cold_master)
    					overall_bias_master.append(naive_all_bias_warm_master)
    					overall_bias_master.append(CFSR_bias_warm_master)
    					overall_bias_master.append(ERAI_bias_warm_master)
    					overall_bias_master.append(ERA5_bias_warm_master)
    					overall_bias_master.append(ERA5_Land_bias_warm_master)
    					overall_bias_master.append(JRA_bias_warm_master)
    					overall_bias_master.append(MERRA2_bias_warm_master)
    					overall_bias_master.append(GLDAS_bias_warm_master)
    					overall_bias_master.append(GLDAS_CLSM_bias_warm_master)

    					overall_bias_master = [i for sub in overall_bias_master for i in sub]

    					dframe_master_bias = pd.DataFrame(data=overall_bias_master,columns=['Bias ($^\circ$C)'])
    					dframe_master_bias['Dataset'] = dataset_master
    					dframe_master_bias['Season'] = season_master


## Create Plot ##
					
    					x_sctr = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]

    					y_sctr_cold = [naive_all_bias_cold_all,np.nan,CFSR_bias_cold_all,np.nan,ERAI_bias_cold_all,np.nan,ERA5_bias_cold_all,np.nan,ERA5_Land_bias_cold_all,np.nan,JRA_bias_cold_all,np.nan,MERRA2_bias_cold_all,np.nan,GLDAS_bias_cold_all,np.nan,GLDAS_CLSM_bias_cold_all,np.nan]
    					y_sctr_warm = [np.nan,naive_all_bias_warm_all,np.nan,CFSR_bias_warm_all,np.nan,ERAI_bias_warm_all,np.nan,ERA5_bias_warm_all,np.nan,ERA5_Land_bias_warm_all,np.nan,JRA_bias_warm_all,np.nan,MERRA2_bias_warm_all,np.nan,GLDAS_bias_warm_all,np.nan,GLDAS_CLSM_bias_warm_all]
    					sctr_dataset = ['Naive-Blend','Naive-Blend','CFSR','CFSR','ERA-Interim','ERA-Interim','ERA5','ERA5','ERA5-Land','ERA5-Land','JRA55','JRA55','MERRA2','MERRA2','GLDAS-Noah','GLDAS-Noah','GLDAS-CLSM','GLDAS-CLSM']
    					sctr_season = ['Cold','Warm','Cold','Warm','Cold','Warm','Cold','Warm','Cold','Warm','Cold','Warm','Cold','Warm','Cold','Warm','Cold','Warm']

    					markers = {"Cold":'s', 'Warm':'x'}
    
    					dframe_bias_scatter_cold = pd.DataFrame(data=y_sctr_cold,columns=['Bias ($^\circ$C)'])
    					dframe_bias_scatter_cold['Dataset'] = sctr_dataset
    					dframe_bias_scatter_cold['Season'] = sctr_season

    					dframe_bias_scatter_warm = pd.DataFrame(data=y_sctr_warm,columns=['Bias ($^\circ$C)'])
    					dframe_bias_scatter_warm['Dataset'] = sctr_dataset
    					dframe_bias_scatter_warm['Season'] = sctr_season

    					bias_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/plots/monte_carlo_boxplots/'+str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_ERA5_'+str(temp_thr_n)+'_bias_monte_carlo.png'])
    					ax_bias =  sns.boxplot(x = dframe_master_bias['Dataset'], y = dframe_master_bias['Bias ($^\circ$C)'], hue = dframe_master_bias['Season'])
    					ax_bias.set_ylim(-15,5)
    					ax_bias.set_yticks(np.arange(-15,5 + 1, 1))
    					ax_bias_sctr_cold = sns.scatterplot(x=dframe_bias_scatter_cold['Dataset'], y=dframe_bias_scatter_cold['Bias ($^\circ$C)'],s=30,style=dframe_bias_scatter_cold['Season'], legend = False)
    					ax_bias_sctr_cold.set_ylim(-15,5)
    					ax_bias_sctr_cold.set_yticks(np.arange(-15,5 + 1, 1))
    					ax_bias_sctr_warm = sns.scatterplot(x=dframe_bias_scatter_warm['Dataset'], y=dframe_bias_scatter_warm['Bias ($^\circ$C)'],s=30,style=dframe_bias_scatter_warm['Season'], legend = False)
    					ax_bias_sctr_warm.set_ylim(-15,5)
    					ax_bias_sctr_warm.set_yticks(np.arange(-15,5 + 1, 1))
    					plt.setp(ax_bias.get_xticklabels(), rotation=20)
    					plt.savefig(bias_fil)
    					plt.close()


###### SDEV ####   					

    					overall_sdev_master = []
    					overall_sdev_master.append(naive_all_sdev_cold_master)
    					overall_sdev_master.append(CFSR_sdev_cold_master)
    					overall_sdev_master.append(ERAI_sdev_cold_master)
    					overall_sdev_master.append(ERA5_sdev_cold_master)
    					overall_sdev_master.append(ERA5_Land_sdev_cold_master)
    					overall_sdev_master.append(JRA_sdev_cold_master)
    					overall_sdev_master.append(MERRA2_sdev_cold_master)
    					overall_sdev_master.append(GLDAS_sdev_cold_master)
    					overall_sdev_master.append(GLDAS_CLSM_sdev_cold_master)
    					overall_sdev_master.append(naive_all_sdev_warm_master)
    					overall_sdev_master.append(CFSR_sdev_warm_master)
    					overall_sdev_master.append(ERAI_sdev_warm_master)
    					overall_sdev_master.append(ERA5_sdev_warm_master)
    					overall_sdev_master.append(ERA5_Land_sdev_warm_master)
    					overall_sdev_master.append(JRA_sdev_warm_master)
    					overall_sdev_master.append(MERRA2_sdev_warm_master)
    					overall_sdev_master.append(GLDAS_sdev_warm_master)
    					overall_sdev_master.append(GLDAS_CLSM_sdev_warm_master)

    					overall_sdev_master = [i for sub in overall_sdev_master for i in sub]

    					dframe_master_sdev = pd.DataFrame(data=overall_sdev_master,columns=['Standard Deviation ($^\circ$C)'])
    					dframe_master_sdev['Dataset'] = dataset_master
    					dframe_master_sdev['Season'] = season_master


## Create Plot ##
					
    					x_sctr = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]

    					y_sctr_cold = [naive_all_sdev_cold_all,np.nan,CFSR_sdev_cold_all,np.nan,ERAI_sdev_cold_all,np.nan,ERA5_sdev_cold_all,np.nan,ERA5_Land_sdev_cold_all,np.nan,JRA_sdev_cold_all,np.nan,MERRA2_sdev_cold_all,np.nan,GLDAS_sdev_cold_all,np.nan,GLDAS_CLSM_sdev_cold_all,np.nan]
    					y_sctr_warm = [np.nan,naive_all_sdev_warm_all,np.nan,CFSR_sdev_warm_all,np.nan,ERAI_sdev_warm_all,np.nan,ERA5_sdev_warm_all,np.nan,ERA5_Land_sdev_warm_all,np.nan,JRA_sdev_warm_all,np.nan,MERRA2_sdev_warm_all,np.nan,GLDAS_sdev_warm_all,np.nan,GLDAS_CLSM_sdev_warm_all]
    					sctr_dataset = ['Naive-Blend','Naive-Blend','CFSR','CFSR','ERA-Interim','ERA-Interim','ERA5','ERA5','ERA5-Land','ERA5-Land','JRA55','JRA55','MERRA2','MERRA2','GLDAS-Noah','GLDAS-Noah','GLDAS-CLSM','GLDAS-CLSM']
    					sctr_season = ['Cold','Warm','Cold','Warm','Cold','Warm','Cold','Warm','Cold','Warm','Cold','Warm','Cold','Warm','Cold','Warm','Cold','Warm']

    					markers = {"Cold":'s', 'Warm':'x'}
    
    					dframe_sdev_scatter_cold = pd.DataFrame(data=y_sctr_cold,columns=['Standard Deviation ($^\circ$C)'])
    					dframe_sdev_scatter_cold['Dataset'] = sctr_dataset
    					dframe_sdev_scatter_cold['Season'] = sctr_season

    					dframe_sdev_scatter_warm = pd.DataFrame(data=y_sctr_warm,columns=['Standard Deviation ($^\circ$C)'])
    					dframe_sdev_scatter_warm['Dataset'] = sctr_dataset
    					dframe_sdev_scatter_warm['Season'] = sctr_season

    					sdev_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/plots/monte_carlo_boxplots/'+str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_ERA5_'+str(temp_thr_n)+'_sdev_monte_carlo.png'])
    					ax_sdev =  sns.boxplot(x = dframe_master_sdev['Dataset'], y = dframe_master_sdev['Standard Deviation ($^\circ$C)'], hue = dframe_master_sdev['Season'])
    					ax_sdev.set_ylim(0,10)
    					ax_sdev.set_yticks(np.arange(0,10 + 1, 1))
    					ax_sdev_sctr_cold = sns.scatterplot(x=dframe_sdev_scatter_cold['Dataset'], y=dframe_sdev_scatter_cold['Standard Deviation ($^\circ$C)'],s=30,style=dframe_sdev_scatter_cold['Season'], legend = False)
    					ax_sdev_sctr_cold.set_ylim(0,10)
    					ax_sdev_sctr_cold.set_yticks(np.arange(0,10 + 1, 1))
    					ax_sdev_sctr_warm = sns.scatterplot(x=dframe_sdev_scatter_warm['Dataset'], y=dframe_sdev_scatter_warm['Standard Deviation ($^\circ$C)'],s=30,style=dframe_sdev_scatter_warm['Season'], legend = False)
    					ax_sdev_sctr_warm.set_ylim(0,10)
    					ax_sdev_sctr_warm.set_yticks(np.arange(0,10 + 1, 1))
    					plt.setp(ax_sdev.get_xticklabels(), rotation=20)
    					plt.savefig(sdev_fil)
    					plt.close()

###### RMSE ####

    					overall_rmse_master = []
    					overall_rmse_master.append(naive_all_rmse_cold_master)
    					overall_rmse_master.append(CFSR_rmse_cold_master)
    					overall_rmse_master.append(ERAI_rmse_cold_master)
    					overall_rmse_master.append(ERA5_rmse_cold_master)
    					overall_rmse_master.append(ERA5_Land_rmse_cold_master)
    					overall_rmse_master.append(JRA_rmse_cold_master)
    					overall_rmse_master.append(MERRA2_rmse_cold_master)
    					overall_rmse_master.append(GLDAS_rmse_cold_master)
    					overall_rmse_master.append(GLDAS_CLSM_rmse_cold_master)
    					overall_rmse_master.append(naive_all_rmse_warm_master)
    					overall_rmse_master.append(CFSR_rmse_warm_master)
    					overall_rmse_master.append(ERAI_rmse_warm_master)
    					overall_rmse_master.append(ERA5_rmse_warm_master)
    					overall_rmse_master.append(ERA5_Land_rmse_warm_master)
    					overall_rmse_master.append(JRA_rmse_warm_master)
    					overall_rmse_master.append(MERRA2_rmse_warm_master)
    					overall_rmse_master.append(GLDAS_rmse_warm_master)
    					overall_rmse_master.append(GLDAS_CLSM_rmse_warm_master)

    					overall_rmse_master = [i for sub in overall_rmse_master for i in sub]

    					dframe_master_rmse = pd.DataFrame(data=overall_rmse_master,columns=['RMSE ($^\circ$C)'])
    					dframe_master_rmse['Dataset'] = dataset_master
    					dframe_master_rmse['Season'] = season_master


## Create Plot ##
					
    					x_sctr = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]

    					y_sctr_cold = [naive_all_rmse_cold_all,np.nan,CFSR_rmse_cold_all,np.nan,ERAI_rmse_cold_all,np.nan,ERA5_rmse_cold_all,np.nan,ERA5_Land_rmse_cold_all,np.nan,JRA_rmse_cold_all,np.nan,MERRA2_rmse_cold_all,np.nan,GLDAS_rmse_cold_all,np.nan,GLDAS_CLSM_rmse_cold_all,np.nan]
    					y_sctr_warm = [np.nan,naive_all_rmse_warm_all,np.nan,CFSR_rmse_warm_all,np.nan,ERAI_rmse_warm_all,np.nan,ERA5_rmse_warm_all,np.nan,ERA5_Land_rmse_warm_all,np.nan,JRA_rmse_warm_all,np.nan,MERRA2_rmse_warm_all,np.nan,GLDAS_rmse_warm_all,np.nan,GLDAS_CLSM_rmse_warm_all]
    					sctr_dataset = ['Naive-Blend','Naive-Blend','CFSR','CFSR','ERA-Interim','ERA-Interim','ERA5','ERA5','ERA5-Land','ERA5-Land','JRA55','JRA55','MERRA2','MERRA2','GLDAS-Noah','GLDAS-Noah','GLDAS-CLSM','GLDAS-CLSM']
    					sctr_season = ['Cold','Warm','Cold','Warm','Cold','Warm','Cold','Warm','Cold','Warm','Cold','Warm','Cold','Warm','Cold','Warm','Cold','Warm']

    					markers = {"Cold":'s', 'Warm':'x'}
    
    					dframe_rmse_scatter_cold = pd.DataFrame(data=y_sctr_cold,columns=['RMSE ($^\circ$C)'])
    					dframe_rmse_scatter_cold['Dataset'] = sctr_dataset
    					dframe_rmse_scatter_cold['Season'] = sctr_season

    					dframe_rmse_scatter_warm = pd.DataFrame(data=y_sctr_warm,columns=['RMSE ($^\circ$C)'])
    					dframe_rmse_scatter_warm['Dataset'] = sctr_dataset
    					dframe_rmse_scatter_warm['Season'] = sctr_season

    					rmse_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/plots/monte_carlo_boxplots/'+str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_ERA5_'+str(temp_thr_n)+'_rmse_monte_carlo.png'])
    					ax_rmse =  sns.boxplot(x = dframe_master_rmse['Dataset'], y = dframe_master_rmse['RMSE ($^\circ$C)'], hue = dframe_master_rmse['Season'])
    					ax_rmse.set_ylim(0,16)
    					ax_rmse.set_yticks(np.arange(0,16 + 1, 1))
    					ax_rmse_sctr_cold = sns.scatterplot(x=dframe_rmse_scatter_cold['Dataset'], y=dframe_rmse_scatter_cold['RMSE ($^\circ$C)'],s=30,style=dframe_rmse_scatter_cold['Season'], legend = False)
    					ax_rmse_sctr_cold.set_ylim(0,16)
    					ax_rmse_sctr_cold.set_yticks(np.arange(0,16 + 1, 1))
    					ax_rmse_sctr_warm = sns.scatterplot(x=dframe_rmse_scatter_warm['Dataset'], y=dframe_rmse_scatter_warm['RMSE ($^\circ$C)'],s=30,style=dframe_rmse_scatter_warm['Season'], legend = False)
    					ax_rmse_sctr_warm.set_ylim(0,16)
    					ax_rmse_sctr_warm.set_yticks(np.arange(0,16 + 1, 1))
    					plt.setp(ax_rmse.get_xticklabels(), rotation=20)
    					plt.savefig(rmse_fil)
    					plt.close()

###### Pearson Correlation ####

    					overall_corr_master = []
    					overall_corr_master.append(naive_all_corr_cold_master)
    					overall_corr_master.append(CFSR_corr_cold_master)
    					overall_corr_master.append(ERAI_corr_cold_master)
    					overall_corr_master.append(ERA5_corr_cold_master)
    					overall_corr_master.append(ERA5_Land_corr_cold_master)
    					overall_corr_master.append(JRA_corr_cold_master)
    					overall_corr_master.append(MERRA2_corr_cold_master)
    					overall_corr_master.append(GLDAS_corr_cold_master)
    					overall_corr_master.append(GLDAS_CLSM_corr_cold_master)
    					overall_corr_master.append(naive_all_corr_warm_master)
    					overall_corr_master.append(CFSR_corr_warm_master)
    					overall_corr_master.append(ERAI_corr_warm_master)
    					overall_corr_master.append(ERA5_corr_warm_master)
    					overall_corr_master.append(ERA5_Land_corr_warm_master)
    					overall_corr_master.append(JRA_corr_warm_master)
    					overall_corr_master.append(MERRA2_corr_warm_master)
    					overall_corr_master.append(GLDAS_corr_warm_master)
    					overall_corr_master.append(GLDAS_CLSM_corr_warm_master)

    					overall_corr_master = [i for sub in overall_corr_master for i in sub]

    					dframe_master_corr = pd.DataFrame(data=overall_corr_master,columns=['Pearson Correlation'])
    					dframe_master_corr['Dataset'] = dataset_master
    					dframe_master_corr['Season'] = season_master


## Create Plot ##
					
    					x_sctr = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]

    					y_sctr_cold = [naive_all_corr_cold_all,np.nan,CFSR_corr_cold_all,np.nan,ERAI_corr_cold_all,np.nan,ERA5_corr_cold_all,np.nan,ERA5_Land_corr_cold_all,np.nan,JRA_corr_cold_all,np.nan,MERRA2_corr_cold_all,np.nan,GLDAS_corr_cold_all,np.nan,GLDAS_CLSM_corr_cold_all,np.nan]
    					y_sctr_warm = [np.nan,naive_all_corr_warm_all,np.nan,CFSR_corr_warm_all,np.nan,ERAI_corr_warm_all,np.nan,ERA5_corr_warm_all,np.nan,ERA5_Land_corr_warm_all,np.nan,JRA_corr_warm_all,np.nan,MERRA2_corr_warm_all,np.nan,GLDAS_corr_warm_all,np.nan,GLDAS_CLSM_corr_warm_all]
    					sctr_dataset = ['Naive-Blend','Naive-Blend','CFSR','CFSR','ERA-Interim','ERA-Interim','ERA5','ERA5','ERA5-Land','ERA5-Land','JRA55','JRA55','MERRA2','MERRA2','GLDAS-Noah','GLDAS-Noah','GLDAS-CLSM','GLDAS-CLSM']
    					sctr_season = ['Cold','Warm','Cold','Warm','Cold','Warm','Cold','Warm','Cold','Warm','Cold','Warm','Cold','Warm','Cold','Warm','Cold','Warm']

    					markers = {"Cold":'s', 'Warm':'x'}
    
    					dframe_corr_scatter_cold = pd.DataFrame(data=y_sctr_cold,columns=['Pearson Correlation'])
    					dframe_corr_scatter_cold['Dataset'] = sctr_dataset
    					dframe_corr_scatter_cold['Season'] = sctr_season

    					dframe_corr_scatter_warm = pd.DataFrame(data=y_sctr_warm,columns=['Pearson Correlation'])
    					dframe_corr_scatter_warm['Dataset'] = sctr_dataset
    					dframe_corr_scatter_warm['Season'] = sctr_season

    					corr_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/plots/monte_carlo_boxplots/'+str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_ERA5_'+str(temp_thr_n)+'_corr_monte_carlo.png'])
    					ax_corr =  sns.boxplot(x = dframe_master_corr['Dataset'], y = dframe_master_corr['Pearson Correlation'], hue = dframe_master_corr['Season'])
    					ax_corr.set_ylim(0,1.0)
    					ax_corr.set_yticks(np.arange(0.,1.0 + 0.1, 0.1))
    					ax_corr_sctr_cold = sns.scatterplot(x=dframe_corr_scatter_cold['Dataset'], y=dframe_corr_scatter_cold['Pearson Correlation'],s=30,style=dframe_corr_scatter_cold['Season'], legend = False)
    					ax_corr_sctr_cold.set_ylim(0,1.0)
    					ax_corr_sctr_cold.set_yticks(np.arange(0,1.0 + 0.1, 0.1))
    					ax_corr_sctr_warm = sns.scatterplot(x=dframe_corr_scatter_warm['Dataset'], y=dframe_corr_scatter_warm['Pearson Correlation'],s=30,style=dframe_corr_scatter_warm['Season'], legend = False)
    					ax_corr_sctr_warm.set_ylim(0,1.0)
    					ax_corr_sctr_warm.set_yticks(np.arange(0,1.0 + 0.1, 0.1))
    					plt.setp(ax_corr.get_xticklabels(), rotation=20)
    					plt.savefig(corr_fil)
    					plt.close()



					






























