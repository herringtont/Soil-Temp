import os
import glob
import netCDF4
import csv
import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import matplotlib.patches as mpl_patches
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
from matplotlib.offsetbox import AnchoredText


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

# Convert longitudes to -180 to 180 degrees
def LonTo180(d_lon):
   dlon = ((d_lon + 180) % 360) - 180
   return dlon

############# Set Directories ############

naive_type = ['simple_average']
olr = ['zscore']#['outliers','zscore','IQR']
lyr = ['top_30cm','30cm_300cm']
thr = ['100']#['0','25','50','75','100']
rmp_type = ['con']#['nn','bil','con']
remap_type = 'remapcon'
tmp_type = ['raw_temp']
temp_thr = ['-2C']#['0C','-2C','-5C','-10C']

ERA5_air = 'temperature'

air_temp = 'Air_Temp_degC'

############## Grab Reanalysis Data ############
#
#for i in rmp_type:
#    rmp_type_i = i
#    remap_type = ''.join(['remap'+rmp_type_i])
#    rnys_dir = ''.join(['/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/obs_depths/common_grid_CLSM/'+str(remap_type)+'/common_date/masked_files/'])
#    air_dir = ''.join(['/mnt/data/users/herringtont/soil_temp/reanalysis/2m_AirTemp/rename/land_only/common_grid_CLSM/'+str(remap_type)+'/masked_files/degC/'])
#    rnys_gcell_dir = ''.join([rnys_dir,'grid_cell_level/'])
#
#    BEST_fi_air = "".join(['/mnt/data/users/herringtont/soil_temp/reanalysis/2m_AirTemp/rename/land_only/common_grid_CLSM/remapcon/masked_files/BEST_2m_air.nc'])
#    CFSR_fi = "".join([rnys_dir,"CFSR_all.nc"])
#    CFSR_fi_air = "".join([air_dir,"CFSR_all_2m_air.nc"])
#    MERRA2_fi = "".join([rnys_dir,"MERRA2.nc"])
#    MERRA2_fi_air = "".join([air_dir,"MERRA2_2m_air.nc"])    
#    ERA5_fi = "".join([rnys_dir,"ERA5.nc"])
#    ERA5_fi_air = "".join([air_dir,"ERA5_2m_air.nc"])       
#    ERA5_MAAT_fi = ''.join(['/mnt/data/users/herringtont/soil_temp/reanalysis/2m_AirTemp/rename/land_only/common_grid_CLSM/'+str(remap_type)+'/masked_files/MAAT/BEST_2m_air_MAAT_1981_2010_clim.nc'])
#    ERA5_Land_fi = "".join([rnys_dir,"ERA5_Land.nc"])
#    ERA5_Land_fi_air = "".join([air_dir,"ERA5-Land_2m_air.nc"]) 
#    ERAI_fi = "".join([rnys_dir,"ERA-Interim.nc"])
#    ERAI_fi_air = "".join([air_dir,"ERA-Interim_2m_air.nc"])
#    JRA_fi = "".join([rnys_dir,"JRA55.nc"])
#    JRA_fi_air = "".join([air_dir,"JRA55_2m_air.nc"])
#    GLDAS_fi = "".join([rnys_dir,"GLDAS.nc"])
#    GLDAS_fi_air = "".join([air_dir,"GLDAS_2m_air.nc"])
#    GLDAS_CLSM_fi = "".join([rnys_dir,"GLDAS_CLSM.nc"])
#    GLDAS_CLSM_fi_air = "".join([air_dir,"GLDAS_CLSM_2m_air.nc"])
#
#    for j in naive_type:
#    	naive_type_j = j
#    	naive_dir_raw = ''.join(['/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/obs_depths/common_grid_CLSM/'+str(remap_type)+'/common_date/masked_files/'])
#    	naive_fi = ''.join([naive_dir_raw+str(remap_type)+'_'+'naive_blend.nc'])
#    	naive_fi_noJRA = ''.join([naive_dir_raw+str(remap_type)+'_'+'naive_blend_noJRA55.nc'])
#    	naive_fi_noJRAold = ''.join([naive_dir_raw+str(remap_type)+'_'+'naive_blend_noJRA55_old.nc'])
#    	naive_fi_all = ''.join([naive_dir_raw+str(remap_type)+'_'+'naive_blend_ERA5L_GLDAS_CLSM.nc'])
#    	naive_fi_all_air = ''.join([air_dir+'ensmean_2m_air.nc'])	
#    	for k in olr:
#    		olr_k = k
#
#
################## Timeseries Master Arrays #################
#
#    		date_master_timeseries_cold = []
#    		gcell_master_timeseries_cold = []
#    		layer_master_timeseries_cold = []
#    		permafrost_type_timeseries_cold = []
#    		continent_master_timeseries_cold = []
#    		station_master_timeseries_cold = []
#    		BEST_air_master_timeseries_cold = []
#    		naive_all_master_timeseries_cold = []
#    		naive_all_air_master_timeseries_cold = []
#    		CFSR_master_timeseries_cold = []
#    		CFSR_air_master_timeseries_cold = []    					
#    		ERAI_master_timeseries_cold = []
#    		ERAI_air_master_timeseries_cold = []
#    		ERA5_master_timeseries_cold = []
#    		ERA5_air_master_timeseries_cold = []
#    		ERA5_Land_master_timeseries_cold = []
#    		ERA5_Land_air_master_timeseries_cold = []
#    		JRA_master_timeseries_cold = []
#    		JRA_air_master_timeseries_cold = []
#    		MERRA2_master_timeseries_cold = []
#    		MERRA2_air_master_timeseries_cold = []
#    		GLDAS_master_timeseries_cold = []
#    		GLDAS_air_master_timeseries_cold = []
#    		GLDAS_CLSM_master_timeseries_cold = []
#    		GLDAS_CLSM_air_master_timeseries_cold = []
#
#
#    		date_master_timeseries_warm = []
#    		gcell_master_timeseries_warm = []
#    		layer_master_timeseries_warm = []
#    		permafrost_type_timeseries_warm = []
#    		continent_master_timeseries_warm = []
#    		station_master_timeseries_warm = []
#    		BEST_air_master_timeseries_warm = []
#    		naive_all_master_timeseries_warm = []
#    		naive_all_air_master_timeseries_warm = []
#    		CFSR_master_timeseries_warm = []
#    		CFSR_air_master_timeseries_warm = []    					
#    		ERAI_master_timeseries_warm = []
#    		ERAI_air_master_timeseries_warm = []
#    		ERA5_master_timeseries_warm = []
#    		ERA5_air_master_timeseries_warm = []
#    		ERA5_Land_master_timeseries_warm = []
#    		ERA5_Land_air_master_timeseries_warm = []					
#    		JRA_master_timeseries_warm = []
#    		JRA_air_master_timeseries_warm = []
#    		MERRA2_master_timeseries_warm = []
#    		MERRA2_air_master_timeseries_warm = []
#    		GLDAS_master_timeseries_warm = []
#    		GLDAS_air_master_timeseries_warm = []
#    		GLDAS_CLSM_master_timeseries_warm = []
#    		GLDAS_CLSM_air_master_timeseries_warm = []
#
#    		for l in lyr:
#    			lyr_l = l
#    			if (lyr_l == 'top_30cm'):
#    				CFSR_layer = "Soil_Temp_TOP30"
#    				CFSR2_layer = "Soil_Temp_TOP30"
#    				GLDAS_layer = "Soil_Temp_TOP30"
#    				GLDAS_CLSM_layer = "Soil_Temp_TOP30"
#    				ERA5_layer = "Soil_Temp_TOP30"
#    				ERA5_Land_layer = "Soil_Temp_TOP30"
#    				ERAI_layer = "Soil_Temp_TOP30"
#    				JRA_layer = "Soil_Temp_TOP30"
#    				MERRA2_layer = "Soil_Temp_TOP30"
#    				Naive_layer = "Soil_Temp_TOP30"
#
#    				in_situ_layer = 'top_30cm'
#
#    			if (lyr_l == '30cm_300cm'):
#    				CFSR_layer = "Soil_Temp_30cm_300cm"
#    				CFSR2_layer = "Soil_Temp_30cm_300cm"
#    				GLDAS_layer = "Soil_Temp_30cm_300cm"
#    				GLDAS_CLSM_layer = "Soil_Temp_30cm_300cm"
#    				ERA5_layer = "Soil_Temp_30cm_300cm"
#    				ERA5_Land_layer = "Soil_Temp_30cm_300cm"
#    				ERAI_layer = "Soil_Temp_30cm_300cm"
#    				JRA_layer = "Soil_Temp_30cm_300cm"
#    				MERRA2_layer = "Soil_Temp_30cm_300cm"
#    				Naive_layer = "Soil_Temp_30cm_300cm"
#
#    				in_situ_layer = '30_299.9'
#
#
#
#
#
#    			for m in thr:
#    				thr_m = m
#    				insitu_dir =  ''.join(['/mnt/data/users/herringtont/soil_temp/In-Situ/All/spatial_average_anom/'+str(remap_type)+'/no_outliers/'+str(olr_k)+'/'+str(in_situ_layer)+'/thr_'+str(thr_m)+'/CLSM/Sep2021/'])
#
#    				for o in temp_thr:
#    					temp_thr_o = o
#
#    					if (temp_thr_o == "0C"):
#    						tmp_val = 0
#
#    					if (temp_thr_o == "-2C"):
#    						tmp_val = -2
#
#    					if (temp_thr_o == "-5C"):
#    						tmp_val = -5
#
#    					if (temp_thr_o == "-10C"):
#    						tmp_val = -10
#
#
#    					print("Remap Type:",remap_type)
#    					print("Layer:",lyr_l)
#    					print("Temp Threshold:", temp_thr_o)
###### Create Master Arrays #####
#    					gcell_master_cold_temp = []
#    					gcell_master_warm_temp = []
#    					gcell_master_cold_date = []
#    					gcell_master_warm_date = []
#    					gcell_master = []
#    					lat_master_cold_temp = []
#    					lat_master_warm_temp = []
#    					lat_master_cold_date = []
#    					lat_master_warm_date = []
#    					lon_master_cold_temp = []
#    					lon_master_warm_temp = []
#    					lon_master_cold_date = []
#    					lon_master_warm_date = []
#    					lon_master_stn = []
#    					lat_master = []
#    					lon_master = []
#    					date_master = []
#    					len_anom_master = []
#    					len_raw_master = []
#    					sample_size_master_cold_temp = []
#    					sample_size_master_warm_temp = []
#    					sample_size_master_cold_date = []
#    					sample_size_master_warm_date = []
#
#    					station_data_master_cold_temp = []
#    					naive_data_master_cold_temp = []
#    					naive_noJRA_data_master_cold_temp = []
#    					naive_noJRAold_data_master_cold_temp = []
#    					naive_all_data_master_cold_temp = []
#    					CFSR_data_master_cold_temp = []
#    					ERAI_data_master_cold_temp = []
#    					ERA5_data_master_cold_temp = []
#    					ERA5_Land_data_master_cold_temp = []
#    					JRA_data_master_cold_temp = []
#    					MERRA2_data_master_cold_temp = []
#    					GLDAS_data_master_cold_temp = []
#    					GLDAS_CLSM_data_master_cold_temp = []
#
#    					station_data_master_warm_temp = []
#    					naive_data_master_warm_temp = []
#    					naive_noJRA_data_master_warm_temp = []
#    					naive_noJRAold_data_master_warm_temp = []
#    					naive_all_data_master_warm_temp = []
#    					CFSR_data_master_warm_temp = []
#    					ERAI_data_master_warm_temp = []
#    					ERA5_data_master_warm_temp = []
#    					ERA5_Land_data_master_warm_temp = []
#    					JRA_data_master_warm_temp = []
#    					MERRA2_data_master_warm_temp = []
#    					GLDAS_data_master_warm_temp = []
#    					GLDAS_CLSM_data_master_warm_temp = []
#
#    					station_data_air_master_cold_temp = []
#    					naive_data_air_master_cold_temp = []
#    					naive_noJRA_data_air_master_cold_temp = []
#    					naive_noJRAold_data_air_master_cold_temp = []
#    					naive_all_data_air_master_cold_temp = []
#    					BEST_data_air_master_cold_temp = []
#    					CFSR_data_air_master_cold_temp = []
#    					ERAI_data_air_master_cold_temp = []
#    					ERA5_data_air_master_cold_temp = []
#    					ERA5_Land_data_air_master_cold_temp = []
#    					JRA_data_air_master_cold_temp = []
#    					MERRA2_data_air_master_cold_temp = []
#    					GLDAS_data_air_master_cold_temp = []
#    					GLDAS_CLSM_data_air_master_cold_temp = []
#
#    					station_data_air_master_warm_temp = []
#    					naive_data_air_master_warm_temp = []
#    					naive_noJRA_data_air_master_warm_temp = []
#    					naive_noJRAold_data_air_master_warm_temp = []
#    					naive_all_data_air_master_warm_temp = []
#    					BEST_data_air_master_warm_temp = []
#    					CFSR_data_air_master_warm_temp = []
#    					ERAI_data_air_master_warm_temp = []
#    					ERA5_data_air_master_warm_temp = []
#    					ERA5_Land_data_air_master_warm_temp = []
#    					JRA_data_air_master_warm_temp = []
#    					MERRA2_data_air_master_warm_temp = []
#    					GLDAS_data_air_master_warm_temp = []
#    					GLDAS_CLSM_data_air_master_warm_temp = []
#
#    					cold_temp_season_master = []
#    					warm_temp_season_master = []
#
#    					cold_temp_continent_master = []
#    					warm_temp_continent_master = []
#
#    					cold_temp_RS_2002_permafrost_master = []
#    					warm_temp_RS_2002_permafrost_master = []
#
#    					cold_temp_Brown_1970_permafrost_master = []
#    					warm_temp_Brown_1970_permafrost_master = []
#
#    					cold_temp_season_air_master = []
#    					warm_temp_season_air_master = []
#
#    					cold_temp_continent_air_master = []
#    					warm_temp_continent_air_master = []
#
#    					cold_temp_RS_2002_permafrost_air_master = []
#    					warm_temp_RS_2002_permafrost_air_master = []
#
#    					cold_temp_Brown_1970_permafrost_air_master = []
#    					warm_temp_Brown_1970_permafrost_air_master = []
#
#    					cold_temp_MAAT = []
#    					warm_temp_MAAT = []
#
#
#
#
################## loop through in-situ files ###############
#    					#print(type(CFSR_anom))
#    					#pathlist = os_sorted(os.listdir(insitu_dir))
#    					pathlist = os.listdir(insitu_dir)
#    					pathlist_sorted = natural_sort(pathlist)
#    					for path in pathlist_sorted:
#    						insitu_fil = ''.join([insitu_dir,path])
#    						print(insitu_fil)
#    						dframe_insitu = pd.read_csv(insitu_fil)
#    						dattim = dframe_insitu['Date'].values
#    						DateTime = [datetime.datetime.strptime(x,'%Y-%m-%d') for x in dattim]
#    						soil_temp = dframe_insitu['Spatial Avg Temp']
#    						gcell = dframe_insitu['Grid Cell'].iloc[0]
#    						lat_cen = dframe_insitu['Central Lat'].iloc[0]
#    						lon_cen = dframe_insitu['Central Lon'].iloc[0]
#
#    						print("Grid Cell:",gcell, " Lon Cen:",str(lon_cen))
# 
#    						#if (lat_cen < 70 and -136 < lon_cen and lon_cen < -109) : #skip grid cells based in NWT
#    							#print("Skipping Grid Cell",gcell," because in NWT")
#    							#continue
#
#    						if(gcell == 33777): #skip grid cell 3377 because of data issues
#    							continue
#    
#    						CFSR_fil = xr.open_dataset(CFSR_fi)
#    						ERAI_fil = xr.open_dataset(ERAI_fi)
#    						ERA5_fil = xr.open_dataset(ERA5_fi)
#    						ERA5_Land_fil = xr.open_dataset(ERA5_Land_fi)
#    						JRA_fil = xr.open_dataset(JRA_fi)
#    						MERRA2_fil = xr.open_dataset(MERRA2_fi)
#    						GLDAS_fil = xr.open_dataset(GLDAS_fi)
#    						GLDAS_CLSM_fil = xr.open_dataset(GLDAS_CLSM_fi)
#
#    						BEST_fil_air = xr.open_dataset(BEST_fi_air)
#    						CFSR_fil_air = xr.open_dataset(CFSR_fi_air)
#    						ERAI_fil_air = xr.open_dataset(ERAI_fi_air)
#    						ERA5_fil_air = xr.open_dataset(ERA5_fi_air)
#    						ERA5_Land_fil_air = xr.open_dataset(ERA5_Land_fi_air)
#    						JRA_fil_air = xr.open_dataset(JRA_fi_air)
#    						MERRA2_fil_air = xr.open_dataset(MERRA2_fi_air)
#    						GLDAS_fil_air = xr.open_dataset(GLDAS_fi_air)
#    						GLDAS_CLSM_fil_air = xr.open_dataset(GLDAS_CLSM_fi_air)
#    						ERA5_MAAT_fil = xr.open_dataset(ERA5_MAAT_fi)
#    						#print(BEST_fil_air)
#
#    						CFSR_stemp = CFSR_fil[CFSR_layer] - 273.15
#    						ERAI_stemp = ERAI_fil[ERAI_layer] - 273.15
#    						ERA5_stemp = ERA5_fil[ERA5_layer] - 273.15
#    						ERA5_Land_stemp = ERA5_Land_fil[ERA5_Land_layer] - 273.15
#    						JRA_stemp = JRA_fil[JRA_layer] - 273.15
#    						MERRA2_stemp = MERRA2_fil[MERRA2_layer] -273.15
#    						GLDAS_stemp = GLDAS_fil[GLDAS_layer] - 273.15
#    						GLDAS_CLSM_stemp = GLDAS_CLSM_fil[GLDAS_CLSM_layer] - 273.15
#
#    						BEST_air = BEST_fil_air['temperature']
#    						CFSR_air = CFSR_fil_air[air_temp]
#    						ERAI_air = ERAI_fil_air[air_temp]
#    						ERA5_air = ERA5_fil_air[air_temp]
#    						ERA5_Land_air = ERA5_Land_fil_air[air_temp]
#    						JRA_air = JRA_fil_air[air_temp]
#    						MERRA2_air = MERRA2_fil_air[air_temp]
#    						GLDAS_air = GLDAS_fil_air[air_temp]
#    						GLDAS_CLSM_air = GLDAS_CLSM_fil_air[air_temp]
#
#    						naive_fil = xr.open_dataset(naive_fi)
#    						naive_fil_noJRA = xr.open_dataset(naive_fi_noJRA)
#    						naive_fil_noJRAold = xr.open_dataset(naive_fi_noJRAold)
#    						naive_fil_all = xr.open_dataset(naive_fi_all)
#    						naive_fil_all_air = xr.open_dataset(naive_fi_all_air)
#    						naive_stemp = naive_fil[Naive_layer] - 273.15
#    						naive_stemp_noJRA = naive_fil_noJRA[Naive_layer] -273.15
#    						naive_stemp_noJRAold = naive_fil_noJRAold[Naive_layer] -273.15
#    						naive_stemp_all = naive_fil_all[Naive_layer] -273.15
#
#    						naive_air_all = naive_fil_all_air[air_temp]
#						
#    						ERA5_tair = ERA5_fil_air[air_temp]
#    						ERA5_MAAT = ERA5_MAAT_fil['temperature']
#
#    						CFSR_stemp_gcell = CFSR_stemp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
#    						ERAI_stemp_gcell = ERAI_stemp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
#    						ERA5_stemp_gcell = ERA5_stemp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
#    						ERA5_Land_stemp_gcell = ERA5_Land_stemp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
#    						JRA_stemp_gcell = JRA_stemp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
#    						MERRA2_stemp_gcell = MERRA2_stemp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
#    						GLDAS_stemp_gcell = GLDAS_stemp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
#    						GLDAS_CLSM_stemp_gcell = GLDAS_CLSM_stemp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
#    						naive_stemp_gcell = naive_stemp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
#    						naive_noJRA_stemp_gcell = naive_stemp_noJRA.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
#    						naive_noJRAold_stemp_gcell = naive_stemp_noJRAold.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
#    						naive_all_stemp_gcell = naive_stemp_all.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
#
#
#    						#print(BEST_air)
#    						#print(str(lat_cen),str(lon_cen))
#
#    						BEST_air_gcell = BEST_air.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
#    						CFSR_air_gcell = CFSR_air.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
#    						ERAI_air_gcell = ERAI_air.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
#    						ERA5_air_gcell = ERA5_air.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
#    						ERA5_Land_air_gcell = ERA5_Land_air.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
#    						JRA_air_gcell = JRA_air.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
#    						MERRA2_air_gcell = MERRA2_air.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
#    						GLDAS_air_gcell = GLDAS_air.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
#    						GLDAS_CLSM_air_gcell = GLDAS_CLSM_air.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
#    						naive_all_air_gcell = naive_air_all.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
#
#    						#print(JRA_air_gcell)
#    						#print(BEST_air_gcell)
#    						#print(CFSR_stemp_gcell)
#    						#print(naive_all_stemp_gcell)
#
#    						ERA5_tair_gcell = ERA5_tair.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
#    						ERA5_MAAT_gcell = ERA5_MAAT.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True).values
#    						ERA5_MAAT_gcell = ERA5_MAAT_gcell[0]
#    						#print(ERA5_MAAT_gcell)
#    						rnys_dattim = naive_fil['time']
#    						rnys_datetime = rnys_dattim.dt.strftime('%Y-%m-%d')    					
#    						len_rnys_dattim = len(rnys_dattim) - 1
#    						rnys_sdate = rnys_dattim.isel(time=0).values
#    						rnys_sdate_str = str(rnys_sdate)
#    						rnys_sdate_dt = datetime.datetime.strptime(rnys_sdate_str[0:10],'%Y-%m-%d')
#    						rnys_edate = rnys_dattim.isel(time=len_rnys_dattim).values
#    						rnys_edate_str = str(rnys_edate)
#    						rnys_edate_dt = datetime.datetime.strptime(rnys_edate_str[0:10],'%Y-%m-%d')
#
#    						CFSR_dattim = CFSR_fil['time']
#    						CFSR_datetime = CFSR_dattim.dt.strftime('%Y-%m-%d')
#    						len_CFSR_dattim = len(CFSR_dattim) - 1 
#    						CFSR_sdate = CFSR_dattim.isel(time=0).values
#    						CFSR_sdate_str = str(CFSR_sdate)
#    						CFSR_sdate_dt = datetime.datetime.strptime(CFSR_sdate_str[0:10],'%Y-%m-%d')
#    						CFSR_edate = CFSR_dattim.isel(time=len_CFSR_dattim).values
#    						CFSR_edate_str = str(CFSR_edate)
#    						CFSR_edate_dt = datetime.datetime.strptime(CFSR_edate_str[0:10],'%Y-%m-%d')
#
#    						#print(ERA5_MAAT_gcell)
#
#
##################### grab collocated temperature data from reanalysis files #######################
#
#    						CFSR_temp_master = []
#    						JRA_temp_master = []
#    						ERAI_temp_master = []
#    						ERA5_temp_master = []
#    						BEST_air_master = []
#    						ERA5_air_master = []
#    						ERA5_Land_temp_master = []
#    						MERRA2_temp_master = []
#    						GLDAS_temp_master = []
#    						GLDAS_CLSM_temp_master = []
#    						CFSR_air_master = []
#    						JRA_air_master = []
#    						ERAI_air_master = []
#    						ERA5_Land_air_master = []
#    						MERRA2_air_master = []
#    						GLDAS_air_master = []
#    						GLDAS_CLSM_air_master = []
#    						TC_temp_master = []
#    						naive_temp_master = []
#    						naive_noJRA_temp_master = []
#    						naive_noJRAold_temp_master = []
#    						naive_all_temp_master = []
#    						naive_all_air_master = []
#    						station_temp_master = []
#    						station_anom_master = []
#    						date_temp_master = []
#
#    						for n in range(0,len(DateTime)):
#    							DateTime_m = DateTime[n]
#    							Year_m = DateTime_m.year
#    							Month_m = DateTime_m.month
#
#    							dattim_m = dattim[n]
#    							if(DateTime_m < rnys_sdate_dt): #skip dates before 1980-01-01
#    								continue
#    							if(DateTime_m > rnys_edate_dt): #skip all dates beyond last reanalysis date
#    								continue
#    							naive_temp_dt = naive_stemp_gcell.sel(time=DateTime_m).values.tolist()
#    							naive_noJRA_temp_dt = naive_noJRA_stemp_gcell.sel(time=DateTime_m).values.tolist()
#    							naive_noJRAold_temp_dt = naive_noJRAold_stemp_gcell.sel(time=DateTime_m).values.tolist()
#    							naive_all_temp_dt = naive_all_stemp_gcell.sel(time=DateTime_m).values.tolist()
#    							naive_all_air_dt = naive_all_air_gcell.sel(time=DateTime_m).values.tolist()
#    							#print(DateTime_m)
#    							#print(rnys_edate_dt)
#    							if(str(naive_temp_dt) == "nan"):
#    								naive_temp_dt = np.nan  
#    							if(str(naive_noJRA_temp_dt) == "nan"):
#    								naive_noJRA_temp_dt = np.nan
#    							if(str(naive_noJRAold_temp_dt) == "nan"):
#    								naive_noJRAold_temp_dt = np.nan
#    							if(str(naive_all_temp_dt) == "nan"):
#    								naive_all_temp_dt = np.nan
#    							if(str(naive_all_air_dt) == "nan"):
#    								naive_all_air_dt = np.nan  						
#    							dframe_insitu_dt = dframe_insitu[dframe_insitu['Date'] == dattim_m]
#    							station_temp_dt = dframe_insitu_dt['Spatial Avg Temp'].values
#    							extracted_value = station_temp_dt[0]
#    							if(str(station_temp_dt) == "nan"):
#    								station_temp_dt = np.nan
#    							station_temp_master.append(float(extracted_value))
#    							station_anom_dt = dframe_insitu_dt['Spatial Avg Anom'].values.tolist()
#    							if(str(station_anom_dt) == "nan"):
#    								station_anom_dt = np.nan
#    							station_anom_master.append(station_anom_dt)									
#    							CFSR_temp_dt = CFSR_stemp_gcell.sel(time=DateTime_m).values.tolist()
#    							if(str(CFSR_temp_dt) == "nan"):
#    								CFSR_temp_dt = np.nan
#    							CFSR_temp_master.append(CFSR_temp_dt)    						
#    							JRA_temp_dt = JRA_stemp_gcell.sel(time=DateTime_m).values.tolist()
#    							#print(JRA_temp_dt)
#    							if(str(JRA_temp_dt) == "nan"):
#    								JRA_temp_dt = np.nan
#    							JRA_temp_master.append(JRA_temp_dt)      							
#    							ERAI_temp_dt = ERAI_stemp_gcell.sel(time=DateTime_m).values.tolist()
#    							if(str(ERAI_temp_dt) == "nan"):
#    								ERAI_temp_dt = np.nan
#    							ERAI_temp_master.append(ERAI_temp_dt)
#    							ERA5_temp_dt = ERA5_stemp_gcell.sel(time=DateTime_m).values.tolist()
#    							if(str(ERA5_temp_dt) == "nan"):
#    								ERA5_temp_dt = np.nan
#    							ERA5_temp_master.append(ERA5_temp_dt)
#    							ERA5_air_dt = ERA5_air_gcell.sel(time=DateTime_m).values.tolist()
#    							if(str(ERA5_air_dt) == "nan"):
#    								ERA5_air_dt = np.nan
#    							ERA5_air_master.append(ERA5_air_dt)
#    							ERA5_Land_temp_dt = ERA5_Land_stemp_gcell.sel(time=DateTime_m).values.tolist()
#    							if(str(ERA5_Land_temp_dt) == "nan"):
#    								ERA5_Land_temp_dt = np.nan
#    							ERA5_Land_temp_master.append(ERA5_Land_temp_dt)
#    							MERRA2_temp_dt = MERRA2_stemp_gcell.sel(time=DateTime_m).values.tolist()
#    							if(str(MERRA2_temp_dt) == "nan"):
#    								MERRA2_temp_dt = np.nan
#    							MERRA2_temp_master.append(MERRA2_temp_dt)
#    							GLDAS_temp_dt = GLDAS_stemp_gcell.sel(time=DateTime_m).values.tolist()
#    							if(str(GLDAS_temp_dt) == "nan"):
#    								GLDAS_temp_dt = np.nan
#    							GLDAS_temp_master.append(GLDAS_temp_dt)
#    							GLDAS_CLSM_temp_dt = GLDAS_CLSM_stemp_gcell.sel(time=DateTime_m).values.tolist()
#    							if(str(GLDAS_CLSM_temp_dt) == "nan"):
#    								GLDAS_CLSM_temp_dt = np.nan
#    							GLDAS_CLSM_temp_master.append(GLDAS_CLSM_temp_dt)
#    							BEST_air_dt = BEST_air_gcell.sel(time=DateTime_m).values.tolist()
#    							if(str(BEST_air_dt) == "nan"):
#    								BEST_air_dt = np.nan
#    							BEST_air_master.append(BEST_air_dt)
#    							CFSR_air_dt = CFSR_air_gcell.sel(time=DateTime_m).values.tolist()
#    							if(str(CFSR_air_dt) == "nan"):
#    								CFSR_air_dt = np.nan
#    							CFSR_air_master.append(CFSR_air_dt)    						
#    							JRA_air_dt = JRA_air_gcell.sel(time=DateTime_m).values.tolist()
#    							if(str(JRA_air_dt) == "nan"):
#    								JRA_air_dt = np.nan
#    							JRA_air_master.append(JRA_air_dt)      							
#    							ERAI_air_dt = ERAI_air_gcell.sel(time=DateTime_m).values.tolist()
#    							if(str(ERAI_air_dt) == "nan"):
#    								ERAI_air_dt = np.nan
#    							ERAI_air_master.append(ERAI_air_dt)
#
#    							ERA5_Land_air_dt = ERA5_Land_air_gcell.sel(time=DateTime_m).values.tolist()
#    							if(str(ERA5_Land_air_dt) == "nan"):
#    								ERA5_Land_air_dt = np.nan
#    							ERA5_Land_air_master.append(ERA5_Land_air_dt)
#    							MERRA2_air_dt = MERRA2_air_gcell.sel(time=DateTime_m).values.tolist()
#    							if(str(MERRA2_air_dt) == "nan"):
#    								MERRA2_air_dt = np.nan
#    							MERRA2_air_master.append(MERRA2_air_dt)
#    							GLDAS_air_dt = GLDAS_air_gcell.sel(time=DateTime_m).values.tolist()
#    							if(str(GLDAS_air_dt) == "nan"):
#    								GLDAS_air_dt = np.nan
#    							GLDAS_air_master.append(GLDAS_air_dt)
#    							GLDAS_CLSM_air_dt = GLDAS_CLSM_air_gcell.sel(time=DateTime_m).values.tolist()
#    							if(str(GLDAS_CLSM_air_dt) == "nan"):
#    								GLDAS_CLSM_air_dt = np.nan
#    							GLDAS_CLSM_air_master.append(GLDAS_CLSM_air_dt)
#    							date_temp_master.append(dattim_m)    						
#    							naive_temp_master.append(naive_temp_dt)            							    						
#    							naive_noJRA_temp_master.append(naive_noJRA_temp_dt)
#    							naive_noJRAold_temp_master.append(naive_noJRAold_temp_dt)
#    							naive_all_temp_master.append(naive_all_temp_dt)
#    							naive_all_air_master.append(naive_all_air_dt)
#
#    						station_temp_master = np.array(station_temp_master)
#    						station_anom_master = np.array(station_anom_master)
#    						date_temp_master = np.array(date_temp_master)
#    						CFSR_temp_master = np.array(CFSR_temp_master)
#    						ERAI_temp_master = np.array(ERAI_temp_master)
#    						ERA5_temp_master = np.array(ERA5_temp_master)
#    						ERA5_air_master = np.array(ERA5_air_master)
#    						ERA5_Land_temp_master = np.array(ERA5_Land_temp_master)
#    						JRA_temp_master = np.array(JRA_temp_master)
#    						MERRA2_temp_master = np.array(MERRA2_temp_master)
#    						GLDAS_temp_master = np.array(GLDAS_temp_master)
#    						GLDAS_CLSM_temp_master = np.array(GLDAS_CLSM_temp_master)
#
#    						BEST_air_master = np.array(BEST_air_master)
#    						CFSR_air_master = np.array(CFSR_air_master)
#    						ERAI_air_master = np.array(ERAI_air_master)
#    						ERA5_Land_air_master = np.array(ERA5_Land_air_master)
#    						JRA_air_master = np.array(JRA_air_master)
#    						MERRA2_air_master = np.array(MERRA2_air_master)
#    						GLDAS_air_master = np.array(GLDAS_air_master)
#    						GLDAS_CLSM_air_master = np.array(GLDAS_CLSM_air_master)
#
#    						naive_temp_master = np.array(naive_temp_master)
#    						naive_noJRA_temp_master = np.array(naive_noJRA_temp_master)
#    						naive_noJRAold_temp_master = np.array(naive_noJRAold_temp_master)
#    						naive_all_temp_master = np.array(naive_all_temp_master)
#    						naive_all_air_master = np.array(naive_all_air_master)
#
#    						#print(naive_all_air_master)
#						
#    						naive_no_nan = naive_all_temp_master[~np.isnan(naive_all_temp_master)]
#
#    						#print(naive_no_nan)
#
#    						CFSR_no_nan = CFSR_temp_master[~np.isnan(CFSR_temp_master)]
#    						#print(CFSR_no_nan)
#
#    						if(DateTime[0]>CFSR_edate_dt or DateTime[len(DateTime) -1] < CFSR_sdate_dt): #skip if the CFSR dates and station dates do not overlap
#    							print('Grid Cell Skipped - CFSR dates do not overlap')
#    							continue
#
#    					
#    						if(len(naive_no_nan) == 0 or len(CFSR_no_nan) == 0): #skip if there are NaN values in blended data
#    							print('Grid Cell Skipped - No Naive Blended Data')
#    							continue
#
#
################## Determine Continent of Grid Cell #####################
#
#    						if (-179.5 <= lon_cen <= -15): # if longitude is between -179.5 and -15, it will be considered part of North America and Greenland
#    							if ((gcell == 16527) or (gcell == 18754) or (gcell == 19201) or (gcell == 19348) or (gcell == 18903)):
#    								continent = "Greenland"
#    								print('Grid Cell Skipped - Greenland Location')
#    								continue
#    							else:
#    								continent = "North_America"
#    								print(lat_cen, lon_cen)
#    						elif (-15 < lon_cen < 179.5): # else it will be considered part of Eurasia
#    							continent = "Eurasia"
#
#
#
################# Determine Permafrost Type ################
#
#    						print(ERA5_MAAT_gcell)
#
#    						if (ERA5_MAAT_gcell <= -8):
#    							RS_2002_permafrost = 'continuous'
#    							Brown_1970_permafrost = 'continuous'
#
#    						elif (-7 >= ERA5_MAAT_gcell > -8):
#    							RS_2002_permafrost = 'continuous'
#    							Brown_1970_permafrost = 'discontinuous'
#
#    						elif (-2 >= ERA5_MAAT_gcell > -7):
#    							RS_2002_permafrost = 'discontinuous'
#    							Brown_1970_permafrost = 'discontinuous'
#
#    						elif (-1 >= ERA5_MAAT_gcell > -2):
#    							RS_2002_permafrost = 'none'
#    							Brown_1970_permafrost = 'discontinuous'
#
#    						elif (ERA5_MAAT_gcell > -1):
#    							RS_2002_permafrost = 'none'
#    							Brown_1970_permafrost = 'none'
#
#
#
################### Separate by cold and warm season (by temperature) ####################
#
####### Cold Season (Soil Temp <= -2) #####
#
#    						#print(type(BEST_air_master))
#    						cold_season_index = np.where(BEST_air_master <= tmp_val)
#    						cold_idx = cold_season_index[0]
#
#
#    						#print(cold_idx)
#    						date_cold_season = []
#    						station_temp_cold_season = []
#    						date_cold_season = []
#    						naive_temp_cold_season = []
#    						naive_noJRA_temp_cold_season = []
#    						naive_noJRAold_temp_cold_season = []
#    						naive_all_temp_cold_season = []
#    						CFSR_temp_cold_season = []
#    						ERAI_temp_cold_season = []
#    						ERA5_temp_cold_season = []
#    						ERA5_Land_temp_cold_season = []
#    						JRA_temp_cold_season = []
#    						MERRA2_temp_cold_season = []
#    						GLDAS_temp_cold_season = []
#    						GLDAS_CLSM_temp_cold_season = []
#
#    						date_warm_season = []
#    						station_air_cold_season = []
#    						naive_air_cold_season = []
#    						naive_noJRA_air_cold_season = []
#    						naive_noJRAold_air_cold_season = []
#    						naive_all_air_cold_season = []
#    						BEST_air_cold_season = []						
#    						CFSR_air_cold_season = []
#    						ERAI_air_cold_season = []
#    						ERA5_air_cold_season = []
#    						ERA5_Land_air_cold_season = []
#    						JRA_air_cold_season = []
#    						MERRA2_air_cold_season = []
#    						GLDAS_air_cold_season = []
#    						GLDAS_CLSM_air_cold_season = []
#    						for x in cold_idx:
#    							date_x = date_temp_master[x].tolist()
#    							date_cold_season.append(date_x)
#    							station_x = station_temp_master[x].tolist()
#    							station_temp_cold_season.append(station_x)
#    							naive_x = naive_temp_master[x].tolist()
#    							naive_temp_cold_season.append(naive_x)
#    							naive_noJRA_x = naive_noJRA_temp_master[x].tolist()
#    							naive_noJRA_temp_cold_season.append(naive_noJRA_x)
#    							naive_noJRAold_x = naive_noJRAold_temp_master[x].tolist()
#    							naive_noJRAold_temp_cold_season.append(naive_noJRAold_x)
#    							naive_all_x = naive_all_temp_master[x].tolist()
#    							naive_all_temp_cold_season.append(naive_all_x)
#    							CFSR_x = CFSR_temp_master[x].tolist()
#    							CFSR_temp_cold_season.append(CFSR_x)
#    							ERAI_x = ERAI_temp_master[x].tolist()
#    							ERAI_temp_cold_season.append(ERAI_x)
#    							ERA5_x = ERA5_temp_master[x].tolist()
#    							ERA5_temp_cold_season.append(ERA5_x)
#    							ERA5_Land_x = ERA5_Land_temp_master[x].tolist()
#    							ERA5_Land_temp_cold_season.append(ERA5_Land_x)
#    							JRA_x = JRA_temp_master[x].tolist()
#    							JRA_temp_cold_season.append(JRA_x)
#    							MERRA2_x = MERRA2_temp_master[x].tolist()
#    							MERRA2_temp_cold_season.append(MERRA2_x)
#    							GLDAS_x = GLDAS_temp_master[x].tolist()
#    							GLDAS_temp_cold_season.append(GLDAS_x)
#    							GLDAS_CLSM_x = GLDAS_CLSM_temp_master[x].tolist()
#    							GLDAS_CLSM_temp_cold_season.append(GLDAS_CLSM_x)
#					   				
#    							naive_all_air_air_x = naive_all_air_master[x].tolist()
#    							naive_all_air_cold_season.append(naive_all_air_air_x)
#    							BEST_air_x = BEST_air_master[x].tolist()
#    							BEST_air_cold_season.append(BEST_air_x)
#    							CFSR_air_x = CFSR_air_master[x].tolist()
#    							CFSR_air_cold_season.append(CFSR_air_x)
#    							ERAI_air_x = ERAI_air_master[x].tolist()
#    							ERAI_air_cold_season.append(ERAI_air_x)
#    							ERA5_air_x = ERA5_air_master[x].tolist()
#    							ERA5_air_cold_season.append(ERA5_air_x)
#    							ERA5_Land_air_x = ERA5_Land_air_master[x].tolist()
#    							ERA5_Land_air_cold_season.append(ERA5_Land_air_x)
#    							JRA_air_x = JRA_air_master[x].tolist()
#    							JRA_air_cold_season.append(JRA_air_x)
#    							MERRA2_air_x = MERRA2_air_master[x].tolist()
#    							MERRA2_air_cold_season.append(MERRA2_air_x)
#    							GLDAS_air_x = GLDAS_air_master[x].tolist()
#    							GLDAS_air_cold_season.append(GLDAS_air_x)
#    							GLDAS_CLSM_air_x = GLDAS_CLSM_air_master[x].tolist()
#    							GLDAS_CLSM_air_cold_season.append(GLDAS_CLSM_air_x)
#
#
#    						if(len(naive_temp_cold_season) < 1 or len(station_temp_cold_season) < 1 or len(CFSR_temp_cold_season) < 1):
#    							print('Grid Cell Skipped - Length of Cold Season Less than 1')
#    							continue
#
#    						dframe_cold_season_temp = pd.DataFrame(data = date_cold_season, columns=['Date'])
#    						dframe_cold_season_temp['Station'] = station_temp_cold_season
#    						dframe_cold_season_temp['Grid Cell'] = gcell
#    						dframe_cold_season_temp['Layer'] = lyr_l
#    						dframe_cold_season_temp['Lat'] = lat_cen
#    						dframe_cold_season_temp['Lon'] = lon_cen
#    						dframe_cold_season_temp['Naive Blend'] = naive_temp_cold_season
#    						dframe_cold_season_temp['Naive Blend no JRA55'] = naive_noJRA_temp_cold_season
#    						dframe_cold_season_temp['Naive Blend no JRA55 Old'] = naive_noJRAold_temp_cold_season
#    						dframe_cold_season_temp['Naive Blend All'] = naive_all_temp_cold_season
#    						dframe_cold_season_temp['CFSR'] = CFSR_temp_cold_season
#    						dframe_cold_season_temp['ERA-Interim'] = ERAI_temp_cold_season
#    						dframe_cold_season_temp['ERA5'] = ERA5_temp_cold_season
#    						dframe_cold_season_temp['ERA5-Land'] = ERA5_Land_temp_cold_season
#    						dframe_cold_season_temp['JRA55'] = JRA_temp_cold_season
#    						dframe_cold_season_temp['MERRA2'] = MERRA2_temp_cold_season
#    						dframe_cold_season_temp['GLDAS-Noah'] = GLDAS_temp_cold_season
#    						dframe_cold_season_temp['GLDAS-CLSM'] = GLDAS_CLSM_temp_cold_season
#    						dframe_cold_season_temp['Naive Blend All Air Temp'] = naive_all_air_cold_season
#    						dframe_cold_season_temp['BEST Air Temp'] = BEST_air_cold_season
#    						dframe_cold_season_temp['CFSR Air Temp'] = CFSR_air_cold_season
#    						dframe_cold_season_temp['ERA-Interim Air Temp'] = ERAI_air_cold_season
#    						dframe_cold_season_temp['ERA5 Air Temp'] = ERA5_air_cold_season
#    						dframe_cold_season_temp['ERA5-Land Air Temp'] = ERA5_Land_air_cold_season
#    						dframe_cold_season_temp['JRA55 Air Temp'] = JRA_air_cold_season
#    						dframe_cold_season_temp['MERRA2 Air Temp'] = MERRA2_air_cold_season
#    						dframe_cold_season_temp['GLDAS-Noah Air Temp'] = GLDAS_air_cold_season
#    						dframe_cold_season_temp['GLDAS-CLSM Air Temp'] = GLDAS_CLSM_air_cold_season
#    						dframe_cold_season_temp = dframe_cold_season_temp.dropna()
#    						dframe_cold_season_temp['N'] = len(dframe_cold_season_temp)
#    						dframe_cold_season_temp['Season'] = 'Cold'
#    						dframe_cold_season_temp['Continent'] = continent
#    						dframe_cold_season_temp['RS 2002 Permafrost'] = RS_2002_permafrost
#    						dframe_cold_season_temp['Brown 1970 Permafrost'] = Brown_1970_permafrost
#    						dframe_cold_season_temp['ERA5 MAAT'] = ERA5_MAAT_gcell
#
#    						#print(dframe_cold_season_temp)
#    						 
#    						if(len(dframe_cold_season_temp) < 1):
#    							print('Grid Cell Skipped -  Length of Cold Season Less than 1')
#    							continue
#
#
#    
####### warm Season (Soil Temp <= -2) #####
#
#    						warm_season_index = np.where(BEST_air_master > tmp_val)
#    						warm_idx = warm_season_index[0]
#
#    						date_warm_season = []
#    						station_temp_warm_season = []
#    						naive_temp_warm_season = []
#    						naive_noJRA_temp_warm_season = []
#    						naive_noJRAold_temp_warm_season = []
#    						naive_all_temp_warm_season = []
#    						CFSR_temp_warm_season = []
#    						ERAI_temp_warm_season = []
#    						ERA5_temp_warm_season = []
#    						ERA5_Land_temp_warm_season = []
#    						JRA_temp_warm_season = []
#    						MERRA2_temp_warm_season = []
#    						GLDAS_temp_warm_season = []
#    						GLDAS_CLSM_temp_warm_season = []
#
#    						station_air_warm_season = []
#    						naive_air_warm_season = []
#    						naive_noJRA_air_warm_season = []
#    						naive_noJRAold_air_warm_season = []
#    						naive_all_air_warm_season = []
#    						BEST_air_warm_season = []
#    						CFSR_air_warm_season = []
#    						ERAI_air_warm_season = []
#    						ERA5_air_warm_season = []
#    						ERA5_Land_air_warm_season = []
#    						JRA_air_warm_season = []
#    						MERRA2_air_warm_season = []
#    						GLDAS_air_warm_season = []
#    						GLDAS_CLSM_air_warm_season = []
#    						for x in warm_idx:
#    							date_x = date_temp_master[x].tolist()
#    							date_warm_season.append(date_x)
#    							station_x = station_temp_master[x].tolist()
#    							station_temp_warm_season.append(station_x)
#    							naive_x = naive_temp_master[x].tolist()
#    							naive_temp_warm_season.append(naive_x)
#    							naive_noJRA_x = naive_noJRA_temp_master[x].tolist()
#    							naive_noJRA_temp_warm_season.append(naive_noJRA_x)
#    							naive_noJRAold_x = naive_noJRAold_temp_master[x].tolist()
#    							naive_noJRAold_temp_warm_season.append(naive_noJRAold_x)
#    							naive_all_x = naive_all_temp_master[x].tolist()
#    							naive_all_temp_warm_season.append(naive_all_x)
#    							CFSR_x = CFSR_temp_master[x].tolist()
#    							CFSR_temp_warm_season.append(CFSR_x)
#    							ERAI_x = ERAI_temp_master[x].tolist()
#    							ERAI_temp_warm_season.append(ERAI_x)
#    							ERA5_x = ERA5_temp_master[x].tolist()
#    							ERA5_temp_warm_season.append(ERA5_x)
#    							ERA5_Land_x = ERA5_Land_temp_master[x].tolist()
#    							ERA5_Land_temp_warm_season.append(ERA5_Land_x)
#    							JRA_x = JRA_temp_master[x].tolist()
#    							JRA_temp_warm_season.append(JRA_x)
#    							MERRA2_x = MERRA2_temp_master[x].tolist()
#    							MERRA2_temp_warm_season.append(MERRA2_x)
#    							GLDAS_x = GLDAS_temp_master[x].tolist()
#    							GLDAS_temp_warm_season.append(GLDAS_x)
#    							GLDAS_CLSM_x = GLDAS_CLSM_temp_master[x].tolist()
#    							GLDAS_CLSM_temp_warm_season.append(GLDAS_CLSM_x)
#					   				
#
#    							naive_all_air_air_x = naive_all_air_master[x].tolist()
#    							naive_all_air_warm_season.append(naive_all_air_air_x)
#    							BEST_air_x = BEST_air_master[x].tolist()
#    							BEST_air_warm_season.append(BEST_air_x)
#    							CFSR_air_x = CFSR_air_master[x].tolist()
#    							CFSR_air_warm_season.append(CFSR_air_x)
#    							ERAI_air_x = ERAI_air_master[x].tolist()
#    							ERAI_air_warm_season.append(ERAI_air_x)
#    							ERA5_air_x = ERA5_air_master[x].tolist()
#    							ERA5_air_warm_season.append(ERA5_air_x)
#    							ERA5_Land_air_x = ERA5_Land_air_master[x].tolist()
#    							ERA5_Land_air_warm_season.append(ERA5_Land_air_x)
#    							JRA_air_x = JRA_air_master[x].tolist()
#    							JRA_air_warm_season.append(JRA_air_x)
#    							MERRA2_air_x = MERRA2_air_master[x].tolist()
#    							MERRA2_air_warm_season.append(MERRA2_air_x)
#    							GLDAS_air_x = GLDAS_air_master[x].tolist()
#    							GLDAS_air_warm_season.append(GLDAS_air_x)
#    							GLDAS_CLSM_air_x = GLDAS_CLSM_air_master[x].tolist()
#    							GLDAS_CLSM_air_warm_season.append(GLDAS_CLSM_air_x)
#
#    						if(len(naive_temp_warm_season) < 1 or len(station_temp_warm_season) < 1 or len(CFSR_temp_warm_season) < 1):
#    							print('Grid Cell Skipped - Length of warm Season Less than 1')
#    							continue
#
#
#
#						#print("Continent:",continent)
#
#    						dframe_warm_season_temp = pd.DataFrame(data = date_warm_season, columns=['Date'])
#    						dframe_warm_season_temp['Station'] = station_temp_warm_season
#    						dframe_warm_season_temp['Grid Cell'] = gcell
#    						dframe_warm_season_temp['Layer'] = lyr_l
#    						dframe_warm_season_temp['Lat'] = lat_cen
#    						dframe_warm_season_temp['Lon'] = lon_cen
#    						dframe_warm_season_temp['Naive Blend'] = naive_temp_warm_season
#    						dframe_warm_season_temp['Naive Blend no JRA55'] = naive_noJRA_temp_warm_season
#    						dframe_warm_season_temp['Naive Blend no JRA55 Old'] = naive_noJRAold_temp_warm_season
#    						dframe_warm_season_temp['Naive Blend All'] = naive_all_temp_warm_season
#    						dframe_warm_season_temp['CFSR'] = CFSR_temp_warm_season
#    						dframe_warm_season_temp['ERA-Interim'] = ERAI_temp_warm_season
#    						dframe_warm_season_temp['ERA5'] = ERA5_temp_warm_season
#    						dframe_warm_season_temp['ERA5-Land'] = ERA5_Land_temp_warm_season
#    						dframe_warm_season_temp['JRA55'] = JRA_temp_warm_season
#    						dframe_warm_season_temp['MERRA2'] = MERRA2_temp_warm_season
#    						dframe_warm_season_temp['GLDAS-Noah'] = GLDAS_temp_warm_season
#    						dframe_warm_season_temp['GLDAS-CLSM'] = GLDAS_CLSM_temp_warm_season
#    						dframe_warm_season_temp['Naive Blend All Air Temp'] = naive_all_air_warm_season
#    						dframe_warm_season_temp['BEST Air Temp'] = BEST_air_warm_season    						
#    						dframe_warm_season_temp['CFSR Air Temp'] = CFSR_air_warm_season
#    						dframe_warm_season_temp['ERA-Interim Air Temp'] = ERAI_air_warm_season
#    						dframe_warm_season_temp['ERA5 Air Temp'] = ERA5_air_warm_season
#    						dframe_warm_season_temp['ERA5-Land Air Temp'] = ERA5_Land_air_warm_season
#    						dframe_warm_season_temp['JRA55 Air Temp'] = JRA_air_warm_season
#    						dframe_warm_season_temp['MERRA2 Air Temp'] = MERRA2_air_warm_season
#    						dframe_warm_season_temp['GLDAS-Noah Air Temp'] = GLDAS_air_warm_season
#    						dframe_warm_season_temp['GLDAS-CLSM Air Temp'] = GLDAS_CLSM_air_warm_season
#    						dframe_warm_season_temp = dframe_warm_season_temp.dropna()
#    						dframe_warm_season_temp['N'] = len(dframe_warm_season_temp)
#    						dframe_warm_season_temp['Season'] = 'Warm'
#    						dframe_warm_season_temp['Continent'] = continent
#    						dframe_warm_season_temp['RS 2002 Permafrost'] = RS_2002_permafrost
#    						dframe_warm_season_temp['Brown 1970 Permafrost'] = Brown_1970_permafrost
#    						dframe_warm_season_temp['ERA5 MAAT'] = ERA5_MAAT_gcell   
#    						if(len(dframe_warm_season_temp) < 1):
#    							print('Grid Cell Skipped -  Length of warm Season Less than 1')
#    							continue
#
#
############################ Create Final Timeseries #################################
#
#    						date_master_timeseries_cold.append(dframe_cold_season_temp['Date'])    						
#    						gcell_master_timeseries_cold.append(dframe_cold_season_temp['Grid Cell'])
#    						layer_master_timeseries_cold.append(dframe_cold_season_temp['Layer'])
#    						permafrost_type_timeseries_cold.append(dframe_cold_season_temp['RS 2002 Permafrost'])
#    						continent_master_timeseries_cold.append(dframe_cold_season_temp['Continent'])
#    						station_master_timeseries_cold.append(dframe_cold_season_temp['Station'])
#    						naive_all_master_timeseries_cold.append(dframe_cold_season_temp['Naive Blend All'])
#    						CFSR_master_timeseries_cold.append(dframe_cold_season_temp['CFSR'])
#    						ERAI_master_timeseries_cold.append(dframe_cold_season_temp['ERA-Interim'])
#    						ERA5_master_timeseries_cold.append(dframe_cold_season_temp['ERA5'])
#    						ERA5_Land_master_timeseries_cold.append(dframe_cold_season_temp['ERA5-Land'])
#    						JRA_master_timeseries_cold.append(dframe_cold_season_temp['JRA55'])
#    						MERRA2_master_timeseries_cold.append(dframe_cold_season_temp['MERRA2'])
#    						GLDAS_master_timeseries_cold.append(dframe_cold_season_temp['GLDAS-Noah'])
#    						GLDAS_CLSM_master_timeseries_cold.append(dframe_cold_season_temp['GLDAS-CLSM'])
#    						BEST_air_master_timeseries_cold.append(dframe_cold_season_temp['BEST Air Temp'])
#    						naive_all_air_master_timeseries_cold.append(dframe_cold_season_temp['Naive Blend All Air Temp'])
#    						CFSR_air_master_timeseries_cold.append(dframe_cold_season_temp['CFSR Air Temp'])
#    						ERAI_air_master_timeseries_cold.append(dframe_cold_season_temp['ERA-Interim Air Temp'])
#    						ERA5_air_master_timeseries_cold.append(dframe_cold_season_temp['ERA5 Air Temp'])
#    						ERA5_Land_air_master_timeseries_cold.append(dframe_cold_season_temp['ERA5-Land Air Temp'])
#    						JRA_air_master_timeseries_cold.append(dframe_cold_season_temp['JRA55 Air Temp'])
#    						MERRA2_air_master_timeseries_cold.append(dframe_cold_season_temp['MERRA2 Air Temp'])
#    						GLDAS_air_master_timeseries_cold.append(dframe_cold_season_temp['GLDAS-Noah Air Temp'])
#    						GLDAS_CLSM_air_master_timeseries_cold.append(dframe_cold_season_temp['GLDAS-CLSM Air Temp'])
#
#
#    						date_master_timeseries_warm.append(dframe_warm_season_temp['Date'])    						
#    						gcell_master_timeseries_warm.append(dframe_warm_season_temp['Grid Cell'])
#    						layer_master_timeseries_warm.append(dframe_warm_season_temp['Layer'])
#    						permafrost_type_timeseries_warm.append(dframe_warm_season_temp['RS 2002 Permafrost'])
#    						continent_master_timeseries_warm.append(dframe_warm_season_temp['Continent'])
#    						station_master_timeseries_warm.append(dframe_warm_season_temp['Station'])
#    						naive_all_master_timeseries_warm.append(dframe_warm_season_temp['Naive Blend All'])
#    						CFSR_master_timeseries_warm.append(dframe_warm_season_temp['CFSR'])
#    						ERAI_master_timeseries_warm.append(dframe_warm_season_temp['ERA-Interim'])
#    						ERA5_master_timeseries_warm.append(dframe_warm_season_temp['ERA5'])
#    						ERA5_Land_master_timeseries_warm.append(dframe_warm_season_temp['ERA5-Land'])
#    						JRA_master_timeseries_warm.append(dframe_warm_season_temp['JRA55'])
#    						MERRA2_master_timeseries_warm.append(dframe_warm_season_temp['MERRA2'])
#    						GLDAS_master_timeseries_warm.append(dframe_warm_season_temp['GLDAS-Noah'])
#    						GLDAS_CLSM_master_timeseries_warm.append(dframe_warm_season_temp['GLDAS-CLSM'])
#    						BEST_air_master_timeseries_warm.append(dframe_warm_season_temp['BEST Air Temp'])
#    						naive_all_air_master_timeseries_warm.append(dframe_warm_season_temp['Naive Blend All Air Temp'])
#    						CFSR_air_master_timeseries_warm.append(dframe_warm_season_temp['CFSR Air Temp'])
#    						ERAI_air_master_timeseries_warm.append(dframe_warm_season_temp['ERA-Interim Air Temp'])
#    						ERA5_air_master_timeseries_warm.append(dframe_warm_season_temp['ERA5 Air Temp'])
#    						ERA5_Land_air_master_timeseries_warm.append(dframe_warm_season_temp['ERA5-Land Air Temp'])
#    						JRA_air_master_timeseries_warm.append(dframe_warm_season_temp['JRA55 Air Temp'])
#    						MERRA2_air_master_timeseries_warm.append(dframe_warm_season_temp['MERRA2 Air Temp'])
#    						GLDAS_air_master_timeseries_warm.append(dframe_warm_season_temp['GLDAS-Noah Air Temp'])
#    						GLDAS_CLSM_air_master_timeseries_warm.append(dframe_warm_season_temp['GLDAS-CLSM Air Temp'])
#
#
#date_master_timeseries_cold = [i for sub in date_master_timeseries_cold for i in sub]
#gcell_master_timeseries_cold = [i for sub in gcell_master_timeseries_cold for i in sub]
#layer_master_timeseries_cold = [i for sub in layer_master_timeseries_cold for i in sub]
#permafrost_timeseries_cold = [i for sub in permafrost_type_timeseries_cold for i in sub]
#continent_master_timeseries_cold = [i for sub in continent_master_timeseries_cold for i in sub]
#station_master_timeseries_cold = [i for sub in station_master_timeseries_cold for i in sub]
#naive_all_master_timeseries_cold = [i for sub in naive_all_master_timeseries_cold for i in sub]
#CFSR_master_timeseries_cold = [i for sub in CFSR_master_timeseries_cold for i in sub]
#ERAI_master_timeseries_cold = [i for sub in ERAI_master_timeseries_cold for i in sub]
#ERA5_master_timeseries_cold = [i for sub in ERA5_master_timeseries_cold for i in sub]
#JRA_master_timeseries_cold = [i for sub in JRA_master_timeseries_cold for i in sub]
#MERRA2_master_timeseries_cold = [i for sub in MERRA2_master_timeseries_cold for i in sub]
#GLDAS_master_timeseries_cold = [i for sub in GLDAS_master_timeseries_cold for i in sub]
#GLDAS_CLSM_master_timeseries_cold = [i for sub in GLDAS_CLSM_master_timeseries_cold for i in sub]
#BEST_air_master_timeseries_cold = [i for sub in BEST_air_master_timeseries_cold for i in sub]
#naive_all_air_master_timeseries_cold = [i for sub in naive_all_air_master_timeseries_cold for i in sub]
#CFSR_air_master_timeseries_cold = [i for sub in CFSR_air_master_timeseries_cold for i in sub]
#ERAI_air_master_timeseries_cold = [i for sub in ERAI_air_master_timeseries_cold for i in sub]
#ERA5_air_master_timeseries_cold = [i for sub in ERA5_air_master_timeseries_cold for i in sub]
#JRA_air_master_timeseries_cold = [i for sub in JRA_air_master_timeseries_cold for i in sub]
#MERRA2_air_master_timeseries_cold = [i for sub in MERRA2_air_master_timeseries_cold for i in sub]
#GLDAS_air_master_timeseries_cold = [i for sub in GLDAS_air_master_timeseries_cold for i in sub]
#GLDAS_CLSM_air_master_timeseries_cold = [i for sub in GLDAS_CLSM_air_master_timeseries_cold for i in sub]
#
#print(date_master_timeseries_cold)
#print(CFSR_master_timeseries_cold)
#
#
#df_timeseries_cold = pd.DataFrame(data=date_master_timeseries_cold, columns=['Date'])
#df_timeseries_cold['Grid Cell'] = gcell_master_timeseries_cold
#df_timeseries_cold['Layer'] = layer_master_timeseries_cold
#df_timeseries_cold['Permafrost Type'] = permafrost_timeseries_cold
#df_timeseries_cold['Continent'] = continent_master_timeseries_cold
#df_timeseries_cold['Station'] = station_master_timeseries_cold
#df_timeseries_cold['Ensemble Mean'] = naive_all_master_timeseries_cold
#df_timeseries_cold['ERA-Interim'] = ERAI_master_timeseries_cold
#df_timeseries_cold['ERA5'] = ERA5_master_timeseries_cold
#df_timeseries_cold['JRA55'] = JRA_master_timeseries_cold
#df_timeseries_cold['MERRA2'] = MERRA2_master_timeseries_cold
#df_timeseries_cold['GLDAS-Noah'] = GLDAS_master_timeseries_cold
#df_timeseries_cold['GLDAS-CLSM'] = GLDAS_CLSM_master_timeseries_cold
#df_timeseries_cold['BEST Air Temp'] = BEST_air_master_timeseries_cold
#df_timeseries_cold['Ensemble Mean Air Temp Air Temp'] = naive_all_air_master_timeseries_cold
#df_timeseries_cold['ERA-Interim Air Temp'] = ERAI_air_master_timeseries_cold
#df_timeseries_cold['ERA5 Air Temp'] = ERA5_air_master_timeseries_cold
#df_timeseries_cold['JRA55 Air Temp'] = JRA_air_master_timeseries_cold
#df_timeseries_cold['MERRA2 Air Temp'] = MERRA2_air_master_timeseries_cold
#df_timeseries_cold['GLDAS-Noah Air Temp'] = GLDAS_air_master_timeseries_cold
#df_timeseries_cold['GLDAS-CLSM Air Temp'] = GLDAS_CLSM_air_master_timeseries_cold
#
#
#cold_timeseries_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_timeseries/Sep2021/'+str(remap_type)+'_dframe_timeseries_BEST_'+str(temp_thr_o)+'_CMOS_CLSM_subset_permafrost_Sep2021_cold.csv'])
#
#df_timeseries_cold.to_csv(cold_timeseries_fil,index=False)
#
#date_master_timeseries_warm = [i for sub in date_master_timeseries_warm for i in sub]
#gcell_master_timeseries_warm = [i for sub in gcell_master_timeseries_warm for i in sub]
#layer_master_timeseries_warm = [i for sub in layer_master_timeseries_warm for i in sub]
#permafrost_timeseries_warm = [i for sub in permafrost_type_timeseries_warm for i in sub]
#continent_master_timeseries_warm = [i for sub in continent_master_timeseries_warm for i in sub]
#station_master_timeseries_warm = [i for sub in station_master_timeseries_warm for i in sub]
#naive_all_master_timeseries_warm = [i for sub in naive_all_master_timeseries_warm for i in sub]
#CFSR_master_timeseries_warm = [i for sub in CFSR_master_timeseries_warm for i in sub]
#ERAI_master_timeseries_warm = [i for sub in ERAI_master_timeseries_warm for i in sub]
#ERA5_master_timeseries_warm = [i for sub in ERA5_master_timeseries_warm for i in sub]
#JRA_master_timeseries_warm = [i for sub in JRA_master_timeseries_warm for i in sub]
#MERRA2_master_timeseries_warm = [i for sub in MERRA2_master_timeseries_warm for i in sub]
#GLDAS_master_timeseries_warm = [i for sub in GLDAS_master_timeseries_warm for i in sub]
#GLDAS_CLSM_master_timeseries_warm = [i for sub in GLDAS_CLSM_master_timeseries_warm for i in sub]
#BEST_air_master_timeseries_warm = [i for sub in BEST_air_master_timeseries_warm for i in sub]
#naive_all_air_master_timeseries_warm = [i for sub in naive_all_air_master_timeseries_warm for i in sub]
#CFSR_air_master_timeseries_warm = [i for sub in CFSR_air_master_timeseries_warm for i in sub]
#ERAI_air_master_timeseries_warm = [i for sub in ERAI_air_master_timeseries_warm for i in sub]
#ERA5_air_master_timeseries_warm = [i for sub in ERA5_air_master_timeseries_warm for i in sub]
#JRA_air_master_timeseries_warm = [i for sub in JRA_air_master_timeseries_warm for i in sub]
#MERRA2_air_master_timeseries_warm = [i for sub in MERRA2_air_master_timeseries_warm for i in sub]
#GLDAS_air_master_timeseries_warm = [i for sub in GLDAS_air_master_timeseries_warm for i in sub]
#GLDAS_CLSM_air_master_timeseries_warm = [i for sub in GLDAS_CLSM_air_master_timeseries_warm for i in sub]
#
#
#df_timeseries_warm = pd.DataFrame(data=date_master_timeseries_warm, columns=['Date'])
#df_timeseries_warm['Grid Cell'] = gcell_master_timeseries_warm
#df_timeseries_warm['Layer'] = layer_master_timeseries_warm
#df_timeseries_warm['Permafrost Type'] = permafrost_timeseries_warm
#df_timeseries_warm['Continent'] = continent_master_timeseries_warm
#df_timeseries_warm['Station'] = station_master_timeseries_warm
#df_timeseries_warm['Ensemble Mean'] = naive_all_master_timeseries_warm
#df_timeseries_warm['ERA-Interim'] = ERAI_master_timeseries_warm
#df_timeseries_warm['ERA5'] = ERA5_master_timeseries_warm
#df_timeseries_warm['JRA55'] = JRA_master_timeseries_warm
#df_timeseries_warm['MERRA2'] = MERRA2_master_timeseries_warm
#df_timeseries_warm['GLDAS-Noah'] = GLDAS_master_timeseries_warm
#df_timeseries_warm['GLDAS-CLSM'] = GLDAS_CLSM_master_timeseries_warm
#df_timeseries_warm['BEST Air Temp'] = BEST_air_master_timeseries_warm
#df_timeseries_warm['Ensemble Mean Air Temp Air Temp'] = naive_all_air_master_timeseries_warm
#df_timeseries_warm['ERA-Interim Air Temp'] = ERAI_air_master_timeseries_warm
#df_timeseries_warm['ERA5 Air Temp'] = ERA5_air_master_timeseries_warm
#df_timeseries_warm['JRA55 Air Temp'] = JRA_air_master_timeseries_warm
#df_timeseries_warm['MERRA2 Air Temp'] = MERRA2_air_master_timeseries_warm
#df_timeseries_warm['GLDAS-Noah Air Temp'] = GLDAS_air_master_timeseries_warm
#df_timeseries_warm['GLDAS-CLSM Air Temp'] = GLDAS_CLSM_air_master_timeseries_warm
#
#warm_timeseries_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_timeseries/Sep2021/'+str(remap_type)+'_dframe_timeseries_BEST_'+str(temp_thr_o)+'_CMOS_CLSM_subset_permafrost_Sep2021_warm.csv'])
#
#df_timeseries_warm.to_csv(warm_timeseries_fil,index=False)

################ Create TimeSeries Plots ###################

all_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_timeseries/Sep2021/remapcon_dframe_timeseries_BEST_-2C_CMOS_CLSM_subset_permafrost_Sep2021_cold_warm.csv'])

dframe_all = pd.read_csv(all_fil)
date_all = dframe_all['Date']

dates_all = date_all.astype(str)
datetime_all = pd.to_datetime(dates_all)

dframe_all['Datetime'] = datetime_all
dframe_all = dframe_all.set_index(pd.DatetimeIndex(dframe_all['Datetime']))

#### loop through grid cells ####

gcells_all = dframe_all['Grid Cell'].values

gcells_uq = np.unique(gcells_all) # grab unique grid cells to loop through

#gcells_uq = '2211'

print(gcells_uq)


for i in gcells_uq:   #loop through grid cells and grab grid cell level data

    gcell_i = i
    dframe_gcell_all = dframe_all[dframe_all['Grid Cell'] == gcell_i]

    lat_all = dframe_gcell_all['Lat'].values
    lon_all = dframe_gcell_all['Lon'].values
    lat_fig = lat_all[0]
    lon_fig = lon_all[0]
    
    permafrost_type_all = dframe_gcell_all['Permafrost Type'].values
    permafrost_type_all = permafrost_type_all[0]

    if (permafrost_type_all == 'continuous' or permafrost_type_all == 'discontinuous'):
    	permafrost_type = 'permafrost'

    elif (permafrost_type_all == 'none'):
    	permafrost_type = 'none'


    dframe_gcell_all_top = dframe_gcell_all[dframe_gcell_all['Layer'] == 'top_30cm']
    dframe_gcell_all_btm = dframe_gcell_all[dframe_gcell_all['Layer'] == '30cm_300cm']

    dframe_gcell_all_top = dframe_gcell_all_top.resample('M').first().fillna(value=np.nan)
    dframe_gcell_all_btm = dframe_gcell_all_btm.resample('M').first().fillna(value=np.nan)

    date_all_top = dframe_gcell_all_top['Date'].values
    date_all_btm = dframe_gcell_all_btm['Date'].values

    dates_all_top = date_all_top.astype(str)
    dates_all_btm = date_all_btm.astype(str)
    datetime_all_top = pd.to_datetime(dates_all_top)
    datetime_all_btm = pd.to_datetime(dates_all_btm)
    
    station_top = dframe_gcell_all_top['Station'].values
    station_btm = dframe_gcell_all_btm['Station'].values

    naive_all_top = dframe_gcell_all_top['Ensemble Mean'].values
    naive_all_btm = dframe_gcell_all_btm['Ensemble Mean'].values

    CFSR_top = dframe_gcell_all_top['CFSR'].values
    CFSR_btm = dframe_gcell_all_btm['CFSR'].values

    ERAI_top = dframe_gcell_all_top['ERA-Interim'].values
    ERAI_btm = dframe_gcell_all_btm['ERA-Interim'].values

    ERA5_top = dframe_gcell_all_top['ERA5'].values
    ERA5_btm = dframe_gcell_all_btm['ERA5'].values

    ERA5_Land_top = dframe_gcell_all_top['ERA5-Land'].values
    ERA5_Land_btm = dframe_gcell_all_btm['ERA5-Land'].values

    JRA_top = dframe_gcell_all_top['JRA55'].values
    JRA_btm = dframe_gcell_all_btm['JRA55'].values

    MERRA2_top = dframe_gcell_all_top['MERRA2'].values
    MERRA2_btm = dframe_gcell_all_btm['MERRA2'].values

    GLDAS_top = dframe_gcell_all_top['GLDAS-Noah'].values
    GLDAS_btm = dframe_gcell_all_btm['GLDAS-Noah'].values

    GLDAS_CLSM_top = dframe_gcell_all_top['GLDAS-CLSM'].values
    GLDAS_CLSM_btm = dframe_gcell_all_btm['GLDAS-CLSM'].values

## grab dates and create date range that covers all dates across top/btm layers
    dates_uq = np.unique(datetime_all)
    #print(dates_uq)


    dates_uq_str = dates_uq.astype(str)
    datetimes_uq = pd.to_datetime(dates_uq_str).values 

    min_date = min(datetimes_uq)
    max_date = max(datetimes_uq)

    date_format = mdates.DateFormatter('%Y-%m')



## create 2x2 figure with cold season on left, warm season on right; near surface on top and depth at bottom ##

    fig,axs=plt.subplots(nrows=2,ncols=1,sharex='all',figsize=(20,20))
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20)

    ymin = -40
    ymax = 30

    ax1 = plt.subplot(2,1,1) #near surface, cold season
    ax1.plot(datetime_all_top,station_top,label='Station',marker='p',markerfacecolor='violet',markersize=10.0,color='violet',linewidth=10.0)
    ax1.plot(datetime_all_top,naive_all_top,label='Ensemble Mean',marker='p',markerfacecolor='hotpink',markersize=8.0,color='hotpink',linewidth=8.0)
    ax1.plot(datetime_all_top,CFSR_top,label='CFSR',marker='s',markerfacecolor='darkviolet',markersize=3.0,color='darkviolet',linewidth=3.0)
    ax1.plot(datetime_all_top,ERAI_top,label="ERA-Interim",marker='v',markerfacecolor='springgreen',markersize=3.0,color='springgreen',linewidth=3.0,linestyle='dotted')
    ax1.plot(datetime_all_top,ERA5_top,label="ERA5",marker='^',markerfacecolor='greenyellow',markersize=3.0,color='greenyellow',linewidth=3.0,linestyle='dashed')
    ax1.plot(datetime_all_top,ERA5_Land_top,label="ERA5-Land",marker='^',markerfacecolor='dodgerblue',markersize=3.0,color='dodgerblue',linewidth=3.0)
    ax1.plot(datetime_all_top,JRA_top,label="JRA-55",marker='*',markerfacecolor='red',markersize=3.0,color='red',linewidth=3.0)
    ax1.plot(datetime_all_top,MERRA2_top,label="MERRA2",marker='D',markerfacecolor='goldenrod',markersize=3.0,color='goldenrod',linewidth=3.0)
    ax1.plot(datetime_all_top,GLDAS_top,label="GLDAS-Noah",marker='x',markerfacecolor='black',markersize=3.0,color='black',linewidth=3.0)
    ax1.plot(datetime_all_top,GLDAS_CLSM_top,label="GLDAS-CLSM",marker='x',markerfacecolor='darkgrey',markersize=3.0,color='darkgrey',linewidth=3.0)
    ax1.set(xlim=[min_date,max_date])


    handles1 = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)]
    ax1text = []
    ax1text.append('Near Surface')
    ax1.legend(handles1, ax1text, loc='upper left', fontsize=20, fancybox=False, framealpha=0, handlelength=0, handletextpad=0)

    ax1.tick_params(axis="x", labelsize=20)
    ax1.tick_params(axis="y", labelsize=20)
    
    ax1.set_xlim(min_date,max_date)
    ax1.set_ylim(ymin,ymax)

    ax3 = plt.subplot(2,1,2) #depth, cold season
    ax3.plot(datetime_all_btm,station_btm,label='Station',marker='p',markerfacecolor='violet',markersize=10.0,color='violet',linewidth=10.0)
    ax3.plot(datetime_all_btm,naive_all_btm,label='Ensemble Mean',marker='p',markerfacecolor='hotpink',markersize=8.0,color='hotpink',linewidth=8.0)
    ax3.plot(datetime_all_btm,CFSR_btm,label='CFSR',marker='s',markerfacecolor='darkviolet',markersize=3.0,color='darkviolet',linewidth=3.0)
    ax3.plot(datetime_all_btm,ERAI_btm,label="ERA-Interim",marker='v',markerfacecolor='springgreen',markersize=3.0,color='springgreen',linewidth=3.0,linestyle='dotted')
    ax3.plot(datetime_all_btm,ERA5_btm,label="ERA5",marker='^',markerfacecolor='greenyellow',markersize=3.0,color='greenyellow',linewidth=3.0,linestyle='dashed')
    ax3.plot(datetime_all_btm,ERA5_Land_btm,label="ERA5-Land",marker='^',markerfacecolor='dodgerblue',markersize=3.0,color='dodgerblue',linewidth=3.0)
    ax3.plot(datetime_all_btm,JRA_btm,label="JRA-55",marker='*',markerfacecolor='red',markersize=3.0,color='red',linewidth=3.0)
    ax3.plot(datetime_all_btm,MERRA2_btm,label="MERRA2",marker='D',markerfacecolor='goldenrod',markersize=3.0,color='goldenrod',linewidth=3.0)
    ax3.plot(datetime_all_btm,GLDAS_btm,label="GLDAS-Noah",marker='x',markerfacecolor='black',markersize=3.0,color='black',linewidth=3.0)
    ax3.plot(datetime_all_btm,GLDAS_CLSM_btm,label="GLDAS-CLSM",marker='x',markerfacecolor='darkgrey',markersize=3.0,color='darkgrey',linewidth=3.0)
     
    ax3.set(xlim=[min_date,max_date])

    handles3 = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)]
    ax3text = []
    ax3text.append('Depth, Cold Season')
    ax3.legend(handles3, ax3text, loc='upper left', fontsize=20, fancybox=False, framealpha=0, handlelength=0, handletextpad=0)

    ax3.tick_params(axis="x", labelsize=20)
    ax3.tick_params(axis="y", labelsize=20)
    
    ax3.set_xlim(min_date,max_date)
    ax3.set_ylim(ymin,ymax)


    lines = []
    labels = []
    for ax in fig.get_axes():
    	axLine, axLabel = ax.get_legend_handles_labels()
    	lines.extend(axLine)
    	labels.extend(axLabel)



    fig.add_subplot(111,frameon=False)
    plt.tick_params(labelcolor='none',bottom=False,left=False)
    legend = fig.legend(lines[0:10],labels[0:10],loc="upper right",fontsize=25,title="Legend")
    legend.get_title().set_fontsize('20') #legend 'Title' fontsize

    plt.xlabel('Date', fontweight='bold',fontsize=20)
    plt.ylabel('Soil Temperature($^\circ$ C)',fontweight='bold', fontsize=20) 

    fig_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/plots/naive_blend_timeseries/Sep2021/2_panel/'+str(permafrost_type)+'/'+str(remap_type)+'_2_panel_grid_cell_'+str(gcell_i)+'.png'])
    plt.tight_layout()




    at = AnchoredText("lat:"+str(lat_fig)+"($^\circ$N) , lon:"+str(lon_fig)+"($^\circ$E)", prop=dict(size=20), frameon=False, loc='lower left')
    #at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax1.add_artist(at)

    plt.setp(ax1.get_xticklabels(),rotation=45,ha='right')
    plt.setp(ax3.get_xticklabels(),rotation=45,ha='right')

    plt.gca().xaxis.set_major_formatter(date_format)
    plt.savefig(fig_fil)
    plt.close()
