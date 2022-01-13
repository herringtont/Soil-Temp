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
from scipy.stats import pearsonr
from matplotlib.ticker import (MultipleLocator, AutoLocator, AutoMinorLocator)


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
tmp_type = ['raw_temp']


ERA5_air = 'Air_Temp'


############# Grab Reanalysis Data ############

for i in rmp_type:
    rmp_type_i = i
    remap_type = ''.join(['remap'+rmp_type_i])
    rnys_dir = ''.join(['/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/obs_depths/common_grid/'+str(remap_type)+'/common_date/'])

    air_dir = ''.join(['/mnt/data/users/herringtont/soil_temp/reanalysis/2m_AirTemp/rename/land_only/common_grid/'+str(remap_type)+'/'])

    CFSR_fi = "".join([rnys_dir,"CFSR_all.nc"])
    MERRA2_fi = "".join([rnys_dir,"MERRA2.nc"])
    ERA5_fi = "".join([rnys_dir,"ERA5.nc"])
    ERA5_air_fi = "".join([air_dir,"ERA5_2m_air.nc"])
    ERA5_Land_fi = "".join([rnys_dir,"ERA5_Land.nc"])
    ERAI_fi = "".join([rnys_dir,"ERA-Interim.nc"])
    JRA_fi = "".join([rnys_dir,"JRA55.nc"])
    GLDAS_fi = "".join([rnys_dir,"GLDAS.nc"])
    GLDAS_CLSM_fi = "".join([rnys_dir,"GLDAS_CLSM.nc"])
    ERA5_air_fi = "".join([air_dir,"ERA5_2m_air.nc"])

    for j in naive_type:
    	naive_type_j = j
    	naive_dir_raw = ''.join(['/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/obs_depths/common_grid/'+str(remap_type)+'/common_date/'])
    	naive_fi = ''.join([naive_dir_raw+str(remap_type)+'_'+'naive_blend.nc'])
    	naive_noJRA_fi = ''.join([naive_dir_raw+str(remap_type)+'_'+'naive_blend_noJRA55.nc'])
    	naive_noJRAold_fi = ''.join([naive_dir_raw+str(remap_type)+'_'+'naive_blend_noJRA55_old.nc'])
    	naive_all_fi = ''.join([naive_dir_raw+str(remap_type)+'_'+'naive_blend_ERA5L_GLDAS_CLSM.nc'])
    	for k in olr:
    		olr_k = k

    		for l in lyr:
    			lyr_l = l
    			if (lyr_l == 'top_30cm'):
    				CFSR_layer = "Soil_Temp_TOP30"
    				CFSR2_layer = "Soil_Temp_TOP30"
    				GLDAS_layer = "Soil_Temp_TOP30"
    				GLDAS_CLSM_layer = "Soil_Temp_TOP30"
    				ERA5_layer = "Soil_Temp_TOP30"
    				ERA5_Land_layer = "Soil_Temp_TOP30"
    				ERAI_layer = "Soil_Temp_TOP30"
    				JRA_layer = "Soil_Temp_TOP30"
    				MERRA2_layer = "Soil_Temp_TOP30"
    				Naive_layer = "Soil_Temp_TOP30"

    				in_situ_layer = 'top_30cm'

    			if (lyr_l == '30cm_300cm'):
    				CFSR_layer = "Soil_Temp_30cm_300cm"
    				CFSR2_layer = "Soil_Temp_30cm_300cm"
    				GLDAS_layer = "Soil_Temp_30cm_300cm"
    				GLDAS_CLSM_layer = "Soil_Temp_30cm_300cm"
    				ERA5_layer = "Soil_Temp_30cm_300cm"
    				ERA5_Land_layer = "Soil_Temp_30cm_300cm"
    				ERAI_layer = "Soil_Temp_30cm_300cm"
    				JRA_layer = "Soil_Temp_30cm_300cm"
    				MERRA2_layer = "Soil_Temp_30cm_300cm"
    				Naive_layer = "Soil_Temp_30cm_300cm"

    				in_situ_layer = '30_299.9'


    			print("Remap Type:",remap_type)
    			print("Layer:",lyr_l)

    			for m in thr:
    				thr_m = m
    				insitu_dir =  ''.join(['/mnt/data/users/herringtont/soil_temp/In-Situ/All/spatial_average_anom/'+str(remap_type)+'/no_outliers/'+str(olr_k)+'/'+str(in_situ_layer)+'/thr_'+str(thr_m)+'/'])
##### Create Master Arrays #####
    				grid_cell_master = []
    				lat_master = []
    				lon_master = []
    				date_master = []
    				sample_size_master = []
    				sites_incl_master = []

    				station_data_master = []
    				naive_data_master = []
    				naive_noJRA_data_master = []
    				naive_noJRAold_data_master = []
    				naive_all_data_master = []
    				CFSR_data_master = []
    				ERAI_data_master = []
    				ERA5_data_master = []
    				ERA5_Land_data_master = []
    				JRA_data_master = []
    				MERRA2_data_master = []
    				GLDAS_data_master = []
    				GLDAS_CLSM_data_master = []

################# loop through in-situ files ###############
    				#print(type(CFSR_anom))
    				#pathlist = os_sorted(os.listdir(insitu_dir))
    				pathlist = os.listdir(insitu_dir)
    				pathlist_sorted = natural_sort(pathlist)

    				for path in pathlist_sorted:
    					insitu_fil = ''.join([insitu_dir,path])
    					if (os.path.isdir(insitu_fil)):
    						continue #skip all sub-directories
    					#print(insitu_fil)
    					dframe_insitu = pd.read_csv(insitu_fil)
    					dattim = dframe_insitu['Date'].values
    					DateTime = [datetime.datetime.strptime(x,'%Y-%m-%d') for x in dattim]
    					soil_temp = dframe_insitu['Spatial Avg Temp']
    					gcell = dframe_insitu['Grid Cell'].iloc[0]
    					lat_cen = dframe_insitu['Central Lat'].iloc[0]
    					lon_cen = dframe_insitu['Central Lon'].iloc[0]
    					lon_cen_180 = (lon_cen + 180) % 360 - 180
    					sites_incl = dframe_insitu['Sites Incl'].values
    					avg_sites = np.mean(sites_incl)

    					print("Grid Cell:",gcell)
    
    					CFSR_fil = xr.open_dataset(CFSR_fi)
    					ERAI_fil = xr.open_dataset(ERAI_fi)
    					ERA5_fil = xr.open_dataset(ERA5_fi)
    					ERA5_Land_fil = xr.open_dataset(ERA5_Land_fi)
    					JRA_fil = xr.open_dataset(JRA_fi)
    					MERRA2_fil = xr.open_dataset(MERRA2_fi)
    					GLDAS_fil = xr.open_dataset(GLDAS_fi)
    					GLDAS_CLSM_fil = xr.open_dataset(GLDAS_CLSM_fi)
    					ERA5_air_fil = xr.open_dataset(ERA5_air_fi)


    					CFSR_stemp = CFSR_fil[CFSR_layer] - 273.15
    					ERAI_stemp = ERAI_fil[ERAI_layer] - 273.15
    					ERA5_stemp = ERA5_fil[ERA5_layer] - 273.15
    					ERA5_Land_stemp = ERA5_Land_fil[ERA5_Land_layer] - 273.15
    					JRA_stemp = JRA_fil[JRA_layer] - 273.15
    					MERRA2_stemp = MERRA2_fil[MERRA2_layer] -273.15
    					GLDAS_stemp = GLDAS_fil[GLDAS_layer] - 273.15
    					GLDAS_CLSM_stemp = GLDAS_CLSM_fil[GLDAS_CLSM_layer] - 273.15

    					ERA5_tair = ERA5_air_fil[ERA5_air] - 273.15

    					naive_fil = xr.open_dataset(naive_fi)
    					naive_stemp = naive_fil[Naive_layer] - 273.15

    					naive_noJRA_fil = xr.open_dataset(naive_noJRA_fi)
    					naive_noJRA_stemp = naive_noJRA_fil[Naive_layer] - 273.15

    					naive_noJRAold_fil = xr.open_dataset(naive_noJRAold_fi)
    					naive_noJRAold_stemp = naive_noJRAold_fil[Naive_layer] - 273.15

    					naive_all_fil = xr.open_dataset(naive_all_fi)
    					naive_all_stemp = naive_all_fil[Naive_layer] - 273.15

  					#print(type(CFSR_stemp))

    					CFSR_stemp_gcell = CFSR_stemp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    					ERAI_stemp_gcell = ERAI_stemp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    					ERA5_stemp_gcell = ERA5_stemp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    					ERA5_Land_stemp_gcell = ERA5_Land_stemp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    					JRA_stemp_gcell = JRA_stemp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    					MERRA2_stemp_gcell = MERRA2_stemp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    					GLDAS_stemp_gcell = GLDAS_stemp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    					GLDAS_CLSM_stemp_gcell = GLDAS_CLSM_stemp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    					naive_stemp_gcell = naive_stemp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    					naive_noJRA_stemp_gcell = naive_noJRA_stemp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    					naive_noJRAold_stemp_gcell = naive_noJRAold_stemp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    					naive_all_stemp_gcell = naive_all_stemp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)					
    					ERA5_tair_gcell = ERA5_tair.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)



    					rnys_dattim = naive_fil['time']
    					rnys_datetime = rnys_dattim.dt.strftime('%Y-%m-%d')    					
    					len_rnys_dattim = len(rnys_dattim) - 1
    					rnys_sdate = rnys_dattim.isel(time=0).values
    					rnys_sdate_str = str(rnys_sdate)
    					rnys_sdate_dt = datetime.datetime.strptime(rnys_sdate_str[0:10],'%Y-%m-%d')
    					rnys_edate = rnys_dattim.isel(time=len_rnys_dattim).values
    					rnys_edate_str = str(rnys_edate)
    					rnys_edate_dt = datetime.datetime.strptime(rnys_edate_str[0:10],'%Y-%m-%d')

    					CFSR_dattim = CFSR_fil['time']
    					CFSR_datetime = CFSR_dattim.dt.strftime('%Y-%m-%d')
    					len_CFSR_dattim = len(CFSR_dattim) - 1 
    					CFSR_sdate = CFSR_dattim.isel(time=0).values
    					CFSR_sdate_str = str(CFSR_sdate)
    					CFSR_sdate_dt = datetime.datetime.strptime(CFSR_sdate_str[0:10],'%Y-%m-%d')
    					CFSR_edate = CFSR_dattim.isel(time=len_CFSR_dattim).values
    					CFSR_edate_str = str(CFSR_edate)
    					CFSR_edate_dt = datetime.datetime.strptime(CFSR_edate_str[0:10],'%Y-%m-%d')

#################### grab collocated temperature data from reanalysis files #######################

    					CFSR_temp_master = []
    					JRA_temp_master = []
    					ERAI_temp_master = []
    					ERA5_temp_master = []
    					ERA5_air_master = []
    					ERA5_Land_temp_master = []
    					MERRA2_temp_master = []
    					GLDAS_temp_master = []
    					GLDAS_CLSM_temp_master = []
    					naive_temp_master = []
    					naive_noJRA_temp_master = []
    					naive_noJRAold_temp_master = []
    					naive_all_temp_master = []
    					station_temp_master = []
    					station_anom_master = []
    					date_temp_master = []

    					for n in range(0,len(DateTime)):
    						DateTime_m = DateTime[n]
    						dattim_m = dattim[n]
    						if(DateTime_m < rnys_sdate_dt): #skip all dates before 1980
    							continue
    						if(DateTime_m > rnys_edate_dt): #skip all dates beyond last reanalysis date
    							continue
    						naive_temp_dt = naive_stemp_gcell.sel(time=DateTime_m).values.tolist()
    						if(str(naive_temp_dt) == "nan"):
    							naive_temp_dt = np.nan 
    						naive_noJRA_temp_dt = naive_noJRA_stemp_gcell.sel(time=DateTime_m).values.tolist()
    						if(str(naive_noJRA_temp_dt) == "nan"):
    							naive_noJRA_temp_dt = np.nan 
    						naive_noJRAold_temp_dt = naive_noJRAold_stemp_gcell.sel(time=DateTime_m).values.tolist()
    						if(str(naive_noJRAold_temp_dt) == "nan"):
    							naive_noJRAold_temp_dt = np.nan
    						naive_all_temp_dt = naive_all_stemp_gcell.sel(time=DateTime_m).values.tolist()
    						if(str(naive_all_temp_dt) == "nan"):
    							naive_all_temp_dt = np.nan  						
    						dframe_insitu_dt = dframe_insitu[dframe_insitu['Date'] == dattim_m]
    						station_temp_dt = dframe_insitu_dt['Spatial Avg Temp'].values
    						extracted_value = station_temp_dt[0]
    						if(str(station_temp_dt) == "nan"):
    							station_temp_dt = np.nan
    						station_temp_master.append(float(extracted_value))
    						station_anom_dt = dframe_insitu_dt['Spatial Avg Anom'].values.tolist()
    						if(str(station_anom_dt) == "nan"):
    							station_anom_dt = np.nan
    						station_anom_master.append(station_anom_dt)									
    						CFSR_temp_dt = CFSR_stemp_gcell.sel(time=DateTime_m).values.tolist()
    						if(str(CFSR_temp_dt) == "nan"):
    							CFSR_temp_dt = np.nan
    						CFSR_temp_master.append(CFSR_temp_dt)    						
    						JRA_temp_dt = JRA_stemp_gcell.sel(time=DateTime_m).values.tolist()
    						if(str(JRA_temp_dt) == "nan"):
    							JRA_temp_dt = np.nan
    						JRA_temp_master.append(JRA_temp_dt)      							
    						ERAI_temp_dt = ERAI_stemp_gcell.sel(time=DateTime_m).values.tolist()
    						if(str(ERAI_temp_dt) == "nan"):
    							ERAI_temp_dt = np.nan
    						ERAI_temp_master.append(ERAI_temp_dt)
    						ERA5_air_dt = ERA5_tair_gcell.sel(time=DateTime_m).values.tolist()
    						if(str(ERA5_air_dt) == "nan"):
    							ERA5_air_dt = np.nan
    						ERA5_air_master.append(ERA5_air_dt)
    						ERA5_temp_dt = ERA5_stemp_gcell.sel(time=DateTime_m).values.tolist()
    						if(str(ERA5_temp_dt) == "nan"):
    							ERA5_temp_dt = np.nan
    						ERA5_temp_master.append(ERA5_temp_dt)
    						ERA5_Land_temp_dt = ERA5_Land_stemp_gcell.sel(time=DateTime_m).values.tolist()
    						if(str(ERA5_Land_temp_dt) == "nan"):
    							ERA5_Land_temp_dt = np.nan
    						ERA5_Land_temp_master.append(ERA5_Land_temp_dt)
    						MERRA2_temp_dt = MERRA2_stemp_gcell.sel(time=DateTime_m).values.tolist()
    						if(str(MERRA2_temp_dt) == "nan"):
    							MERRA2_temp_dt = np.nan
    						MERRA2_temp_master.append(MERRA2_temp_dt)
    						GLDAS_temp_dt = GLDAS_stemp_gcell.sel(time=DateTime_m).values.tolist()
    						if(str(GLDAS_temp_dt) == "nan"):
    							GLDAS_temp_dt = np.nan
    						GLDAS_temp_master.append(GLDAS_temp_dt)
    						GLDAS_CLSM_temp_dt = GLDAS_CLSM_stemp_gcell.sel(time=DateTime_m).values.tolist()
    						if(str(GLDAS_CLSM_temp_dt) == "nan"):
    							GLDAS_CLSM_temp_dt = np.nan
    						GLDAS_CLSM_temp_master.append(GLDAS_CLSM_temp_dt)
    						date_temp_master.append(dattim_m)    						
    						naive_temp_master.append(naive_temp_dt)            							    						
    						naive_noJRA_temp_master.append(naive_noJRA_temp_dt) 
    						naive_noJRAold_temp_master.append(naive_noJRAold_temp_dt)
    						naive_all_temp_master.append(naive_all_temp_dt)

    					station_temp_master = np.array(station_temp_master)
    					station_anom_master = np.array(station_anom_master)
    					date_temp_master = np.array(date_temp_master)
    					CFSR_temp_master = np.array(CFSR_temp_master)
    					ERAI_temp_master = np.array(ERAI_temp_master)
    					ERA5_temp_master = np.array(ERA5_temp_master)
    					ERA5_air_master = np.array(ERA5_air_master)
    					ERA5_Land_temp_master = np.array(ERA5_Land_temp_master)
    					JRA_temp_master = np.array(JRA_temp_master)
    					MERRA2_temp_master = np.array(MERRA2_temp_master)
    					GLDAS_temp_master = np.array(GLDAS_temp_master)
    					GLDAS_CLSM_temp_master = np.array(GLDAS_CLSM_temp_master)
    					naive_temp_master = np.array(naive_temp_master)
    					naive_noJRA_temp_master = np.array(naive_noJRA_temp_master)
    					naive_noJRAold_temp_master = np.array(naive_noJRAold_temp_master)
    					naive_all_temp_master = np.array(naive_all_temp_master)
    					naive_no_nan = naive_all_temp_master[~np.isnan(naive_all_temp_master)]

    					#print(naive_no_nan,TC_no_nan)

    					CFSR_no_nan = CFSR_temp_master[~np.isnan(CFSR_temp_master)]
    					#print(CFSR_no_nan)

    					if(DateTime[0]>CFSR_edate_dt or DateTime[len(DateTime) -1] < CFSR_sdate_dt): #skip if the CFSR dates and station dates do not overlap
    						print('Grid Cell Skipped - CFSR dates do not overlap')
    						continue

    					
    					if(len(naive_no_nan) == 0 or len(CFSR_no_nan) == 0): #skip if there are NaN values in blended data
    						print('Grid Cell Skipped - No Naive Blended Data')
    						continue

    					


################## Separate by cold and warm season (by temperature) ####################

###### Cold Season (Soil Temp <= -2) #####

    					cold_season_index = np.where(ERA5_air_master <= -2)
    					cold_idx = cold_season_index[0]

    					station_temp_cold_season = []
    					naive_temp_cold_season = []
    					naive_noJRA_temp_cold_season = []
    					naive_noJRAold_temp_cold_season = []
    					naive_all_temp_cold_season = []
    					CFSR_temp_cold_season = []
    					ERAI_temp_cold_season = []
    					ERA5_temp_cold_season = []
    					ERA5_Land_temp_cold_season = []
    					JRA_temp_cold_season = []
    					MERRA2_temp_cold_season = []
    					GLDAS_temp_cold_season = []
    					GLDAS_CLSM_temp_cold_season = []
    					for x in cold_idx:
    						station_x = station_temp_master[x].tolist()
    						station_temp_cold_season.append(station_x)
    						naive_x = naive_temp_master[x].tolist()
    						naive_temp_cold_season.append(naive_x)
    						naive_noJRA_x = naive_noJRA_temp_master[x].tolist()
    						naive_noJRA_temp_cold_season.append(naive_noJRA_x)
    						naive_noJRAold_x = naive_noJRAold_temp_master[x].tolist()
    						naive_noJRAold_temp_cold_season.append(naive_noJRAold_x)
    						naive_all_x = naive_all_temp_master[x].tolist()
    						naive_all_temp_cold_season.append(naive_all_x)
    						CFSR_x = CFSR_temp_master[x].tolist()
    						CFSR_temp_cold_season.append(CFSR_x)
    						ERAI_x = ERAI_temp_master[x].tolist()
    						ERAI_temp_cold_season.append(ERAI_x)
    						ERA5_x = ERA5_temp_master[x].tolist()
    						ERA5_temp_cold_season.append(ERA5_x)
    						ERA5_Land_x = ERA5_Land_temp_master[x].tolist()
    						ERA5_Land_temp_cold_season.append(ERA5_Land_x)
    						JRA_x = JRA_temp_master[x].tolist()
    						JRA_temp_cold_season.append(JRA_x)
    						MERRA2_x = MERRA2_temp_master[x].tolist()
    						MERRA2_temp_cold_season.append(MERRA2_x)
    						GLDAS_x = GLDAS_temp_master[x].tolist()
    						GLDAS_temp_cold_season.append(GLDAS_x)
    						GLDAS_CLSM_x = GLDAS_temp_master[x].tolist()
    						GLDAS_CLSM_temp_cold_season.append(GLDAS_x)
					   				

    					if(len(naive_temp_cold_season) < 1 or len(station_temp_cold_season) < 1 or len(CFSR_temp_cold_season) < 1):
    						print('Grid Cell Skipped - Length of Cold Season Less than 1')
    						continue

    					dframe_cold_season_temp = pd.DataFrame(data = station_temp_cold_season, columns=['Station'])
    					dframe_cold_season_temp['Grid Cell'] = gcell
    					dframe_cold_season_temp['Lat'] = lat_cen
    					dframe_cold_season_temp['Lon'] = lon_cen
    					dframe_cold_season_temp['Naive Blend'] = naive_temp_cold_season
    					dframe_cold_season_temp['Naive Blend no JRA55'] = naive_noJRA_temp_cold_season
    					dframe_cold_season_temp['Naive Blend no JRA55 Old'] = naive_noJRAold_temp_cold_season
    					dframe_cold_season_temp['Naive Blend All'] = naive_all_temp_cold_season
    					dframe_cold_season_temp['CFSR'] = CFSR_temp_cold_season
    					dframe_cold_season_temp['ERA-Interim'] = ERAI_temp_cold_season
    					dframe_cold_season_temp['ERA5'] = ERA5_temp_cold_season
    					dframe_cold_season_temp['ERA5-Land'] = ERA5_Land_temp_cold_season
    					dframe_cold_season_temp['JRA55'] = JRA_temp_cold_season
    					dframe_cold_season_temp['MERRA2'] = MERRA2_temp_cold_season
    					dframe_cold_season_temp['GLDAS-Noah'] = GLDAS_temp_cold_season
    					dframe_cold_season_temp['GLDAS-CLSM'] = GLDAS_CLSM_temp_cold_season
    					dframe_cold_season_temp = dframe_cold_season_temp.dropna()
    					dframe_cold_season_temp['N'] = len(dframe_cold_season_temp)
    					dframe_cold_season_temp['Season'] = 'Cold' 
    					if(len(dframe_cold_season_temp) < 1):
    						print('Grid Cell Skipped - Length of Cold Season Less than 1')
    						continue

###### warm Season (Soil Temp <= -2) #####
    					warm_season_index = np.where(ERA5_air_master > -2)
    					warm_idx = warm_season_index[0]

    					station_temp_warm_season = []
    					naive_temp_warm_season = []
    					naive_noJRA_temp_warm_season = []
    					naive_noJRAold_temp_warm_season = []
    					naive_all_temp_warm_season = []
    					CFSR_temp_warm_season = []
    					ERAI_temp_warm_season = []
    					ERA5_temp_warm_season = []
    					ERA5_Land_temp_warm_season = []
    					JRA_temp_warm_season = []
    					MERRA2_temp_warm_season = []
    					GLDAS_temp_warm_season = []
    					GLDAS_CLSM_temp_warm_season = []
    					for x in warm_idx:
    						station_x = station_temp_master[x].tolist()
    						station_temp_warm_season.append(station_x)
    						naive_x = naive_temp_master[x].tolist()
    						naive_temp_warm_season.append(naive_x)
    						naive_noJRA_x = naive_noJRA_temp_master[x].tolist()
    						naive_noJRA_temp_warm_season.append(naive_noJRA_x)
    						naive_noJRAold_x = naive_noJRAold_temp_master[x].tolist()
    						naive_noJRAold_temp_warm_season.append(naive_noJRAold_x)
    						naive_all_x = naive_all_temp_master[x].tolist()
    						naive_all_temp_warm_season.append(naive_all_x)
    						CFSR_x = CFSR_temp_master[x].tolist()
    						CFSR_temp_warm_season.append(CFSR_x)
    						ERAI_x = ERAI_temp_master[x].tolist()
    						ERAI_temp_warm_season.append(ERAI_x)
    						ERA5_x = ERA5_temp_master[x].tolist()
    						ERA5_temp_warm_season.append(ERA5_x)
    						ERA5_Land_x = ERA5_Land_temp_master[x].tolist()
    						ERA5_Land_temp_warm_season.append(ERA5_Land_x)
    						JRA_x = JRA_temp_master[x].tolist()
    						JRA_temp_warm_season.append(JRA_x)
    						MERRA2_x = MERRA2_temp_master[x].tolist()
    						MERRA2_temp_warm_season.append(MERRA2_x)
    						GLDAS_x = GLDAS_temp_master[x].tolist()
    						GLDAS_temp_warm_season.append(GLDAS_x)
    						GLDAS_CLSM_x = GLDAS_temp_master[x].tolist()
    						GLDAS_CLSM_temp_warm_season.append(GLDAS_x)
					   				

    					if(len(naive_temp_warm_season) < 1 or len(station_temp_warm_season) < 1 or len(CFSR_temp_warm_season) < 1):
    						print('Grid Cell Skipped - Length of warm Season Less than 1')
    						continue

    					dframe_warm_season_temp = pd.DataFrame(data = station_temp_warm_season, columns=['Station'])
    					dframe_warm_season_temp['Grid Cell'] = gcell
    					dframe_warm_season_temp['Lat'] = lat_cen
    					dframe_warm_season_temp['Lon'] = lon_cen
    					dframe_warm_season_temp['Naive Blend'] = naive_temp_warm_season
    					dframe_warm_season_temp['Naive Blend no JRA55'] = naive_noJRA_temp_warm_season
    					dframe_warm_season_temp['Naive Blend no JRA55 Old'] = naive_noJRAold_temp_warm_season
    					dframe_warm_season_temp['Naive Blend All'] = naive_all_temp_warm_season
    					dframe_warm_season_temp['CFSR'] = CFSR_temp_warm_season
    					dframe_warm_season_temp['ERA-Interim'] = ERAI_temp_warm_season
    					dframe_warm_season_temp['ERA5'] = ERA5_temp_warm_season
    					dframe_warm_season_temp['ERA5-Land'] = ERA5_Land_temp_warm_season
    					dframe_warm_season_temp['JRA55'] = JRA_temp_warm_season
    					dframe_warm_season_temp['MERRA2'] = MERRA2_temp_warm_season
    					dframe_warm_season_temp['GLDAS-Noah'] = GLDAS_temp_warm_season
    					dframe_warm_season_temp['GLDAS-CLSM'] = GLDAS_CLSM_temp_warm_season
    					dframe_warm_season_temp = dframe_warm_season_temp.dropna()
    					dframe_warm_season_temp['N'] = len(dframe_warm_season_temp)
    					dframe_warm_season_temp['Season'] = 'warm' 
    					if(len(dframe_warm_season_temp) < 1):
    						print('Grid Cell Skipped - Length of warm Season Less than 1')
    						continue

    					dframe_stn = pd.DataFrame(data = station_temp_master, columns=['Station'])
    					#dframe_stn.insert(0,'Grid Cell', gcell)
    					#dframe_stn.insert(1,'Central Lat', lat_cen)
    					#dframe_stn.insert(2,'Central Lon', lon_cen)
    					dframe_stn.insert(0,'Date', date_temp_master)
    					dframe_stn['Naive Blend'] = naive_temp_master
    					dframe_stn['Naive Blend no JRA55'] = naive_noJRA_temp_master
    					dframe_stn['Naive Blend no JRA55 Old'] = naive_noJRAold_temp_master
    					dframe_stn['Naive Blend All'] = naive_all_temp_master
    					dframe_stn['CFSR'] = CFSR_temp_master				
    					dframe_stn['ERA-Interim'] = ERAI_temp_master
    					dframe_stn['ERA5'] = ERA5_temp_master
    					dframe_stn['ERA5-Land'] = ERA5_Land_temp_master
    					dframe_stn['JRA-55'] = JRA_temp_master
    					dframe_stn['MERRA2'] = MERRA2_temp_master
    					dframe_stn['GLDAS-Noah'] = GLDAS_temp_master
    					dframe_stn['GLDAS-CLSM'] = GLDAS_CLSM_temp_master

    					min_year = np.datetime64(min(date_temp_master)) - np.timedelta64(365,'D')
    					max_year = np.datetime64(max(date_temp_master)) + np.timedelta64(365,'D')
    					dt_rng = pd.date_range(min_year,max_year, freq='MS')

    					DateTime = pd.DatetimeIndex(dframe_stn['Date'])
    					dframe_stn2 = dframe_stn.set_index(DateTime)
    					#print(dframe_stn2)
    					dframe_stn3 = dframe_stn2.reindex(dt_rng)
    					#print(dframe_stn3)

    					dframe_stn3.insert(0,'Grid Cell', gcell)
    					dframe_stn3.insert(1, 'Central Lat', lat_cen)
    					dframe_stn3.insert(2, 'Central Lon', lon_cen_180)
    					DateTime2 = dframe_stn3.index.to_pydatetime()
    					DateTime3 = [z.strftime("%Y-%m-%d") for z in DateTime2]
    					dframe_stn3.insert(3, 'Sites Incl', avg_sites) 				 
    					dframe_stn3.insert(4,'DateTime',DateTime3)

    					#print(dframe_stn3)
    					grid_cell_master.append(dframe_stn3['Grid Cell'].values.tolist())
    					lat_master.append(dframe_stn3['Central Lat'].values.tolist())
    					lon_master.append(dframe_stn3['Central Lon'].values.tolist())
    					date_master.append(dframe_stn3['DateTime'].values.tolist())
    					sites_incl_master.append(dframe_stn3['Sites Incl'].values.tolist())
    					station_data_master.append(dframe_stn3['Station'].values.tolist())
    					naive_data_master.append(dframe_stn3['Naive Blend'].values.tolist())
    					naive_noJRA_data_master.append(dframe_stn3['Naive Blend no JRA55'].values.tolist())
    					naive_noJRAold_data_master.append(dframe_stn3['Naive Blend no JRA55 Old'].values.tolist())
    					naive_all_data_master.append(dframe_stn3['Naive Blend All'].values.tolist())
    					CFSR_data_master.append(dframe_stn3['CFSR'].values.tolist())
    					ERAI_data_master.append(dframe_stn3['ERA-Interim'].values.tolist())
    					ERA5_data_master.append(dframe_stn3['ERA5'].values.tolist())
    					ERA5_Land_data_master.append(dframe_stn3['ERA5-Land'].values.tolist())
    					JRA_data_master.append(dframe_stn3['JRA-55'].values.tolist())
    					MERRA2_data_master.append(dframe_stn3['MERRA2'].values.tolist())				
    					GLDAS_data_master.append(dframe_stn3['GLDAS-Noah'].values.tolist())
    					GLDAS_CLSM_data_master.append(dframe_stn3['GLDAS-CLSM'].values.tolist())

    				grid_cell_master = [i for sub in grid_cell_master for i in sub]
    				lat_master = [i for sub in lat_master for i in sub]
    				lon_master = [i for sub in lon_master for i in sub]
    				sites_incl_master = [i for sub in sites_incl_master for i in sub]
    				date_master = [i for sub in date_master for i in sub]
    				station_data_master = [i for sub in station_data_master for i in sub]
    				naive_data_master = [i for sub in naive_data_master for i in sub]
    				naive_noJRA_data_master = [i for sub in naive_noJRA_data_master for i in sub]
    				naive_noJRAold_data_master = [i for sub in naive_noJRAold_data_master for i in sub]
    				naive_all_data_master = [i for sub in naive_all_data_master for i in sub]
    				CFSR_data_master = [i for sub in CFSR_data_master for i in sub]
    				ERAI_data_master = [i for sub in ERAI_data_master for i in sub]
    				ERA5_data_master = [i for sub in ERA5_data_master for i in sub]
    				ERA5_Land_data_master = [i for sub in ERA5_Land_data_master for i in sub]
    				JRA_data_master = [i for sub in JRA_data_master for i in sub]
    				MERRA2_data_master = [i for sub in MERRA2_data_master for i in sub]
    				GLDAS_data_master = [i for sub in GLDAS_data_master for i in sub]
    				GLDAS_CLSM_data_master = [i for sub in GLDAS_CLSM_data_master for i in sub]

    				dframe_master = pd.DataFrame(data = station_data_master, columns=['Station'])
    				dframe_master.insert(0,'Grid Cell', grid_cell_master)
    				dframe_master.insert(1,'Central Lat', lat_master)
    				dframe_master.insert(2,'Central Lon', lon_master)
    				dframe_master.insert(3,'Sites Incl', sites_incl_master)
    				dframe_master.insert(4,'Date', date_master)
    				dframe_master['Naive Blend'] = naive_data_master
    				dframe_master['Naive Blend no JRA55'] = naive_noJRA_data_master
    				dframe_master['Naive Blend no JRA55 Old'] = naive_noJRAold_data_master
    				dframe_master['Naive Blend All'] = naive_all_data_master
    				dframe_master['CFSR'] = CFSR_data_master				
    				dframe_master['ERA-Interim'] = ERAI_data_master
    				dframe_master['ERA5'] = ERA5_data_master
    				dframe_master['ERA5-Land'] = ERA5_Land_data_master
    				dframe_master['JRA-55'] = JRA_data_master
    				dframe_master['MERRA2'] = MERRA2_data_master
    				dframe_master['GLDAS-Noah'] = GLDAS_data_master
    				dframe_master['GLDAS-CLSM'] = GLDAS_CLSM_data_master

    				timeseries_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_timeseries/new_data/new_depth/'+str(remap_type)+"_"+str(naive_type_j)+'_'+str(olr_k)+'_'+str(lyr_l)+'_'+str(thr_m)+'_data_CMOS_newdepth.csv'])

    				dframe_master.to_csv(timeseries_fil,index=False)
    				gcell_uq = np.unique(grid_cell_master)
    				gcell_uq = gcell_uq[~np.isnan(gcell_uq)]
    				#print(gcell_uq)    			


################ number of figures required ###############
#
#    				numgrid = len(gcell_uq)
#
#### for a figure with 12 subplots ###
#
#    				if (numgrid%12 == 0):
#    					numfig = int(len(grid_uq)/12)
#    					lastfig = 12
#
#    				elif (numgrid%12 != 0):
#    					numfig = int((numgrid//12)+1)
#    					lastfig = int(numgrid%12)
#
#    				if (lastfig%3 == 0):
#    					numrow = int(lastfig/3)
#
#    				elif (lastfig%3 != 0):
#    					numrow = int((lastfig//3)+1)    			
#
#    				ymin = -50
#    				ymax = 25
#    				#xmin = np.datetime64(datetime.date(1990,1,1),'Y')
#    				#xmax = np.datetime64(datetime.date(2020,1,1),'Y')
#
#
########################### create subplots ###################################
#    				for a in range (0,numfig):
#    					fig = plt.figure()
#    					fig,axs = plt.subplots(nrows = 4, ncols = 3, sharex = 'col', sharey = 'row', figsize=(16,12)) # create a figure with 3x4 subplots
#	
#################### grab data for each site to plot ######################
#    					if (a == (numfig-1)):
#    						mxrg = lastfig+1
#    					elif (a < (numfig-1)):
#    						mxrg = 13
#    					test = lastfig%3
#    					#print("MOD =",test)
#    					if (lastfig%3 == 0): #if there is no remainder
#    						numrow = int(lastfig/3)
#    					elif (lastfig%3 != 0):
#    						numrow = int((lastfig//3)+1)
#    					#print("last row =",numrow)
#    					totfig = numrow*3 
#    					min_grid = gcell_uq[(a*12)]
#    					min_grid = round(min_grid,0)
#    					max_grid = gcell_uq[(a*12)+(mxrg-2)]
#    					max_grid = round(max_grid,0)
#    					#print(min_grid,max_grid)
#    					line_labels = ["Station","Naive Blend","CFSR", "ERA-Interim", "ERA5", "ERA5-Land" "JRA-55", "MERRA2", "GLDAS-Noah", "GLDAS-CLSM"]
#    					for b in range (1,mxrg): # 12 subplots per figure unless last figure    					    		
#    						j0 = b-1
#    						#print(j0)
#    						jgrd = (a*12) + j0
#    						#print(jgrd)
#    						jgrid = gcell_uq[jgrd]
#    						#print(jgrid)
#    						dframe_gcell = dframe_master[dframe_master['Grid Cell'] == jgrid]
#    						#print(dframe_gcell)
#    						lat_grid = np.round(dframe_gcell['Central Lat'].iloc[1],2)
#    						lon_grid = np.round(dframe_gcell['Central Lon'].iloc[1],2)
#    						sites_grid = np.round(dframe_gcell['Sites Incl'].iloc[1],2)   				
#    						date_grid = dframe_gcell['Date'].values
#    						#print(type(date_grid))
#    						#print(date_grid)
#    						date_grid2 = [datetime.datetime.strptime(x,'%Y-%m-%d') for x in date_grid]
#    						gcell_str = round(jgrid,0)
#    						stemp_stn = dframe_gcell['Station'].values
#    						stemp_naive = dframe_gcell['Naive Blend'].values
#    						stemp_CFSR = dframe_gcell['CFSR'].values
#    						stemp_ERAI = dframe_gcell['ERA-Interim'].values
#    						stemp_ERA5 = dframe_gcell['ERA5'].values
#    						stemp_ERA5_Land = dframe_gcell['ERA5-Land'].values
#    						stemp_JRA = dframe_gcell['JRA-55'].values
#    						stemp_MERRA2 = dframe_gcell['MERRA2'].values
#    						stemp_GLDAS = dframe_gcell['GLDAS-Noah'].values
#    						stemp_GLDAS_CLSM = dframe_gcell['GLDAS-CLSM'].values
#
#    						xmin = np.datetime64(min(date_grid2),'Y')
#    						xmin2 = min(date_grid2)
#    						max_year = np.datetime64(max(date_grid2))
#    						xmax = np.datetime64(max_year, 'Y')
#    						xmax2 = max(date_grid2)
#    						len_timeseries = relativedelta(xmax2,xmin2).years
#
#    						#print(xmin)
#    						#print(xmax)
#
#    						if (a < (numfig-1)): #if figure has 12 subplots
#    							ax = plt.subplot(4,3,b)
#    						elif (a == (numfig-1)): #else if there are less than 12 subplots in figure
#    							ax = plt.subplot(numrow,3,b)
#    						ax.plot(date_grid2,stemp_stn,label="Station",marker='o',markerfacecolor='dimgrey',markersize=2.5,color='dimgrey',linewidth=2.75)
#    						ax.plot(date_grid2,stemp_naive,label="Naive Blend",marker='p',markerfacecolor='dodgerblue',markersize=2.0,color='dodgerblue',linewidth=2.0)
#    						ax.plot(date_grid2,stemp_CFSR,label="CFSR",marker='s',markerfacecolor='m',markersize=.5,color='m',linewidth=.5)
#    						ax.plot(date_grid2,stemp_ERAI,label="ERA-Interim",marker='v',markerfacecolor='limegreen',markersize=.5,color='limegreen',linewidth=0.5)
#    						ax.plot(date_grid2,stemp_ERA5,label="ERA5",marker='^',markerfacecolor='cyan',markersize=.5,color='cyan',linewidth=.5)
#    						ax.plot(date_grid2,stemp_ERA5_Land,label="ERA5-Land",marker='^',markerfacecolor='skyblue',markersize=.5,color='skyblue',linewidth=.5, linestyle='dashdot')
#    						ax.plot(date_grid2,stemp_JRA,label="JRA-55",marker='*',markerfacecolor='red',markersize=.5,color='red',linewidth=.5)
#    						ax.plot(date_grid2,stemp_MERRA2,label="MERRA2",marker='D',markerfacecolor='goldenrod',markersize=.5,color='goldenrod',linewidth=.5)
#    						ax.plot(date_grid2,stemp_GLDAS,label="GLDAS-Noah",marker='x',markerfacecolor='black',markersize=.5,color='black',linewidth=.5)
#    						ax.plot(date_grid2,stemp_GLDAS_CLSM,label="GLDAS-CLSM",marker='x',markerfacecolor='darkgrey',markersize=.5,color='darkgrey',linewidth=.5, linestyle='dashdot')
#
#    						if (len_timeseries > 5):
#    							ax.xaxis.set_major_locator(mdates.YearLocator(5)) #major tick every 5 years
#    							ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y')) #only show the year
#    							ax.xaxis.set_minor_locator(mdates.YearLocator(1)) #minor tick every year   					
#    							ax.yaxis.set_major_locator(MultipleLocator(5)) #every 5 degrees will be a major tick
#    							ax.yaxis.set_minor_locator(MultipleLocator(1)) #every 1 degrees will be a minor tick
#    						elif (len_timeseries <=5):
#    							ax.xaxis.set_major_locator(mdates.YearLocator(1)) #major tick every year
#    							ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y')) #only show the year
#    							ax.xaxis.set_minor_locator(mdates.MonthLocator(1)) #minor tick every month   					
#    							ax.yaxis.set_major_locator(MultipleLocator(5)) #every 5 degrees will be a major tick
#    							ax.yaxis.set_minor_locator(MultipleLocator(1)) #every 1 degrees will be a minor tick
#    						ax.set_xlim(xmin,xmax)
#    						ax.set_ylim(ymin,ymax)
#    						handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)] 
#    						axtext = []
#    						axtext.append('Grid Cell: '+str(gcell_str)+', Lat: '+str(lat_grid)+'$^\circ$N, Lon:'+str(lon_grid)+'$^\circ$'+'E'+', Avg Sites: '+str(sites_grid))
#    						ax.legend(handles, axtext, loc='best', fontsize = 'small', fancybox=False, framealpha=0, handlelength=0, handletextpad=0) 
#    						lines = []
#    						labels = []
#    						for ax in fig.get_axes():
#    							axLine, axLabel = ax.get_legend_handles_labels()
#    							lines.extend(axLine)
#    							labels.extend(axLabel)
#						
#    					if (a == (numfig-1)):
#    						for c in range(mxrg,totfig+1):
#    							plt.subplot(numrow,3,c).set_visible(False)
#						
#    					fig.add_subplot(111, frameon=False) #create large subplot which will include the plot labels for plots
#    					plt.tick_params(labelcolor='none',bottom=False,left=False) #hide ticks
#    					plt.xlabel('Date',fontweight='bold')
#    					plt.ylabel('Soil Temperature($^\circ$ C)',fontweight='bold')
#    					fig.legend(lines[0:10],labels[0:10],loc="right",title="Legend")
#				
#    					if ( a < (numfig -1)):    			
#    						plt.tight_layout()
#					   
#    					L1fil = "".join(["/mnt/data/users/herringtont/soil_temp/plots/naive_blend_timeseries/"+str(remap_type)+"_"+str(naive_type_j)+'_'+str(olr_k)+"_"+str(lyr_l)+"_thr"+str(thr_m)+"_grid"+str(min_grid)+"_grid"+str(max_grid)+".png"])
#    					print(L1fil)
#    					plt.savefig(L1fil)
#    					plt.close()












