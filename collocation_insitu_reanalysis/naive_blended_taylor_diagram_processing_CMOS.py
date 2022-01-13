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

naive_type = ['simple_average']
olr = ['zscore']#['outliers','zscore','IQR']
lyr = ['top_30cm','30cm_300cm']
thr = ['100']#['0','25','50','75','100']
rmp_type = ['con']#['nn','bil','con']
tmp_type = ['raw_temp']
temp_thr = ['-2C']#['0C','-2C','-5C','-10C']

ERA5_air = 'Air_Temp'
############# Grab Reanalysis Data ############

for i in rmp_type:
    rmp_type_i = i
    remap_type = ''.join(['remap'+rmp_type_i])
    rnys_dir = ''.join(['/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/obs_depths/common_grid/'+str(remap_type)+'/common_date/'])
    air_dir = ''.join(['/mnt/data/users/herringtont/soil_temp/reanalysis/2m_AirTemp/rename/land_only/common_grid/'+str(remap_type)+'/'])
    rnys_gcell_dir = ''.join([rnys_dir,'grid_cell_level/'])

    CFSR_fi = "".join([rnys_dir,"CFSR_all.nc"])
    MERRA2_fi = "".join([rnys_dir,"MERRA2.nc"])
    ERA5_fi = "".join([rnys_dir,"ERA5.nc"])
    ERA5_air_fi = "".join([air_dir,"ERA5_2m_air.nc"])
    ERA5_Land_fi = "".join([rnys_dir,"ERA5_Land.nc"])
    ERAI_fi = "".join([rnys_dir,"ERA-Interim.nc"])
    JRA_fi = "".join([rnys_dir,"JRA55.nc"])
    GLDAS_fi = "".join([rnys_dir,"GLDAS.nc"])
    GLDAS_CLSM_fi = "".join([rnys_dir,"GLDAS_CLSM.nc"])

    for j in naive_type:
    	naive_type_j = j
    	naive_dir_raw = ''.join(['/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/obs_depths/common_grid/'+str(remap_type)+'/common_date/'])
    	naive_fi = ''.join([naive_dir_raw+str(remap_type)+'_'+'naive_blend.nc'])
    	naive_fi_noJRA = ''.join([naive_dir_raw+str(remap_type)+'_'+'naive_blend_noJRA55.nc'])
    	naive_fi_noJRAold = ''.join([naive_dir_raw+str(remap_type)+'_'+'naive_blend_noJRA55_old.nc'])
    	naive_fi_all = ''.join([naive_dir_raw+str(remap_type)+'_'+'naive_blend_ERA5L_GLDAS_CLSM.nc'])	
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


    			for m in thr:
    				thr_m = m
    				insitu_dir =  ''.join(['/mnt/data/users/herringtont/soil_temp/In-Situ/All/spatial_average_anom/'+str(remap_type)+'/no_outliers/'+str(olr_k)+'/'+str(in_situ_layer)+'/thr_'+str(thr_m)+'/'])

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
##### Create Master Arrays #####
    					gcell_master_cold_temp = []
    					gcell_master_warm_temp = []
    					gcell_master_cold_date = []
    					gcell_master_warm_date = []
    					gcell_master = []
    					lat_master_cold_temp = []
    					lat_master_warm_temp = []
    					lat_master_cold_date = []
    					lat_master_warm_date = []
    					lon_master_cold_temp = []
    					lon_master_warm_temp = []
    					lon_master_cold_date = []
    					lon_master_warm_date = []
    					lon_master_stn = []
    					lat_master = []
    					lon_master = []
    					date_master = []
    					len_anom_master = []
    					len_raw_master = []
    					sample_size_master_cold_temp = []
    					sample_size_master_warm_temp = []
    					sample_size_master_cold_date = []
    					sample_size_master_warm_date = []

    					station_data_master_cold_temp = []
    					naive_data_master_cold_temp = []
    					naive_noJRA_data_master_cold_temp = []
    					naive_noJRAold_data_master_cold_temp = []
    					naive_all_data_master_cold_temp = []
    					CFSR_data_master_cold_temp = []
    					ERAI_data_master_cold_temp = []
    					ERA5_data_master_cold_temp = []
    					ERA5_Land_data_master_cold_temp = []
    					JRA_data_master_cold_temp = []
    					MERRA2_data_master_cold_temp = []
    					GLDAS_data_master_cold_temp = []
    					GLDAS_CLSM_data_master_cold_temp = []

    					station_data_master_warm_temp = []
    					naive_data_master_warm_temp = []
    					naive_noJRA_data_master_warm_temp = []
    					naive_noJRAold_data_master_warm_temp = []
    					naive_all_data_master_warm_temp = []
    					CFSR_data_master_warm_temp = []
    					ERAI_data_master_warm_temp = []
    					ERA5_data_master_warm_temp = []
    					ERA5_Land_data_master_warm_temp = []
    					JRA_data_master_warm_temp = []
    					MERRA2_data_master_warm_temp = []
    					GLDAS_data_master_warm_temp = []
    					GLDAS_CLSM_data_master_warm_temp = []

    					station_data_master_cold_date = []
    					naive_data_master_cold_date = []
    					naive_noJRA_data_master_cold_date = []
    					naive_noJRAold_data_master_cold_date = []
    					naive_all_data_master_cold_date = []
    					CFSR_data_master_cold_date = []
    					ERAI_data_master_cold_date = []
    					ERA5_data_master_cold_date = []
    					ERA5_Land_data_master_cold_date = []
    					JRA_data_master_cold_date = []
    					MERRA2_data_master_cold_date = []
    					GLDAS_data_master_cold_date = []
    					GLDAS_CLSM_data_master_cold_date = []

    					station_data_master_warm_date = []
    					naive_data_master_warm_date = []
    					naive_noJRA55_data_master_warm_date = []
    					naive_noJRA55old_data_master_warm_date = []
    					naive_all_data_master_warm_date = []
    					CFSR_data_master_warm_date = []
    					ERAI_data_master_warm_date = []
    					ERA5_data_master_warm_date = []
    					ERA5_Land_data_master_warm_date = []
    					JRA_data_master_warm_date = []
    					MERRA2_data_master_warm_date = []
    					GLDAS_data_master_warm_date = []
    					GLDAS_CLSM_data_master_warm_date = []

    					cold_temp_season_master = []
    					warm_temp_season_master = []

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


    						naive_fil = xr.open_dataset(naive_fi)
    						naive_fil_noJRA = xr.open_dataset(naive_fi_noJRA)
    						naive_fil_noJRAold = xr.open_dataset(naive_fi_noJRAold)
    						naive_fil_all = xr.open_dataset(naive_fi_all)
    						naive_stemp = naive_fil[Naive_layer] - 273.15
    						naive_stemp_noJRA = naive_fil_noJRA[Naive_layer] -273.15
    						naive_stemp_noJRAold = naive_fil_noJRAold[Naive_layer] -273.15
    						naive_stemp_all = naive_fil_all[Naive_layer] -273.15
						
    						ERA5_tair = ERA5_air_fil[ERA5_air] - 273.15

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
    						naive_noJRA_stemp_gcell = naive_stemp_noJRA.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    						naive_noJRAold_stemp_gcell = naive_stemp_noJRAold.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    						naive_all_stemp_gcell = naive_stemp_all.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)

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
    						TC_temp_master = []
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
    							if(DateTime_m < rnys_sdate_dt): #skip dates before 1980-01-01
    								continue
    							if(DateTime_m > rnys_edate_dt): #skip all dates beyond last reanalysis date
    								continue
    							naive_temp_dt = naive_stemp_gcell.sel(time=DateTime_m).values.tolist()
    							naive_noJRA_temp_dt = naive_noJRA_stemp_gcell.sel(time=DateTime_m).values.tolist()
    							naive_noJRAold_temp_dt = naive_noJRAold_stemp_gcell.sel(time=DateTime_m).values.tolist()
    							naive_all_temp_dt = naive_all_stemp_gcell.sel(time=DateTime_m).values.tolist()
    							#print(DateTime_m)
    							#print(rnys_edate_dt)
    							if(str(naive_temp_dt) == "nan"):
    								naive_temp_dt = np.nan  
    							if(str(naive_noJRA_temp_dt) == "nan"):
    								naive_noJRA_temp_dt = np.nan
    							if(str(naive_noJRAold_temp_dt) == "nan"):
    								naive_noJRAold_temp_dt = np.nan
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
    							ERA5_temp_dt = ERA5_stemp_gcell.sel(time=DateTime_m).values.tolist()
    							if(str(ERA5_temp_dt) == "nan"):
    								ERA5_temp_dt = np.nan
    							ERA5_temp_master.append(ERA5_temp_dt)
    							ERA5_air_dt = ERA5_tair_gcell.sel(time=DateTime_m).values.tolist()
    							if(str(ERA5_air_dt) == "nan"):
    								ERA5_air_dt = np.nan
    							ERA5_air_master.append(ERA5_air_dt)
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

    						cold_season_index = np.where(ERA5_air_master <= tmp_val)
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
    							GLDAS_CLSM_x = GLDAS_CLSM_temp_master[x].tolist()
    							GLDAS_CLSM_temp_cold_season.append(GLDAS_CLSM_x)
					   				

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
    							print('Grid Cell Skipped -  Length of Cold Season Less than 1')
    							continue

###### warm Season (Soil Temp <= -2) #####

    						warm_season_index = np.where(ERA5_air_master > tmp_val)
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
    							GLDAS_CLSM_x = GLDAS_CLSM_temp_master[x].tolist()
    							GLDAS_CLSM_temp_warm_season.append(GLDAS_CLSM_x)
					   				

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
    						dframe_warm_season_temp['Season'] = 'Warm' 
    						if(len(dframe_warm_season_temp) < 1):
    							print('Grid Cell Skipped -  Length of warm Season Less than 1')
    							continue

    						print(dframe_warm_season_temp)

########################### Create Scatterplot Dataframe ############################


    						gcell_master_cold_temp.append(dframe_cold_season_temp['Grid Cell'].values.tolist())
    						gcell_master_warm_temp.append(dframe_warm_season_temp['Grid Cell'].values.tolist())
				
    						lat_master_cold_temp.append(dframe_cold_season_temp['Lat'].values.tolist())
    						lat_master_warm_temp.append(dframe_warm_season_temp['Lat'].values.tolist())
				
    						lon_master_cold_temp.append(dframe_cold_season_temp['Lon'].values.tolist())
    						lon_master_warm_temp.append(dframe_warm_season_temp['Lon'].values.tolist())				
				 
    						sample_size_master_cold_temp.append(dframe_cold_season_temp['N'].values.tolist())
    						sample_size_master_warm_temp.append(dframe_warm_season_temp['N'].values.tolist())


    						station_data_master_cold_temp.append(dframe_cold_season_temp['Station'].values.tolist())
    						station_data_master_warm_temp.append(dframe_warm_season_temp['Station'].values.tolist())


    						naive_data_master_cold_temp.append(dframe_cold_season_temp['Naive Blend'].values.tolist())
    						naive_data_master_warm_temp.append(dframe_warm_season_temp['Naive Blend'].values.tolist())

    						naive_noJRAold_data_master_cold_temp.append(dframe_cold_season_temp['Naive Blend no JRA55 Old'].values.tolist())
    						naive_noJRAold_data_master_warm_temp.append(dframe_warm_season_temp['Naive Blend no JRA55 Old'].values.tolist())

    						naive_noJRA_data_master_cold_temp.append(dframe_cold_season_temp['Naive Blend no JRA55'].values.tolist())
    						naive_noJRA_data_master_warm_temp.append(dframe_warm_season_temp['Naive Blend no JRA55'].values.tolist())

    						naive_all_data_master_cold_temp.append(dframe_cold_season_temp['Naive Blend All'].values.tolist())
    						naive_all_data_master_warm_temp.append(dframe_warm_season_temp['Naive Blend All'].values.tolist())

    						CFSR_data_master_cold_temp.append(dframe_cold_season_temp['CFSR'].values.tolist())
    						CFSR_data_master_warm_temp.append(dframe_warm_season_temp['CFSR'].values.tolist())

    						ERAI_data_master_cold_temp.append(dframe_cold_season_temp['ERA-Interim'].values.tolist())
    						ERAI_data_master_warm_temp.append(dframe_warm_season_temp['ERA-Interim'].values.tolist())

    						ERA5_data_master_cold_temp.append(dframe_cold_season_temp['ERA5'].values.tolist())
    						ERA5_data_master_warm_temp.append(dframe_warm_season_temp['ERA5'].values.tolist())

    						ERA5_Land_data_master_cold_temp.append(dframe_cold_season_temp['ERA5-Land'].values.tolist())
    						ERA5_Land_data_master_warm_temp.append(dframe_warm_season_temp['ERA5-Land'].values.tolist())

    						JRA_data_master_cold_temp.append(dframe_cold_season_temp['JRA55'].values.tolist())
    						JRA_data_master_warm_temp.append(dframe_warm_season_temp['JRA55'].values.tolist())

    						MERRA2_data_master_cold_temp.append(dframe_cold_season_temp['MERRA2'].values.tolist())
    						MERRA2_data_master_warm_temp.append(dframe_warm_season_temp['MERRA2'].values.tolist())

    						GLDAS_data_master_cold_temp.append(dframe_cold_season_temp['GLDAS-Noah'].values.tolist())
    						GLDAS_data_master_warm_temp.append(dframe_warm_season_temp['GLDAS-Noah'].values.tolist())

    						GLDAS_CLSM_data_master_cold_temp.append(dframe_cold_season_temp['GLDAS-CLSM'].values.tolist())
    						GLDAS_CLSM_data_master_warm_temp.append(dframe_warm_season_temp['GLDAS-CLSM'].values.tolist())

    						cold_temp_season_master.append(dframe_cold_season_temp['Season'].values.tolist())
    						warm_temp_season_master.append(dframe_warm_season_temp['Season'].values.tolist())

    					gcell_master_cold_temp = [i for sub in gcell_master_cold_temp for i in sub]
    					gcell_master_warm_temp = [i for sub in gcell_master_warm_temp for i in sub]

    					lat_master_cold_temp = [i for sub in lat_master_cold_temp for i in sub]
    					lat_master_warm_temp = [i for sub in lat_master_warm_temp for i in sub]

    					lon_master_cold_temp = [i for sub in lon_master_cold_temp for i in sub]
    					lon_master_warm_temp = [i for sub in lon_master_warm_temp for i in sub]

    					sample_size_master_cold_temp = [i for sub in sample_size_master_cold_temp for i in sub]
    					sample_size_master_warm_temp = [i for sub in sample_size_master_warm_temp for i in sub]

    					station_data_master_cold_temp = [i for sub in station_data_master_cold_temp for i in sub]
    					station_data_master_warm_temp = [i for sub in station_data_master_warm_temp for i in sub]

    					naive_data_master_cold_temp = [i for sub in naive_data_master_cold_temp for i in sub]
    					naive_data_master_warm_temp = [i for sub in naive_data_master_warm_temp for i in sub]

    					naive_noJRA_data_master_cold_temp = [i for sub in naive_noJRA_data_master_cold_temp for i in sub]
    					naive_noJRA_data_master_warm_temp = [i for sub in naive_noJRA_data_master_warm_temp for i in sub]

    					naive_noJRAold_data_master_cold_temp = [i for sub in naive_noJRAold_data_master_cold_temp for i in sub]
    					naive_noJRAold_data_master_warm_temp = [i for sub in naive_noJRAold_data_master_warm_temp for i in sub]

    					naive_all_data_master_cold_temp = [i for sub in naive_all_data_master_cold_temp for i in sub]
    					naive_all_data_master_warm_temp = [i for sub in naive_all_data_master_warm_temp for i in sub]

    					CFSR_data_master_cold_temp = [i for sub in CFSR_data_master_cold_temp for i in sub]
    					CFSR_data_master_warm_temp = [i for sub in CFSR_data_master_warm_temp for i in sub]

    					ERAI_data_master_cold_temp = [i for sub in ERAI_data_master_cold_temp for i in sub]
    					ERAI_data_master_warm_temp = [i for sub in ERAI_data_master_warm_temp for i in sub]

    					ERA5_data_master_cold_temp = [i for sub in ERA5_data_master_cold_temp for i in sub]
    					ERA5_data_master_warm_temp = [i for sub in ERA5_data_master_warm_temp for i in sub]

    					ERA5_Land_data_master_cold_temp = [i for sub in ERA5_Land_data_master_cold_temp for i in sub]
    					ERA5_Land_data_master_warm_temp = [i for sub in ERA5_Land_data_master_warm_temp for i in sub]

    					JRA_data_master_cold_temp = [i for sub in JRA_data_master_cold_temp for i in sub]
    					JRA_data_master_warm_temp = [i for sub in JRA_data_master_warm_temp for i in sub]

    					MERRA2_data_master_cold_temp = [i for sub in MERRA2_data_master_cold_temp for i in sub]
    					MERRA2_data_master_warm_temp = [i for sub in MERRA2_data_master_warm_temp for i in sub]

    					GLDAS_data_master_cold_temp = [i for sub in GLDAS_data_master_cold_temp for i in sub]
    					GLDAS_data_master_warm_temp = [i for sub in GLDAS_data_master_warm_temp for i in sub]

    					GLDAS_CLSM_data_master_cold_temp = [i for sub in GLDAS_CLSM_data_master_cold_temp for i in sub]
    					GLDAS_CLSM_data_master_warm_temp = [i for sub in GLDAS_CLSM_data_master_warm_temp for i in sub]

    					cold_temp_season_master = [i for sub in cold_temp_season_master for i in sub]
    					warm_temp_season_master = [i for sub in warm_temp_season_master for i in sub]

    					dframe_cold_season_temp_master = pd.DataFrame(data=gcell_master_cold_temp, columns=['Grid Cell'])
    					dframe_cold_season_temp_master['Central Lat'] = lat_master_cold_temp
    					dframe_cold_season_temp_master['Central Lon'] = lon_master_cold_temp
    					dframe_cold_season_temp_master['N'] = sample_size_master_cold_temp
    					dframe_cold_season_temp_master['Station'] = station_data_master_cold_temp			
    					dframe_cold_season_temp_master['Naive Blend'] = naive_data_master_cold_temp
    					dframe_cold_season_temp_master['Naive Blend no JRA55'] = naive_noJRA_data_master_cold_temp
    					dframe_cold_season_temp_master['Naive Blend no JRA55 Old'] = naive_noJRAold_data_master_cold_temp
    					dframe_cold_season_temp_master['Naive Blend All'] = naive_all_data_master_cold_temp					
    					dframe_cold_season_temp_master['CFSR'] = CFSR_data_master_cold_temp
    					dframe_cold_season_temp_master['ERA-Interim'] = ERAI_data_master_cold_temp
    					dframe_cold_season_temp_master['ERA5'] = ERA5_data_master_cold_temp
    					dframe_cold_season_temp_master['ERA5-Land'] = ERA5_Land_data_master_cold_temp
    					dframe_cold_season_temp_master['JRA55'] = JRA_data_master_cold_temp
    					dframe_cold_season_temp_master['MERRA2'] = MERRA2_data_master_cold_temp
    					dframe_cold_season_temp_master['GLDAS-Noah'] = GLDAS_data_master_cold_temp
    					dframe_cold_season_temp_master['GLDAS-CLSM'] = GLDAS_CLSM_data_master_cold_temp
    					dframe_cold_season_temp_master['Season'] = cold_temp_season_master

    					dframe_warm_season_temp_master = pd.DataFrame(data=gcell_master_warm_temp, columns=['Grid Cell'])
    					dframe_warm_season_temp_master['Central Lat'] = lat_master_warm_temp
    					dframe_warm_season_temp_master['Central Lon'] = lon_master_warm_temp
    					dframe_warm_season_temp_master['N'] = sample_size_master_warm_temp
    					dframe_warm_season_temp_master['Station'] = station_data_master_warm_temp			
    					dframe_warm_season_temp_master['Naive Blend'] = naive_data_master_warm_temp
    					dframe_warm_season_temp_master['Naive Blend no JRA55'] = naive_noJRA_data_master_warm_temp
    					dframe_warm_season_temp_master['Naive Blend no JRA55 Old'] = naive_noJRAold_data_master_warm_temp
    					dframe_warm_season_temp_master['Naive Blend All'] = naive_all_data_master_warm_temp
    					dframe_warm_season_temp_master['CFSR'] = CFSR_data_master_warm_temp
    					dframe_warm_season_temp_master['ERA-Interim'] = ERAI_data_master_warm_temp
    					dframe_warm_season_temp_master['ERA5'] = ERA5_data_master_warm_temp
    					dframe_warm_season_temp_master['ERA5-Land'] = ERA5_Land_data_master_warm_temp
    					dframe_warm_season_temp_master['JRA55'] = JRA_data_master_warm_temp
    					dframe_warm_season_temp_master['MERRA2'] = MERRA2_data_master_warm_temp
    					dframe_warm_season_temp_master['GLDAS-Noah'] = GLDAS_data_master_warm_temp
    					dframe_warm_season_temp_master['GLDAS-CLSM'] = GLDAS_CLSM_data_master_warm_temp
    					dframe_warm_season_temp_master['Season'] = warm_temp_season_master


    					cold_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/new_depth/'+str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_dframe_cold_season_temp_master_ERA5_'+str(temp_thr_o)+'_CMOS_newdepth.csv'])
    					dframe_cold_season_temp_master.to_csv(cold_fil,index=False)


    					warm_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data//new_depth/'+str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_dframe_warm_season_temp_master_ERA5_'+str(temp_thr_o)+'_CMOS_newdepth.csv'])
    					dframe_warm_season_temp_master.to_csv(warm_fil,index=False)

    					scatterplot_master_station = []
    					scatterplot_master_naive = []
    					scatterplot_master_naive_noJRA = []
    					scatterplot_master_naive_noJRAold = []
    					scatterplot_master_naive_all = []
    					scatterplot_master_CFSR = []
    					scatterplot_master_ERAI = []
    					scatterplot_master_ERA5 = []
    					scatterplot_master_ERA5_Land = []
    					scatterplot_master_JRA = []
    					scatterplot_master_MERRA2 = []
    					scatterplot_master_GLDAS = []
    					scatterplot_master_GLDAS_CLSM = []
    					scatterplot_master_season = []


    					scatterplot_master_station.append(dframe_cold_season_temp_master['Station'].values.tolist())
    					scatterplot_master_station.append(dframe_warm_season_temp_master['Station'].values.tolist())
    					scatterplot_master_naive.append(dframe_cold_season_temp_master['Naive Blend'].values.tolist())
    					scatterplot_master_naive.append(dframe_warm_season_temp_master['Naive Blend'].values.tolist())
    					scatterplot_master_naive_noJRA.append(dframe_cold_season_temp_master['Naive Blend no JRA55'].values.tolist())
    					scatterplot_master_naive_noJRA.append(dframe_warm_season_temp_master['Naive Blend no JRA55'].values.tolist())
    					scatterplot_master_naive_noJRAold.append(dframe_cold_season_temp_master['Naive Blend no JRA55 Old'].values.tolist())
    					scatterplot_master_naive_noJRAold.append(dframe_warm_season_temp_master['Naive Blend no JRA55 Old'].values.tolist())
    					scatterplot_master_naive_all.append(dframe_cold_season_temp_master['Naive Blend All'].values.tolist())
    					scatterplot_master_naive_all.append(dframe_warm_season_temp_master['Naive Blend All'].values.tolist())
    					scatterplot_master_CFSR.append(dframe_cold_season_temp_master['CFSR'].values.tolist())
    					scatterplot_master_CFSR.append(dframe_warm_season_temp_master['CFSR'].values.tolist())
    					scatterplot_master_ERAI.append(dframe_cold_season_temp_master['ERA-Interim'].values.tolist())
    					scatterplot_master_ERAI.append(dframe_warm_season_temp_master['ERA-Interim'].values.tolist())
    					scatterplot_master_ERA5.append(dframe_cold_season_temp_master['ERA5'].values.tolist())
    					scatterplot_master_ERA5.append(dframe_warm_season_temp_master['ERA5'].values.tolist())
    					scatterplot_master_ERA5_Land.append(dframe_cold_season_temp_master['ERA5-Land'].values.tolist())
    					scatterplot_master_ERA5_Land.append(dframe_warm_season_temp_master['ERA5-Land'].values.tolist())
    					scatterplot_master_JRA.append(dframe_cold_season_temp_master['JRA55'].values.tolist())
    					scatterplot_master_JRA.append(dframe_warm_season_temp_master['JRA55'].values.tolist())
    					scatterplot_master_MERRA2.append(dframe_cold_season_temp_master['MERRA2'].values.tolist())
    					scatterplot_master_MERRA2.append(dframe_warm_season_temp_master['MERRA2'].values.tolist())
    					scatterplot_master_GLDAS.append(dframe_cold_season_temp_master['GLDAS-Noah'].values.tolist())
    					scatterplot_master_GLDAS.append(dframe_warm_season_temp_master['GLDAS-Noah'].values.tolist())
    					scatterplot_master_GLDAS_CLSM.append(dframe_cold_season_temp_master['GLDAS-CLSM'].values.tolist())
    					scatterplot_master_GLDAS_CLSM.append(dframe_warm_season_temp_master['GLDAS-CLSM'].values.tolist())
    					scatterplot_master_season.append(dframe_cold_season_temp_master['Season'].values.tolist())
    					scatterplot_master_season.append(dframe_warm_season_temp_master['Season'].values.tolist())

    					scatterplot_master_station = [ i for sub in scatterplot_master_station for i in sub]
    					scatterplot_master_naive = [i for sub in scatterplot_master_naive for i in sub]
    					scatterplot_master_naive_noJRA = [i for sub in scatterplot_master_naive_noJRA for i in sub]
    					scatterplot_master_naive_noJRAold = [i for sub in scatterplot_master_naive_noJRAold for i in sub]
    					scatterplot_master_naive_all = [i for sub in scatterplot_master_naive_all for i in sub]
    					scatterplot_master_CFSR = [i for sub in scatterplot_master_CFSR for i in sub]
    					scatterplot_master_ERAI = [i for sub in scatterplot_master_ERAI for i in sub]
    					scatterplot_master_ERA5 = [i for sub in scatterplot_master_ERA5 for i in sub]
    					scatterplot_master_ERA5_Land= [i for sub in scatterplot_master_ERA5_Land for i in sub]
    					scatterplot_master_JRA = [i for sub in scatterplot_master_JRA for i in sub]
    					scatterplot_master_MERRA2 = [i for sub in scatterplot_master_MERRA2 for i in sub]
    					scatterplot_master_GLDAS = [i for sub in scatterplot_master_GLDAS for i in sub]
    					scatterplot_master_GLDAS_CLSM = [i for sub in scatterplot_master_GLDAS_CLSM for i in sub]
    					scatterplot_master_season = [i for sub in scatterplot_master_season for i in sub]

    					dframe_scatterplot_master = pd.DataFrame(data=scatterplot_master_station, columns=['Station'])
    					dframe_scatterplot_master['Naive Blend'] = scatterplot_master_naive
    					dframe_scatterplot_master['Naive Blend no JRA55'] = scatterplot_master_naive_noJRA
    					dframe_scatterplot_master['Naive Blend no JRA55 Old'] = scatterplot_master_naive_noJRAold
    					dframe_scatterplot_master['Naive Blend All'] = scatterplot_master_naive_all
    					dframe_scatterplot_master['CFSR'] = scatterplot_master_CFSR
    					dframe_scatterplot_master['ERA-Interim'] = scatterplot_master_ERAI
    					dframe_scatterplot_master['ERA5'] = scatterplot_master_ERA5
    					dframe_scatterplot_master['ERA5-Land'] = scatterplot_master_ERA5_Land
    					dframe_scatterplot_master['JRA55'] = scatterplot_master_JRA
    					dframe_scatterplot_master['MERRA2'] = scatterplot_master_MERRA2
    					dframe_scatterplot_master['GLDAS-Noah'] = scatterplot_master_GLDAS
    					dframe_scatterplot_master['GLDAS-CLSM'] = scatterplot_master_GLDAS_CLSM
    					dframe_scatterplot_master['Season'] = scatterplot_master_season

    					print(dframe_scatterplot_master)

    					scatter_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/new_depth/'+str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_dframe_scatterplot_ERA5_'+str(temp_thr_o)+'_CMOS_newdepth.csv'])
    					dframe_scatterplot_master.to_csv(scatter_fil,index=False)



