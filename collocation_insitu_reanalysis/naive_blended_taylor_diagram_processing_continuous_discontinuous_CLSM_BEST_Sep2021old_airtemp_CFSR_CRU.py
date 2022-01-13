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
tmp_type = ['raw_temp']
temp_thr = ['-2C']#['0C','-2C','-5C','-10C']

ERA5_air = 'Air_Temp_degC'

air_temp = 'Air_Temp_degC'
############# Grab Reanalysis Data ############

for i in rmp_type:
    rmp_type_i = i
    remap_type = ''.join(['remap'+rmp_type_i])
    rnys_dir = ''.join(['/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/obs_depths/common_grid_CFSR/'+str(remap_type)+'/common_date/masked_files/'])
    air_dir = ''.join(['/mnt/data/users/herringtont/soil_temp/reanalysis/2m_AirTemp/rename/land_only/common_grid_CFSR/'+str(remap_type)+'/masked_files/degC/'])

    BEST_fi_air = "".join(['/mnt/data/users/herringtont/soil_temp/reanalysis/2m_AirTemp/rename/land_only/common_grid_CFSR/remapcon/masked_files/degC/CRU_2m_air.nc'])
    CFSR_fi = "".join([rnys_dir,"CFSR_all.nc"])
    CFSR_fi_air = "".join([air_dir,"CFSR_all_2m_air.nc"])
    ERA5_fi = "".join([rnys_dir,"ERA5.nc"])
    ERA5_fi_air = "".join([air_dir,"ERA5_2m_air.nc"])       
    ERA5_MAAT_fi = ''.join(['/mnt/data/users/herringtont/soil_temp/reanalysis/2m_AirTemp/rename/land_only/common_grid_CFSR/'+str(remap_type)+'/masked_files/MAAT/CRU_2m_air_MAAT_1981_2010_clim.nc'])
    ERA5_Land_fi = "".join([rnys_dir,"ERA5_Land.nc"])
    ERA5_Land_fi_air = "".join([air_dir,"ERA5-Land_2m_air.nc"]) 
    GLDAS_fi = "".join([rnys_dir,"GLDAS.nc"])
    GLDAS_fi_air = "".join([air_dir,"GLDAS_2m_air.nc"])

    for j in naive_type:
    	naive_type_j = j
    	naive_dir_raw = ''.join(['/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/obs_depths/common_grid_CFSR/'+str(remap_type)+'/common_date/masked_files/'])
    	naive_fi_all = ''.join([naive_dir_raw+'ensmean_blend.nc'])
    	naive_fi_all_air = ''.join([air_dir+'ensmean_2m_air.nc'])	
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
    				insitu_dir =  ''.join(['/mnt/data/users/herringtont/soil_temp/In-Situ/All/spatial_average_anom/'+str(remap_type)+'/no_outliers/'+str(olr_k)+'/'+str(in_situ_layer)+'/thr_'+str(thr_m)+'/CFSR/Sep2021/'])

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
    					naive_all_data_master_warm_temp = []
    					CFSR_data_master_warm_temp = []
    					ERAI_data_master_warm_temp = []
    					ERA5_data_master_warm_temp = []
    					ERA5_Land_data_master_warm_temp = []
    					JRA_data_master_warm_temp = []
    					MERRA2_data_master_warm_temp = []
    					GLDAS_data_master_warm_temp = []
    					GLDAS_CLSM_data_master_warm_temp = []

    					station_data_air_master_cold_temp = []
    					naive_data_air_master_cold_temp = []
    					naive_noJRA_data_air_master_cold_temp = []
    					naive_noJRAold_data_air_master_cold_temp = []
    					naive_all_data_air_master_cold_temp = []
    					BEST_data_air_master_cold_temp = []
    					CFSR_data_air_master_cold_temp = []
    					ERAI_data_air_master_cold_temp = []
    					ERA5_data_air_master_cold_temp = []
    					ERA5_Land_data_air_master_cold_temp = []
    					JRA_data_air_master_cold_temp = []
    					MERRA2_data_air_master_cold_temp = []
    					GLDAS_data_air_master_cold_temp = []
    					GLDAS_CLSM_data_air_master_cold_temp = []

    					station_data_air_master_warm_temp = []
    					naive_data_air_master_warm_temp = []
    					naive_noJRA_data_air_master_warm_temp = []
    					naive_noJRAold_data_air_master_warm_temp = []
    					naive_all_data_air_master_warm_temp = []
    					BEST_data_air_master_warm_temp = []
    					CFSR_data_air_master_warm_temp = []
    					ERAI_data_air_master_warm_temp = []
    					ERA5_data_air_master_warm_temp = []
    					ERA5_Land_data_air_master_warm_temp = []
    					JRA_data_air_master_warm_temp = []
    					MERRA2_data_air_master_warm_temp = []
    					GLDAS_data_air_master_warm_temp = []
    					GLDAS_CLSM_data_air_master_warm_temp = []

    					cold_temp_season_master = []
    					warm_temp_season_master = []

    					cold_temp_continent_master = []
    					warm_temp_continent_master = []

    					cold_temp_RS_2002_permafrost_master = []
    					warm_temp_RS_2002_permafrost_master = []

    					cold_temp_Brown_1970_permafrost_master = []
    					warm_temp_Brown_1970_permafrost_master = []

    					cold_temp_season_air_master = []
    					warm_temp_season_air_master = []

    					cold_temp_continent_air_master = []
    					warm_temp_continent_air_master = []

    					cold_temp_RS_2002_permafrost_air_master = []
    					warm_temp_RS_2002_permafrost_air_master = []

    					cold_temp_Brown_1970_permafrost_air_master = []
    					warm_temp_Brown_1970_permafrost_air_master = []

    					cold_temp_MAAT = []
    					warm_temp_MAAT = []
################# loop through in-situ files ###############
    					#print(type(CFSR_anom))
    					#pathlist = os_sorted(os.listdir(insitu_dir))
    					pathlist = os.listdir(insitu_dir)
    					pathlist_sorted = natural_sort(pathlist)
    					for path in pathlist_sorted:
    						insitu_fil = ''.join([insitu_dir,path])
    						print(insitu_fil)
    						dframe_insitu = pd.read_csv(insitu_fil)
    						dattim = dframe_insitu['Date'].values
    						DateTime = [datetime.datetime.strptime(x,'%Y-%m-%d') for x in dattim]
    						soil_temp = dframe_insitu['Spatial Avg Temp']
    						gcell = dframe_insitu['Grid Cell'].iloc[0]
    						lat_cen = dframe_insitu['Central Lat'].iloc[0]
    						lon_cen = dframe_insitu['Central Lon'].iloc[0]

    						print("Grid Cell:",gcell, " Lat Cen:",str(lat_cen)," Lon Cen:",str(lon_cen))
 
    						#if (lat_cen < 70 and -136 < lon_cen and lon_cen < -109) : #skip grid cells based in NWT
    							#print("Skipping Grid Cell",gcell," because in NWT")
    							#continue

    						#if(gcell == 33777): #skip grid cell 3377 because of data issues
    							#continue
    
    						CFSR_fil = xr.open_dataset(CFSR_fi)
    						ERA5_fil = xr.open_dataset(ERA5_fi)
    						ERA5_Land_fil = xr.open_dataset(ERA5_Land_fi)
    						GLDAS_fil = xr.open_dataset(GLDAS_fi)

    						BEST_fil_air = xr.open_dataset(BEST_fi_air)
    						CFSR_fil_air = xr.open_dataset(CFSR_fi_air)
    						ERA5_fil_air = xr.open_dataset(ERA5_fi_air)
    						ERA5_Land_fil_air = xr.open_dataset(ERA5_Land_fi_air)
    						GLDAS_fil_air = xr.open_dataset(GLDAS_fi_air)
    						ERA5_MAAT_fil = xr.open_dataset(ERA5_MAAT_fi)
    						#print(BEST_fil_air)

    						CFSR_stemp = CFSR_fil[CFSR_layer] - 273.15
    						ERA5_stemp = ERA5_fil[ERA5_layer] - 273.15
    						ERA5_Land_stemp = ERA5_Land_fil[ERA5_Land_layer] - 273.15
    						GLDAS_stemp = GLDAS_fil[GLDAS_layer] - 273.15

    						BEST_air = BEST_fil_air['Air_Temp_degC']
    						CFSR_air = CFSR_fil_air[air_temp]
    						ERA5_air = ERA5_fil_air[air_temp]
    						ERA5_Land_air = ERA5_Land_fil_air[air_temp]
    						GLDAS_air = GLDAS_fil_air[air_temp]

    						naive_fil_all = xr.open_dataset(naive_fi_all)
    						naive_fil_all_air = xr.open_dataset(naive_fi_all_air)
    						naive_stemp_all = naive_fil_all[Naive_layer] -273.15

    						naive_air_all = naive_fil_all_air[air_temp]
						
    						ERA5_tair = ERA5_fil_air[air_temp]
    						ERA5_MAAT = ERA5_MAAT_fil['Air_Temp_degC']

    						CFSR_stemp_gcell = CFSR_stemp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    						ERA5_stemp_gcell = ERA5_stemp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    						ERA5_Land_stemp_gcell = ERA5_Land_stemp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    						GLDAS_stemp_gcell = GLDAS_stemp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    						naive_all_stemp_gcell = naive_stemp_all.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)


    						#print(BEST_air)
    						#print(str(lat_cen),str(lon_cen))

    						BEST_air_gcell = BEST_air.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    						CFSR_air_gcell = CFSR_air.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    						ERA5_air_gcell = ERA5_air.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    						ERA5_Land_air_gcell = ERA5_Land_air.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    						GLDAS_air_gcell = GLDAS_air.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    						naive_all_air_gcell = naive_air_all.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)

    						#print(CFSR_stemp_gcell)
    						#print(naive_all_stemp_gcell)

    						ERA5_tair_gcell = ERA5_tair.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    						ERA5_MAAT_gcell = ERA5_MAAT.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True).values
    						ERA5_MAAT_gcell = ERA5_MAAT_gcell[0]
    						#print(ERA5_MAAT_gcell)
    						rnys_dattim = naive_fil_all['time']
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

    						#print(ERA5_MAAT_gcell)


#################### grab collocated temperature data from reanalysis files #######################

    						CFSR_temp_master = []
    						JRA_temp_master = []
    						ERAI_temp_master = []
    						ERA5_temp_master = []
    						BEST_air_master = []
    						ERA5_air_master = []
    						ERA5_Land_temp_master = []
    						MERRA2_temp_master = []
    						GLDAS_temp_master = []
    						GLDAS_CLSM_temp_master = []
    						CFSR_air_master = []
    						JRA_air_master = []
    						ERAI_air_master = []
    						ERA5_Land_air_master = []
    						MERRA2_air_master = []
    						GLDAS_air_master = []
    						GLDAS_CLSM_air_master = []
    						TC_temp_master = []
    						naive_temp_master = []
    						naive_noJRA_temp_master = []
    						naive_noJRAold_temp_master = []
    						naive_all_temp_master = []
    						naive_all_air_master = []
    						station_temp_master = []
    						station_anom_master = []
    						date_temp_master = []

    						for n in range(0,len(DateTime)):
    							DateTime_m = DateTime[n]
    							Year_m = DateTime_m.year
    							Month_m = DateTime_m.month

    							dattim_m = dattim[n]
    							if(DateTime_m < rnys_sdate_dt): #skip dates before 1980-01-01
    								continue
    							if(DateTime_m > rnys_edate_dt): #skip all dates beyond last reanalysis date
    								continue
    							naive_all_temp_dt = naive_all_stemp_gcell.sel(time=DateTime_m).values.tolist()
    							naive_all_air_dt = naive_all_air_gcell.sel(time=DateTime_m).values.tolist()
    							#print(DateTime_m)
    							#print(rnys_edate_dt)
    							if(str(naive_all_temp_dt) == "nan"):
    								naive_all_temp_dt = np.nan
    							if(str(naive_all_air_dt) == "nan"):
    								naive_all_air_dt = np.nan  						
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
    							ERA5_temp_dt = ERA5_stemp_gcell.sel(time=DateTime_m).values.tolist()
    							if(str(ERA5_temp_dt) == "nan"):
    								ERA5_temp_dt = np.nan
    							ERA5_temp_master.append(ERA5_temp_dt)
    							ERA5_air_dt = ERA5_air_gcell.sel(time=DateTime_m).values.tolist()
    							if(str(ERA5_air_dt) == "nan"):
    								ERA5_air_dt = np.nan
    							ERA5_air_master.append(ERA5_air_dt)
    							ERA5_Land_temp_dt = ERA5_Land_stemp_gcell.sel(time=DateTime_m).values.tolist()
    							if(str(ERA5_Land_temp_dt) == "nan"):
    								ERA5_Land_temp_dt = np.nan
    							ERA5_Land_temp_master.append(ERA5_Land_temp_dt)
    							GLDAS_temp_dt = GLDAS_stemp_gcell.sel(time=DateTime_m).values.tolist()
    							if(str(GLDAS_temp_dt) == "nan"):
    								GLDAS_temp_dt = np.nan
    							GLDAS_temp_master.append(GLDAS_temp_dt)
    							BEST_air_dt = BEST_air_gcell.sel(time=DateTime_m).values.tolist()
    							if(str(BEST_air_dt) == "nan"):
    								BEST_air_dt = np.nan
    							BEST_air_master.append(BEST_air_dt)
    							CFSR_air_dt = CFSR_air_gcell.sel(time=DateTime_m).values.tolist()
    							if(str(CFSR_air_dt) == "nan"):
    								CFSR_air_dt = np.nan
    							CFSR_air_master.append(CFSR_air_dt)    						

    							ERA5_Land_air_dt = ERA5_Land_air_gcell.sel(time=DateTime_m).values.tolist()
    							if(str(ERA5_Land_air_dt) == "nan"):
    								ERA5_Land_air_dt = np.nan
    							ERA5_Land_air_master.append(ERA5_Land_air_dt)
    							GLDAS_air_dt = GLDAS_air_gcell.sel(time=DateTime_m).values.tolist()
    							if(str(GLDAS_air_dt) == "nan"):
    								GLDAS_air_dt = np.nan
    							GLDAS_air_master.append(GLDAS_air_dt)
    							date_temp_master.append(dattim_m)    						
    							naive_all_temp_master.append(naive_all_temp_dt)
    							naive_all_air_master.append(naive_all_air_dt)

    						station_temp_master = np.array(station_temp_master)
    						station_anom_master = np.array(station_anom_master)
    						date_temp_master = np.array(date_temp_master)
    						CFSR_temp_master = np.array(CFSR_temp_master)
    						ERA5_temp_master = np.array(ERA5_temp_master)
    						ERA5_air_master = np.array(ERA5_air_master)
    						ERA5_Land_temp_master = np.array(ERA5_Land_temp_master)
    						GLDAS_temp_master = np.array(GLDAS_temp_master)

    						BEST_air_master = np.array(BEST_air_master)
    						CFSR_air_master = np.array(CFSR_air_master)
    						ERA5_Land_air_master = np.array(ERA5_Land_air_master)
    						GLDAS_air_master = np.array(GLDAS_air_master)

    						naive_all_temp_master = np.array(naive_all_temp_master)
    						naive_all_air_master = np.array(naive_all_air_master)
						
    						naive_no_nan = naive_all_temp_master[~np.isnan(naive_all_temp_master)]

    						#print(naive_no_nan)

    						CFSR_no_nan = CFSR_temp_master[~np.isnan(CFSR_temp_master)]
    						#print(CFSR_no_nan)

    						if(DateTime[0]>CFSR_edate_dt or DateTime[len(DateTime) -1] < CFSR_sdate_dt): #skip if the CFSR dates and station dates do not overlap
    							print('Grid Cell Skipped - CFSR dates do not overlap')
    							continue

    					
    						if(len(naive_no_nan) == 0 or len(CFSR_no_nan) == 0): #skip if there are NaN values in blended data
    							print('Grid Cell Skipped - No Naive Blended Data')
    							continue


################# Determine Continent of Grid Cell #####################
    						if (-179.5 <= lon_cen <= -15): # if longitude is between -179.5 and -15, it will be considered part of North America and Greenland
    							continent = "North_America"
    							#print(lat_cen, lon_cen)
    						elif (-15 < lon_cen < 179.5): # else it will be considered part of Eurasia
    							continent = "Eurasia"



################ Determine Permafrost Type ################

    						print(ERA5_MAAT_gcell)

    						if (str(ERA5_MAAT_gcell) == "nan"):
    							print('Grid Cell Skipped - air temperature is NAN')
    							continue

    						elif (ERA5_MAAT_gcell <= -8):
    							RS_2002_permafrost = 'continuous'
    							Brown_1970_permafrost = 'continuous'


    						elif (-7 >= ERA5_MAAT_gcell > -8):
    							RS_2002_permafrost = 'continuous'
    							Brown_1970_permafrost = 'discontinuous'

    						elif (-2 >= ERA5_MAAT_gcell > -7):
    							RS_2002_permafrost = 'discontinuous'
    							Brown_1970_permafrost = 'discontinuous'

    						elif (-1 >= ERA5_MAAT_gcell > -2):
    							RS_2002_permafrost = 'none'
    							Brown_1970_permafrost = 'discontinuous'

    						elif (ERA5_MAAT_gcell > -1):
    							RS_2002_permafrost = 'none'
    							Brown_1970_permafrost = 'none'



################## Separate by cold and warm season (by temperature) ####################

###### Cold Season (Soil Temp <= -2) #####

    						#print(type(BEST_air_master))
    						cold_season_index = np.where(BEST_air_master <= tmp_val)
    						cold_idx = cold_season_index[0]

    						#print(cold_idx)

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

    						station_air_cold_season = []
    						naive_air_cold_season = []
    						naive_noJRA_air_cold_season = []
    						naive_noJRAold_air_cold_season = []
    						naive_all_air_cold_season = []
    						BEST_air_cold_season = []						
    						CFSR_air_cold_season = []
    						ERAI_air_cold_season = []
    						ERA5_air_cold_season = []
    						ERA5_Land_air_cold_season = []
    						JRA_air_cold_season = []
    						MERRA2_air_cold_season = []
    						GLDAS_air_cold_season = []
    						GLDAS_CLSM_air_cold_season = []
    						for x in cold_idx:
    							station_x = station_temp_master[x].tolist()
    							station_temp_cold_season.append(station_x)
    							naive_all_x = naive_all_temp_master[x].tolist()
    							naive_all_temp_cold_season.append(naive_all_x)
    							CFSR_x = CFSR_temp_master[x].tolist()
    							CFSR_temp_cold_season.append(CFSR_x)
    							ERA5_x = ERA5_temp_master[x].tolist()
    							ERA5_temp_cold_season.append(ERA5_x)
    							ERA5_Land_x = ERA5_Land_temp_master[x].tolist()
    							ERA5_Land_temp_cold_season.append(ERA5_Land_x)
    							GLDAS_x = GLDAS_temp_master[x].tolist()
    							GLDAS_temp_cold_season.append(GLDAS_x)
					   				
    							naive_all_air_air_x = naive_all_air_master[x].tolist()
    							naive_all_air_cold_season.append(naive_all_air_air_x)
    							BEST_air_x = BEST_air_master[x].tolist()
    							BEST_air_cold_season.append(BEST_air_x)
    							CFSR_air_x = CFSR_air_master[x].tolist()
    							CFSR_air_cold_season.append(CFSR_air_x)
    							ERA5_air_x = ERA5_air_master[x].tolist()
    							ERA5_air_cold_season.append(ERA5_air_x)
    							ERA5_Land_air_x = ERA5_Land_air_master[x].tolist()
    							ERA5_Land_air_cold_season.append(ERA5_Land_air_x)
    							GLDAS_air_x = GLDAS_air_master[x].tolist()
    							GLDAS_air_cold_season.append(GLDAS_air_x)


    						if(len(naive_all_temp_cold_season) < 1 or len(station_temp_cold_season) < 1 or len(CFSR_temp_cold_season) < 1):
    							print('Grid Cell Skipped - Length of Cold Season Less than 1')
    							continue

    						dframe_cold_season_temp = pd.DataFrame(data = station_temp_cold_season, columns=['Station'])
    						dframe_cold_season_temp['Grid Cell'] = gcell
    						dframe_cold_season_temp['Lat'] = lat_cen
    						dframe_cold_season_temp['Lon'] = lon_cen
    						dframe_cold_season_temp['Naive Blend All'] = naive_all_temp_cold_season
    						dframe_cold_season_temp['CFSR'] = CFSR_temp_cold_season
    						dframe_cold_season_temp['ERA5'] = ERA5_temp_cold_season
    						dframe_cold_season_temp['ERA5-Land'] = ERA5_Land_temp_cold_season
    						dframe_cold_season_temp['GLDAS-Noah'] = GLDAS_temp_cold_season
    						dframe_cold_season_temp['Naive Blend All Air Temp'] = naive_all_air_cold_season
    						dframe_cold_season_temp['BEST Air Temp'] = BEST_air_cold_season
    						dframe_cold_season_temp['CFSR Air Temp'] = CFSR_air_cold_season
    						dframe_cold_season_temp['ERA5 Air Temp'] = ERA5_air_cold_season
    						dframe_cold_season_temp['ERA5-Land Air Temp'] = ERA5_Land_air_cold_season
    						dframe_cold_season_temp['GLDAS-Noah Air Temp'] = GLDAS_air_cold_season
    						dframe_cold_season_temp = dframe_cold_season_temp.dropna()
    						dframe_cold_season_temp['N'] = len(dframe_cold_season_temp)
    						dframe_cold_season_temp['Season'] = 'Cold'
    						dframe_cold_season_temp['Continent'] = continent
    						dframe_cold_season_temp['RS 2002 Permafrost'] = RS_2002_permafrost
    						dframe_cold_season_temp['Brown 1970 Permafrost'] = Brown_1970_permafrost
    						dframe_cold_season_temp['ERA5 MAAT'] = ERA5_MAAT_gcell

    						#print(dframe_cold_season_temp)
    						 
    						if(len(dframe_cold_season_temp) < 1):
    							print('Grid Cell Skipped -  Length of Cold Season Less than 1')
    							continue


    
###### warm Season (Soil Temp <= -2) #####

    						warm_season_index = np.where(BEST_air_master > tmp_val)
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

    						station_air_warm_season = []
    						naive_air_warm_season = []
    						naive_noJRA_air_warm_season = []
    						naive_noJRAold_air_warm_season = []
    						naive_all_air_warm_season = []
    						BEST_air_warm_season = []
    						CFSR_air_warm_season = []
    						ERAI_air_warm_season = []
    						ERA5_air_warm_season = []
    						ERA5_Land_air_warm_season = []
    						JRA_air_warm_season = []
    						MERRA2_air_warm_season = []
    						GLDAS_air_warm_season = []
    						GLDAS_CLSM_air_warm_season = []
    						for x in warm_idx:
    							station_x = station_temp_master[x].tolist()
    							station_temp_warm_season.append(station_x)
    							naive_all_x = naive_all_temp_master[x].tolist()
    							naive_all_temp_warm_season.append(naive_all_x)
    							CFSR_x = CFSR_temp_master[x].tolist()
    							CFSR_temp_warm_season.append(CFSR_x)
    							ERA5_x = ERA5_temp_master[x].tolist()
    							ERA5_temp_warm_season.append(ERA5_x)
    							ERA5_Land_x = ERA5_Land_temp_master[x].tolist()
    							ERA5_Land_temp_warm_season.append(ERA5_Land_x)
    							GLDAS_x = GLDAS_temp_master[x].tolist()
    							GLDAS_temp_warm_season.append(GLDAS_x)
					   				

    							naive_all_air_air_x = naive_all_air_master[x].tolist()
    							naive_all_air_warm_season.append(naive_all_air_air_x)
    							BEST_air_x = BEST_air_master[x].tolist()
    							BEST_air_warm_season.append(BEST_air_x)
    							CFSR_air_x = CFSR_air_master[x].tolist()
    							CFSR_air_warm_season.append(CFSR_air_x)
    							ERA5_air_x = ERA5_air_master[x].tolist()
    							ERA5_air_warm_season.append(ERA5_air_x)
    							ERA5_Land_air_x = ERA5_Land_air_master[x].tolist()
    							ERA5_Land_air_warm_season.append(ERA5_Land_air_x)
    							GLDAS_air_x = GLDAS_air_master[x].tolist()
    							GLDAS_air_warm_season.append(GLDAS_air_x)

    						if(len(naive_all_temp_warm_season) < 1 or len(station_temp_warm_season) < 1 or len(CFSR_temp_warm_season) < 1):
    							print('Grid Cell Skipped - Length of warm Season Less than 1')
    							continue



						#print("Continent:",continent)

    						dframe_warm_season_temp = pd.DataFrame(data = station_temp_warm_season, columns=['Station'])
    						dframe_warm_season_temp['Grid Cell'] = gcell
    						dframe_warm_season_temp['Lat'] = lat_cen
    						dframe_warm_season_temp['Lon'] = lon_cen
    						dframe_warm_season_temp['Naive Blend All'] = naive_all_temp_warm_season
    						dframe_warm_season_temp['CFSR'] = CFSR_temp_warm_season
    						dframe_warm_season_temp['ERA5'] = ERA5_temp_warm_season
    						dframe_warm_season_temp['ERA5-Land'] = ERA5_Land_temp_warm_season
    						dframe_warm_season_temp['GLDAS-Noah'] = GLDAS_temp_warm_season
    						dframe_warm_season_temp['Naive Blend All Air Temp'] = naive_all_air_warm_season
    						dframe_warm_season_temp['BEST Air Temp'] = BEST_air_warm_season    						
    						dframe_warm_season_temp['CFSR Air Temp'] = CFSR_air_warm_season
    						dframe_warm_season_temp['ERA5 Air Temp'] = ERA5_air_warm_season
    						dframe_warm_season_temp['ERA5-Land Air Temp'] = ERA5_Land_air_warm_season
    						dframe_warm_season_temp['GLDAS-Noah Air Temp'] = GLDAS_air_warm_season
    						dframe_warm_season_temp = dframe_warm_season_temp.dropna()
    						dframe_warm_season_temp['N'] = len(dframe_warm_season_temp)
    						dframe_warm_season_temp['Season'] = 'Warm'
    						dframe_warm_season_temp['Continent'] = continent
    						dframe_warm_season_temp['RS 2002 Permafrost'] = RS_2002_permafrost
    						dframe_warm_season_temp['Brown 1970 Permafrost'] = Brown_1970_permafrost
    						dframe_warm_season_temp['ERA5 MAAT'] = ERA5_MAAT_gcell   
    						if(len(dframe_warm_season_temp) < 1):
    							print('Grid Cell Skipped -  Length of warm Season Less than 1')
    							continue

    						#print(dframe_warm_season_temp)

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


    						naive_all_data_master_cold_temp.append(dframe_cold_season_temp['Naive Blend All'].values.tolist())
    						naive_all_data_master_warm_temp.append(dframe_warm_season_temp['Naive Blend All'].values.tolist())

    						CFSR_data_master_cold_temp.append(dframe_cold_season_temp['CFSR'].values.tolist())
    						CFSR_data_master_warm_temp.append(dframe_warm_season_temp['CFSR'].values.tolist())

    						ERA5_data_master_cold_temp.append(dframe_cold_season_temp['ERA5'].values.tolist())
    						ERA5_data_master_warm_temp.append(dframe_warm_season_temp['ERA5'].values.tolist())

    						ERA5_Land_data_master_cold_temp.append(dframe_cold_season_temp['ERA5-Land'].values.tolist())
    						ERA5_Land_data_master_warm_temp.append(dframe_warm_season_temp['ERA5-Land'].values.tolist())

    						GLDAS_data_master_cold_temp.append(dframe_cold_season_temp['GLDAS-Noah'].values.tolist())
    						GLDAS_data_master_warm_temp.append(dframe_warm_season_temp['GLDAS-Noah'].values.tolist())

    						cold_temp_season_master.append(dframe_cold_season_temp['Season'].values.tolist())
    						warm_temp_season_master.append(dframe_warm_season_temp['Season'].values.tolist())

    						cold_temp_continent_master.append(dframe_cold_season_temp['Continent'].values.tolist())
    						warm_temp_continent_master.append(dframe_warm_season_temp['Continent'].values.tolist())

    						cold_temp_RS_2002_permafrost_master.append(dframe_cold_season_temp['RS 2002 Permafrost'].values.tolist())
    						warm_temp_RS_2002_permafrost_master.append(dframe_warm_season_temp['RS 2002 Permafrost'].values.tolist())

    						cold_temp_Brown_1970_permafrost_master.append(dframe_cold_season_temp['Brown 1970 Permafrost'].values.tolist())
    						warm_temp_Brown_1970_permafrost_master.append(dframe_warm_season_temp['Brown 1970 Permafrost'].values.tolist())


    						naive_all_data_air_master_cold_temp.append(dframe_cold_season_temp['Naive Blend All Air Temp'].values.tolist())
    						naive_all_data_air_master_warm_temp.append(dframe_warm_season_temp['Naive Blend All Air Temp'].values.tolist())

    						BEST_data_air_master_cold_temp.append(dframe_cold_season_temp['BEST Air Temp'].values.tolist())
    						BEST_data_air_master_warm_temp.append(dframe_warm_season_temp['BEST Air Temp'].values.tolist())
    						CFSR_data_air_master_cold_temp.append(dframe_cold_season_temp['CFSR Air Temp'].values.tolist())
    						CFSR_data_air_master_warm_temp.append(dframe_warm_season_temp['CFSR Air Temp'].values.tolist())

    						ERA5_data_air_master_cold_temp.append(dframe_cold_season_temp['ERA5 Air Temp'].values.tolist())
    						ERA5_data_air_master_warm_temp.append(dframe_warm_season_temp['ERA5 Air Temp'].values.tolist())

    						ERA5_Land_data_air_master_cold_temp.append(dframe_cold_season_temp['ERA5-Land Air Temp'].values.tolist())
    						ERA5_Land_data_air_master_warm_temp.append(dframe_warm_season_temp['ERA5-Land Air Temp'].values.tolist())


    						GLDAS_data_air_master_cold_temp.append(dframe_cold_season_temp['GLDAS-Noah Air Temp'].values.tolist())
    						GLDAS_data_air_master_warm_temp.append(dframe_warm_season_temp['GLDAS-Noah Air Temp'].values.tolist())


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

    					naive_all_data_master_cold_temp = [i for sub in naive_all_data_master_cold_temp for i in sub]
    					naive_all_data_master_warm_temp = [i for sub in naive_all_data_master_warm_temp for i in sub]

    					CFSR_data_master_cold_temp = [i for sub in CFSR_data_master_cold_temp for i in sub]
    					CFSR_data_master_warm_temp = [i for sub in CFSR_data_master_warm_temp for i in sub]

    					ERA5_data_master_cold_temp = [i for sub in ERA5_data_master_cold_temp for i in sub]
    					ERA5_data_master_warm_temp = [i for sub in ERA5_data_master_warm_temp for i in sub]

    					ERA5_Land_data_master_cold_temp = [i for sub in ERA5_Land_data_master_cold_temp for i in sub]
    					ERA5_Land_data_master_warm_temp = [i for sub in ERA5_Land_data_master_warm_temp for i in sub]

    					GLDAS_data_master_cold_temp = [i for sub in GLDAS_data_master_cold_temp for i in sub]
    					GLDAS_data_master_warm_temp = [i for sub in GLDAS_data_master_warm_temp for i in sub]


    					naive_all_data_air_master_cold_temp = [i for sub in naive_all_data_air_master_cold_temp for i in sub]
    					naive_all_data_air_master_warm_temp = [i for sub in naive_all_data_air_master_warm_temp for i in sub]

    					BEST_data_air_master_cold_temp = [i for sub in BEST_data_air_master_cold_temp for i in sub]
    					BEST_data_air_master_warm_temp = [i for sub in BEST_data_air_master_warm_temp for i in sub]

    					CFSR_data_air_master_cold_temp = [i for sub in CFSR_data_air_master_cold_temp for i in sub]
    					CFSR_data_air_master_warm_temp = [i for sub in CFSR_data_air_master_warm_temp for i in sub]

    					ERA5_data_air_master_cold_temp = [i for sub in ERA5_data_air_master_cold_temp for i in sub]
    					ERA5_data_air_master_warm_temp = [i for sub in ERA5_data_air_master_warm_temp for i in sub]

    					ERA5_Land_data_air_master_cold_temp = [i for sub in ERA5_Land_data_air_master_cold_temp for i in sub]
    					ERA5_Land_data_air_master_warm_temp = [i for sub in ERA5_Land_data_air_master_warm_temp for i in sub]

    					GLDAS_data_air_master_cold_temp = [i for sub in GLDAS_data_air_master_cold_temp for i in sub]
    					GLDAS_data_air_master_warm_temp = [i for sub in GLDAS_data_air_master_warm_temp for i in sub]

    					cold_temp_season_master = [i for sub in cold_temp_season_master for i in sub]
    					warm_temp_season_master = [i for sub in warm_temp_season_master for i in sub]

    					cold_temp_continent_master = [i for sub in cold_temp_continent_master for i in sub]
    					warm_temp_continent_master = [i for sub in warm_temp_continent_master for i in sub]

    					cold_temp_RS_2002_permafrost_master = [i for sub in cold_temp_RS_2002_permafrost_master for i in sub]
    					warm_temp_RS_2002_permafrost_master = [i for sub in warm_temp_RS_2002_permafrost_master for i in sub]

    					cold_temp_Brown_1970_permafrost_master = [i for sub in cold_temp_Brown_1970_permafrost_master for i in sub]
    					warm_temp_Brown_1970_permafrost_master = [i for sub in warm_temp_Brown_1970_permafrost_master for i in sub]

    					dframe_cold_season_temp_master = pd.DataFrame(data=gcell_master_cold_temp, columns=['Grid Cell'])
    					dframe_cold_season_temp_master['Central Lat'] = lat_master_cold_temp
    					dframe_cold_season_temp_master['Central Lon'] = lon_master_cold_temp
    					dframe_cold_season_temp_master['N'] = sample_size_master_cold_temp
    					dframe_cold_season_temp_master['Station'] = station_data_master_cold_temp			
    					dframe_cold_season_temp_master['Ensemble Mean'] = naive_all_data_master_cold_temp					
    					dframe_cold_season_temp_master['CFSR'] = CFSR_data_master_cold_temp
    					dframe_cold_season_temp_master['ERA5'] = ERA5_data_master_cold_temp
    					dframe_cold_season_temp_master['ERA5-Land'] = ERA5_Land_data_master_cold_temp
    					dframe_cold_season_temp_master['GLDAS-Noah'] = GLDAS_data_master_cold_temp
    					dframe_cold_season_temp_master['Ensemble Mean Air'] = naive_all_data_air_master_cold_temp
    					dframe_cold_season_temp_master['GHCN Air'] = BEST_data_air_master_cold_temp
    					dframe_cold_season_temp_master['CFSR Air'] = CFSR_data_air_master_cold_temp
    					dframe_cold_season_temp_master['ERA5 Air'] = ERA5_data_air_master_cold_temp
    					dframe_cold_season_temp_master['ERA5-Land Air'] = ERA5_Land_data_air_master_cold_temp
    					dframe_cold_season_temp_master['GLDAS-Noah Air'] = GLDAS_data_air_master_cold_temp
    					dframe_cold_season_temp_master['Season'] = cold_temp_season_master
    					dframe_cold_season_temp_master['Continent'] = cold_temp_continent_master
    					dframe_cold_season_temp_master['RS 2002 Permafrost'] = cold_temp_RS_2002_permafrost_master
    					dframe_cold_season_temp_master['Brown 1970 Permafrost'] = cold_temp_Brown_1970_permafrost_master
    					dframe_cold_season_temp_master = dframe_cold_season_temp_master.dropna()

    					dframe_warm_season_temp_master = pd.DataFrame(data=gcell_master_warm_temp, columns=['Grid Cell'])
    					dframe_warm_season_temp_master['Central Lat'] = lat_master_warm_temp
    					dframe_warm_season_temp_master['Central Lon'] = lon_master_warm_temp
    					dframe_warm_season_temp_master['N'] = sample_size_master_warm_temp
    					dframe_warm_season_temp_master['Station'] = station_data_master_warm_temp			
    					dframe_warm_season_temp_master['Ensemble Mean'] = naive_all_data_master_warm_temp
    					dframe_warm_season_temp_master['CFSR'] = CFSR_data_master_warm_temp
    					dframe_warm_season_temp_master['ERA5'] = ERA5_data_master_warm_temp
    					dframe_warm_season_temp_master['ERA5-Land'] = ERA5_Land_data_master_warm_temp
    					dframe_warm_season_temp_master['GLDAS-Noah'] = GLDAS_data_master_warm_temp
    					dframe_warm_season_temp_master['Ensemble Mean Air'] = naive_all_data_air_master_warm_temp
    					dframe_warm_season_temp_master['GHCN Air'] = BEST_data_air_master_warm_temp
    					dframe_warm_season_temp_master['CFSR Air'] = CFSR_data_air_master_warm_temp
    					dframe_warm_season_temp_master['ERA5 Air'] = ERA5_data_air_master_warm_temp
    					dframe_warm_season_temp_master['ERA5-Land Air'] = ERA5_Land_data_air_master_warm_temp
    					dframe_warm_season_temp_master['GLDAS-Noah Air'] = GLDAS_data_air_master_warm_temp
    					dframe_warm_season_temp_master['Season'] = warm_temp_season_master
    					dframe_warm_season_temp_master['Continent'] = warm_temp_continent_master
    					dframe_warm_season_temp_master['RS 2002 Permafrost'] = warm_temp_RS_2002_permafrost_master
    					dframe_warm_season_temp_master['Brown 1970 Permafrost'] = warm_temp_Brown_1970_permafrost_master
    					dframe_warm_season_temp_master = dframe_warm_season_temp_master.dropna()

    					print(dframe_warm_season_temp_master)

    					cold_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/CLSM_res/subset/'+str(olr_k)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_dframe_cold_season_temp_master_CRU_subset_permafrost_Sep2021_airtemp_CFSR.csv'])
    					dframe_cold_season_temp_master.to_csv(cold_fil,index=False)


    					warm_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/CLSM_res/subset/'+str(olr_k)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_dframe_warm_season_temp_master_CRU_subset_permafrost_Sep2021_airtemp_CFSR.csv'])
    					dframe_warm_season_temp_master.to_csv(warm_fil,index=False)

    					scatterplot_master_gcell = []
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
    					scatterplot_master_naive_all_air = []
    					scatterplot_master_BEST_air = []
    					scatterplot_master_CFSR_air = []
    					scatterplot_master_ERAI_air = []
    					scatterplot_master_ERA5_air = []
    					scatterplot_master_ERA5_Land_air = []
    					scatterplot_master_JRA_air = []
    					scatterplot_master_MERRA2_air = []
    					scatterplot_master_GLDAS_air = []
    					scatterplot_master_GLDAS_CLSM_air = []
    					scatterplot_master_season = []
    					scatterplot_master_continent = []
    					scatterplot_master_RS_2002_permafrost = []
    					scatterplot_master_Brown_1970_permafrost = []

    					scatterplot_master_gcell.append(dframe_cold_season_temp_master['Grid Cell'].values.tolist())
    					scatterplot_master_gcell.append(dframe_warm_season_temp_master['Grid Cell'].values.tolist())
    					scatterplot_master_station.append(dframe_cold_season_temp_master['Station'].values.tolist())
    					scatterplot_master_station.append(dframe_warm_season_temp_master['Station'].values.tolist())
    					scatterplot_master_naive_all.append(dframe_cold_season_temp_master['Ensemble Mean'].values.tolist())
    					scatterplot_master_naive_all.append(dframe_warm_season_temp_master['Ensemble Mean'].values.tolist())
    					scatterplot_master_CFSR.append(dframe_cold_season_temp_master['CFSR'].values.tolist())
    					scatterplot_master_CFSR.append(dframe_warm_season_temp_master['CFSR'].values.tolist())
    					scatterplot_master_ERA5.append(dframe_cold_season_temp_master['ERA5'].values.tolist())
    					scatterplot_master_ERA5.append(dframe_warm_season_temp_master['ERA5'].values.tolist())
    					scatterplot_master_ERA5_Land.append(dframe_cold_season_temp_master['ERA5-Land'].values.tolist())
    					scatterplot_master_ERA5_Land.append(dframe_warm_season_temp_master['ERA5-Land'].values.tolist())
    					scatterplot_master_GLDAS.append(dframe_cold_season_temp_master['GLDAS-Noah'].values.tolist())
    					scatterplot_master_GLDAS.append(dframe_warm_season_temp_master['GLDAS-Noah'].values.tolist())
    					scatterplot_master_naive_all_air.append(dframe_cold_season_temp_master['Ensemble Mean Air'].values.tolist())
    					scatterplot_master_naive_all_air.append(dframe_warm_season_temp_master['Ensemble Mean Air'].values.tolist())
    					scatterplot_master_BEST_air.append(dframe_cold_season_temp_master['GHCN Air'].values.tolist())
    					scatterplot_master_BEST_air.append(dframe_warm_season_temp_master['GHCN Air'].values.tolist())
    					scatterplot_master_CFSR_air.append(dframe_cold_season_temp_master['CFSR Air'].values.tolist())
    					scatterplot_master_CFSR_air.append(dframe_warm_season_temp_master['CFSR Air'].values.tolist())
    					scatterplot_master_ERA5_air.append(dframe_cold_season_temp_master['ERA5 Air'].values.tolist())
    					scatterplot_master_ERA5_air.append(dframe_warm_season_temp_master['ERA5 Air'].values.tolist())
    					scatterplot_master_ERA5_Land_air.append(dframe_cold_season_temp_master['ERA5-Land Air'].values.tolist())
    					scatterplot_master_ERA5_Land_air.append(dframe_warm_season_temp_master['ERA5-Land Air'].values.tolist())
    					scatterplot_master_GLDAS_air.append(dframe_cold_season_temp_master['GLDAS-Noah Air'].values.tolist())
    					scatterplot_master_GLDAS_air.append(dframe_warm_season_temp_master['GLDAS-Noah Air'].values.tolist())
    					scatterplot_master_season.append(dframe_cold_season_temp_master['Season'].values.tolist())
    					scatterplot_master_season.append(dframe_warm_season_temp_master['Season'].values.tolist())
    					scatterplot_master_continent.append(dframe_cold_season_temp_master['Continent'].values.tolist())
    					scatterplot_master_continent.append(dframe_warm_season_temp_master['Continent'].values.tolist())
    					scatterplot_master_RS_2002_permafrost.append(dframe_cold_season_temp_master['RS 2002 Permafrost'].values.tolist())
    					scatterplot_master_RS_2002_permafrost.append(dframe_warm_season_temp_master['RS 2002 Permafrost'].values.tolist())
    					scatterplot_master_Brown_1970_permafrost.append(dframe_cold_season_temp_master['Brown 1970 Permafrost'].values.tolist())
    					scatterplot_master_Brown_1970_permafrost.append(dframe_warm_season_temp_master['Brown 1970 Permafrost'].values.tolist())

    					scatterplot_master_gcell = [ i for sub in scatterplot_master_gcell for i in sub]
    					scatterplot_master_station = [ i for sub in scatterplot_master_station for i in sub]
    					scatterplot_master_naive_all = [i for sub in scatterplot_master_naive_all for i in sub]
    					scatterplot_master_CFSR = [i for sub in scatterplot_master_CFSR for i in sub]
    					scatterplot_master_ERA5 = [i for sub in scatterplot_master_ERA5 for i in sub]
    					scatterplot_master_ERA5_Land= [i for sub in scatterplot_master_ERA5_Land for i in sub]
    					scatterplot_master_GLDAS = [i for sub in scatterplot_master_GLDAS for i in sub]
    					scatterplot_master_naive_all_air = [i for sub in scatterplot_master_naive_all_air for i in sub]
    					scatterplot_master_BEST_air = [i for sub in scatterplot_master_BEST_air for i in sub]
    					scatterplot_master_CFSR_air = [i for sub in scatterplot_master_CFSR_air for i in sub]
    					scatterplot_master_ERA5_air = [i for sub in scatterplot_master_ERA5_air for i in sub]
    					scatterplot_master_ERA5_Land_air = [i for sub in scatterplot_master_ERA5_Land_air for i in sub]
    					scatterplot_master_GLDAS_air = [i for sub in scatterplot_master_GLDAS_air for i in sub]
    					scatterplot_master_season = [i for sub in scatterplot_master_season for i in sub]
    					scatterplot_master_continent = [i for sub in scatterplot_master_continent for i in sub]
    					scatterplot_master_RS_2002_permafrost = [i for sub in scatterplot_master_RS_2002_permafrost for i in sub]
    					scatterplot_master_Brown_1970_permafrost = [i for sub in scatterplot_master_Brown_1970_permafrost for i in sub]

    					dframe_scatterplot_master = pd.DataFrame(data=scatterplot_master_station, columns=['Station'])
    					dframe_scatterplot_master['Ensemble Mean'] = scatterplot_master_naive_all
    					dframe_scatterplot_master['CFSR'] = scatterplot_master_CFSR
    					dframe_scatterplot_master['ERA5'] = scatterplot_master_ERA5
    					dframe_scatterplot_master['ERA5-Land'] = scatterplot_master_ERA5_Land
    					dframe_scatterplot_master['GLDAS-Noah'] = scatterplot_master_GLDAS
    					dframe_scatterplot_master['Ensemble Mean Air'] = scatterplot_master_naive_all_air
    					dframe_scatterplot_master['GHCN Air'] = scatterplot_master_BEST_air
    					dframe_scatterplot_master['CFSR Air'] = scatterplot_master_CFSR_air
    					dframe_scatterplot_master['ERA5 Air'] = scatterplot_master_ERA5_air
    					dframe_scatterplot_master['ERA5-Land Air'] = scatterplot_master_ERA5_Land_air
    					dframe_scatterplot_master['GLDAS-Noah Air'] = scatterplot_master_GLDAS_air
    					dframe_scatterplot_master['Grid Cell'] = scatterplot_master_gcell
    					dframe_scatterplot_master['Season'] = scatterplot_master_season
    					dframe_scatterplot_master['Continent'] = scatterplot_master_continent
    					dframe_scatterplot_master['RS 2002 Permafrost'] = scatterplot_master_RS_2002_permafrost
    					dframe_scatterplot_master['Brown 1970 Permafrost'] = scatterplot_master_Brown_1970_permafrost

    					dframe_scatterplot_master = dframe_scatterplot_master.dropna()

    					print(dframe_scatterplot_master)

    					scatter_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/CLSM_res/subset/'+str(olr_k)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_dframe_scatterplot_CRU_subset_permafrost_Sep2021_air_CFSR.csv'])
    					dframe_scatterplot_master.to_csv(scatter_fil,index=False)






















