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
lyr = ['top_30cm']
thr = ['100']#['0','25','50','75','100']
rmp_type = ['nn']#['nn','bil']
tmp_type = ['raw_temp']

CFSR_layer = "Air_Temp"
CFSR2_layer = "Air_Temp"
GLDAS_layer = "Air_Temp"
ERA5_layer = "Air_Temp"
ERAI_layer = "Air_Temp"
JRA_layer = "Air_Temp"
MERRA2_layer = "Air_Temp"
Naive_layer = "Air_Temp"


############# Grab Reanalysis Data ############

for i in rmp_type:
    rmp_type_i = i
    remap_type = ''.join(['remap'+rmp_type_i])
    rnys_dir = ''.join(['/mnt/data/users/herringtont/soil_temp/reanalysis/2m_AirTemp/rename/land_only/common_grid/'+str(remap_type)+'/'])
    rnys_gcell_dir = ''.join([rnys_dir,'grid_cell_level/'])

    CFSR_fi = "".join([rnys_dir,"CFSR_all_2m_air.nc"])
    MERRA2_fi = "".join([rnys_dir,"MERRA2_2m_air.nc"])
    ERA5_fi = "".join([rnys_dir,"ERA5_2m_air.nc"])
    ERAI_fi = "".join([rnys_dir,"ERA-Interim_2m_air.nc"])
    JRA_fi = "".join([rnys_dir,"JRA55_2m_air.nc"])
    GLDAS_fi = "".join([rnys_dir,"GLDAS_2m_air.nc"])

    for j in naive_type:
    	naive_type_j = j
    	naive_dir_raw = ''.join(['/mnt/data/users/herringtont/soil_temp/reanalysis/2m_AirTemp/rename/land_only/common_grid/'+str(remap_type)+'/'])
    	naive_fi = ''.join([naive_dir_raw+str(remap_type)+'_'+'Naive_2m_air.nc'])
    	for k in olr:
    		olr_k = k


    		for l in thr:
    			thr_l = l
    			insitu_dir =  ''.join(['/mnt/data/users/herringtont/soil_temp/In-Situ/All/spatial_average_anom/'+str(remap_type)+'/no_outliers/'+str(olr_k)+'/top_30cm/thr_'+str(thr_l)+'/'])
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
    			CFSR_data_master_cold_temp = []
    			ERAI_data_master_cold_temp = []
    			ERA5_data_master_cold_temp = []
    			JRA_data_master_cold_temp = []
    			MERRA2_data_master_cold_temp = []
    			GLDAS_data_master_cold_temp = []
    			cold_temp_season_master = []

    			station_data_master_warm_temp = []
    			naive_data_master_warm_temp = []
    			CFSR_data_master_warm_temp = []
    			ERAI_data_master_warm_temp = []
    			ERA5_data_master_warm_temp = []
    			JRA_data_master_warm_temp = []
    			MERRA2_data_master_warm_temp = []
    			GLDAS_data_master_warm_temp = []
    			warm_temp_season_master = []

################# loop through in-situ files ###############
    			#print(type(CFSR_anom))
    			#pathlist = os_sorted(os.listdir(insitu_dir))
    			pathlist = os.listdir(insitu_dir)
    			pathlist_sorted = natural_sort(pathlist)
    			for path in pathlist_sorted:
    				insitu_fil = ''.join([insitu_dir,path])
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
    				JRA_fil = xr.open_dataset(JRA_fi)
    				MERRA2_fil = xr.open_dataset(MERRA2_fi)
    				GLDAS_fil = xr.open_dataset(GLDAS_fi)

    				CFSR_temp = CFSR_fil[CFSR_layer] - 273.15
    				ERAI_temp = ERAI_fil[ERAI_layer] - 273.15
    				ERA5_temp = ERA5_fil[ERA5_layer] - 273.15
    				JRA_temp = JRA_fil[JRA_layer] - 273.15
    				MERRA2_temp = MERRA2_fil[MERRA2_layer] -273.15
    				GLDAS_temp = GLDAS_fil[GLDAS_layer] - 273.15

    				naive_fil = xr.open_dataset(naive_fi)
    				naive_temp = naive_fil[Naive_layer] -273.15


    				#print(type(CFSR_temp))

    				CFSR_temp_gcell = CFSR_temp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    				ERAI_temp_gcell = ERAI_temp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    				ERA5_temp_gcell = ERA5_temp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    				JRA_temp_gcell = JRA_temp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    				MERRA2_temp_gcell = MERRA2_temp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    				GLDAS_temp_gcell = GLDAS_temp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    				naive_temp_gcell = naive_temp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)

    				rnys_dattim = naive_fil['time']
    				rnys_datetime = rnys_dattim.dt.strftime('%Y-%m-%d')    					
    				len_rnys_dattim = len(rnys_dattim) - 1
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
    				MERRA2_temp_master = []
    				GLDAS_temp_master = []
    				TC_temp_master = []
    				naive_temp_master = []
    				station_temp_master = []
    				station_anom_master = []
    				date_temp_master = []

    				for n in range(0,len(DateTime)):
    					DateTime_m = DateTime[n]
    					dattim_m = dattim[n]
    					if(DateTime_m > rnys_edate_dt): #skip all dates beyond last reanalysis date
    						continue
    					naive_temp_dt = naive_temp_gcell.sel(time=DateTime_m).values.tolist()
    					#print(DateTime_m)
    					#print(rnys_edate_dt)
    					if(str(naive_temp_dt) == "nan"):
    						naive_temp_dt = np.nan  						
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
    					CFSR_temp_dt = CFSR_temp_gcell.sel(time=DateTime_m).values.tolist()
    					if(str(CFSR_temp_dt) == "nan"):
    						CFSR_temp_dt = np.nan
    					CFSR_temp_master.append(CFSR_temp_dt)    						
    					JRA_temp_dt = JRA_temp_gcell.sel(time=DateTime_m).values.tolist()
    					if(str(JRA_temp_dt) == "nan"):
    						JRA_temp_dt = np.nan
    					JRA_temp_master.append(JRA_temp_dt)      							
    					ERAI_temp_dt = ERAI_temp_gcell.sel(time=DateTime_m).values.tolist()
    					if(str(ERAI_temp_dt) == "nan"):
    						ERAI_temp_dt = np.nan
    					ERAI_temp_master.append(ERAI_temp_dt)
    					ERA5_temp_dt = ERA5_temp_gcell.sel(time=DateTime_m).values.tolist()
    					if(str(ERA5_temp_dt) == "nan"):
    						ERA5_temp_dt = np.nan
    					ERA5_temp_master.append(ERA5_temp_dt)
    					MERRA2_temp_dt = MERRA2_temp_gcell.sel(time=DateTime_m).values.tolist()
    					if(str(MERRA2_temp_dt) == "nan"):
    						MERRA2_temp_dt = np.nan
    					MERRA2_temp_master.append(MERRA2_temp_dt)
    					GLDAS_temp_dt = GLDAS_temp_gcell.sel(time=DateTime_m).values.tolist()
    					if(str(GLDAS_temp_dt) == "nan"):
    						GLDAS_temp_dt = np.nan
    					GLDAS_temp_master.append(GLDAS_temp_dt)
    					date_temp_master.append(dattim_m)    						
    					naive_temp_master.append(naive_temp_dt)            							    						


    				station_temp_master = np.array(station_temp_master)
    				station_anom_master = np.array(station_anom_master)
    				date_temp_master = np.array(date_temp_master)
    				CFSR_temp_master = np.array(CFSR_temp_master)
    				ERAI_temp_master = np.array(ERAI_temp_master)
    				ERA5_temp_master = np.array(ERA5_temp_master)
    				JRA_temp_master = np.array(JRA_temp_master)
    				MERRA2_temp_master = np.array(MERRA2_temp_master)
    				GLDAS_temp_master = np.array(GLDAS_temp_master)
    				naive_temp_master = np.array(naive_temp_master)

    				#print(naive_temp_master)


    				naive_no_nan = naive_temp_master[~np.isnan(naive_temp_master)]

    				#print(naive_no_nan,TC_no_nan)

    				CFSR_no_nan = CFSR_temp_master[~np.isnan(CFSR_temp_master)]
    				#print(CFSR_no_nan)

    				if(DateTime[0]>CFSR_edate_dt or DateTime[len(DateTime) -1] < CFSR_sdate_dt): #skip if the CFSR dates and station dates do not overlap
    					print('Grid Cell Skipped')
    					continue

    					
    				if(len(naive_no_nan) == 0 or len(CFSR_no_nan) == 0): #skip if there are NaN values in blended data
    					print('Grid Cell Skipped')
    					continue

################## Separate by cold and warm season (by temperature) ####################

###### Cold Season (Air Temp <= -2) #####

    				cold_season_station_index = np.where(ERA5_temp_master <= -5)
    				cold_idx = cold_season_station_index[0]

    				station_temp_cold_season = []
    				naive_temp_cold_season = []
    				CFSR_temp_cold_season = []
    				ERAI_temp_cold_season = []
    				ERA5_temp_cold_season = []
    				JRA_temp_cold_season = []
    				MERRA2_temp_cold_season = []
    				GLDAS_temp_cold_season = []
    				for x in cold_idx:
    					station_x = station_temp_master[x].tolist()
    					station_temp_cold_season.append(station_x)
    					naive_x = naive_temp_master[x].tolist()
    					naive_temp_cold_season.append(naive_x)
    					CFSR_x = CFSR_temp_master[x].tolist()
    					CFSR_temp_cold_season.append(CFSR_x)
    					ERAI_x = ERAI_temp_master[x].tolist()
    					ERAI_temp_cold_season.append(ERAI_x)
    					ERA5_x = ERA5_temp_master[x].tolist()
    					ERA5_temp_cold_season.append(ERA5_x)
    					JRA_x = JRA_temp_master[x].tolist()
    					JRA_temp_cold_season.append(JRA_x)
    					MERRA2_x = MERRA2_temp_master[x].tolist()
    					MERRA2_temp_cold_season.append(MERRA2_x)
    					GLDAS_x = GLDAS_temp_master[x].tolist()
    					GLDAS_temp_cold_season.append(GLDAS_x)
					   				

    				if(len(naive_temp_cold_season) < 5 or len(station_temp_cold_season) < 5 or len(CFSR_temp_cold_season) < 5):
    					print('Grid Cell Skipped')
    					continue

    				cold_season_name = 'Cold'
    				warm_season_name = 'Warm'

    				dframe_cold_season_temp = pd.DataFrame(data = station_temp_cold_season, columns=['Station'])
    				dframe_cold_season_temp['Grid Cell'] = gcell
    				dframe_cold_season_temp['Lat'] = lat_cen
    				dframe_cold_season_temp['Lon'] = lon_cen
    				dframe_cold_season_temp['Naive Blend'] = naive_temp_cold_season
    				dframe_cold_season_temp['CFSR'] = CFSR_temp_cold_season
    				dframe_cold_season_temp['ERA-Interim'] = ERAI_temp_cold_season
    				dframe_cold_season_temp['ERA5'] = ERA5_temp_cold_season
    				dframe_cold_season_temp['JRA55'] = JRA_temp_cold_season
    				dframe_cold_season_temp['MERRA2'] = MERRA2_temp_cold_season
    				dframe_cold_season_temp['GLDAS'] = GLDAS_temp_cold_season
    				dframe_cold_season_temp = dframe_cold_season_temp.dropna()
    				dframe_cold_season_temp['N'] = len(dframe_cold_season_temp)
    				dframe_cold_season_temp['Season'] = cold_season_name
    				if(len(dframe_cold_season_temp) < 5):
    					print('Grid Cell Skipped')
    					continue

###### Warm Season (ERA5 Air Temp > -2) #####
    				warm_season_station_index = np.where(ERA5_temp_master > -5)
    				warm_idx = warm_season_station_index[0]

    				station_temp_warm_season = []
    				naive_temp_warm_season = []
    				CFSR_temp_warm_season = []
    				ERAI_temp_warm_season = []
    				ERA5_temp_warm_season = []
    				JRA_temp_warm_season = []
    				MERRA2_temp_warm_season = []
    				GLDAS_temp_warm_season = []
    				for x in warm_idx:
    					station_x = station_temp_master[x].tolist()
    					station_temp_warm_season.append(station_x)
    					naive_x = naive_temp_master[x].tolist()
    					naive_temp_warm_season.append(naive_x)
    					CFSR_x = CFSR_temp_master[x].tolist()
    					CFSR_temp_warm_season.append(CFSR_x)
    					ERAI_x = ERAI_temp_master[x].tolist()
    					ERAI_temp_warm_season.append(ERAI_x)
    					ERA5_x = ERA5_temp_master[x].tolist()
    					ERA5_temp_warm_season.append(ERA5_x)
    					JRA_x = JRA_temp_master[x].tolist()
    					JRA_temp_warm_season.append(JRA_x)
    					MERRA2_x = MERRA2_temp_master[x].tolist()
    					MERRA2_temp_warm_season.append(MERRA2_x)
    					GLDAS_x = GLDAS_temp_master[x].tolist()
    					GLDAS_temp_warm_season.append(GLDAS_x)
					   				
    				if(len(naive_temp_warm_season) < 5 or len(station_temp_cold_season) < 5 or len(CFSR_temp_warm_season) < 5):
    					print('Grid Cell Skipped')
    					continue

    				dframe_warm_season_temp = pd.DataFrame(data = station_temp_warm_season, columns=['Station'])
    				dframe_warm_season_temp['Grid Cell'] = gcell
    				dframe_warm_season_temp['Lat'] = lat_cen
    				dframe_warm_season_temp['Lon'] = lon_cen				
    				dframe_warm_season_temp['Naive Blend'] = naive_temp_warm_season
    				dframe_warm_season_temp['CFSR'] = CFSR_temp_warm_season
    				dframe_warm_season_temp['ERA-Interim'] = ERAI_temp_warm_season
    				dframe_warm_season_temp['ERA5'] = ERA5_temp_warm_season
    				dframe_warm_season_temp['JRA55'] = JRA_temp_warm_season
    				dframe_warm_season_temp['MERRA2'] = MERRA2_temp_warm_season
    				dframe_warm_season_temp['GLDAS'] = GLDAS_temp_warm_season
    				dframe_warm_season_temp = dframe_warm_season_temp.dropna()
    				dframe_warm_season_temp['N'] = len(dframe_warm_season_temp)
    				dframe_warm_season_temp['Season'] =  warm_season_name
    				if(len(dframe_warm_season_temp) < 5):
    					print('Grid Cell Skipped')
    					continue


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

    				CFSR_data_master_cold_temp.append(dframe_cold_season_temp['CFSR'].values.tolist())
    				CFSR_data_master_warm_temp.append(dframe_warm_season_temp['CFSR'].values.tolist())

    				ERAI_data_master_cold_temp.append(dframe_cold_season_temp['ERA-Interim'].values.tolist())
    				ERAI_data_master_warm_temp.append(dframe_warm_season_temp['ERA-Interim'].values.tolist())

    				ERA5_data_master_cold_temp.append(dframe_cold_season_temp['ERA5'].values.tolist())
    				ERA5_data_master_warm_temp.append(dframe_warm_season_temp['ERA5'].values.tolist())

    				JRA_data_master_cold_temp.append(dframe_cold_season_temp['JRA55'].values.tolist())
    				JRA_data_master_warm_temp.append(dframe_warm_season_temp['JRA55'].values.tolist())

    				MERRA2_data_master_cold_temp.append(dframe_cold_season_temp['MERRA2'].values.tolist())
    				MERRA2_data_master_warm_temp.append(dframe_warm_season_temp['MERRA2'].values.tolist())

    				GLDAS_data_master_cold_temp.append(dframe_cold_season_temp['GLDAS'].values.tolist())
    				GLDAS_data_master_warm_temp.append(dframe_warm_season_temp['GLDAS'].values.tolist())

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

    			CFSR_data_master_cold_temp = [i for sub in CFSR_data_master_cold_temp for i in sub]
    			CFSR_data_master_warm_temp = [i for sub in CFSR_data_master_warm_temp for i in sub]

    			ERAI_data_master_cold_temp = [i for sub in ERAI_data_master_cold_temp for i in sub]
    			ERAI_data_master_warm_temp = [i for sub in ERAI_data_master_warm_temp for i in sub]

    			ERA5_data_master_cold_temp = [i for sub in ERA5_data_master_cold_temp for i in sub]
    			ERA5_data_master_warm_temp = [i for sub in ERA5_data_master_warm_temp for i in sub]

    			JRA_data_master_cold_temp = [i for sub in JRA_data_master_cold_temp for i in sub]
    			JRA_data_master_warm_temp = [i for sub in JRA_data_master_warm_temp for i in sub]

    			MERRA2_data_master_cold_temp = [i for sub in MERRA2_data_master_cold_temp for i in sub]
    			MERRA2_data_master_warm_temp = [i for sub in MERRA2_data_master_warm_temp for i in sub]

    			GLDAS_data_master_cold_temp = [i for sub in GLDAS_data_master_cold_temp for i in sub]
    			GLDAS_data_master_warm_temp = [i for sub in GLDAS_data_master_warm_temp for i in sub]

    			cold_temp_season_master = [i for sub in cold_temp_season_master for i in sub]
    			warm_temp_season_master = [i for sub in warm_temp_season_master for i in sub]

    			dframe_cold_season_temp_master = pd.DataFrame(data=gcell_master_cold_temp, columns=['Grid Cell'])
    			dframe_cold_season_temp_master['Central Lat'] = lat_master_cold_temp
    			dframe_cold_season_temp_master['Central Lon'] = lon_master_cold_temp
    			dframe_cold_season_temp_master['N'] = sample_size_master_cold_temp			
    			dframe_cold_season_temp_master['Naive Blend'] = naive_data_master_cold_temp
    			dframe_cold_season_temp_master['CFSR'] = CFSR_data_master_cold_temp
    			dframe_cold_season_temp_master['ERA-Interim'] = ERAI_data_master_cold_temp
    			dframe_cold_season_temp_master['ERA5'] = ERA5_data_master_cold_temp
    			dframe_cold_season_temp_master['JRA55'] = JRA_data_master_cold_temp
    			dframe_cold_season_temp_master['MERRA2'] = MERRA2_data_master_cold_temp
    			dframe_cold_season_temp_master['GLDAS'] = GLDAS_data_master_cold_temp
    			dframe_cold_season_temp_master['Season'] = cold_temp_season_master

    			print(dframe_cold_season_temp_master)

    			dframe_warm_season_temp_master = pd.DataFrame(data=gcell_master_warm_temp, columns=['Grid Cell'])
    			dframe_warm_season_temp_master['Central Lat'] = lat_master_warm_temp
    			dframe_warm_season_temp_master['Central Lon'] = lon_master_warm_temp
    			dframe_warm_season_temp_master['N'] = sample_size_master_warm_temp			
    			dframe_warm_season_temp_master['Naive Blend'] = naive_data_master_warm_temp
    			dframe_warm_season_temp_master['CFSR'] = CFSR_data_master_warm_temp
    			dframe_warm_season_temp_master['ERA-Interim'] = ERAI_data_master_warm_temp
    			dframe_warm_season_temp_master['ERA5'] = ERA5_data_master_warm_temp
    			dframe_warm_season_temp_master['JRA55'] = JRA_data_master_warm_temp
    			dframe_warm_season_temp_master['MERRA2'] = MERRA2_data_master_warm_temp
    			dframe_warm_season_temp_master['GLDAS'] = GLDAS_data_master_warm_temp
    			dframe_warm_season_temp_master['Season'] = warm_temp_season_master

    			print(dframe_warm_season_temp_master)

    			scatterplot_master_naive = []
    			scatterplot_master_CFSR = []
    			scatterplot_master_ERAI = []
    			scatterplot_master_ERA5 = []
    			scatterplot_master_JRA = []
    			scatterplot_master_MERRA2 = []
    			scatterplot_master_GLDAS = []
    			scatterplot_master_season = []

    			scatterplot_master_naive.append(dframe_cold_season_temp_master['Naive Blend'].values.tolist())
    			scatterplot_master_naive.append(dframe_warm_season_temp_master['Naive Blend'].values.tolist())
    			scatterplot_master_CFSR.append(dframe_cold_season_temp_master['CFSR'].values.tolist())
    			scatterplot_master_CFSR.append(dframe_warm_season_temp_master['CFSR'].values.tolist())
    			scatterplot_master_ERAI.append(dframe_cold_season_temp_master['ERA-Interim'].values.tolist())
    			scatterplot_master_ERAI.append(dframe_warm_season_temp_master['ERA-Interim'].values.tolist())
    			scatterplot_master_ERA5.append(dframe_cold_season_temp_master['ERA5'].values.tolist())
    			scatterplot_master_ERA5.append(dframe_warm_season_temp_master['ERA5'].values.tolist())
    			scatterplot_master_JRA.append(dframe_cold_season_temp_master['JRA55'].values.tolist())
    			scatterplot_master_JRA.append(dframe_warm_season_temp_master['JRA55'].values.tolist())
    			scatterplot_master_MERRA2.append(dframe_cold_season_temp_master['MERRA2'].values.tolist())
    			scatterplot_master_MERRA2.append(dframe_warm_season_temp_master['MERRA2'].values.tolist())
    			scatterplot_master_GLDAS.append(dframe_cold_season_temp_master['GLDAS'].values.tolist())
    			scatterplot_master_GLDAS.append(dframe_warm_season_temp_master['GLDAS'].values.tolist())
    			scatterplot_master_season.append(dframe_cold_season_temp_master['Season'].values.tolist())
    			scatterplot_master_season.append(dframe_warm_season_temp_master['Season'].values.tolist())

    			scatterplot_master_naive = [i for sub in scatterplot_master_naive for i in sub]
    			scatterplot_master_CFSR = [i for sub in scatterplot_master_CFSR for i in sub]
    			scatterplot_master_ERAI = [i for sub in scatterplot_master_ERAI for i in sub]
    			scatterplot_master_ERA5 = [i for sub in scatterplot_master_ERA5 for i in sub]
    			scatterplot_master_JRA = [i for sub in scatterplot_master_JRA for i in sub]
    			scatterplot_master_MERRA2 = [i for sub in scatterplot_master_MERRA2 for i in sub]
    			scatterplot_master_GLDAS = [i for sub in scatterplot_master_GLDAS for i in sub]
    			scatterplot_master_season = [i for sub in scatterplot_master_season for i in sub]

    			dframe_scatterplot_master = pd.DataFrame(data=scatterplot_master_naive, columns=['Naive Blend'])
    			dframe_scatterplot_master['CFSR'] = scatterplot_master_CFSR
    			dframe_scatterplot_master['ERA-Interim'] = scatterplot_master_ERAI
    			dframe_scatterplot_master['ERA5'] = scatterplot_master_ERA5
    			dframe_scatterplot_master['JRA55'] = scatterplot_master_JRA
    			dframe_scatterplot_master['MERRA2'] = scatterplot_master_MERRA2
    			dframe_scatterplot_master['GLDAS'] = scatterplot_master_GLDAS
    			dframe_scatterplot_master['Season'] = scatterplot_master_season

    			print(dframe_scatterplot_master)

###################### Calculate Correlation Matrices #########################

    			fig,axs = plt.subplots(nrows=1, ncols=2, sharex='col', sharey='row', figsize=(15,20))
    			cbar_kws = {"orientation":"horizontal","shrink":1}
###### Cold Season (Temp) #######

    			dframe_cold_season_temp_corr = dframe_cold_season_temp_master[['Naive Blend','CFSR','ERA-Interim','ERA5','JRA55','MERRA2','GLDAS']]
    			cold_season_temp_corrMatrix = dframe_cold_season_temp_corr.corr()

    			ax1 = plt.subplot(121)
    			corr1 = sn.heatmap(cold_season_temp_corrMatrix,annot=True,square='True',vmin=0,vmax=1,cbar_kws=cbar_kws)
    			ax1.set_title('Cold Season Correlation Matrix')

###### Warm Season (Temp) #######
    			dframe_warm_season_temp_corr = dframe_warm_season_temp_master[['Naive Blend','CFSR','ERA-Interim','ERA5','JRA55','MERRA2','GLDAS']]
    			warm_season_temp_corrMatrix = dframe_warm_season_temp_corr.corr()

    			ax2 = plt.subplot(122)
    			corr2 = sn.heatmap(warm_season_temp_corrMatrix,annot=True,square='True',vmin=0,vmax=1,cbar_kws=cbar_kws)
    			ax2.set_title('Warm Season Correlation Matrix')

    			plt.savefig('/mnt/data/users/herringtont/soil_temp/plots/naive_blend_correlation/'+str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_2m_air_temp_thr_'+str(thr_l)+'_correlation_byseason_ERA5_-5C.png')
    			plt.close()

			
###################### Create Scatterplot Matrices ####################

###### Cold Season (Temp) ######
    			scatter1 = sn.pairplot(dframe_scatterplot_master, hue='Season')
    			for ax in scatter1.axes.flat:
    				ax.set_xlim(-40,30)
    				ax.set_ylim(-40,30)
    			plt.savefig('/mnt/data/users/herringtont/soil_temp/plots/naive_blend_scatterplots/'+str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_2m_air_temp_thr_'+str(thr_l)+'_scatterplot_all_temp_ERA5_-5C.png')
    			plt.close()

