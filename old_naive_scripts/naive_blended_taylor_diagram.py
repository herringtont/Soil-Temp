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

CFSR_layer = "Soil_Temp_TOP30"
CFSR2_layer = "Soil_Temp_TOP30"
GLDAS_layer = "Soil_Temp_TOP30"
ERA5_layer = "Soil_Temp_TOP30"
ERAI_layer = "Soil_Temp_TOP30"
JRA_layer = "Soil_Temp"
MERRA2_layer = "Soil_Temp_TOP30"
Naive_layer = "Soil_Temp_TOP30"


############# Grab Reanalysis Data ############

for i in rmp_type:
    rmp_type_i = i
    remap_type = ''.join(['remap'+rmp_type_i])
    rnys_dir = ''.join(['/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/common_grid/'+str(remap_type)+'/common_date/'])
    rnys_gcell_dir = ''.join([rnys_dir,'grid_cell_level/'])

    CFSR_fi = "".join([rnys_dir,"CFSR_all.nc"])
    MERRA2_fi = "".join([rnys_dir,"MERRA2.nc"])
    ERA5_fi = "".join([rnys_dir,"ERA5.nc"])
    ERAI_fi = "".join([rnys_dir,"ERA-Interim.nc"])
    JRA_fi = "".join([rnys_dir,"JRA55.nc"])
    GLDAS_fi = "".join([rnys_dir,"GLDAS.nc"])

    for j in naive_type:
    	naive_type_j = j
    	naive_dir_raw = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_product/'+str(naive_type_j)+'/raw_temp/'+str(remap_type)+'/'])
    	naive_fi = ''.join([naive_dir_raw+str(remap_type)+'_'+'Naive_stemp_TOP30cm.nc'])
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

    			station_data_master_warm_temp = []
    			naive_data_master_warm_temp = []
    			CFSR_data_master_warm_temp = []
    			ERAI_data_master_warm_temp = []
    			ERA5_data_master_warm_temp = []
    			JRA_data_master_warm_temp = []
    			MERRA2_data_master_warm_temp = []
    			GLDAS_data_master_warm_temp = []

    			station_data_master_cold_date = []
    			naive_data_master_cold_date = []
    			CFSR_data_master_cold_date = []
    			ERAI_data_master_cold_date = []
    			ERA5_data_master_cold_date = []
    			JRA_data_master_cold_date = []
    			MERRA2_data_master_cold_date = []
    			GLDAS_data_master_cold_date = []

    			station_data_master_warm_date = []
    			naive_data_master_warm_date = []
    			CFSR_data_master_warm_date = []
    			ERAI_data_master_warm_date = []
    			ERA5_data_master_warm_date = []
    			JRA_data_master_warm_date = []
    			MERRA2_data_master_warm_date = []
    			GLDAS_data_master_warm_date = []

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

    				CFSR_stemp = CFSR_fil[CFSR_layer] - 273.15
    				ERAI_stemp = ERAI_fil[ERAI_layer] - 273.15
    				ERA5_stemp = ERA5_fil[ERA5_layer] - 273.15
    				JRA_stemp = JRA_fil[JRA_layer] - 273.15
    				MERRA2_stemp = MERRA2_fil[MERRA2_layer] -273.15
    				GLDAS_stemp = GLDAS_fil[GLDAS_layer] - 273.15

    				naive_fil = xr.open_dataset(naive_fi)
    				naive_stemp = naive_fil[Naive_layer]


    				#print(type(CFSR_stemp))

    				CFSR_stemp_gcell = CFSR_stemp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    				ERAI_stemp_gcell = ERAI_stemp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    				ERA5_stemp_gcell = ERA5_stemp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    				JRA_stemp_gcell = JRA_stemp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    				MERRA2_stemp_gcell = MERRA2_stemp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    				GLDAS_stemp_gcell = GLDAS_stemp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    				naive_stemp_gcell = naive_stemp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)

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
    					naive_temp_dt = naive_stemp_gcell.sel(time=DateTime_m).values.tolist()
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
    					MERRA2_temp_dt = MERRA2_stemp_gcell.sel(time=DateTime_m).values.tolist()
    					if(str(MERRA2_temp_dt) == "nan"):
    						MERRA2_temp_dt = np.nan
    					MERRA2_temp_master.append(MERRA2_temp_dt)
    					GLDAS_temp_dt = GLDAS_stemp_gcell.sel(time=DateTime_m).values.tolist()
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

###### Cold Season (Soil Temp <= -2) #####

    				cold_season_station_index = np.where(station_temp_master <= -2)
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
					   				

    				if(len(naive_temp_cold_season) == 0 or len(station_temp_cold_season) == 0 or len(CFSR_temp_cold_season) == 0):
    					print('Grid Cell Skipped')
    					continue

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
    				if(len(dframe_cold_season_temp) < 30):
    					print('Grid Cell Skipped')
    					continue

###### Warm Season (Soil Temp > -2) #####
    				warm_season_station_index = np.where(station_temp_master > -2)
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
    				if(len(dframe_warm_season_temp) < 30):
    					print('Grid Cell Skipped')
    					continue


################# Separate by cold and warm season (by month) #######################

##### Create Collocated Temperature Dataframe #####
    				dframe_temp_gcell_stn_raw = pd.DataFrame(data=date_temp_master,columns=['Date'])
    				dframe_temp_gcell_stn_raw['DateTime'] = [datetime.datetime.strptime(i,'%Y-%m-%d') for i in date_temp_master]
    				dframe_temp_gcell_stn_raw['Grid Cell'] = gcell
    				dframe_temp_gcell_stn_raw['Lat'] = lat_cen
    				dframe_temp_gcell_stn_raw['Lon'] = lon_cen
    				dframe_temp_gcell_stn_raw['Station'] = station_temp_master
    				dframe_temp_gcell_stn_raw['Naive Blend'] = naive_temp_master
    				dframe_temp_gcell_stn_raw['CFSR'] = CFSR_temp_master
    				dframe_temp_gcell_stn_raw['ERA-Interim'] = ERAI_temp_master
    				dframe_temp_gcell_stn_raw['ERA5'] = ERA5_temp_master
    				dframe_temp_gcell_stn_raw['JRA55'] = JRA_temp_master
    				dframe_temp_gcell_stn_raw['MERRA2'] = MERRA2_temp_master
    				dframe_temp_gcell_stn_raw['GLDAS'] = GLDAS_temp_master
    				dframe_temp_gcell_stn_raw = dframe_temp_gcell_stn_raw.set_index('DateTime')
    				dframe_temp_gcell_stn_raw = dframe_temp_gcell_stn_raw[['Grid Cell','Lat','Lon','Station','Naive Blend','CFSR','ERA-Interim','ERA5','JRA55','MERRA2','GLDAS']]


##### Cold Season (Nov - May) #####
    				dframe_cold_season_date = dframe_temp_gcell_stn_raw[(dframe_temp_gcell_stn_raw.index.month == 11) | (dframe_temp_gcell_stn_raw.index.month == 12) | (dframe_temp_gcell_stn_raw.index.month == 1) | (dframe_temp_gcell_stn_raw.index.month == 2) | (dframe_temp_gcell_stn_raw.index.month == 3) | (dframe_temp_gcell_stn_raw.index.month == 4) | (dframe_temp_gcell_stn_raw.index.month == 5)]
    				dframe_cold_season_date = dframe_cold_season_date.dropna()
    				dframe_cold_season_date['N'] = len(dframe_cold_season_date)
##### Warm Season (Jun - Oct) #####
    				dframe_warm_season_date = dframe_temp_gcell_stn_raw[(dframe_temp_gcell_stn_raw.index.month == 6) | (dframe_temp_gcell_stn_raw.index.month == 7) | (dframe_temp_gcell_stn_raw.index.month == 8) | (dframe_temp_gcell_stn_raw.index.month == 9) | (dframe_temp_gcell_stn_raw.index.month == 10)]    				
    				dframe_warm_season_date = dframe_warm_season_date.dropna()
    				dframe_warm_season_date['N'] = len(dframe_warm_season_date)

    				if(len(dframe_cold_season_date) < 30 or len(dframe_warm_season_date) < 30):
    					print('Grid Cell Skipped')
    					continue
					
    				#print(dframe_cold_season_temp)
    				#print(dframe_warm_season_temp)
    				#print(dframe_cold_season_date)
    				#print(dframe_warm_season_date)

    				gcell_master_cold_temp.append(dframe_cold_season_temp['Grid Cell'].values.tolist())
    				gcell_master_warm_temp.append(dframe_warm_season_temp['Grid Cell'].values.tolist())
    				gcell_master_cold_date.append(dframe_cold_season_date['Grid Cell'].values.tolist())
    				gcell_master_warm_date.append(dframe_warm_season_date['Grid Cell'].values.tolist())
				
    				lat_master_cold_temp.append(dframe_cold_season_temp['Lat'].values.tolist())
    				lat_master_warm_temp.append(dframe_warm_season_temp['Lat'].values.tolist())
    				lat_master_cold_date.append(dframe_cold_season_date['Lat'].values.tolist())
    				lat_master_warm_date.append(dframe_warm_season_date['Lat'].values.tolist())
				
    				lon_master_cold_temp.append(dframe_cold_season_temp['Lon'].values.tolist())
    				lon_master_warm_temp.append(dframe_warm_season_temp['Lon'].values.tolist())
    				lon_master_cold_date.append(dframe_cold_season_date['Lon'].values.tolist())
    				lon_master_warm_date.append(dframe_warm_season_date['Lon'].values.tolist())				
				 
    				sample_size_master_cold_temp.append(dframe_cold_season_temp['N'].values.tolist())
    				sample_size_master_warm_temp.append(dframe_warm_season_temp['N'].values.tolist())
    				sample_size_master_cold_date.append(dframe_cold_season_date['N'].values.tolist())
    				sample_size_master_warm_date.append(dframe_warm_season_date['N'].values.tolist())

    				station_data_master_cold_temp.append(dframe_cold_season_temp['Station'].values.tolist())
    				station_data_master_warm_temp.append(dframe_warm_season_temp['Station'].values.tolist())
    				station_data_master_cold_date.append(dframe_cold_season_date['Station'].values.tolist())
    				station_data_master_warm_date.append(dframe_warm_season_date['Station'].values.tolist())

    				naive_data_master_cold_temp.append(dframe_cold_season_temp['Naive Blend'].values.tolist())
    				naive_data_master_warm_temp.append(dframe_warm_season_temp['Naive Blend'].values.tolist())
    				naive_data_master_cold_date.append(dframe_cold_season_date['Naive Blend'].values.tolist())
    				naive_data_master_warm_date.append(dframe_warm_season_date['Naive Blend'].values.tolist())

    				CFSR_data_master_cold_temp.append(dframe_cold_season_temp['CFSR'].values.tolist())
    				CFSR_data_master_warm_temp.append(dframe_warm_season_temp['CFSR'].values.tolist())
    				CFSR_data_master_cold_date.append(dframe_cold_season_date['CFSR'].values.tolist())
    				CFSR_data_master_warm_date.append(dframe_warm_season_date['CFSR'].values.tolist())

    				ERAI_data_master_cold_temp.append(dframe_cold_season_temp['ERA-Interim'].values.tolist())
    				ERAI_data_master_warm_temp.append(dframe_warm_season_temp['ERA-Interim'].values.tolist())
    				ERAI_data_master_cold_date.append(dframe_cold_season_date['ERA-Interim'].values.tolist())
    				ERAI_data_master_warm_date.append(dframe_warm_season_date['ERA-Interim'].values.tolist())

    				ERA5_data_master_cold_temp.append(dframe_cold_season_temp['ERA5'].values.tolist())
    				ERA5_data_master_warm_temp.append(dframe_warm_season_temp['ERA5'].values.tolist())
    				ERA5_data_master_cold_date.append(dframe_cold_season_date['ERA5'].values.tolist())
    				ERA5_data_master_warm_date.append(dframe_warm_season_date['ERA5'].values.tolist())

    				JRA_data_master_cold_temp.append(dframe_cold_season_temp['JRA55'].values.tolist())
    				JRA_data_master_warm_temp.append(dframe_warm_season_temp['JRA55'].values.tolist())
    				JRA_data_master_cold_date.append(dframe_cold_season_date['JRA55'].values.tolist())
    				JRA_data_master_warm_date.append(dframe_warm_season_date['JRA55'].values.tolist())

    				MERRA2_data_master_cold_temp.append(dframe_cold_season_temp['MERRA2'].values.tolist())
    				MERRA2_data_master_warm_temp.append(dframe_warm_season_temp['MERRA2'].values.tolist())
    				MERRA2_data_master_cold_date.append(dframe_cold_season_date['MERRA2'].values.tolist())
    				MERRA2_data_master_warm_date.append(dframe_warm_season_date['MERRA2'].values.tolist())

    				GLDAS_data_master_cold_temp.append(dframe_cold_season_temp['GLDAS'].values.tolist())
    				GLDAS_data_master_warm_temp.append(dframe_warm_season_temp['GLDAS'].values.tolist())
    				GLDAS_data_master_cold_date.append(dframe_cold_season_date['GLDAS'].values.tolist())
    				GLDAS_data_master_warm_date.append(dframe_warm_season_date['GLDAS'].values.tolist())

    			gcell_master_cold_temp = [i for sub in gcell_master_cold_temp for i in sub]
    			gcell_master_warm_temp = [i for sub in gcell_master_warm_temp for i in sub]
    			gcell_master_cold_date = [i for sub in gcell_master_cold_date for i in sub]
    			gcell_master_warm_date = [i for sub in gcell_master_warm_date for i in sub]

    			lat_master_cold_temp = [i for sub in lat_master_cold_temp for i in sub]
    			lat_master_warm_temp = [i for sub in lat_master_warm_temp for i in sub]
    			lat_master_cold_date = [i for sub in lat_master_cold_date for i in sub]
    			lat_master_warm_date = [i for sub in lat_master_warm_date for i in sub]

    			lon_master_cold_temp = [i for sub in lon_master_cold_temp for i in sub]
    			lon_master_warm_temp = [i for sub in lon_master_warm_temp for i in sub]
    			lon_master_cold_date = [i for sub in lon_master_cold_date for i in sub]
    			lon_master_warm_date = [i for sub in lon_master_warm_date for i in sub]

    			sample_size_master_cold_temp = [i for sub in sample_size_master_cold_temp for i in sub]
    			sample_size_master_warm_temp = [i for sub in sample_size_master_warm_temp for i in sub]
    			sample_size_master_cold_date = [i for sub in sample_size_master_cold_date for i in sub]
    			sample_size_master_warm_date = [i for sub in sample_size_master_warm_date for i in sub]

    			station_data_master_cold_temp = [i for sub in station_data_master_cold_temp for i in sub]
    			station_data_master_warm_temp = [i for sub in station_data_master_warm_temp for i in sub]
    			station_data_master_cold_date = [i for sub in station_data_master_cold_date for i in sub]
    			station_data_master_warm_date = [i for sub in station_data_master_warm_date for i in sub]

    			naive_data_master_cold_temp = [i for sub in naive_data_master_cold_temp for i in sub]
    			naive_data_master_warm_temp = [i for sub in naive_data_master_warm_temp for i in sub]
    			naive_data_master_cold_date = [i for sub in naive_data_master_cold_date for i in sub]
    			naive_data_master_warm_date = [i for sub in naive_data_master_warm_date for i in sub]

    			CFSR_data_master_cold_temp = [i for sub in CFSR_data_master_cold_temp for i in sub]
    			CFSR_data_master_warm_temp = [i for sub in CFSR_data_master_warm_temp for i in sub]
    			CFSR_data_master_cold_date = [i for sub in CFSR_data_master_cold_date for i in sub]
    			CFSR_data_master_warm_date = [i for sub in CFSR_data_master_warm_date for i in sub]

    			ERAI_data_master_cold_temp = [i for sub in ERAI_data_master_cold_temp for i in sub]
    			ERAI_data_master_warm_temp = [i for sub in ERAI_data_master_warm_temp for i in sub]
    			ERAI_data_master_cold_date = [i for sub in ERAI_data_master_cold_date for i in sub]
    			ERAI_data_master_warm_date = [i for sub in ERAI_data_master_warm_date for i in sub]

    			ERA5_data_master_cold_temp = [i for sub in ERA5_data_master_cold_temp for i in sub]
    			ERA5_data_master_warm_temp = [i for sub in ERA5_data_master_warm_temp for i in sub]
    			ERA5_data_master_cold_date = [i for sub in ERA5_data_master_cold_date for i in sub]
    			ERA5_data_master_warm_date = [i for sub in ERA5_data_master_warm_date for i in sub]

    			JRA_data_master_cold_temp = [i for sub in JRA_data_master_cold_temp for i in sub]
    			JRA_data_master_warm_temp = [i for sub in JRA_data_master_warm_temp for i in sub]
    			JRA_data_master_cold_date = [i for sub in JRA_data_master_cold_date for i in sub]
    			JRA_data_master_warm_date = [i for sub in JRA_data_master_warm_date for i in sub]

    			MERRA2_data_master_cold_temp = [i for sub in MERRA2_data_master_cold_temp for i in sub]
    			MERRA2_data_master_warm_temp = [i for sub in MERRA2_data_master_warm_temp for i in sub]
    			MERRA2_data_master_cold_date = [i for sub in MERRA2_data_master_cold_date for i in sub]
    			MERRA2_data_master_warm_date = [i for sub in MERRA2_data_master_warm_date for i in sub]

    			GLDAS_data_master_cold_temp = [i for sub in GLDAS_data_master_cold_temp for i in sub]
    			GLDAS_data_master_warm_temp = [i for sub in GLDAS_data_master_warm_temp for i in sub]
    			GLDAS_data_master_cold_date = [i for sub in GLDAS_data_master_cold_date for i in sub]
    			GLDAS_data_master_warm_date = [i for sub in GLDAS_data_master_warm_date for i in sub]

    			dframe_cold_season_temp_master = pd.DataFrame(data=gcell_master_cold_temp, columns=['Grid Cell'])
    			dframe_cold_season_temp_master['Central Lat'] = lat_master_cold_temp
    			dframe_cold_season_temp_master['Central Lon'] = lon_master_cold_temp
    			dframe_cold_season_temp_master['N'] = sample_size_master_cold_temp
    			dframe_cold_season_temp_master['Station'] = station_data_master_cold_temp			
    			dframe_cold_season_temp_master['Naive Blend'] = naive_data_master_cold_temp
    			dframe_cold_season_temp_master['CFSR'] = CFSR_data_master_cold_temp
    			dframe_cold_season_temp_master['ERA-Interim'] = ERAI_data_master_cold_temp
    			dframe_cold_season_temp_master['ERA5'] = ERA5_data_master_cold_temp
    			dframe_cold_season_temp_master['JRA55'] = JRA_data_master_cold_temp
    			dframe_cold_season_temp_master['MERRA2'] = MERRA2_data_master_cold_temp
    			dframe_cold_season_temp_master['GLDAS'] = GLDAS_data_master_cold_temp

    			dframe_warm_season_temp_master = pd.DataFrame(data=gcell_master_warm_temp, columns=['Grid Cell'])
    			dframe_warm_season_temp_master['Central Lat'] = lat_master_warm_temp
    			dframe_warm_season_temp_master['Central Lon'] = lon_master_warm_temp
    			dframe_warm_season_temp_master['N'] = sample_size_master_warm_temp
    			dframe_warm_season_temp_master['Station'] = station_data_master_warm_temp			
    			dframe_warm_season_temp_master['Naive Blend'] = naive_data_master_warm_temp
    			dframe_warm_season_temp_master['CFSR'] = CFSR_data_master_warm_temp
    			dframe_warm_season_temp_master['ERA-Interim'] = ERAI_data_master_warm_temp
    			dframe_warm_season_temp_master['ERA5'] = ERA5_data_master_warm_temp
    			dframe_warm_season_temp_master['JRA55'] = JRA_data_master_warm_temp
    			dframe_warm_season_temp_master['MERRA2'] = MERRA2_data_master_warm_temp
    			dframe_warm_season_temp_master['GLDAS'] = GLDAS_data_master_warm_temp

    			dframe_cold_season_date_master = pd.DataFrame(data=gcell_master_cold_date, columns=['Grid Cell'])
    			dframe_cold_season_date_master['Central Lat'] = lat_master_cold_date
    			dframe_cold_season_date_master['Central Lon'] = lon_master_cold_date
    			dframe_cold_season_date_master['N'] = sample_size_master_cold_date
    			dframe_cold_season_date_master['Station'] = station_data_master_cold_date			
    			dframe_cold_season_date_master['Naive Blend'] = naive_data_master_cold_date
    			dframe_cold_season_date_master['CFSR'] = CFSR_data_master_cold_date
    			dframe_cold_season_date_master['ERA-Interim'] = ERAI_data_master_cold_date
    			dframe_cold_season_date_master['ERA5'] = ERA5_data_master_cold_date
    			dframe_cold_season_date_master['JRA55'] = JRA_data_master_cold_date
    			dframe_cold_season_date_master['MERRA2'] = MERRA2_data_master_cold_date
    			dframe_cold_season_date_master['GLDAS'] = GLDAS_data_master_cold_date

    			dframe_warm_season_date_master = pd.DataFrame(data=gcell_master_warm_date, columns=['Grid Cell'])
    			dframe_warm_season_date_master['Central Lat'] = lat_master_warm_date
    			dframe_warm_season_date_master['Central Lon'] = lon_master_warm_date
    			dframe_warm_season_date_master['N'] = sample_size_master_warm_date
    			dframe_warm_season_date_master['Station'] = station_data_master_warm_date			
    			dframe_warm_season_date_master['Naive Blend'] = naive_data_master_warm_date
    			dframe_warm_season_date_master['CFSR'] = CFSR_data_master_warm_date
    			dframe_warm_season_date_master['ERA-Interim'] = ERAI_data_master_warm_date
    			dframe_warm_season_date_master['ERA5'] = ERA5_data_master_warm_date
    			dframe_warm_season_date_master['JRA55'] = JRA_data_master_warm_date
    			dframe_warm_season_date_master['MERRA2'] = MERRA2_data_master_warm_date
    			dframe_warm_season_date_master['GLDAS'] = GLDAS_data_master_warm_date

    			#print(dframe_cold_season_temp_master)
    			#print(dframe_warm_season_temp_master)


##################### Cold Season (Temp) ####################


    			cold_season_temp_gcell = np.unique(dframe_cold_season_temp_master['Grid Cell'].values)


    			label = {'Naive Blend': 'dodgerblue', 'CFSR': 'm','ERA-Interim': 'g', 'ERA5': 'c', 'JRA55': 'r', 'MERRA2': 'y', 'GLDAS': 'k'}
    			SDEV_naive_cold_temp = []
    			SDEV_CFSR_cold_temp = []
    			SDEV_ERAI_cold_temp = []
    			SDEV_ERA5_cold_temp = []
    			SDEV_JRA_cold_temp = []
    			SDEV_MERRA2_cold_temp = []
    			SDEV_GLDAS_cold_temp = []

    			CRMSD_naive_cold_temp = []
    			CRMSD_CFSR_cold_temp = []
    			CRMSD_ERAI_cold_temp = []
    			CRMSD_ERA5_cold_temp = []
    			CRMSD_JRA_cold_temp = []
    			CRMSD_MERRA2_cold_temp = []
    			CRMSD_GLDAS_cold_temp = []

    			CCOEF_naive_cold_temp = []
    			CCOEF_CFSR_cold_temp = []
    			CCOEF_ERAI_cold_temp = []
    			CCOEF_ERA5_cold_temp = []
    			CCOEF_JRA_cold_temp = []
    			CCOEF_MERRA2_cold_temp = []
    			CCOEF_GLDAS_cold_temp = []			
    			for a in cold_season_temp_gcell:
    				grid_cell_a = a
    				dframe_cold_season_temp_master_gcell = dframe_cold_season_temp_master[dframe_cold_season_temp_master['Grid Cell'] == a]
    				#print(dframe_cold_season_temp_master_gcell)
    				station_cold_season_temp_gcell = dframe_cold_season_temp_master_gcell['Station'].values
    				naive_cold_season_temp_gcell = dframe_cold_season_temp_master_gcell['Naive Blend'].values
    				CFSR_cold_season_temp_gcell = dframe_cold_season_temp_master_gcell['CFSR'].values    				
    				ERAI_cold_season_temp_gcell = dframe_cold_season_temp_master_gcell['ERA-Interim'].values
    				ERA5_cold_season_temp_gcell = dframe_cold_season_temp_master_gcell['ERA5'].values
    				JRA_cold_season_temp_gcell = dframe_cold_season_temp_master_gcell['JRA55'].values
    				MERRA2_cold_season_temp_gcell = dframe_cold_season_temp_master_gcell['MERRA2'].values
    				GLDAS_cold_season_temp_gcell = dframe_cold_season_temp_master_gcell['GLDAS'].values
				
    				taylor_stats_naive_cold_temp = sm.taylor_statistics(naive_cold_season_temp_gcell,station_cold_season_temp_gcell)
    				taylor_stats_CFSR_cold_temp = sm.taylor_statistics(CFSR_cold_season_temp_gcell,station_cold_season_temp_gcell)			
    				taylor_stats_ERAI_cold_temp = sm.taylor_statistics(ERAI_cold_season_temp_gcell,station_cold_season_temp_gcell)
    				taylor_stats_ERA5_cold_temp = sm.taylor_statistics(ERA5_cold_season_temp_gcell,station_cold_season_temp_gcell)
    				taylor_stats_JRA_cold_temp = sm.taylor_statistics(JRA_cold_season_temp_gcell,station_cold_season_temp_gcell)
    				taylor_stats_MERRA2_cold_temp = sm.taylor_statistics(MERRA2_cold_season_temp_gcell,station_cold_season_temp_gcell)
    				taylor_stats_GLDAS_cold_temp = sm.taylor_statistics(GLDAS_cold_season_temp_gcell,station_cold_season_temp_gcell)

    				normalized_sdev_naive_cold_temp = taylor_stats_naive_cold_temp['sdev'][1]/taylor_stats_naive_cold_temp['sdev'][0]
    				SDEV_naive_cold_temp.append(normalized_sdev_naive_cold_temp)
    				normalized_sdev_CFSR_cold_temp = taylor_stats_CFSR_cold_temp['sdev'][1]/taylor_stats_CFSR_cold_temp['sdev'][0]
    				SDEV_CFSR_cold_temp.append(normalized_sdev_CFSR_cold_temp)    				
    				normalized_sdev_ERAI_cold_temp = taylor_stats_ERAI_cold_temp['sdev'][1]/taylor_stats_ERAI_cold_temp['sdev'][0]
    				SDEV_ERAI_cold_temp.append(normalized_sdev_ERAI_cold_temp)
    				normalized_sdev_ERA5_cold_temp = taylor_stats_ERA5_cold_temp['sdev'][1]/taylor_stats_ERA5_cold_temp['sdev'][0]
    				SDEV_ERA5_cold_temp.append(normalized_sdev_ERA5_cold_temp)
    				normalized_sdev_JRA_cold_temp = taylor_stats_JRA_cold_temp['sdev'][1]/taylor_stats_JRA_cold_temp['sdev'][0]
    				SDEV_JRA_cold_temp.append(normalized_sdev_JRA_cold_temp)
    				normalized_sdev_MERRA2_cold_temp = taylor_stats_MERRA2_cold_temp['sdev'][1]/taylor_stats_MERRA2_cold_temp['sdev'][0]
    				SDEV_MERRA2_cold_temp.append(normalized_sdev_MERRA2_cold_temp)
    				normalized_sdev_GLDAS_cold_temp = taylor_stats_GLDAS_cold_temp['sdev'][1]/taylor_stats_GLDAS_cold_temp['sdev'][0]
    				SDEV_GLDAS_cold_temp.append(normalized_sdev_GLDAS_cold_temp)

    				CRMSD_naive_cold_temp.append(taylor_stats_naive_cold_temp['crmsd'][1])
    				CRMSD_CFSR_cold_temp.append(taylor_stats_CFSR_cold_temp['crmsd'][1])
    				CRMSD_ERAI_cold_temp.append(taylor_stats_ERAI_cold_temp['crmsd'][1])
    				CRMSD_ERA5_cold_temp.append(taylor_stats_ERA5_cold_temp['crmsd'][1])
    				CRMSD_JRA_cold_temp.append(taylor_stats_JRA_cold_temp['crmsd'][1])
    				CRMSD_MERRA2_cold_temp.append(taylor_stats_MERRA2_cold_temp['crmsd'][1])
    				CRMSD_GLDAS_cold_temp.append(taylor_stats_GLDAS_cold_temp['crmsd'][1])

    				CCOEF_naive_cold_temp.append(taylor_stats_naive_cold_temp['ccoef'][1])
    				CCOEF_CFSR_cold_temp.append(taylor_stats_CFSR_cold_temp['ccoef'][1])
    				CCOEF_ERAI_cold_temp.append(taylor_stats_ERAI_cold_temp['ccoef'][1])
    				CCOEF_ERA5_cold_temp.append(taylor_stats_ERA5_cold_temp['ccoef'][1])
    				CCOEF_JRA_cold_temp.append(taylor_stats_JRA_cold_temp['ccoef'][1])
    				CCOEF_MERRA2_cold_temp.append(taylor_stats_MERRA2_cold_temp['ccoef'][1])
    				CCOEF_GLDAS_cold_temp.append(taylor_stats_GLDAS_cold_temp['ccoef'][1])

    			SDEV_naive_cold_temp2 = np.array(SDEV_naive_cold_temp)
    			SDEV_CFSR_cold_temp2 = np.array(SDEV_CFSR_cold_temp)
    			SDEV_ERAI_cold_temp2 = np.array(SDEV_ERAI_cold_temp)
    			SDEV_ERA5_cold_temp2 = np.array(SDEV_ERA5_cold_temp)
    			SDEV_JRA_cold_temp2 = np.array(SDEV_JRA_cold_temp)
    			SDEV_MERRA2_cold_temp2 = np.array(SDEV_MERRA2_cold_temp)
    			SDEV_GLDAS_cold_temp2 = np.array(SDEV_GLDAS_cold_temp)

    			CRMSD_naive_cold_temp2 = np.array(CRMSD_naive_cold_temp)
    			CRMSD_CFSR_cold_temp2 = np.array(CRMSD_CFSR_cold_temp)
    			CRMSD_ERAI_cold_temp2 = np.array(CRMSD_ERAI_cold_temp)
    			CRMSD_ERA5_cold_temp2 = np.array(CRMSD_ERA5_cold_temp)
    			CRMSD_JRA_cold_temp2 = np.array(CRMSD_JRA_cold_temp)
    			CRMSD_MERRA2_cold_temp2 = np.array(CRMSD_MERRA2_cold_temp)
    			CRMSD_GLDAS_cold_temp2 = np.array(CRMSD_GLDAS_cold_temp)

    			CCOEF_naive_cold_temp2 = np.array(CCOEF_naive_cold_temp)
    			CCOEF_CFSR_cold_temp2 = np.array(CCOEF_CFSR_cold_temp)
    			CCOEF_ERAI_cold_temp2 = np.array(CCOEF_ERAI_cold_temp)
    			CCOEF_ERA5_cold_temp2 = np.array(CCOEF_ERA5_cold_temp)
    			CCOEF_JRA_cold_temp2 = np.array(CCOEF_JRA_cold_temp)
    			CCOEF_MERRA2_cold_temp2 = np.array(CCOEF_MERRA2_cold_temp)
    			CCOEF_GLDAS_cold_temp2 = np.array(CCOEF_GLDAS_cold_temp)

    			sm.taylor_diagram(SDEV_ERAI_cold_temp2, CRMSD_ERAI_cold_temp2,CCOEF_ERAI_cold_temp2,rincSTD=0.25, titleRMS = 'off', showlabelsRMS = 'off', tickRMS =[0.0],tickSTD= np.linspace(0,4,17), markercolor = 'c', alpha = 0.0)
    			sm.taylor_diagram(SDEV_naive_cold_temp2, CRMSD_naive_cold_temp2,CCOEF_naive_cold_temp2,rincSTD=0.25, titleRMS = 'off', showlabelsRMS = 'off', tickRMS =[0.0],tickSTD= np.linspace(0,4,17),markercolor = 'dodgerblue', alpha = 0.0, overlay = 'on')
    			sm.taylor_diagram(SDEV_CFSR_cold_temp2, CRMSD_CFSR_cold_temp2,CCOEF_CFSR_cold_temp2,rincSTD=0.25, titleRMS = 'off', showlabelsRMS = 'off', tickRMS =[0.0],tickSTD= np.linspace(0,4,17), markercolor = 'm', alpha = 0.0, overlay = 'on')
    			sm.taylor_diagram(SDEV_ERA5_cold_temp2, CRMSD_ERA5_cold_temp2,CCOEF_ERA5_cold_temp2,rincSTD=0.25,  titleRMS = 'off', showlabelsRMS = 'off', tickRMS =[0.0],tickSTD= np.linspace(0,4,17), markercolor = 'g', alpha = 0.0, overlay = 'on')
    			sm.taylor_diagram(SDEV_JRA_cold_temp2, CRMSD_JRA_cold_temp2,CCOEF_JRA_cold_temp2,rincSTD=0.25,  titleRMS = 'off', showlabelsRMS = 'off', tickRMS =[0.0],tickSTD= np.linspace(0,4,17), markercolor = 'r', alpha = 0.0, overlay = 'on')
    			sm.taylor_diagram(SDEV_MERRA2_cold_temp2, CRMSD_MERRA2_cold_temp2,CCOEF_MERRA2_cold_temp2,rincSTD=0.25,  titleRMS = 'off', showlabelsRMS = 'off', tickRMS =[0.0],tickSTD= np.linspace(0,4,17), markercolor = 'y', alpha = 0.0, overlay = 'on')
    			sm.taylor_diagram(SDEV_GLDAS_cold_temp2, CRMSD_GLDAS_cold_temp2,CCOEF_GLDAS_cold_temp2,rincSTD=0.25, titleRMS = 'off', showlabelsRMS = 'off', tickRMS =[0.0],tickSTD= np.linspace(0,4,17), markercolor = 'k', alpha = 0.0, overlay = 'on', markerLabel = label)    			


    			plt.savefig('/mnt/data/users/herringtont/soil_temp/plots/taylor_diagrams/'+str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_top_30cm_thr_'+str(thr_l)+'_cold_season_temp_subset.png')


##################### warm Season (Temp) ####################


    			warm_season_temp_gcell = np.unique(dframe_warm_season_temp_master['Grid Cell'].values)


    			label = {'Naive Blend': 'dodgerblue', 'CFSR': 'm','ERA-Interim': 'g', 'ERA5': 'c', 'JRA55': 'r', 'MERRA2': 'y', 'GLDAS': 'k'}
    			SDEV_naive_warm_temp = []
    			SDEV_CFSR_warm_temp = []
    			SDEV_ERAI_warm_temp = []
    			SDEV_ERA5_warm_temp = []
    			SDEV_JRA_warm_temp = []
    			SDEV_MERRA2_warm_temp = []
    			SDEV_GLDAS_warm_temp = []

    			CRMSD_naive_warm_temp = []
    			CRMSD_CFSR_warm_temp = []
    			CRMSD_ERAI_warm_temp = []
    			CRMSD_ERA5_warm_temp = []
    			CRMSD_JRA_warm_temp = []
    			CRMSD_MERRA2_warm_temp = []
    			CRMSD_GLDAS_warm_temp = []

    			CCOEF_naive_warm_temp = []
    			CCOEF_CFSR_warm_temp = []
    			CCOEF_ERAI_warm_temp = []
    			CCOEF_ERA5_warm_temp = []
    			CCOEF_JRA_warm_temp = []
    			CCOEF_MERRA2_warm_temp = []
    			CCOEF_GLDAS_warm_temp = []			
    			for a in warm_season_temp_gcell:
    				grid_cell_a = a
    				dframe_warm_season_temp_master_gcell = dframe_warm_season_temp_master[dframe_warm_season_temp_master['Grid Cell'] == a]
    				#print(dframe_warm_season_temp_master_gcell)
    				station_warm_season_temp_gcell = dframe_warm_season_temp_master_gcell['Station'].values
    				naive_warm_season_temp_gcell = dframe_warm_season_temp_master_gcell['Naive Blend'].values
    				CFSR_warm_season_temp_gcell = dframe_warm_season_temp_master_gcell['CFSR'].values    				
    				ERAI_warm_season_temp_gcell = dframe_warm_season_temp_master_gcell['ERA-Interim'].values
    				ERA5_warm_season_temp_gcell = dframe_warm_season_temp_master_gcell['ERA5'].values
    				JRA_warm_season_temp_gcell = dframe_warm_season_temp_master_gcell['JRA55'].values
    				MERRA2_warm_season_temp_gcell = dframe_warm_season_temp_master_gcell['MERRA2'].values
    				GLDAS_warm_season_temp_gcell = dframe_warm_season_temp_master_gcell['GLDAS'].values
				
    				taylor_stats_naive_warm_temp = sm.taylor_statistics(naive_warm_season_temp_gcell,station_warm_season_temp_gcell)
    				taylor_stats_CFSR_warm_temp = sm.taylor_statistics(CFSR_warm_season_temp_gcell,station_warm_season_temp_gcell)			
    				taylor_stats_ERAI_warm_temp = sm.taylor_statistics(ERAI_warm_season_temp_gcell,station_warm_season_temp_gcell)
    				taylor_stats_ERA5_warm_temp = sm.taylor_statistics(ERA5_warm_season_temp_gcell,station_warm_season_temp_gcell)
    				taylor_stats_JRA_warm_temp = sm.taylor_statistics(JRA_warm_season_temp_gcell,station_warm_season_temp_gcell)
    				taylor_stats_MERRA2_warm_temp = sm.taylor_statistics(MERRA2_warm_season_temp_gcell,station_warm_season_temp_gcell)
    				taylor_stats_GLDAS_warm_temp = sm.taylor_statistics(GLDAS_warm_season_temp_gcell,station_warm_season_temp_gcell)

    				normalized_sdev_naive_warm_temp = taylor_stats_naive_warm_temp['sdev'][1]/taylor_stats_naive_warm_temp['sdev'][0]
    				SDEV_naive_warm_temp.append(normalized_sdev_naive_warm_temp)
    				normalized_sdev_CFSR_warm_temp = taylor_stats_CFSR_warm_temp['sdev'][1]/taylor_stats_CFSR_warm_temp['sdev'][0]
    				SDEV_CFSR_warm_temp.append(normalized_sdev_CFSR_warm_temp)    				
    				normalized_sdev_ERAI_warm_temp = taylor_stats_ERAI_warm_temp['sdev'][1]/taylor_stats_ERAI_warm_temp['sdev'][0]
    				SDEV_ERAI_warm_temp.append(normalized_sdev_ERAI_warm_temp)
    				normalized_sdev_ERA5_warm_temp = taylor_stats_ERA5_warm_temp['sdev'][1]/taylor_stats_ERA5_warm_temp['sdev'][0]
    				SDEV_ERA5_warm_temp.append(normalized_sdev_ERA5_warm_temp)
    				normalized_sdev_JRA_warm_temp = taylor_stats_JRA_warm_temp['sdev'][1]/taylor_stats_JRA_warm_temp['sdev'][0]
    				SDEV_JRA_warm_temp.append(normalized_sdev_JRA_warm_temp)
    				normalized_sdev_MERRA2_warm_temp = taylor_stats_MERRA2_warm_temp['sdev'][1]/taylor_stats_MERRA2_warm_temp['sdev'][0]
    				SDEV_MERRA2_warm_temp.append(normalized_sdev_MERRA2_warm_temp)
    				normalized_sdev_GLDAS_warm_temp = taylor_stats_GLDAS_warm_temp['sdev'][1]/taylor_stats_GLDAS_warm_temp['sdev'][0]
    				SDEV_GLDAS_warm_temp.append(normalized_sdev_GLDAS_warm_temp)

    				CRMSD_naive_warm_temp.append(taylor_stats_naive_warm_temp['crmsd'][1])
    				CRMSD_CFSR_warm_temp.append(taylor_stats_CFSR_warm_temp['crmsd'][1])
    				CRMSD_ERAI_warm_temp.append(taylor_stats_ERAI_warm_temp['crmsd'][1])
    				CRMSD_ERA5_warm_temp.append(taylor_stats_ERA5_warm_temp['crmsd'][1])
    				CRMSD_JRA_warm_temp.append(taylor_stats_JRA_warm_temp['crmsd'][1])
    				CRMSD_MERRA2_warm_temp.append(taylor_stats_MERRA2_warm_temp['crmsd'][1])
    				CRMSD_GLDAS_warm_temp.append(taylor_stats_GLDAS_warm_temp['crmsd'][1])

    				CCOEF_naive_warm_temp.append(taylor_stats_naive_warm_temp['ccoef'][1])
    				CCOEF_CFSR_warm_temp.append(taylor_stats_CFSR_warm_temp['ccoef'][1])
    				CCOEF_ERAI_warm_temp.append(taylor_stats_ERAI_warm_temp['ccoef'][1])
    				CCOEF_ERA5_warm_temp.append(taylor_stats_ERA5_warm_temp['ccoef'][1])
    				CCOEF_JRA_warm_temp.append(taylor_stats_JRA_warm_temp['ccoef'][1])
    				CCOEF_MERRA2_warm_temp.append(taylor_stats_MERRA2_warm_temp['ccoef'][1])
    				CCOEF_GLDAS_warm_temp.append(taylor_stats_GLDAS_warm_temp['ccoef'][1])

    			SDEV_naive_warm_temp2 = np.array(SDEV_naive_warm_temp)
    			SDEV_CFSR_warm_temp2 = np.array(SDEV_CFSR_warm_temp)
    			SDEV_ERAI_warm_temp2 = np.array(SDEV_ERAI_warm_temp)
    			SDEV_ERA5_warm_temp2 = np.array(SDEV_ERA5_warm_temp)
    			SDEV_JRA_warm_temp2 = np.array(SDEV_JRA_warm_temp)
    			SDEV_MERRA2_warm_temp2 = np.array(SDEV_MERRA2_warm_temp)
    			SDEV_GLDAS_warm_temp2 = np.array(SDEV_GLDAS_warm_temp)

    			CRMSD_naive_warm_temp2 = np.array(CRMSD_naive_warm_temp)
    			CRMSD_CFSR_warm_temp2 = np.array(CRMSD_CFSR_warm_temp)
    			CRMSD_ERAI_warm_temp2 = np.array(CRMSD_ERAI_warm_temp)
    			CRMSD_ERA5_warm_temp2 = np.array(CRMSD_ERA5_warm_temp)
    			CRMSD_JRA_warm_temp2 = np.array(CRMSD_JRA_warm_temp)
    			CRMSD_MERRA2_warm_temp2 = np.array(CRMSD_MERRA2_warm_temp)
    			CRMSD_GLDAS_warm_temp2 = np.array(CRMSD_GLDAS_warm_temp)

    			CCOEF_naive_warm_temp2 = np.array(CCOEF_naive_warm_temp)
    			CCOEF_CFSR_warm_temp2 = np.array(CCOEF_CFSR_warm_temp)
    			CCOEF_ERAI_warm_temp2 = np.array(CCOEF_ERAI_warm_temp)
    			CCOEF_ERA5_warm_temp2 = np.array(CCOEF_ERA5_warm_temp)
    			CCOEF_JRA_warm_temp2 = np.array(CCOEF_JRA_warm_temp)
    			CCOEF_MERRA2_warm_temp2 = np.array(CCOEF_MERRA2_warm_temp)
    			CCOEF_GLDAS_warm_temp2 = np.array(CCOEF_GLDAS_warm_temp)

    			sm.taylor_diagram(SDEV_ERAI_warm_temp2, CRMSD_ERAI_warm_temp2,CCOEF_ERAI_warm_temp2,rincSTD=0.25, titleRMS = 'off', showlabelsRMS = 'off', tickRMS =[0.0],tickSTD= np.linspace(0,4,17), markercolor = 'c', alpha = 0.0)
    			sm.taylor_diagram(SDEV_naive_warm_temp2, CRMSD_naive_warm_temp2,CCOEF_naive_warm_temp2,rincSTD=0.25, titleRMS = 'off', showlabelsRMS = 'off', tickRMS =[0.0],tickSTD= np.linspace(0,4,17),markercolor = 'dodgerblue', alpha = 0.0, overlay = 'on')
    			sm.taylor_diagram(SDEV_CFSR_warm_temp2, CRMSD_CFSR_warm_temp2,CCOEF_CFSR_warm_temp2,rincSTD=0.25, titleRMS = 'off', showlabelsRMS = 'off', tickRMS =[0.0],tickSTD= np.linspace(0,4,17), markercolor = 'm', alpha = 0.0, overlay = 'on')
    			sm.taylor_diagram(SDEV_ERA5_warm_temp2, CRMSD_ERA5_warm_temp2,CCOEF_ERA5_warm_temp2,rincSTD=0.25,  titleRMS = 'off', showlabelsRMS = 'off', tickRMS =[0.0],tickSTD= np.linspace(0,4,17), markercolor = 'g', alpha = 0.0, overlay = 'on')
    			sm.taylor_diagram(SDEV_JRA_warm_temp2, CRMSD_JRA_warm_temp2,CCOEF_JRA_warm_temp2,rincSTD=0.25,  titleRMS = 'off', showlabelsRMS = 'off', tickRMS =[0.0],tickSTD= np.linspace(0,4,17), markercolor = 'r', alpha = 0.0, overlay = 'on')
    			sm.taylor_diagram(SDEV_MERRA2_warm_temp2, CRMSD_MERRA2_warm_temp2,CCOEF_MERRA2_warm_temp2,rincSTD=0.25,  titleRMS = 'off', showlabelsRMS = 'off', tickRMS =[0.0],tickSTD= np.linspace(0,4,17), markercolor = 'y', alpha = 0.0, overlay = 'on')
    			sm.taylor_diagram(SDEV_GLDAS_warm_temp2, CRMSD_GLDAS_warm_temp2,CCOEF_GLDAS_warm_temp2,rincSTD=0.25, titleRMS = 'off', showlabelsRMS = 'off', tickRMS =[0.0],tickSTD= np.linspace(0,4,17), markercolor = 'k', alpha = 0.0, overlay = 'on', markerLabel = label)  

    			#plt.show()
    			plt.savefig('/mnt/data/users/herringtont/soil_temp/plots/taylor_diagrams/'+str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_top_30cm_thr'+str(thr_l)+'_warm_season_temp_subset.png')
