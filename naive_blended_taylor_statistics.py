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
olr = ['outliers','zscore','IQR']
lyr = ['top_30cm']
thr = ['0','25','50','75','100']
rmp_type = ['nn','bil']
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
    			gcell_master_stn = []
    			gcell_master = []
    			lat_master_stn = []
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
    			naive_bias_master_cold_temp = []
    			CFSR_bias_master_cold_temp = []
    			ERAI_bias_master_cold_temp = []
    			ERA5_bias_master_cold_temp = []
    			JRA_bias_master_cold_temp = []
    			MERRA2_bias_master_cold_temp = []
    			GLDAS_bias_master_cold_temp = []

    			naive_RMSD_master_cold_temp = []
    			CFSR_RMSD_master_cold_temp = []
    			ERAI_RMSD_master_cold_temp = []
    			ERA5_RMSD_master_cold_temp = []
    			JRA_RMSD_master_cold_temp = []
    			MERRA2_RMSD_master_cold_temp = []
    			GLDAS_RMSD_master_cold_temp = []

    			naive_CRMSD_master_cold_temp = []
    			CFSR_CRMSD_master_cold_temp = []
    			ERAI_CRMSD_master_cold_temp = []
    			ERA5_CRMSD_master_cold_temp = []
    			JRA_CRMSD_master_cold_temp = []
    			MERRA2_CRMSD_master_cold_temp = []
    			GLDAS_CRMSD_master_cold_temp = []

    			station_SDV_master_cold_temp = []
    			naive_SDV_master_cold_temp = []
    			CFSR_SDV_master_cold_temp = []
    			ERAI_SDV_master_cold_temp = []
    			ERA5_SDV_master_cold_temp = []
    			JRA_SDV_master_cold_temp = []
    			MERRA2_SDV_master_cold_temp = []
    			GLDAS_SDV_master_cold_temp = []

    			naive_BSS_master_cold_temp = []
    			CFSR_BSS_master_cold_temp = []
    			ERAI_BSS_master_cold_temp = []
    			ERA5_BSS_master_cold_temp = []
    			JRA_BSS_master_cold_temp = []
    			MERRA2_BSS_master_cold_temp = []
    			GLDAS_BSS_master_cold_temp = []

    			naive_NDSS_master_cold_temp = []
    			CFSR_NDSS_master_cold_temp = []
    			ERAI_NDSS_master_cold_temp = []
    			ERA5_NDSS_master_cold_temp = []
    			JRA_NDSS_master_cold_temp = []
    			MERRA2_NDSS_master_cold_temp = []
    			GLDAS_NDSS_master_cold_temp = []

    			naive_corr_master_cold_temp = []
    			CFSR_corr_master_cold_temp = []
    			ERAI_corr_master_cold_temp = []
    			ERA5_corr_master_cold_temp = []
    			JRA_corr_master_cold_temp = []
    			MERRA2_corr_master_cold_temp = []
    			GLDAS_corr_master_cold_temp = []

    			naive_bias_master_warm_temp = []
    			CFSR_bias_master_warm_temp = []
    			ERAI_bias_master_warm_temp = []
    			ERA5_bias_master_warm_temp = []
    			JRA_bias_master_warm_temp = []
    			MERRA2_bias_master_warm_temp = []
    			GLDAS_bias_master_warm_temp = []

    			naive_RMSD_master_warm_temp = []
    			CFSR_RMSD_master_warm_temp = []
    			ERAI_RMSD_master_warm_temp = []
    			ERA5_RMSD_master_warm_temp = []
    			JRA_RMSD_master_warm_temp = []
    			MERRA2_RMSD_master_warm_temp = []
    			GLDAS_RMSD_master_warm_temp = []

    			naive_CRMSD_master_warm_temp = []
    			CFSR_CRMSD_master_warm_temp = []
    			ERAI_CRMSD_master_warm_temp = []
    			ERA5_CRMSD_master_warm_temp = []
    			JRA_CRMSD_master_warm_temp = []
    			MERRA2_CRMSD_master_warm_temp = []
    			GLDAS_CRMSD_master_warm_temp = []

    			station_SDV_master_warm_temp = []
    			naive_SDV_master_warm_temp = []
    			CFSR_SDV_master_warm_temp = []
    			ERAI_SDV_master_warm_temp = []
    			ERA5_SDV_master_warm_temp = []
    			JRA_SDV_master_warm_temp = []
    			MERRA2_SDV_master_warm_temp = []
    			GLDAS_SDV_master_warm_temp = []

    			naive_BSS_master_warm_temp = []
    			CFSR_BSS_master_warm_temp = []
    			ERAI_BSS_master_warm_temp = []
    			ERA5_BSS_master_warm_temp = []
    			JRA_BSS_master_warm_temp = []
    			MERRA2_BSS_master_warm_temp = []
    			GLDAS_BSS_master_warm_temp = []

    			naive_NDSS_master_warm_temp = []
    			CFSR_NDSS_master_warm_temp = []
    			ERAI_NDSS_master_warm_temp = []
    			ERA5_NDSS_master_warm_temp = []
    			JRA_NDSS_master_warm_temp = []
    			MERRA2_NDSS_master_warm_temp = []
    			GLDAS_NDSS_master_warm_temp = []

    			naive_corr_master_warm_temp = []
    			CFSR_corr_master_warm_temp = []
    			ERAI_corr_master_warm_temp = []
    			ERA5_corr_master_warm_temp = []
    			JRA_corr_master_warm_temp = []
    			MERRA2_corr_master_warm_temp = []
    			GLDAS_corr_master_warm_temp = []

    			naive_bias_master_cold_date = []
    			CFSR_bias_master_cold_date = []
    			ERAI_bias_master_cold_date = []
    			ERA5_bias_master_cold_date = []
    			JRA_bias_master_cold_date = []
    			MERRA2_bias_master_cold_date = []
    			GLDAS_bias_master_cold_date = []

    			naive_RMSD_master_cold_date = []
    			CFSR_RMSD_master_cold_date = []
    			ERAI_RMSD_master_cold_date = []
    			ERA5_RMSD_master_cold_date = []
    			JRA_RMSD_master_cold_date = []
    			MERRA2_RMSD_master_cold_date = []
    			GLDAS_RMSD_master_cold_date = []

    			naive_CRMSD_master_cold_date = []
    			CFSR_CRMSD_master_cold_date = []
    			ERAI_CRMSD_master_cold_date = []
    			ERA5_CRMSD_master_cold_date = []
    			JRA_CRMSD_master_cold_date = []
    			MERRA2_CRMSD_master_cold_date = []
    			GLDAS_CRMSD_master_cold_date = []

    			station_SDV_master_cold_date = []
    			naive_SDV_master_cold_date = []
    			CFSR_SDV_master_cold_date = []
    			ERAI_SDV_master_cold_date = []
    			ERA5_SDV_master_cold_date = []
    			JRA_SDV_master_cold_date = []
    			MERRA2_SDV_master_cold_date = []
    			GLDAS_SDV_master_cold_date = []

    			naive_BSS_master_cold_date = []
    			CFSR_BSS_master_cold_date = []
    			ERAI_BSS_master_cold_date = []
    			ERA5_BSS_master_cold_date = []
    			JRA_BSS_master_cold_date = []
    			MERRA2_BSS_master_cold_date = []
    			GLDAS_BSS_master_cold_date = []

    			naive_NDSS_master_cold_date = []
    			CFSR_NDSS_master_cold_date = []
    			ERAI_NDSS_master_cold_date = []
    			ERA5_NDSS_master_cold_date = []
    			JRA_NDSS_master_cold_date = []
    			MERRA2_NDSS_master_cold_date = []
    			GLDAS_NDSS_master_cold_date = []

    			naive_corr_master_cold_date = []
    			CFSR_corr_master_cold_date = []
    			ERAI_corr_master_cold_date = []
    			ERA5_corr_master_cold_date = []
    			JRA_corr_master_cold_date = []
    			MERRA2_corr_master_cold_date = []
    			GLDAS_corr_master_cold_date = []

    			naive_bias_master_warm_date = []
    			CFSR_bias_master_warm_date = []
    			ERAI_bias_master_warm_date = []
    			ERA5_bias_master_warm_date = []
    			JRA_bias_master_warm_date = []
    			MERRA2_bias_master_warm_date = []
    			GLDAS_bias_master_warm_date = []

    			naive_RMSD_master_warm_date = []
    			CFSR_RMSD_master_warm_date = []
    			ERAI_RMSD_master_warm_date = []
    			ERA5_RMSD_master_warm_date = []
    			JRA_RMSD_master_warm_date = []
    			MERRA2_RMSD_master_warm_date = []
    			GLDAS_RMSD_master_warm_date = []

    			naive_CRMSD_master_warm_date = []
    			CFSR_CRMSD_master_warm_date = []
    			ERAI_CRMSD_master_warm_date = []
    			ERA5_CRMSD_master_warm_date = []
    			JRA_CRMSD_master_warm_date = []
    			MERRA2_CRMSD_master_warm_date = []
    			GLDAS_CRMSD_master_warm_date = []

    			station_SDV_master_warm_date = []
    			naive_SDV_master_warm_date = []
    			CFSR_SDV_master_warm_date = []
    			ERAI_SDV_master_warm_date = []
    			ERA5_SDV_master_warm_date = []
    			JRA_SDV_master_warm_date = []
    			MERRA2_SDV_master_warm_date = []
    			GLDAS_SDV_master_warm_date = []

    			naive_BSS_master_warm_date = []
    			CFSR_BSS_master_warm_date = []
    			ERAI_BSS_master_warm_date = []
    			ERA5_BSS_master_warm_date = []
    			JRA_BSS_master_warm_date = []
    			MERRA2_BSS_master_warm_date = []
    			GLDAS_BSS_master_warm_date = []

    			naive_NDSS_master_warm_date = []
    			CFSR_NDSS_master_warm_date = []
    			ERAI_NDSS_master_warm_date = []
    			ERA5_NDSS_master_warm_date = []
    			JRA_NDSS_master_warm_date = []
    			MERRA2_NDSS_master_warm_date = []
    			GLDAS_NDSS_master_warm_date = []

    			naive_corr_master_warm_date = []
    			CFSR_corr_master_warm_date = []
    			ERAI_corr_master_warm_date = []
    			ERA5_corr_master_warm_date = []
    			JRA_corr_master_warm_date = []
    			MERRA2_corr_master_warm_date = []
    			GLDAS_corr_master_warm_date = []


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
    					continue
    					
    				if(len(naive_no_nan) == 0 or len(CFSR_no_nan) == 0): #skip if there are NaN values in blended data
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
    					continue

    				dframe_cold_season_temp = pd.DataFrame(data = station_temp_cold_season, columns=['Station'])
    				dframe_cold_season_temp['Naive Blend'] = naive_temp_cold_season
    				dframe_cold_season_temp['CFSR'] = CFSR_temp_cold_season
    				dframe_cold_season_temp['ERA-Interim'] = ERAI_temp_cold_season
    				dframe_cold_season_temp['ERA5'] = ERA5_temp_cold_season
    				dframe_cold_season_temp['JRA55'] = JRA_temp_cold_season
    				dframe_cold_season_temp['MERRA2'] = MERRA2_temp_cold_season
    				dframe_cold_season_temp['GLDAS'] = GLDAS_temp_cold_season
    				dframe_cold_season_temp = dframe_cold_season_temp.dropna()

    				if(len(dframe_cold_season_temp) < 30):
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
    					continue


    				dframe_warm_season_temp = pd.DataFrame(data = station_temp_warm_season, columns=['Station'])
    				dframe_warm_season_temp['Naive Blend'] = naive_temp_warm_season
    				dframe_warm_season_temp['CFSR'] = CFSR_temp_warm_season
    				dframe_warm_season_temp['ERA-Interim'] = ERAI_temp_warm_season
    				dframe_warm_season_temp['ERA5'] = ERA5_temp_warm_season
    				dframe_warm_season_temp['JRA55'] = JRA_temp_warm_season
    				dframe_warm_season_temp['MERRA2'] = MERRA2_temp_warm_season
    				dframe_warm_season_temp['GLDAS'] = GLDAS_temp_warm_season
    				dframe_warm_season_temp = dframe_cold_season_temp.dropna()

    				if(len(dframe_warm_season_temp) < 30):
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
    				dframe_temp_gcell_stn_raw = dframe_temp_gcell_stn_raw[['Station','Naive Blend','CFSR','ERA-Interim','ERA5','JRA55','MERRA2','GLDAS']]


##### Cold Season (Nov - May) #####
    				dframe_cold_season_date = dframe_temp_gcell_stn_raw[(dframe_temp_gcell_stn_raw.index.month == 11) | (dframe_temp_gcell_stn_raw.index.month == 12) | (dframe_temp_gcell_stn_raw.index.month == 1) | (dframe_temp_gcell_stn_raw.index.month == 2) | (dframe_temp_gcell_stn_raw.index.month == 3) | (dframe_temp_gcell_stn_raw.index.month == 4) | (dframe_temp_gcell_stn_raw.index.month == 5)]
    				dframe_cold_season_date = dframe_cold_season_date.dropna()
##### Warm Season (Jun - Oct) #####
    				dframe_warm_season_date = dframe_temp_gcell_stn_raw[(dframe_temp_gcell_stn_raw.index.month == 6) | (dframe_temp_gcell_stn_raw.index.month == 7) | (dframe_temp_gcell_stn_raw.index.month == 8) | (dframe_temp_gcell_stn_raw.index.month == 9) | (dframe_temp_gcell_stn_raw.index.month == 10)]    				
    				dframe_warm_season_date = dframe_warm_season_date.dropna()

    				if(len(dframe_cold_season_date) < 30 or len(dframe_warm_season_date) < 30):
    					continue

    				print(dframe_cold_season_temp)
    				print(dframe_warm_season_temp)
    				print(dframe_cold_season_date)
    				print(dframe_warm_season_date)

    				gcell_master.append(gcell)
    				lat_master.append(lat_cen)
    				lon_master.append(lon_cen)
    				sample_size_master_cold_temp.append(len(dframe_cold_season_temp))
    				sample_size_master_warm_temp.append(len(dframe_warm_season_temp))
    				sample_size_master_cold_date.append(len(dframe_cold_season_date))
    				sample_size_master_warm_date.append(len(dframe_warm_season_date))
################## Calculate Relevant Taylor Diagram Metrics for Each Grid Cell ###################

    				station_cold_temp = dframe_cold_season_temp['Station'].values
    				naive_cold_temp = dframe_cold_season_temp['Naive Blend'].values 
    				CFSR_cold_temp = dframe_cold_season_temp['CFSR'].values 
    				ERAI_cold_temp = dframe_cold_season_temp['ERA-Interim'].values
    				ERA5_cold_temp = dframe_cold_season_temp['ERA5'].values
    				JRA_cold_temp = dframe_cold_season_temp['JRA55'].values
    				MERRA2_cold_temp = dframe_cold_season_temp['MERRA2'].values
    				GLDAS_cold_temp = dframe_cold_season_temp['GLDAS'].values				

    				station_warm_temp = dframe_warm_season_temp['Station'].values
    				naive_warm_temp = dframe_warm_season_temp['Naive Blend'].values 
    				CFSR_warm_temp = dframe_warm_season_temp['CFSR'].values 
    				ERAI_warm_temp = dframe_warm_season_temp['ERA-Interim'].values
    				ERA5_warm_temp = dframe_warm_season_temp['ERA5'].values
    				JRA_warm_temp = dframe_warm_season_temp['JRA55'].values
    				MERRA2_warm_temp = dframe_warm_season_temp['MERRA2'].values
    				GLDAS_warm_temp = dframe_warm_season_temp['GLDAS'].values

    				station_cold_date = dframe_cold_season_date['Station'].values
    				naive_cold_date = dframe_cold_season_date['Naive Blend'].values 
    				CFSR_cold_date = dframe_cold_season_date['CFSR'].values 
    				ERAI_cold_date = dframe_cold_season_date['ERA-Interim'].values
    				ERA5_cold_date = dframe_cold_season_date['ERA5'].values
    				JRA_cold_date = dframe_cold_season_date['JRA55'].values
    				MERRA2_cold_date = dframe_cold_season_date['MERRA2'].values
    				GLDAS_cold_date = dframe_cold_season_date['GLDAS'].values

    				station_warm_date = dframe_warm_season_date['Station'].values
    				naive_warm_date = dframe_warm_season_date['Naive Blend'].values 
    				CFSR_warm_date = dframe_warm_season_date['CFSR'].values 
    				ERAI_warm_date = dframe_warm_season_date['ERA-Interim'].values
    				ERA5_warm_date = dframe_warm_season_date['ERA5'].values
    				JRA_warm_date = dframe_warm_season_date['JRA55'].values
    				MERRA2_warm_date = dframe_warm_season_date['MERRA2'].values
    				GLDAS_warm_date = dframe_warm_season_date['GLDAS'].values

### Cold Season (Temp) ###

    				naive_bias_cold_temp = sm.bias(station_cold_temp,naive_cold_temp)
    				naive_bias_master_cold_temp.append(naive_bias_cold_temp)
    				CFSR_bias_cold_temp = sm.bias(station_cold_temp,CFSR_cold_temp)
    				CFSR_bias_master_cold_temp.append(CFSR_bias_cold_temp)
    				ERAI_bias_cold_temp = sm.bias(station_cold_temp,ERAI_cold_temp)
    				ERAI_bias_master_cold_temp.append(ERAI_bias_cold_temp)
    				ERA5_bias_cold_temp = sm.bias(station_cold_temp,ERA5_cold_temp)
    				ERA5_bias_master_cold_temp.append(ERA5_bias_cold_temp)
    				JRA_bias_cold_temp = sm.bias(station_cold_temp,JRA_cold_temp)
    				JRA_bias_master_cold_temp.append(JRA_bias_cold_temp)
    				MERRA2_bias_cold_temp = sm.bias(station_cold_temp,MERRA2_cold_temp)
    				MERRA2_bias_master_cold_temp.append(MERRA2_bias_cold_temp)
    				GLDAS_bias_cold_temp = sm.bias(station_cold_temp,GLDAS_cold_temp)
    				GLDAS_bias_master_cold_temp.append(GLDAS_bias_cold_temp)

    				naive_RMSD_cold_temp = sm.rmsd(station_cold_temp,naive_cold_temp)
    				naive_RMSD_master_cold_temp.append(naive_RMSD_cold_temp)
    				CFSR_RMSD_cold_temp = sm.rmsd(station_cold_temp,CFSR_cold_temp)
    				CFSR_RMSD_master_cold_temp.append(CFSR_RMSD_cold_temp)
    				ERAI_RMSD_cold_temp = sm.rmsd(station_cold_temp,ERAI_cold_temp)
    				ERAI_RMSD_master_cold_temp.append(ERAI_RMSD_cold_temp)
    				ERA5_RMSD_cold_temp = sm.rmsd(station_cold_temp,ERA5_cold_temp)
    				ERA5_RMSD_master_cold_temp.append(ERA5_RMSD_cold_temp)
    				JRA_RMSD_cold_temp = sm.rmsd(station_cold_temp,JRA_cold_temp)
    				JRA_RMSD_master_cold_temp.append(JRA_RMSD_cold_temp)
    				MERRA2_RMSD_cold_temp = sm.rmsd(station_cold_temp,MERRA2_cold_temp)
    				MERRA2_RMSD_master_cold_temp.append(MERRA2_RMSD_cold_temp)
    				GLDAS_RMSD_cold_temp = sm.rmsd(station_cold_temp,GLDAS_cold_temp)
    				GLDAS_RMSD_master_cold_temp.append(GLDAS_RMSD_cold_temp)

    				naive_CRMSD_cold_temp = sm.centered_rms_dev(station_cold_temp,naive_cold_temp)
    				naive_CRMSD_master_cold_temp.append(naive_CRMSD_cold_temp)
    				CFSR_CRMSD_cold_temp = sm.centered_rms_dev(station_cold_temp,CFSR_cold_temp)
    				CFSR_CRMSD_master_cold_temp.append(CFSR_CRMSD_cold_temp)
    				ERAI_CRMSD_cold_temp = sm.centered_rms_dev(station_cold_temp,ERAI_cold_temp)
    				ERAI_CRMSD_master_cold_temp.append(ERAI_CRMSD_cold_temp)
    				ERA5_CRMSD_cold_temp = sm.centered_rms_dev(station_cold_temp,ERA5_cold_temp)
    				ERA5_CRMSD_master_cold_temp.append(ERA5_CRMSD_cold_temp)
    				JRA_CRMSD_cold_temp = sm.centered_rms_dev(station_cold_temp,JRA_cold_temp)
    				JRA_CRMSD_master_cold_temp.append(JRA_CRMSD_cold_temp)
    				MERRA2_CRMSD_cold_temp = sm.centered_rms_dev(station_cold_temp,MERRA2_cold_temp)
    				MERRA2_CRMSD_master_cold_temp.append(MERRA2_CRMSD_cold_temp)
    				GLDAS_CRMSD_cold_temp = sm.centered_rms_dev(station_cold_temp,GLDAS_cold_temp)
    				GLDAS_CRMSD_master_cold_temp.append(GLDAS_CRMSD_cold_temp)

    				station_SDV_cold_temp = np.std(station_cold_temp)
    				station_SDV_master_cold_temp.append(station_SDV_cold_temp)
    				naive_SDV_cold_temp = np.std(naive_cold_temp)
    				naive_SDV_master_cold_temp.append(naive_SDV_cold_temp)
    				CFSR_SDV_cold_temp = np.std(CFSR_cold_temp)
    				CFSR_SDV_master_cold_temp.append(CFSR_SDV_cold_temp)
    				ERAI_SDV_cold_temp = np.std(ERAI_cold_temp)
    				ERAI_SDV_master_cold_temp.append(ERAI_SDV_cold_temp)
    				ERA5_SDV_cold_temp = np.std(ERA5_cold_temp)
    				ERA5_SDV_master_cold_temp.append(ERA5_SDV_cold_temp)
    				JRA_SDV_cold_temp = np.std(JRA_cold_temp)
    				JRA_SDV_master_cold_temp.append(JRA_SDV_cold_temp)
    				MERRA2_SDV_cold_temp = np.std(MERRA2_cold_temp)
    				MERRA2_SDV_master_cold_temp.append(MERRA2_SDV_cold_temp)
    				GLDAS_SDV_cold_temp = np.std(GLDAS_cold_temp)
    				GLDAS_SDV_master_cold_temp.append(GLDAS_SDV_cold_temp)
				
    				naive_NDSS_cold_temp = sm.skill_score_murphy(station_cold_temp,naive_cold_temp)
    				naive_NDSS_master_cold_temp.append(naive_NDSS_cold_temp)
    				CFSR_NDSS_cold_temp = sm.skill_score_murphy(station_cold_temp,CFSR_cold_temp)
    				CFSR_NDSS_master_cold_temp.append(CFSR_NDSS_cold_temp)
    				ERAI_NDSS_cold_temp = sm.skill_score_murphy(station_cold_temp,ERAI_cold_temp)
    				ERAI_NDSS_master_cold_temp.append(ERAI_NDSS_cold_temp)
    				ERA5_NDSS_cold_temp = sm.skill_score_murphy(station_cold_temp,ERA5_cold_temp)
    				ERA5_NDSS_master_cold_temp.append(ERA5_NDSS_cold_temp)
    				JRA_NDSS_cold_temp = sm.skill_score_murphy(station_cold_temp,JRA_cold_temp)
    				JRA_NDSS_master_cold_temp.append(JRA_NDSS_cold_temp)
    				MERRA2_NDSS_cold_temp = sm.skill_score_murphy(station_cold_temp,MERRA2_cold_temp)
    				MERRA2_NDSS_master_cold_temp.append(MERRA2_NDSS_cold_temp)
    				GLDAS_NDSS_cold_temp = sm.skill_score_murphy(station_cold_temp,GLDAS_cold_temp)
    				GLDAS_NDSS_master_cold_temp.append(GLDAS_NDSS_cold_temp)

    				naive_corr_cold_temp, _ = scipy.stats.pearsonr(station_cold_temp,naive_cold_temp)
    				naive_corr_master_cold_temp.append(naive_corr_cold_temp)
    				CFSR_corr_cold_temp, _ = scipy.stats.pearsonr(station_cold_temp,CFSR_cold_temp)
    				CFSR_corr_master_cold_temp.append(CFSR_corr_cold_temp)
    				ERAI_corr_cold_temp, _ = scipy.stats.pearsonr(station_cold_temp,ERAI_cold_temp)
    				ERAI_corr_master_cold_temp.append(ERAI_corr_cold_temp)
    				ERA5_corr_cold_temp, _ = scipy.stats.pearsonr(station_cold_temp,ERA5_cold_temp)
    				ERA5_corr_master_cold_temp.append(ERA5_corr_cold_temp)
    				JRA_corr_cold_temp, _ = scipy.stats.pearsonr(station_cold_temp,JRA_cold_temp)
    				JRA_corr_master_cold_temp.append(JRA_corr_cold_temp)
    				MERRA2_corr_cold_temp, _ = scipy.stats.pearsonr(station_cold_temp,MERRA2_cold_temp)
    				MERRA2_corr_master_cold_temp.append(MERRA2_corr_cold_temp)
    				GLDAS_corr_cold_temp, _ = scipy.stats.pearsonr(station_cold_temp,GLDAS_cold_temp)
    				GLDAS_corr_master_cold_temp.append(GLDAS_corr_cold_temp)

### Warm Season (Temp) ###

    				naive_bias_warm_temp = sm.bias(station_warm_temp,naive_warm_temp)
    				naive_bias_master_warm_temp.append(naive_bias_warm_temp)
    				CFSR_bias_warm_temp = sm.bias(station_warm_temp,CFSR_warm_temp)
    				CFSR_bias_master_warm_temp.append(CFSR_bias_warm_temp)
    				ERAI_bias_warm_temp = sm.bias(station_warm_temp,ERAI_warm_temp)
    				ERAI_bias_master_warm_temp.append(ERAI_bias_warm_temp)
    				ERA5_bias_warm_temp = sm.bias(station_warm_temp,ERA5_warm_temp)
    				ERA5_bias_master_warm_temp.append(ERA5_bias_warm_temp)
    				JRA_bias_warm_temp = sm.bias(station_warm_temp,JRA_warm_temp)
    				JRA_bias_master_warm_temp.append(JRA_bias_warm_temp)
    				MERRA2_bias_warm_temp = sm.bias(station_warm_temp,MERRA2_warm_temp)
    				MERRA2_bias_master_warm_temp.append(MERRA2_bias_warm_temp)
    				GLDAS_bias_warm_temp = sm.bias(station_warm_temp,GLDAS_warm_temp)
    				GLDAS_bias_master_warm_temp.append(GLDAS_bias_warm_temp)

    				naive_RMSD_warm_temp = sm.rmsd(station_warm_temp,naive_warm_temp)
    				naive_RMSD_master_warm_temp.append(naive_RMSD_warm_temp)
    				CFSR_RMSD_warm_temp = sm.rmsd(station_warm_temp,CFSR_warm_temp)
    				CFSR_RMSD_master_warm_temp.append(CFSR_RMSD_warm_temp)
    				ERAI_RMSD_warm_temp = sm.rmsd(station_warm_temp,ERAI_warm_temp)
    				ERAI_RMSD_master_warm_temp.append(ERAI_RMSD_warm_temp)
    				ERA5_RMSD_warm_temp = sm.rmsd(station_warm_temp,ERA5_warm_temp)
    				ERA5_RMSD_master_warm_temp.append(ERA5_RMSD_warm_temp)
    				JRA_RMSD_warm_temp = sm.rmsd(station_warm_temp,JRA_warm_temp)
    				JRA_RMSD_master_warm_temp.append(JRA_RMSD_warm_temp)
    				MERRA2_RMSD_warm_temp = sm.rmsd(station_warm_temp,MERRA2_warm_temp)
    				MERRA2_RMSD_master_warm_temp.append(MERRA2_RMSD_warm_temp)
    				GLDAS_RMSD_warm_temp = sm.rmsd(station_warm_temp,GLDAS_warm_temp)
    				GLDAS_RMSD_master_warm_temp.append(GLDAS_RMSD_warm_temp)

    				naive_CRMSD_warm_temp = sm.centered_rms_dev(station_warm_temp,naive_warm_temp)
    				naive_CRMSD_master_warm_temp.append(naive_CRMSD_warm_temp)
    				CFSR_CRMSD_warm_temp = sm.centered_rms_dev(station_warm_temp,CFSR_warm_temp)
    				CFSR_CRMSD_master_warm_temp.append(CFSR_CRMSD_warm_temp)
    				ERAI_CRMSD_warm_temp = sm.centered_rms_dev(station_warm_temp,ERAI_warm_temp)
    				ERAI_CRMSD_master_warm_temp.append(ERAI_CRMSD_warm_temp)
    				ERA5_CRMSD_warm_temp = sm.centered_rms_dev(station_warm_temp,ERA5_warm_temp)
    				ERA5_CRMSD_master_warm_temp.append(ERA5_CRMSD_warm_temp)
    				JRA_CRMSD_warm_temp = sm.centered_rms_dev(station_warm_temp,JRA_warm_temp)
    				JRA_CRMSD_master_warm_temp.append(JRA_CRMSD_warm_temp)
    				MERRA2_CRMSD_warm_temp = sm.centered_rms_dev(station_warm_temp,MERRA2_warm_temp)
    				MERRA2_CRMSD_master_warm_temp.append(MERRA2_CRMSD_warm_temp)
    				GLDAS_CRMSD_warm_temp = sm.centered_rms_dev(station_warm_temp,GLDAS_warm_temp)
    				GLDAS_CRMSD_master_warm_temp.append(GLDAS_CRMSD_warm_temp)

    				station_SDV_warm_temp = np.std(station_warm_temp)
    				station_SDV_master_warm_temp.append(station_SDV_warm_temp)
    				naive_SDV_warm_temp = np.std(naive_warm_temp)
    				naive_SDV_master_warm_temp.append(naive_SDV_warm_temp)
    				CFSR_SDV_warm_temp = np.std(CFSR_warm_temp)
    				CFSR_SDV_master_warm_temp.append(CFSR_SDV_warm_temp)
    				ERAI_SDV_warm_temp = np.std(ERAI_warm_temp)
    				ERAI_SDV_master_warm_temp.append(ERAI_SDV_warm_temp)
    				ERA5_SDV_warm_temp = np.std(ERA5_warm_temp)
    				ERA5_SDV_master_warm_temp.append(ERA5_SDV_warm_temp)
    				JRA_SDV_warm_temp = np.std(JRA_warm_temp)
    				JRA_SDV_master_warm_temp.append(JRA_SDV_warm_temp)
    				MERRA2_SDV_warm_temp = np.std(MERRA2_warm_temp)
    				MERRA2_SDV_master_warm_temp.append(MERRA2_SDV_warm_temp)
    				GLDAS_SDV_warm_temp = np.std(GLDAS_warm_temp)
    				GLDAS_SDV_master_warm_temp.append(GLDAS_SDV_warm_temp)
				
    				naive_NDSS_warm_temp = sm.skill_score_murphy(station_warm_temp,naive_warm_temp)
    				naive_NDSS_master_warm_temp.append(naive_NDSS_warm_temp)
    				CFSR_NDSS_warm_temp = sm.skill_score_murphy(station_warm_temp,CFSR_warm_temp)
    				CFSR_NDSS_master_warm_temp.append(CFSR_NDSS_warm_temp)
    				ERAI_NDSS_warm_temp = sm.skill_score_murphy(station_warm_temp,ERAI_warm_temp)
    				ERAI_NDSS_master_warm_temp.append(ERAI_NDSS_warm_temp)
    				ERA5_NDSS_warm_temp = sm.skill_score_murphy(station_warm_temp,ERA5_warm_temp)
    				ERA5_NDSS_master_warm_temp.append(ERA5_NDSS_warm_temp)
    				JRA_NDSS_warm_temp = sm.skill_score_murphy(station_warm_temp,JRA_warm_temp)
    				JRA_NDSS_master_warm_temp.append(JRA_NDSS_warm_temp)
    				MERRA2_NDSS_warm_temp = sm.skill_score_murphy(station_warm_temp,MERRA2_warm_temp)
    				MERRA2_NDSS_master_warm_temp.append(MERRA2_NDSS_warm_temp)
    				GLDAS_NDSS_warm_temp = sm.skill_score_murphy(station_warm_temp,GLDAS_warm_temp)
    				GLDAS_NDSS_master_warm_temp.append(GLDAS_NDSS_warm_temp)

    				naive_corr_warm_temp, _ = scipy.stats.pearsonr(station_warm_temp,naive_warm_temp)
    				naive_corr_master_warm_temp.append(naive_corr_warm_temp)
    				CFSR_corr_warm_temp, _ = scipy.stats.pearsonr(station_warm_temp,CFSR_warm_temp)
    				CFSR_corr_master_warm_temp.append(CFSR_corr_warm_temp)
    				ERAI_corr_warm_temp, _ = scipy.stats.pearsonr(station_warm_temp,ERAI_warm_temp)
    				ERAI_corr_master_warm_temp.append(ERAI_corr_warm_temp)
    				ERA5_corr_warm_temp, _ = scipy.stats.pearsonr(station_warm_temp,ERA5_warm_temp)
    				ERA5_corr_master_warm_temp.append(ERA5_corr_warm_temp)
    				JRA_corr_warm_temp, _ = scipy.stats.pearsonr(station_warm_temp,JRA_warm_temp)
    				JRA_corr_master_warm_temp.append(JRA_corr_warm_temp)
    				MERRA2_corr_warm_temp, _ = scipy.stats.pearsonr(station_warm_temp,MERRA2_warm_temp)
    				MERRA2_corr_master_warm_temp.append(MERRA2_corr_warm_temp)
    				GLDAS_corr_warm_temp, _ = scipy.stats.pearsonr(station_warm_temp,GLDAS_warm_temp)
    				GLDAS_corr_master_warm_temp.append(GLDAS_corr_warm_temp)

### Cold Season (Date) ###

    				naive_bias_cold_date = sm.bias(station_cold_date,naive_cold_date)
    				naive_bias_master_cold_date.append(naive_bias_cold_date)
    				CFSR_bias_cold_date = sm.bias(station_cold_date,CFSR_cold_date)
    				CFSR_bias_master_cold_date.append(CFSR_bias_cold_date)
    				ERAI_bias_cold_date = sm.bias(station_cold_date,ERAI_cold_date)
    				ERAI_bias_master_cold_date.append(ERAI_bias_cold_date)
    				ERA5_bias_cold_date = sm.bias(station_cold_date,ERA5_cold_date)
    				ERA5_bias_master_cold_date.append(ERA5_bias_cold_date)
    				JRA_bias_cold_date = sm.bias(station_cold_date,JRA_cold_date)
    				JRA_bias_master_cold_date.append(JRA_bias_cold_date)
    				MERRA2_bias_cold_date = sm.bias(station_cold_date,MERRA2_cold_date)
    				MERRA2_bias_master_cold_date.append(MERRA2_bias_cold_date)
    				GLDAS_bias_cold_date = sm.bias(station_cold_date,GLDAS_cold_date)
    				GLDAS_bias_master_cold_date.append(GLDAS_bias_cold_date)

    				naive_RMSD_cold_date = sm.rmsd(station_cold_date,naive_cold_date)
    				naive_RMSD_master_cold_date.append(naive_RMSD_cold_date)
    				CFSR_RMSD_cold_date = sm.rmsd(station_cold_date,CFSR_cold_date)
    				CFSR_RMSD_master_cold_date.append(CFSR_RMSD_cold_date)
    				ERAI_RMSD_cold_date = sm.rmsd(station_cold_date,ERAI_cold_date)
    				ERAI_RMSD_master_cold_date.append(ERAI_RMSD_cold_date)
    				ERA5_RMSD_cold_date = sm.rmsd(station_cold_date,ERA5_cold_date)
    				ERA5_RMSD_master_cold_date.append(ERA5_RMSD_cold_date)
    				JRA_RMSD_cold_date = sm.rmsd(station_cold_date,JRA_cold_date)
    				JRA_RMSD_master_cold_date.append(JRA_RMSD_cold_date)
    				MERRA2_RMSD_cold_date = sm.rmsd(station_cold_date,MERRA2_cold_date)
    				MERRA2_RMSD_master_cold_date.append(MERRA2_RMSD_cold_date)
    				GLDAS_RMSD_cold_date = sm.rmsd(station_cold_date,GLDAS_cold_date)
    				GLDAS_RMSD_master_cold_date.append(GLDAS_RMSD_cold_date)

    				naive_CRMSD_cold_date = sm.centered_rms_dev(station_cold_date,naive_cold_date)
    				naive_CRMSD_master_cold_date.append(naive_CRMSD_cold_date)
    				CFSR_CRMSD_cold_date = sm.centered_rms_dev(station_cold_date,CFSR_cold_date)
    				CFSR_CRMSD_master_cold_date.append(CFSR_CRMSD_cold_date)
    				ERAI_CRMSD_cold_date = sm.centered_rms_dev(station_cold_date,ERAI_cold_date)
    				ERAI_CRMSD_master_cold_date.append(ERAI_CRMSD_cold_date)
    				ERA5_CRMSD_cold_date = sm.centered_rms_dev(station_cold_date,ERA5_cold_date)
    				ERA5_CRMSD_master_cold_date.append(ERA5_CRMSD_cold_date)
    				JRA_CRMSD_cold_date = sm.centered_rms_dev(station_cold_date,JRA_cold_date)
    				JRA_CRMSD_master_cold_date.append(JRA_CRMSD_cold_date)
    				MERRA2_CRMSD_cold_date = sm.centered_rms_dev(station_cold_date,MERRA2_cold_date)
    				MERRA2_CRMSD_master_cold_date.append(MERRA2_CRMSD_cold_date)
    				GLDAS_CRMSD_cold_date = sm.centered_rms_dev(station_cold_date,GLDAS_cold_date)
    				GLDAS_CRMSD_master_cold_date.append(GLDAS_CRMSD_cold_date)

    				station_SDV_cold_date = np.std(station_cold_date)
    				station_SDV_master_cold_date.append(station_SDV_cold_date)
    				naive_SDV_cold_date = np.std(naive_cold_date)
    				naive_SDV_master_cold_date.append(naive_SDV_cold_date)
    				CFSR_SDV_cold_date = np.std(CFSR_cold_date)
    				CFSR_SDV_master_cold_date.append(CFSR_SDV_cold_date)
    				ERAI_SDV_cold_date = np.std(ERAI_cold_date)
    				ERAI_SDV_master_cold_date.append(ERAI_SDV_cold_date)
    				ERA5_SDV_cold_date = np.std(ERA5_cold_date)
    				ERA5_SDV_master_cold_date.append(ERA5_SDV_cold_date)
    				JRA_SDV_cold_date = np.std(JRA_cold_date)
    				JRA_SDV_master_cold_date.append(JRA_SDV_cold_date)
    				MERRA2_SDV_cold_date = np.std(MERRA2_cold_date)
    				MERRA2_SDV_master_cold_date.append(MERRA2_SDV_cold_date)
    				GLDAS_SDV_cold_date = np.std(GLDAS_cold_date)
    				GLDAS_SDV_master_cold_date.append(GLDAS_SDV_cold_date)
				
    				naive_NDSS_cold_date = sm.skill_score_murphy(station_cold_date,naive_cold_date)
    				naive_NDSS_master_cold_date.append(naive_NDSS_cold_date)
    				CFSR_NDSS_cold_date = sm.skill_score_murphy(station_cold_date,CFSR_cold_date)
    				CFSR_NDSS_master_cold_date.append(CFSR_NDSS_cold_date)
    				ERAI_NDSS_cold_date = sm.skill_score_murphy(station_cold_date,ERAI_cold_date)
    				ERAI_NDSS_master_cold_date.append(ERAI_NDSS_cold_date)
    				ERA5_NDSS_cold_date = sm.skill_score_murphy(station_cold_date,ERA5_cold_date)
    				ERA5_NDSS_master_cold_date.append(ERA5_NDSS_cold_date)
    				JRA_NDSS_cold_date = sm.skill_score_murphy(station_cold_date,JRA_cold_date)
    				JRA_NDSS_master_cold_date.append(JRA_NDSS_cold_date)
    				MERRA2_NDSS_cold_date = sm.skill_score_murphy(station_cold_date,MERRA2_cold_date)
    				MERRA2_NDSS_master_cold_date.append(MERRA2_NDSS_cold_date)
    				GLDAS_NDSS_cold_date = sm.skill_score_murphy(station_cold_date,GLDAS_cold_date)
    				GLDAS_NDSS_master_cold_date.append(GLDAS_NDSS_cold_date)

    				naive_corr_cold_date, _ = scipy.stats.pearsonr(station_cold_date,naive_cold_date)
    				naive_corr_master_cold_date.append(naive_corr_cold_date)
    				CFSR_corr_cold_date, _ = scipy.stats.pearsonr(station_cold_date,CFSR_cold_date)
    				CFSR_corr_master_cold_date.append(CFSR_corr_cold_date)
    				ERAI_corr_cold_date, _ = scipy.stats.pearsonr(station_cold_date,ERAI_cold_date)
    				ERAI_corr_master_cold_date.append(ERAI_corr_cold_date)
    				ERA5_corr_cold_date, _ = scipy.stats.pearsonr(station_cold_date,ERA5_cold_date)
    				ERA5_corr_master_cold_date.append(ERA5_corr_cold_date)
    				JRA_corr_cold_date, _ = scipy.stats.pearsonr(station_cold_date,JRA_cold_date)
    				JRA_corr_master_cold_date.append(JRA_corr_cold_date)
    				MERRA2_corr_cold_date, _ = scipy.stats.pearsonr(station_cold_date,MERRA2_cold_date)
    				MERRA2_corr_master_cold_date.append(MERRA2_corr_cold_date)
    				GLDAS_corr_cold_date, _ = scipy.stats.pearsonr(station_cold_date,GLDAS_cold_date)
    				GLDAS_corr_master_cold_date.append(GLDAS_corr_cold_date)


### Warm Season (Date) ###

    				naive_bias_warm_date = sm.bias(station_warm_date,naive_warm_date)
    				naive_bias_master_warm_date.append(naive_bias_warm_date)
    				CFSR_bias_warm_date = sm.bias(station_warm_date,CFSR_warm_date)
    				CFSR_bias_master_warm_date.append(CFSR_bias_warm_date)
    				ERAI_bias_warm_date = sm.bias(station_warm_date,ERAI_warm_date)
    				ERAI_bias_master_warm_date.append(ERAI_bias_warm_date)
    				ERA5_bias_warm_date = sm.bias(station_warm_date,ERA5_warm_date)
    				ERA5_bias_master_warm_date.append(ERA5_bias_warm_date)
    				JRA_bias_warm_date = sm.bias(station_warm_date,JRA_warm_date)
    				JRA_bias_master_warm_date.append(JRA_bias_warm_date)
    				MERRA2_bias_warm_date = sm.bias(station_warm_date,MERRA2_warm_date)
    				MERRA2_bias_master_warm_date.append(MERRA2_bias_warm_date)
    				GLDAS_bias_warm_date = sm.bias(station_warm_date,GLDAS_warm_date)
    				GLDAS_bias_master_warm_date.append(GLDAS_bias_warm_date)

    				naive_RMSD_warm_date = sm.rmsd(station_warm_date,naive_warm_date)
    				naive_RMSD_master_warm_date.append(naive_RMSD_warm_date)
    				CFSR_RMSD_warm_date = sm.rmsd(station_warm_date,CFSR_warm_date)
    				CFSR_RMSD_master_warm_date.append(CFSR_RMSD_warm_date)
    				ERAI_RMSD_warm_date = sm.rmsd(station_warm_date,ERAI_warm_date)
    				ERAI_RMSD_master_warm_date.append(ERAI_RMSD_warm_date)
    				ERA5_RMSD_warm_date = sm.rmsd(station_warm_date,ERA5_warm_date)
    				ERA5_RMSD_master_warm_date.append(ERA5_RMSD_warm_date)
    				JRA_RMSD_warm_date = sm.rmsd(station_warm_date,JRA_warm_date)
    				JRA_RMSD_master_warm_date.append(JRA_RMSD_warm_date)
    				MERRA2_RMSD_warm_date = sm.rmsd(station_warm_date,MERRA2_warm_date)
    				MERRA2_RMSD_master_warm_date.append(MERRA2_RMSD_warm_date)
    				GLDAS_RMSD_warm_date = sm.rmsd(station_warm_date,GLDAS_warm_date)
    				GLDAS_RMSD_master_warm_date.append(GLDAS_RMSD_warm_date)

    				naive_CRMSD_warm_date = sm.centered_rms_dev(station_warm_date,naive_warm_date)
    				naive_CRMSD_master_warm_date.append(naive_CRMSD_warm_date)
    				CFSR_CRMSD_warm_date = sm.centered_rms_dev(station_warm_date,CFSR_warm_date)
    				CFSR_CRMSD_master_warm_date.append(CFSR_CRMSD_warm_date)
    				ERAI_CRMSD_warm_date = sm.centered_rms_dev(station_warm_date,ERAI_warm_date)
    				ERAI_CRMSD_master_warm_date.append(ERAI_CRMSD_warm_date)
    				ERA5_CRMSD_warm_date = sm.centered_rms_dev(station_warm_date,ERA5_warm_date)
    				ERA5_CRMSD_master_warm_date.append(ERA5_CRMSD_warm_date)
    				JRA_CRMSD_warm_date = sm.centered_rms_dev(station_warm_date,JRA_warm_date)
    				JRA_CRMSD_master_warm_date.append(JRA_CRMSD_warm_date)
    				MERRA2_CRMSD_warm_date = sm.centered_rms_dev(station_warm_date,MERRA2_warm_date)
    				MERRA2_CRMSD_master_warm_date.append(MERRA2_CRMSD_warm_date)
    				GLDAS_CRMSD_warm_date = sm.centered_rms_dev(station_warm_date,GLDAS_warm_date)
    				GLDAS_CRMSD_master_warm_date.append(GLDAS_CRMSD_warm_date)

    				station_SDV_warm_date = np.std(station_warm_date)
    				station_SDV_master_warm_date.append(station_SDV_warm_date)
    				naive_SDV_warm_date = np.std(naive_warm_date)
    				naive_SDV_master_warm_date.append(naive_SDV_warm_date)
    				CFSR_SDV_warm_date = np.std(CFSR_warm_date)
    				CFSR_SDV_master_warm_date.append(CFSR_SDV_warm_date)
    				ERAI_SDV_warm_date = np.std(ERAI_warm_date)
    				ERAI_SDV_master_warm_date.append(ERAI_SDV_warm_date)
    				ERA5_SDV_warm_date = np.std(ERA5_warm_date)
    				ERA5_SDV_master_warm_date.append(ERA5_SDV_warm_date)
    				JRA_SDV_warm_date = np.std(JRA_warm_date)
    				JRA_SDV_master_warm_date.append(JRA_SDV_warm_date)
    				MERRA2_SDV_warm_date = np.std(MERRA2_warm_date)
    				MERRA2_SDV_master_warm_date.append(MERRA2_SDV_warm_date)
    				GLDAS_SDV_warm_date = np.std(GLDAS_warm_date)
    				GLDAS_SDV_master_warm_date.append(GLDAS_SDV_warm_date)
				
    				naive_NDSS_warm_date = sm.skill_score_murphy(station_warm_date,naive_warm_date)
    				naive_NDSS_master_warm_date.append(naive_NDSS_warm_date)
    				CFSR_NDSS_warm_date = sm.skill_score_murphy(station_warm_date,CFSR_warm_date)
    				CFSR_NDSS_master_warm_date.append(CFSR_NDSS_warm_date)
    				ERAI_NDSS_warm_date = sm.skill_score_murphy(station_warm_date,ERAI_warm_date)
    				ERAI_NDSS_master_warm_date.append(ERAI_NDSS_warm_date)
    				ERA5_NDSS_warm_date = sm.skill_score_murphy(station_warm_date,ERA5_warm_date)
    				ERA5_NDSS_master_warm_date.append(ERA5_NDSS_warm_date)
    				JRA_NDSS_warm_date = sm.skill_score_murphy(station_warm_date,JRA_warm_date)
    				JRA_NDSS_master_warm_date.append(JRA_NDSS_warm_date)
    				MERRA2_NDSS_warm_date = sm.skill_score_murphy(station_warm_date,MERRA2_warm_date)
    				MERRA2_NDSS_master_warm_date.append(MERRA2_NDSS_warm_date)
    				GLDAS_NDSS_warm_date = sm.skill_score_murphy(station_warm_date,GLDAS_warm_date)
    				GLDAS_NDSS_master_warm_date.append(GLDAS_NDSS_warm_date)

    				naive_corr_warm_date, _ = scipy.stats.pearsonr(station_warm_date,naive_warm_date)
    				naive_corr_master_warm_date.append(naive_corr_warm_date)
    				CFSR_corr_warm_date, _ = scipy.stats.pearsonr(station_warm_date,CFSR_warm_date)
    				CFSR_corr_master_warm_date.append(CFSR_corr_warm_date)
    				ERAI_corr_warm_date, _ = scipy.stats.pearsonr(station_warm_date,ERAI_warm_date)
    				ERAI_corr_master_warm_date.append(ERAI_corr_warm_date)
    				ERA5_corr_warm_date, _ = scipy.stats.pearsonr(station_warm_date,ERA5_warm_date)
    				ERA5_corr_master_warm_date.append(ERA5_corr_warm_date)
    				JRA_corr_warm_date, _ = scipy.stats.pearsonr(station_warm_date,JRA_warm_date)
    				JRA_corr_master_warm_date.append(JRA_corr_warm_date)
    				MERRA2_corr_warm_date, _ = scipy.stats.pearsonr(station_warm_date,MERRA2_warm_date)
    				MERRA2_corr_master_warm_date.append(MERRA2_corr_warm_date)
    				GLDAS_corr_warm_date, _ = scipy.stats.pearsonr(station_warm_date,GLDAS_warm_date)
    				GLDAS_corr_master_warm_date.append(GLDAS_corr_warm_date)

######### Create Summary Dataframes ##########

### Cold Season Temp ###

    			cold_season_dataframe_temp = pd.DataFrame(data=gcell_master, columns=['Grid Cell'])
    			cold_season_dataframe_temp['Lat'] = lat_master
    			cold_season_dataframe_temp['Lon'] = lon_master
    			cold_season_dataframe_temp['N'] = sample_size_master_cold_temp
    			cold_season_dataframe_temp['Naive Blend Bias'] = naive_bias_master_cold_temp
    			cold_season_dataframe_temp['CFSR Bias'] = CFSR_bias_master_cold_temp
    			cold_season_dataframe_temp['ERA-Interim Bias'] = ERAI_bias_master_cold_temp
    			cold_season_dataframe_temp['ERA5 Bias'] = ERA5_bias_master_cold_temp
    			cold_season_dataframe_temp['JRA55 Bias'] = JRA_bias_master_cold_temp
    			cold_season_dataframe_temp['MERRA2 Bias'] = MERRA2_bias_master_cold_temp
    			cold_season_dataframe_temp['GLDAS Bias'] = GLDAS_bias_master_cold_temp
    			cold_season_dataframe_temp['Naive Blend RMSD'] = naive_RMSD_master_cold_temp
    			cold_season_dataframe_temp['CFSR RMSD'] = CFSR_RMSD_master_cold_temp
    			cold_season_dataframe_temp['ERA-Interim RMSD'] = ERAI_RMSD_master_cold_temp
    			cold_season_dataframe_temp['ERA5 RMSD'] = ERA5_RMSD_master_cold_temp
    			cold_season_dataframe_temp['JRA55 RMSD'] = JRA_RMSD_master_cold_temp
    			cold_season_dataframe_temp['MERRA2 RMSD'] = MERRA2_RMSD_master_cold_temp
    			cold_season_dataframe_temp['GLDAS RMSD'] = GLDAS_RMSD_master_cold_temp
    			cold_season_dataframe_temp['Naive Blend CRMSD'] = naive_CRMSD_master_cold_temp
    			cold_season_dataframe_temp['CFSR CRMSD'] = CFSR_CRMSD_master_cold_temp
    			cold_season_dataframe_temp['ERA-Interim CRMSD'] = ERAI_CRMSD_master_cold_temp
    			cold_season_dataframe_temp['ERA5 CRMSD'] = ERA5_CRMSD_master_cold_temp
    			cold_season_dataframe_temp['JRA55 CRMSD'] = JRA_CRMSD_master_cold_temp
    			cold_season_dataframe_temp['MERRA2 CRMSD'] = MERRA2_CRMSD_master_cold_temp
    			cold_season_dataframe_temp['GLDAS CRMSD'] = GLDAS_CRMSD_master_cold_temp
    			cold_season_dataframe_temp['Station SDV'] = station_SDV_master_cold_temp
    			cold_season_dataframe_temp['Naive Blend SDV'] = naive_SDV_master_cold_temp
    			cold_season_dataframe_temp['CFSR SDV'] = CFSR_SDV_master_cold_temp
    			cold_season_dataframe_temp['ERA-Interim SDV'] = ERAI_SDV_master_cold_temp
    			cold_season_dataframe_temp['ERA5 SDV'] = ERA5_SDV_master_cold_temp
    			cold_season_dataframe_temp['JRA55 SDV'] = JRA_SDV_master_cold_temp
    			cold_season_dataframe_temp['MERRA2 SDV'] = MERRA2_SDV_master_cold_temp
    			cold_season_dataframe_temp['GLDAS SDV'] = GLDAS_SDV_master_cold_temp
    			cold_season_dataframe_temp['Naive Blend NDSS'] = naive_NDSS_master_cold_temp
    			cold_season_dataframe_temp['CFSR NDSS'] = CFSR_NDSS_master_cold_temp
    			cold_season_dataframe_temp['ERA-Interim NDSS'] = ERAI_NDSS_master_cold_temp
    			cold_season_dataframe_temp['ERA5 NDSS'] = ERA5_NDSS_master_cold_temp
    			cold_season_dataframe_temp['JRA55 NDSS'] = JRA_NDSS_master_cold_temp
    			cold_season_dataframe_temp['MERRA2 NDSS'] = MERRA2_NDSS_master_cold_temp
    			cold_season_dataframe_temp['GLDAS NDSS'] = GLDAS_NDSS_master_cold_temp
    			cold_season_dataframe_temp['Naive Blend corr'] = naive_corr_master_cold_temp
    			cold_season_dataframe_temp['CFSR corr'] = CFSR_corr_master_cold_temp
    			cold_season_dataframe_temp['ERA-Interim corr'] = ERAI_corr_master_cold_temp
    			cold_season_dataframe_temp['ERA5 corr'] = ERA5_corr_master_cold_temp
    			cold_season_dataframe_temp['JRA55 corr'] = JRA_corr_master_cold_temp
    			cold_season_dataframe_temp['MERRA2 corr'] = MERRA2_corr_master_cold_temp
    			cold_season_dataframe_temp['GLDAS corr'] = GLDAS_corr_master_cold_temp

    			print(cold_season_dataframe_temp)

    			cold_season_temp_metrics_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/by_temp/'+str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_top_30cm_thr_'+str(thr_l)+'_cold_season_temp_summary.csv'])
    			print(cold_season_temp_metrics_fil)
    			cold_season_dataframe_temp.to_csv(cold_season_temp_metrics_fil,index=False)

### Warm Season Temp ###

    			warm_season_dataframe_temp = pd.DataFrame(data=gcell_master, columns=['Grid Cell'])
    			warm_season_dataframe_temp['Lat'] = lat_master
    			warm_season_dataframe_temp['Lon'] = lon_master
    			warm_season_dataframe_temp['N'] = sample_size_master_warm_temp
    			warm_season_dataframe_temp['Naive Blend Bias'] = naive_bias_master_warm_temp
    			warm_season_dataframe_temp['CFSR Bias'] = CFSR_bias_master_warm_temp
    			warm_season_dataframe_temp['ERA-Interim Bias'] = ERAI_bias_master_warm_temp
    			warm_season_dataframe_temp['ERA5 Bias'] = ERA5_bias_master_warm_temp
    			warm_season_dataframe_temp['JRA55 Bias'] = JRA_bias_master_warm_temp
    			warm_season_dataframe_temp['MERRA2 Bias'] = MERRA2_bias_master_warm_temp
    			warm_season_dataframe_temp['GLDAS Bias'] = GLDAS_bias_master_warm_temp
    			warm_season_dataframe_temp['Naive Blend RMSD'] = naive_RMSD_master_warm_temp
    			warm_season_dataframe_temp['CFSR RMSD'] = CFSR_RMSD_master_warm_temp
    			warm_season_dataframe_temp['ERA-Interim RMSD'] = ERAI_RMSD_master_warm_temp
    			warm_season_dataframe_temp['ERA5 RMSD'] = ERA5_RMSD_master_warm_temp
    			warm_season_dataframe_temp['JRA55 RMSD'] = JRA_RMSD_master_warm_temp
    			warm_season_dataframe_temp['MERRA2 RMSD'] = MERRA2_RMSD_master_warm_temp
    			warm_season_dataframe_temp['GLDAS RMSD'] = GLDAS_RMSD_master_warm_temp
    			warm_season_dataframe_temp['Naive Blend CRMSD'] = naive_CRMSD_master_warm_temp
    			warm_season_dataframe_temp['CFSR CRMSD'] = CFSR_CRMSD_master_warm_temp
    			warm_season_dataframe_temp['ERA-Interim CRMSD'] = ERAI_CRMSD_master_warm_temp
    			warm_season_dataframe_temp['ERA5 CRMSD'] = ERA5_CRMSD_master_warm_temp
    			warm_season_dataframe_temp['JRA55 CRMSD'] = JRA_CRMSD_master_warm_temp
    			warm_season_dataframe_temp['MERRA2 CRMSD'] = MERRA2_CRMSD_master_warm_temp
    			warm_season_dataframe_temp['GLDAS CRMSD'] = GLDAS_CRMSD_master_warm_temp
    			warm_season_dataframe_temp['Station SDV'] = station_SDV_master_warm_temp
    			warm_season_dataframe_temp['Naive Blend SDV'] = naive_SDV_master_warm_temp
    			warm_season_dataframe_temp['CFSR SDV'] = CFSR_SDV_master_warm_temp
    			warm_season_dataframe_temp['ERA-Interim SDV'] = ERAI_SDV_master_warm_temp
    			warm_season_dataframe_temp['ERA5 SDV'] = ERA5_SDV_master_warm_temp
    			warm_season_dataframe_temp['JRA55 SDV'] = JRA_SDV_master_warm_temp
    			warm_season_dataframe_temp['MERRA2 SDV'] = MERRA2_SDV_master_warm_temp
    			warm_season_dataframe_temp['GLDAS SDV'] = GLDAS_SDV_master_warm_temp
    			warm_season_dataframe_temp['Naive Blend NDSS'] = naive_NDSS_master_warm_temp
    			warm_season_dataframe_temp['CFSR NDSS'] = CFSR_NDSS_master_warm_temp
    			warm_season_dataframe_temp['ERA-Interim NDSS'] = ERAI_NDSS_master_warm_temp
    			warm_season_dataframe_temp['ERA5 NDSS'] = ERA5_NDSS_master_warm_temp
    			warm_season_dataframe_temp['JRA55 NDSS'] = JRA_NDSS_master_warm_temp
    			warm_season_dataframe_temp['MERRA2 NDSS'] = MERRA2_NDSS_master_warm_temp
    			warm_season_dataframe_temp['GLDAS NDSS'] = GLDAS_NDSS_master_warm_temp
    			warm_season_dataframe_temp['Naive Blend corr'] = naive_corr_master_warm_temp
    			warm_season_dataframe_temp['CFSR corr'] = CFSR_corr_master_warm_temp
    			warm_season_dataframe_temp['ERA-Interim corr'] = ERAI_corr_master_warm_temp
    			warm_season_dataframe_temp['ERA5 corr'] = ERA5_corr_master_warm_temp
    			warm_season_dataframe_temp['JRA55 corr'] = JRA_corr_master_warm_temp
    			warm_season_dataframe_temp['MERRA2 corr'] = MERRA2_corr_master_warm_temp
    			warm_season_dataframe_temp['GLDAS corr'] = GLDAS_corr_master_warm_temp

    			print(warm_season_dataframe_temp)

    			warm_season_temp_metrics_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/by_temp/'+str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_top_30cm_thr_'+str(thr_l)+'_warm_season_temp_summary.csv'])
    			print(warm_season_temp_metrics_fil)
    			warm_season_dataframe_temp.to_csv(warm_season_temp_metrics_fil,index=False)

### Cold Season Date ###

    			cold_season_dataframe_date = pd.DataFrame(data=gcell_master, columns=['Grid Cell'])
    			cold_season_dataframe_date['Lat'] = lat_master
    			cold_season_dataframe_date['Lon'] = lon_master
    			cold_season_dataframe_date['N'] = sample_size_master_cold_date
    			cold_season_dataframe_date['Naive Blend Bias'] = naive_bias_master_cold_date
    			cold_season_dataframe_date['CFSR Bias'] = CFSR_bias_master_cold_date
    			cold_season_dataframe_date['ERA-Interim Bias'] = ERAI_bias_master_cold_date
    			cold_season_dataframe_date['ERA5 Bias'] = ERA5_bias_master_cold_date
    			cold_season_dataframe_date['JRA55 Bias'] = JRA_bias_master_cold_date
    			cold_season_dataframe_date['MERRA2 Bias'] = MERRA2_bias_master_cold_date
    			cold_season_dataframe_date['GLDAS Bias'] = GLDAS_bias_master_cold_date
    			cold_season_dataframe_date['Naive Blend RMSD'] = naive_RMSD_master_cold_date
    			cold_season_dataframe_date['CFSR RMSD'] = CFSR_RMSD_master_cold_date
    			cold_season_dataframe_date['ERA-Interim RMSD'] = ERAI_RMSD_master_cold_date
    			cold_season_dataframe_date['ERA5 RMSD'] = ERA5_RMSD_master_cold_date
    			cold_season_dataframe_date['JRA55 RMSD'] = JRA_RMSD_master_cold_date
    			cold_season_dataframe_date['MERRA2 RMSD'] = MERRA2_RMSD_master_cold_date
    			cold_season_dataframe_date['GLDAS RMSD'] = GLDAS_RMSD_master_cold_date
    			cold_season_dataframe_date['Naive Blend CRMSD'] = naive_CRMSD_master_cold_date
    			cold_season_dataframe_date['CFSR CRMSD'] = CFSR_CRMSD_master_cold_date
    			cold_season_dataframe_date['ERA-Interim CRMSD'] = ERAI_CRMSD_master_cold_date
    			cold_season_dataframe_date['ERA5 CRMSD'] = ERA5_CRMSD_master_cold_date
    			cold_season_dataframe_date['JRA55 CRMSD'] = JRA_CRMSD_master_cold_date
    			cold_season_dataframe_date['MERRA2 CRMSD'] = MERRA2_CRMSD_master_cold_date
    			cold_season_dataframe_date['GLDAS CRMSD'] = GLDAS_CRMSD_master_cold_date
    			cold_season_dataframe_date['Station SDV'] = station_SDV_master_cold_date
    			cold_season_dataframe_date['Naive Blend SDV'] = naive_SDV_master_cold_date
    			cold_season_dataframe_date['CFSR SDV'] = CFSR_SDV_master_cold_date
    			cold_season_dataframe_date['ERA-Interim SDV'] = ERAI_SDV_master_cold_date
    			cold_season_dataframe_date['ERA5 SDV'] = ERA5_SDV_master_cold_date
    			cold_season_dataframe_date['JRA55 SDV'] = JRA_SDV_master_cold_date
    			cold_season_dataframe_date['MERRA2 SDV'] = MERRA2_SDV_master_cold_date
    			cold_season_dataframe_date['GLDAS SDV'] = GLDAS_SDV_master_cold_date
    			cold_season_dataframe_date['Naive Blend NDSS'] = naive_NDSS_master_cold_date
    			cold_season_dataframe_date['CFSR NDSS'] = CFSR_NDSS_master_cold_date
    			cold_season_dataframe_date['ERA-Interim NDSS'] = ERAI_NDSS_master_cold_date
    			cold_season_dataframe_date['ERA5 NDSS'] = ERA5_NDSS_master_cold_date
    			cold_season_dataframe_date['JRA55 NDSS'] = JRA_NDSS_master_cold_date
    			cold_season_dataframe_date['MERRA2 NDSS'] = MERRA2_NDSS_master_cold_date
    			cold_season_dataframe_date['GLDAS NDSS'] = GLDAS_NDSS_master_cold_date
    			cold_season_dataframe_date['Naive Blend corr'] = naive_corr_master_cold_date
    			cold_season_dataframe_date['CFSR corr'] = CFSR_corr_master_cold_date
    			cold_season_dataframe_date['ERA-Interim corr'] = ERAI_corr_master_cold_date
    			cold_season_dataframe_date['ERA5 corr'] = ERA5_corr_master_cold_date
    			cold_season_dataframe_date['JRA55 corr'] = JRA_corr_master_cold_date
    			cold_season_dataframe_date['MERRA2 corr'] = MERRA2_corr_master_cold_date
    			cold_season_dataframe_date['GLDAS corr'] = GLDAS_corr_master_cold_date


    			print(cold_season_dataframe_date)

    			cold_season_date_metrics_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/by_date/'+str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_top_30cm_thr_'+str(thr_l)+'_cold_season_date_summary.csv'])
    			print(cold_season_date_metrics_fil)
    			cold_season_dataframe_temp.to_csv(cold_season_date_metrics_fil,index=False)

### Warm Season Date ###

    			warm_season_dataframe_date = pd.DataFrame(data=gcell_master, columns=['Grid Cell'])
    			warm_season_dataframe_date['Lat'] = lat_master
    			warm_season_dataframe_date['Lon'] = lon_master
    			warm_season_dataframe_temp['N'] = sample_size_master_warm_date
    			warm_season_dataframe_date['Naive Blend Bias'] = naive_bias_master_warm_date
    			warm_season_dataframe_date['CFSR Bias'] = CFSR_bias_master_warm_date
    			warm_season_dataframe_date['ERA-Interim Bias'] = ERAI_bias_master_warm_date
    			warm_season_dataframe_date['ERA5 Bias'] = ERA5_bias_master_warm_date
    			warm_season_dataframe_date['JRA55 Bias'] = JRA_bias_master_warm_date
    			warm_season_dataframe_date['MERRA2 Bias'] = MERRA2_bias_master_warm_date
    			warm_season_dataframe_date['GLDAS Bias'] = GLDAS_bias_master_warm_date
    			warm_season_dataframe_date['Naive Blend RMSD'] = naive_RMSD_master_warm_date
    			warm_season_dataframe_date['CFSR RMSD'] = CFSR_RMSD_master_warm_date
    			warm_season_dataframe_date['ERA-Interim RMSD'] = ERAI_RMSD_master_warm_date
    			warm_season_dataframe_date['ERA5 RMSD'] = ERA5_RMSD_master_warm_date
    			warm_season_dataframe_date['JRA55 RMSD'] = JRA_RMSD_master_warm_date
    			warm_season_dataframe_date['MERRA2 RMSD'] = MERRA2_RMSD_master_warm_date
    			warm_season_dataframe_date['GLDAS RMSD'] = GLDAS_RMSD_master_warm_date
    			warm_season_dataframe_date['Naive Blend CRMSD'] = naive_CRMSD_master_warm_date
    			warm_season_dataframe_date['CFSR CRMSD'] = CFSR_CRMSD_master_warm_date
    			warm_season_dataframe_date['ERA-Interim CRMSD'] = ERAI_CRMSD_master_warm_date
    			warm_season_dataframe_date['ERA5 CRMSD'] = ERA5_CRMSD_master_warm_date
    			warm_season_dataframe_date['JRA55 CRMSD'] = JRA_CRMSD_master_warm_date
    			warm_season_dataframe_date['MERRA2 CRMSD'] = MERRA2_CRMSD_master_warm_date
    			warm_season_dataframe_date['GLDAS CRMSD'] = GLDAS_CRMSD_master_warm_date
    			warm_season_dataframe_date['Station SDV'] = station_SDV_master_warm_date
    			warm_season_dataframe_date['Naive Blend SDV'] = naive_SDV_master_warm_date
    			warm_season_dataframe_date['CFSR SDV'] = CFSR_SDV_master_warm_date
    			warm_season_dataframe_date['ERA-Interim SDV'] = ERAI_SDV_master_warm_date
    			warm_season_dataframe_date['ERA5 SDV'] = ERA5_SDV_master_warm_date
    			warm_season_dataframe_date['JRA55 SDV'] = JRA_SDV_master_warm_date
    			warm_season_dataframe_date['MERRA2 SDV'] = MERRA2_SDV_master_warm_date
    			warm_season_dataframe_date['GLDAS SDV'] = GLDAS_SDV_master_warm_date
    			warm_season_dataframe_date['Naive Blend NDSS'] = naive_NDSS_master_warm_date
    			warm_season_dataframe_date['CFSR NDSS'] = CFSR_NDSS_master_warm_date
    			warm_season_dataframe_date['ERA-Interim NDSS'] = ERAI_NDSS_master_warm_date
    			warm_season_dataframe_date['ERA5 NDSS'] = ERA5_NDSS_master_warm_date
    			warm_season_dataframe_date['JRA55 NDSS'] = JRA_NDSS_master_warm_date
    			warm_season_dataframe_date['MERRA2 NDSS'] = MERRA2_NDSS_master_warm_date
    			warm_season_dataframe_date['GLDAS NDSS'] = GLDAS_NDSS_master_warm_date
    			warm_season_dataframe_date['Naive Blend corr'] = naive_corr_master_warm_date
    			warm_season_dataframe_date['CFSR corr'] = CFSR_corr_master_warm_date
    			warm_season_dataframe_date['ERA-Interim corr'] = ERAI_corr_master_warm_date
    			warm_season_dataframe_date['ERA5 corr'] = ERA5_corr_master_warm_date
    			warm_season_dataframe_date['JRA55 corr'] = JRA_corr_master_warm_date
    			warm_season_dataframe_date['MERRA2 corr'] = MERRA2_corr_master_warm_date
    			warm_season_dataframe_date['GLDAS corr'] = GLDAS_corr_master_warm_date

    			print(warm_season_dataframe_date)

    			warm_season_date_metrics_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/by_date/'+str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_top_30cm_thr_'+str(thr_l)+'_warm_season_date_summary.csv'])
    			print(warm_season_date_metrics_fil)
    			warm_season_dataframe_temp.to_csv(warm_season_date_metrics_fil,index=False)



