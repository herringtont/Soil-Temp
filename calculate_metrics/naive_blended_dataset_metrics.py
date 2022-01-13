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
rmp_type = ['nn','bil','con']
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

    CFSR_fi_anom = "".join([rnys_dir,"CFSR_anom.nc"])
    MERRA2_fi_anom = "".join([rnys_dir,"MERRA2_anom.nc"])
    ERA5_fi_anom = "".join([rnys_dir,"ERA5_anom.nc"])
    ERAI_fi_anom = "".join([rnys_dir,"ERA-Interim_anom.nc"])
    JRA_fi_anom = "".join([rnys_dir,"JRA55_anom.nc"])
    GLDAS_fi_anom = "".join([rnys_dir,"GLDAS_anom.nc"])
        
    for j in naive_type:
    	naive_type_j = j
    	naive_dir_raw = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_product/'+str(naive_type_j)+'/raw_temp/'+str(remap_type)+'/'])
    	naive_dir_anom = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_product/'+str(naive_type_j)+'/anom/'+str(remap_type)+'/'])
    	naive_fi = ''.join([naive_dir_raw+str(remap_type)+'_'+'Naive_stemp_TOP30cm.nc'])
    	naive_fi_anom = ''.join([naive_dir_anom+str(remap_type)+'_'+'Naive_anom_TOP30cm.nc'])



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



    			naive_temp_master_raw = []
    			CFSR_temp_master_raw = []
    			ERAI_temp_master_raw = []
    			ERA5_temp_master_raw = []
    			JRA_temp_master_raw = []
    			MERRA2_temp_master_raw = []
    			GLDAS_temp_master_raw = []

    			naive_bias_master_raw = []
    			CFSR_bias_master_raw = []
    			ERAI_bias_master_raw = []
    			ERA5_bias_master_raw = []
    			JRA_bias_master_raw = []
    			MERRA2_bias_master_raw = []
    			GLDAS_bias_master_raw = []

    			naive_SDV_master_raw = []
    			CFSR_SDV_master_raw = []
    			ERAI_SDV_master_raw = []
    			ERA5_SDV_master_raw = []
    			JRA_SDV_master_raw = []
    			MERRA2_SDV_master_raw = []
    			GLDAS_SDV_master_raw = []

    			naive_rmse_master_raw = []
    			CFSR_rmse_master_raw = []
    			ERAI_rmse_master_raw = []
    			ERA5_rmse_master_raw = []
    			JRA_rmse_master_raw = []
    			MERRA2_rmse_master_raw = []
    			GLDAS_rmse_master_raw = []

    			naive_ubrmse_master_raw = []
    			CFSR_ubrmse_master_raw = []
    			ERAI_ubrmse_master_raw = []
    			ERA5_ubrmse_master_raw = []
    			JRA_ubrmse_master_raw = []
    			MERRA2_ubrmse_master_raw = []
    			GLDAS_ubrmse_master_raw = []

    			naive_corr_master_raw = []
    			CFSR_corr_master_raw = []
    			ERAI_corr_master_raw = []
    			ERA5_corr_master_raw = []
    			JRA_corr_master_raw = []
    			MERRA2_corr_master_raw = []
    			GLDAS_corr_master_raw = []


    			naive_bias_master_anom = []
    			CFSR_bias_master_anom = []
    			ERAI_bias_master_anom = []
    			ERA5_bias_master_anom = []
    			JRA_bias_master_anom = []
    			MERRA2_bias_master_anom = []
    			GLDAS_bias_master_anom = []
    			delta_corr_master_raw = []

    			naive_SDV_master_anom = []
    			CFSR_SDV_master_anom = []
    			ERAI_SDV_master_anom = []
    			ERA5_SDV_master_anom = []
    			JRA_SDV_master_anom = []
    			MERRA2_SDV_master_anom = []
    			GLDAS_SDV_master_anom = []

    			naive_rmse_master_anom = []
    			CFSR_rmse_master_anom = []
    			ERAI_rmse_master_anom = []
    			ERA5_rmse_master_anom = []
    			JRA_rmse_master_anom = []
    			MERRA2_rmse_master_anom = []
    			GLDAS_rmse_master_anom = []

    			naive_ubrmse_master_anom = []
    			CFSR_ubrmse_master_anom = []
    			ERAI_ubrmse_master_anom = []
    			ERA5_ubrmse_master_anom = []
    			JRA_ubrmse_master_anom = []
    			MERRA2_ubrmse_master_anom = []
    			GLDAS_ubrmse_master_anom = []

    			naive_corr_master_anom = []
    			CFSR_corr_master_anom = []
    			ERAI_corr_master_anom = []
    			ERA5_corr_master_anom = []
    			JRA_corr_master_anom = []
    			MERRA2_corr_master_anom = []
    			GLDAS_corr_master_anom = []


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
    				soil_anom = dframe_insitu['Spatial Avg Anom']
    				gcell = dframe_insitu['Grid Cell'].iloc[0]
    				lat_cen = dframe_insitu['Central Lat'].iloc[0]
    				lon_cen = dframe_insitu['Central Lon'].iloc[0]
    
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

    				CFSR_fil_anom = xr.open_dataset(CFSR_fi_anom)
    				ERAI_fil_anom = xr.open_dataset(ERAI_fi_anom)
    				ERA5_fil_anom = xr.open_dataset(ERA5_fi_anom)
    				JRA_fil_anom = xr.open_dataset(JRA_fi_anom)
    				MERRA2_fil_anom = xr.open_dataset(MERRA2_fi_anom)
    				GLDAS_fil_anom = xr.open_dataset(GLDAS_fi_anom)

    				CFSR_anom = CFSR_fil_anom[CFSR_layer]
    				ERAI_anom = ERAI_fil_anom[ERAI_layer]
    				ERA5_anom = ERA5_fil_anom[ERA5_layer]
    				JRA_anom = JRA_fil_anom[JRA_layer]
    				MERRA2_anom = MERRA2_fil_anom[MERRA2_layer]
    				GLDAS_anom = GLDAS_fil_anom[GLDAS_layer]


    				naive_fil = xr.open_dataset(naive_fi)
    				naive_stemp = naive_fil[Naive_layer]

    				naive_fil_anom = xr.open_dataset(naive_fi_anom)
    				naive_anom = naive_fil[Naive_layer]

    				#print(type(CFSR_stemp))

    				CFSR_stemp_gcell = CFSR_stemp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    				ERAI_stemp_gcell = ERAI_stemp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    				ERA5_stemp_gcell = ERA5_stemp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    				JRA_stemp_gcell = JRA_stemp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    				MERRA2_stemp_gcell = MERRA2_stemp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    				GLDAS_stemp_gcell = GLDAS_stemp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    				naive_stemp_gcell = naive_stemp.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)

    				#print(type(CFSR_anom))

    				CFSR_anom_gcell = CFSR_anom.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    				ERAI_anom_gcell = ERAI_anom.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    				ERA5_anom_gcell = ERA5_anom.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    				JRA_anom_gcell = JRA_anom.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    				MERRA2_anom_gcell = MERRA2_anom.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    				GLDAS_anom_gcell = GLDAS_anom.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)
    				naive_anom_gcell = naive_anom.sel(lat=lat_cen,lon=lon_cen,method='nearest',drop = True)

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

#################### create anomalies for reanalysis files #######################
    				rnysis_anom_master = []
    				rnysis_date_master = []
    				rnysis_name_master = []
    				rnysis_stemp_master = []
    				rnysis = [naive_temp_master,CFSR_temp_master,ERAI_temp_master,ERA5_temp_master,JRA_temp_master,MERRA2_temp_master,GLDAS_temp_master]
    				rnysis_name = ['Naive Blend','CFSR','ERA-Interim','ERA-5','JRA-55','MERRA2','GLDAS']
    				dat_rowlist = [datetime.datetime.strptime(x,'%Y-%m-%d') for x in date_temp_master]
    				dat_rowlist2 = date_temp_master
    				num_rows = len(dat_rowlist)

    				for m in range(0,7):
    					rnysisi = rnysis[m]
    					rnysis_namei = rnysis_name[m]
    					#print("Reanalysis Product:",rnysis_namei)
    					#print(rnysisi)
    					climatology = dict()
    					clim_averages = dict()
    					stemp_mstr = []
    					stemp_anom_master = []
    					date_mstr = []
    					name_mstr = []
    					for month in range(1,13):
    						month_key = f"{month:02}"
    						climatology[month_key] = list()

    					for n in range(0,num_rows):
					###add month data to list based on key
    						dat_row = dat_rowlist[n]
    						stemp_row = rnysisi[n]
    						month_key = dat_row.strftime("%m")
    						climatology[month_key].append(stemp_row)

    					climatology_keys = list(climatology.keys())
    					climatology_keys2 = np.array(climatology_keys).flatten()
    						#print(climatology)
					
    					for key in climatology:
					###take averages and write to averages dictionary
    						current_total = 0
    						len_current_list = 0
    						current_list = climatology[key]
    						for temp in current_list:
    							if (temp == np.nan):
    								current_total = current_total + 0
    								len_current_list = len_current_list + 0
    							else:
    								current_total = current_total + temp
    								len_current_list = len_current_list + 1
    						if (len_current_list == 0):
    							average = np.nan
    						else:
    							average = current_total/len_current_list
    						clim_averages[key] = average
    						#print(average)
							
    					clim_avg = list(clim_averages.values())
    					#print(clim_averages)
						
    					for o in range (0, num_rows):
    						stemp_rw = rnysisi[o]
    						dat_row = dat_rowlist[o]
    						dat_row_mon = dat_row.month
    						dat_row_mons = f"{dat_row_mon:02}"
    						#print(stemp_rw,dat_row_mon,clim_averages[dat_row_mons])
    						stemp_anom = stemp_rw - clim_averages[dat_row_mons]

    						rnysis_anom_master.append(stemp_anom)
    						rnysis_date_master.append(dat_row)					
    						rnysis_name_master.append(rnysis_namei)
    						rnysis_stemp_master.append(stemp_rw)

    					#print(rnysis_stemp_master)

    				naive_no_nan = naive_temp_master[~np.isnan(naive_temp_master)]

    				#print(naive_no_nan,TC_no_nan)

    				CFSR_no_nan = CFSR_temp_master[~np.isnan(CFSR_temp_master)]
    				#print(CFSR_no_nan)

    				if(DateTime[0]>CFSR_edate_dt or DateTime[len(DateTime) -1] < CFSR_sdate_dt): #skip if the CFSR dates and station dates do not overlap
    					continue
    					
    				if(len(naive_no_nan) == 0 or len(CFSR_no_nan) == 0): #skip if there are NaN values in blended data
    					continue

####### Station Collocated Anomalies #####

    				dframe_anom_master = pd.DataFrame(data=rnysis_date_master, columns=['Date'])
    				dframe_anom_master['Name'] = rnysis_name_master
    				dframe_anom_master['Raw Temp'] = rnysis_stemp_master
    				dframe_anom_master['Anom'] = rnysis_anom_master
    				dframe_anom_master.dropna(inplace=True)
    				len_dframe_anom = len(dframe_anom_master)

    				station_anom = station_anom_master
    				naive_anom = dframe_anom_master[dframe_anom_master['Name'] == 'Naive Blend']
    				naive_anom = naive_anom['Anom'].values
    				CFSR_anom = dframe_anom_master[dframe_anom_master['Name'] == 'CFSR']
    				CFSR_anom = CFSR_anom['Anom'].values
    				ERAI_anom = dframe_anom_master[dframe_anom_master['Name'] == 'ERA-Interim']
    				ERAI_anom = ERAI_anom['Anom'].values
    				ERA5_anom = dframe_anom_master[dframe_anom_master['Name'] == 'ERA-5']
    				ERA5_anom = ERA5_anom['Anom'].values
    				JRA_anom = dframe_anom_master[dframe_anom_master['Name'] == 'JRA-55']
    				JRA_anom = JRA_anom['Anom'].values					
    				MERRA2_anom = dframe_anom_master[dframe_anom_master['Name'] == 'MERRA2']
    				MERRA2_anom = MERRA2_anom['Anom'].values
    				GLDAS_anom = dframe_anom_master[dframe_anom_master['Name'] == 'GLDAS']
    				GLDAS_anom = GLDAS_anom['Anom'].values

    				if (len(CFSR_anom) == 0):
    					continue

    				dframe_anom_master2 = pd.DataFrame(data=date_temp_master, columns=['Date'])
    				dframe_anom_master2['Station'] = station_anom_master					
    				dframe_anom_master2['Naive Blend'] = naive_anom
    				dframe_anom_master2['CFSR'] = CFSR_anom
    				dframe_anom_master2['ERA-Interim'] = ERAI_anom
    				dframe_anom_master2['ERA5'] = ERA5_anom
    				dframe_anom_master2['JRA55'] = JRA_anom
    				dframe_anom_master2['MERRA2'] = MERRA2_anom
    				dframe_anom_master2['GLDAS'] = GLDAS_anom
    				dframe_anom_master2.dropna(inplace=True)
    				len_dframe_anom = len(dframe_anom_master2)
    				#print(dframe_anom_master2)
    				if (len_dframe_anom == 0): #skip if length of non-NaN dframe is 0
    					continue

    				dframe_raw_master = pd.DataFrame(data=date_temp_master, columns=['Date'])
    				dframe_raw_master['Station'] = station_temp_master
    				dframe_raw_master['Naive Blend'] = naive_temp_master
    				dframe_raw_master['CFSR'] = CFSR_temp_master
    				dframe_raw_master['ERA-Interim'] = ERAI_temp_master
    				dframe_raw_master['ERA5'] = ERA5_temp_master
    				dframe_raw_master['JRA55'] = JRA_temp_master
    				dframe_raw_master['MERRA2'] = MERRA2_temp_master
    				dframe_raw_master['GLDAS'] = GLDAS_temp_master
    				dframe_raw_master.dropna(inplace=True)
    				len_dframe_raw = len(dframe_raw_master)
    				if (len_dframe_raw == 0): #skip if length of non-NaN dframe is 0
    					continue	

############### Calculate Biases ############
    				station_raw_temp = dframe_raw_master['Station'].values
    				naive_raw_temp = dframe_raw_master['Naive Blend'].values
    				CFSR_raw_temp = dframe_raw_master['CFSR'].values
    				ERAI_raw_temp = dframe_raw_master['ERA-Interim'].values
    				ERA5_raw_temp = dframe_raw_master['ERA5'].values
    				JRA_raw_temp = dframe_raw_master['JRA55'].values
    				MERRA2_raw_temp = dframe_raw_master['MERRA2'].values
    				GLDAS_raw_temp = dframe_raw_master['GLDAS'].values

    				station_anom = dframe_anom_master2['Station'].values
    				naive_anom = dframe_anom_master2['Naive Blend'].values
    				CFSR_anom = dframe_anom_master2['CFSR'].values
    				ERAI_anom = dframe_anom_master2['ERA-Interim'].values
    				ERA5_anom = dframe_anom_master2['ERA5'].values
    				JRA_anom = dframe_anom_master2['JRA55'].values
    				MERRA2_anom = dframe_anom_master2['MERRA2'].values
    				GLDAS_anom = dframe_anom_master2['GLDAS'].values


    				gcell = dframe_insitu['Grid Cell'].iloc[0]
    				gcell_master_stn.append(gcell)
    				lat_cen = dframe_insitu['Central Lat'].iloc[0]
    				lat_master_stn.append(lat_cen)
    				lon_cen = dframe_insitu['Central Lon'].iloc[0]
    				lon_master_stn.append(lon_cen)
    				len_raw_master.append(len_dframe_raw)
    				len_anom_master.append(len_dframe_anom)

###### Raw Temp #####

    				naive_bias_raw = bias(naive_raw_temp, station_raw_temp)
    				naive_bias_master_raw.append(naive_bias_raw)

    				CFSR_bias_raw = bias(CFSR_raw_temp, station_raw_temp)
    				CFSR_bias_master_raw.append(CFSR_bias_raw)

    				ERAI_bias_raw = bias(ERAI_raw_temp, station_raw_temp)
    				ERAI_bias_master_raw.append(ERAI_bias_raw)

    				ERA5_bias_raw = bias(ERA5_raw_temp, station_raw_temp)
    				ERA5_bias_master_raw.append(ERA5_bias_raw)

    				JRA_bias_raw = bias(JRA_raw_temp, station_raw_temp)
    				JRA_bias_master_raw.append(JRA_bias_raw)

    				MERRA2_bias_raw = bias(MERRA2_raw_temp, station_raw_temp)
    				MERRA2_bias_master_raw.append(MERRA2_bias_raw)

    				GLDAS_bias_raw = bias(GLDAS_raw_temp, station_raw_temp)
    				GLDAS_bias_master_raw.append(GLDAS_bias_raw)

###### Anomalies #####

    				naive_bias_anom = bias(naive_anom, station_anom)
    				naive_bias_master_anom.append(naive_bias_anom)

    				CFSR_bias_anom = bias(CFSR_anom, station_anom)
    				CFSR_bias_master_anom.append(CFSR_bias_anom)

    				ERAI_bias_anom = bias(ERAI_anom, station_anom)
    				ERAI_bias_master_anom.append(ERAI_bias_anom)

    				ERA5_bias_anom = bias(ERA5_anom, station_anom)
    				ERA5_bias_master_anom.append(ERA5_bias_anom)

    				JRA_bias_anom = bias(JRA_anom, station_anom)
    				JRA_bias_master_anom.append(JRA_bias_anom)

    				MERRA2_bias_anom = bias(MERRA2_anom, station_anom)
    				MERRA2_bias_master_anom.append(MERRA2_bias_anom)

    				GLDAS_bias_anom = bias(GLDAS_anom, station_anom)
    				GLDAS_bias_master_anom.append(GLDAS_bias_anom)

############### Calculate normalized standard deviations (relative to in-situ) ############

###### Raw Temp #####

    				naive_SDV_raw = SDVnorm(naive_raw_temp, station_raw_temp)
    				naive_SDV_master_raw.append(naive_SDV_raw)

    				CFSR_SDV_raw = SDVnorm(CFSR_raw_temp, station_raw_temp)
    				CFSR_SDV_master_raw.append(CFSR_SDV_raw)

    				ERAI_SDV_raw = SDVnorm(ERAI_raw_temp, station_raw_temp)
    				ERAI_SDV_master_raw.append(ERAI_SDV_raw)

    				ERA5_SDV_raw = SDVnorm(ERA5_raw_temp, station_raw_temp)
    				ERA5_SDV_master_raw.append(ERA5_SDV_raw)

    				JRA_SDV_raw = SDVnorm(JRA_raw_temp, station_raw_temp)
    				JRA_SDV_master_raw.append(JRA_SDV_raw)

    				MERRA2_SDV_raw = SDVnorm(MERRA2_raw_temp, station_raw_temp)
    				MERRA2_SDV_master_raw.append(MERRA2_SDV_raw)

    				GLDAS_SDV_raw = SDVnorm(GLDAS_raw_temp, station_raw_temp)
    				GLDAS_SDV_master_raw.append(GLDAS_SDV_raw)

###### Anomalies #####

    				naive_SDV_anom = SDVnorm(naive_anom, station_anom)
    				naive_SDV_master_anom.append(naive_SDV_anom)

    				CFSR_SDV_anom = SDVnorm(CFSR_anom, station_anom)
    				CFSR_SDV_master_anom.append(CFSR_SDV_anom)

    				ERAI_SDV_anom = SDVnorm(ERAI_anom, station_anom)
    				ERAI_SDV_master_anom.append(ERAI_SDV_anom)

    				ERA5_SDV_anom = SDVnorm(ERA5_anom, station_anom)
    				ERA5_SDV_master_anom.append(ERA5_SDV_anom)

    				JRA_SDV_anom = SDVnorm(JRA_anom, station_anom)
    				JRA_SDV_master_anom.append(JRA_SDV_anom)

    				MERRA2_SDV_anom = SDVnorm(MERRA2_anom, station_anom)
    				MERRA2_SDV_master_anom.append(MERRA2_SDV_anom)

    				GLDAS_SDV_anom = SDVnorm(GLDAS_anom, station_anom)
    				GLDAS_SDV_master_anom.append(GLDAS_SDV_anom)


############## Calculate RMSE and ubRMSE for products ##############

###### Raw Temp #####
    				y_true_raw = station_raw_temp
    				y_naive_raw = naive_raw_temp
    				y_CFSR_raw = CFSR_raw_temp
    				y_ERAI_raw = ERAI_raw_temp
    				y_ERA5_raw = ERA5_raw_temp
    				y_JRA_raw = JRA_raw_temp
    				y_MERRA2_raw = MERRA2_raw_temp
    				y_GLDAS_raw = GLDAS_raw_temp   			
    				#print("Station Data:")
    				#print(DateTime)    					
    				#print(len(y_true_raw))
    				#print("CFSR Data:")
    				#print(CFSR_datetime)
    				#print(len(y_CFSR_raw))

    				naive_rmse_raw = mean_squared_error(y_true_raw, y_naive_raw, squared=False)
    				naive_rmse_master_raw.append(naive_rmse_raw)

    				CFSR_rmse_raw = mean_squared_error(y_true_raw, y_CFSR_raw, squared=False)
    				CFSR_rmse_master_raw.append(CFSR_rmse_raw)

    				ERAI_rmse_raw = mean_squared_error(y_true_raw, y_ERAI_raw, squared=False)
    				ERAI_rmse_master_raw.append(ERAI_rmse_raw)

    				ERA5_rmse_raw = mean_squared_error(y_true_raw, y_ERA5_raw, squared=False)
    				ERA5_rmse_master_raw.append(ERA5_rmse_raw)

    				JRA_rmse_raw = mean_squared_error(y_true_raw, y_JRA_raw, squared=False)
    				JRA_rmse_master_raw.append(JRA_rmse_raw)

    				MERRA2_rmse_raw = mean_squared_error(y_true_raw, y_MERRA2_raw, squared=False)
    				MERRA2_rmse_master_raw.append(MERRA2_rmse_raw)

    				GLDAS_rmse_raw = mean_squared_error(y_true_raw, y_GLDAS_raw, squared=False)    			
    				GLDAS_rmse_master_raw.append(GLDAS_rmse_raw)

    				naive_ubrmse_raw = ubrmsd(y_true_raw, y_naive_raw)
    				naive_ubrmse_master_raw.append(naive_ubrmse_raw)
    			
    				CFSR_ubrmse_raw = ubrmsd(y_true_raw, y_CFSR_raw)
    				CFSR_ubrmse_master_raw.append(CFSR_ubrmse_raw)

    				ERAI_ubrmse_raw = ubrmsd(y_true_raw, y_ERAI_raw)
    				ERAI_ubrmse_master_raw.append(ERAI_ubrmse_raw)

    				ERA5_ubrmse_raw = ubrmsd(y_true_raw, y_ERA5_raw)
    				ERA5_ubrmse_master_raw.append(ERA5_ubrmse_raw)

    				JRA_ubrmse_raw = ubrmsd(y_true_raw, y_JRA_raw)
    				JRA_ubrmse_master_raw.append(JRA_ubrmse_raw)

    				MERRA2_ubrmse_raw = ubrmsd(y_true_raw, y_MERRA2_raw)
    				MERRA2_ubrmse_master_raw.append(MERRA2_ubrmse_raw)

    				GLDAS_ubrmse_raw = ubrmsd(y_true_raw, y_GLDAS_raw)
    				GLDAS_ubrmse_master_raw.append(GLDAS_ubrmse_raw) 


###### Anomalies #####

    				y_true_anom = station_anom
    				y_naive_anom = naive_anom
    				y_CFSR_anom = CFSR_anom
    				y_ERAI_anom = ERAI_anom
    				y_ERA5_anom = ERA5_anom
    				y_JRA_anom = JRA_anom
    				y_MERRA2_anom = MERRA2_anom
    				y_GLDAS_anom = GLDAS_anom   			

    				naive_rmse_anom = mean_squared_error(y_true_anom, y_naive_anom, squared=False)
    				naive_rmse_master_anom.append(naive_rmse_anom)

    				CFSR_rmse_anom = mean_squared_error(y_true_anom, y_CFSR_anom, squared=False)
    				CFSR_rmse_master_anom.append(CFSR_rmse_anom)

    				ERAI_rmse_anom = mean_squared_error(y_true_anom, y_ERAI_anom, squared=False)
    				ERAI_rmse_master_anom.append(ERAI_rmse_anom)

    				ERA5_rmse_anom = mean_squared_error(y_true_anom, y_ERA5_anom, squared=False)
    				ERA5_rmse_master_anom.append(ERA5_rmse_anom)

    				JRA_rmse_anom = mean_squared_error(y_true_anom, y_JRA_anom, squared=False)
    				JRA_rmse_master_anom.append(JRA_rmse_anom)

    				MERRA2_rmse_anom = mean_squared_error(y_true_anom, y_MERRA2_anom, squared=False)
    				MERRA2_rmse_master_anom.append(MERRA2_rmse_anom)

    				GLDAS_rmse_anom = mean_squared_error(y_true_anom, y_GLDAS_anom, squared=False)    			
    				GLDAS_rmse_master_anom.append(GLDAS_rmse_anom)


    				naive_ubrmse_anom = ubrmsd(y_true_anom, y_naive_anom)
    				naive_ubrmse_master_anom.append(naive_ubrmse_anom)
    			
    				CFSR_ubrmse_anom = ubrmsd(y_true_anom, y_CFSR_anom)
    				CFSR_ubrmse_master_anom.append(CFSR_ubrmse_anom)

    				ERAI_ubrmse_anom = ubrmsd(y_true_anom, y_ERAI_anom)
    				ERAI_ubrmse_master_anom.append(ERAI_ubrmse_anom)

    				ERA5_ubrmse_anom = ubrmsd(y_true_anom, y_ERA5_anom)
    				ERA5_ubrmse_master_anom.append(ERA5_ubrmse_anom)

    				JRA_ubrmse_anom = ubrmsd(y_true_anom, y_JRA_anom)
    				JRA_ubrmse_master_anom.append(JRA_ubrmse_anom)

    				MERRA2_ubrmse_anom = ubrmsd(y_true_anom, y_MERRA2_anom)
    				MERRA2_ubrmse_master_anom.append(MERRA2_ubrmse_anom)

    				GLDAS_ubrmse_anom = ubrmsd(y_true_anom, y_GLDAS_anom)
    				GLDAS_ubrmse_master_anom.append(GLDAS_ubrmse_anom)

    				#print(TC_raw_temp)
    				#print(station_raw_temp)


################## Calculate Pearson Correlations ####################

##### Raw Temperatures #####
    				naive_corr_raw, _ = pearsonr(naive_raw_temp, station_raw_temp)
    				naive_corr_master_raw.append(naive_corr_raw)
    				CFSR_corr_raw, _ = pearsonr(CFSR_raw_temp, station_raw_temp)
    				CFSR_corr_master_raw.append(CFSR_corr_raw)
    				ERAI_corr_raw, _ = pearsonr(ERAI_raw_temp, station_raw_temp)
    				ERAI_corr_master_raw.append(ERAI_corr_raw)
    				ERA5_corr_raw, _ = pearsonr(ERA5_raw_temp, station_raw_temp)
    				ERA5_corr_master_raw.append(ERA5_corr_raw)
    				JRA_corr_raw, _ = pearsonr(JRA_raw_temp, station_raw_temp)
    				JRA_corr_master_raw.append(JRA_corr_raw)
    				MERRA2_corr_raw, _ = pearsonr(MERRA2_raw_temp, station_raw_temp)
    				MERRA2_corr_master_raw.append(MERRA2_corr_raw)
    				GLDAS_corr_raw, _ = pearsonr(GLDAS_raw_temp, station_raw_temp)
    				GLDAS_corr_master_raw.append(GLDAS_corr_raw)

##### Anomalies #####

    				naive_corr_anom, _ = pearsonr(naive_anom, station_anom)
    				naive_corr_master_anom.append(naive_corr_anom)
    				CFSR_corr_anom, _ = pearsonr(CFSR_anom, station_anom)
    				CFSR_corr_master_anom.append(CFSR_corr_anom)
    				ERAI_corr_anom, _ = pearsonr(ERAI_anom, station_anom)
    				ERAI_corr_master_anom.append(ERAI_corr_anom)
    				ERA5_corr_anom, _ = pearsonr(ERA5_anom, station_anom)
    				ERA5_corr_master_anom.append(ERA5_corr_anom)
    				JRA_corr_anom, _ = pearsonr(JRA_anom, station_anom)
    				JRA_corr_master_anom.append(JRA_corr_anom)
    				MERRA2_corr_anom, _ = pearsonr(MERRA2_anom, station_anom)
    				MERRA2_corr_master_anom.append(MERRA2_corr_anom)
    				GLDAS_corr_anom, _ = pearsonr(GLDAS_anom, station_anom)
    				GLDAS_corr_master_anom.append(GLDAS_corr_anom)  					
										    					


################## Create Summary Statistics Dataframes ##############

    			df_summary_raw = pd.DataFrame(data=gcell_master_stn, columns=['Grid Cell'])
    			df_summary_raw['Central Lat'] = lat_master_stn
    			df_summary_raw['Central Lon'] = lon_master_stn
    			df_summary_raw['N'] = len_raw_master
    			df_summary_raw['Naive Blend Bias'] = naive_bias_master_raw
    			df_summary_raw['CFSR Bias'] = CFSR_bias_master_raw
    			df_summary_raw['ERA-Interim Bias'] = ERAI_bias_master_raw
    			df_summary_raw['ERA5 Bias'] = ERA5_bias_master_raw
    			df_summary_raw['JRA-55 Bias'] = JRA_bias_master_raw
    			df_summary_raw['MERRA2 Bias'] = MERRA2_bias_master_raw
    			df_summary_raw['GLDAS Bias'] = GLDAS_bias_master_raw

    			df_summary_raw['Naive Blend SDV'] = naive_SDV_master_raw
    			df_summary_raw['CFSR SDV'] = CFSR_SDV_master_raw
    			df_summary_raw['ERA-Interim SDV'] = ERAI_SDV_master_raw
    			df_summary_raw['ERA5 SDV'] = ERA5_SDV_master_raw
    			df_summary_raw['JRA-55 SDV'] = JRA_SDV_master_raw
    			df_summary_raw['MERRA2 SDV'] = MERRA2_SDV_master_raw
    			df_summary_raw['GLDAS SDV'] = GLDAS_SDV_master_raw

    			df_summary_raw['Naive Blend RMSE'] = naive_rmse_master_raw
    			df_summary_raw['CFSR RMSE'] = CFSR_rmse_master_raw
    			df_summary_raw['ERA-Interim RMSE'] = ERAI_rmse_master_raw
    			df_summary_raw['ERA5 RMSE'] = ERA5_rmse_master_raw
    			df_summary_raw['JRA-55 RMSE'] = JRA_rmse_master_raw
    			df_summary_raw['MERRA2 RMSE'] = MERRA2_rmse_master_raw
    			df_summary_raw['GLDAS RMSE'] = GLDAS_rmse_master_raw

    			df_summary_raw['Naive Blend ubRMSE'] = naive_ubrmse_master_raw
    			df_summary_raw['CFSR ubRMSE'] = CFSR_ubrmse_master_raw
    			df_summary_raw['ERA-Interim ubRMSE'] = ERAI_ubrmse_master_raw
    			df_summary_raw['ERA5 ubRMSE'] = ERA5_ubrmse_master_raw
    			df_summary_raw['JRA-55 ubRMSE'] = JRA_ubrmse_master_raw
    			df_summary_raw['MERRA2 ubRMSE'] = MERRA2_ubrmse_master_raw
    			df_summary_raw['GLDAS ubRMSE'] = GLDAS_ubrmse_master_raw

    			df_summary_raw['Naive Blend corr'] = naive_corr_master_raw
    			df_summary_raw['CFSR corr'] = CFSR_corr_master_raw
    			df_summary_raw['ERA-Interim corr'] = ERAI_corr_master_raw
    			df_summary_raw['ERA5 corr'] = ERA5_corr_master_raw
    			df_summary_raw['JRA-55 corr'] = JRA_corr_master_raw
    			df_summary_raw['MERRA2 corr'] = MERRA2_corr_master_raw
    			df_summary_raw['GLDAS corr'] = GLDAS_corr_master_raw

    			print(df_summary_raw)

    			df_summary_anom = pd.DataFrame(data=gcell_master_stn, columns=['Grid Cell'])
    			df_summary_anom['Central Lat'] = lat_master_stn
    			df_summary_anom['Central Lon'] = lon_master_stn
    			df_summary_anom['N'] = len_anom_master
    			df_summary_anom['Naive Blend Bias'] = naive_bias_master_anom
    			df_summary_anom['CFSR Bias'] = CFSR_bias_master_anom
    			df_summary_anom['ERA-Interim Bias'] = ERAI_bias_master_anom
    			df_summary_anom['ERA5 Bias'] = ERA5_bias_master_anom
    			df_summary_anom['JRA-55 Bias'] = JRA_bias_master_anom
    			df_summary_anom['MERRA2 Bias'] = MERRA2_bias_master_anom
    			df_summary_anom['GLDAS Bias'] = GLDAS_bias_master_anom

    			df_summary_anom['Naive Blend SDV'] = naive_SDV_master_anom
    			df_summary_anom['CFSR SDV'] = CFSR_SDV_master_anom
    			df_summary_anom['ERA-Interim SDV'] = ERAI_SDV_master_anom
    			df_summary_anom['ERA5 SDV'] = ERA5_SDV_master_anom
    			df_summary_anom['JRA-55 SDV'] = JRA_SDV_master_anom
    			df_summary_anom['MERRA2 SDV'] = MERRA2_SDV_master_anom
    			df_summary_anom['GLDAS SDV'] = GLDAS_SDV_master_anom

    			df_summary_anom['Naive Blend RMSE'] = naive_rmse_master_anom
    			df_summary_anom['CFSR RMSE'] = CFSR_rmse_master_anom
    			df_summary_anom['ERA-Interim RMSE'] = ERAI_rmse_master_anom
    			df_summary_anom['ERA5 RMSE'] = ERA5_rmse_master_anom
    			df_summary_anom['JRA-55 RMSE'] = JRA_rmse_master_anom
    			df_summary_anom['MERRA2 RMSE'] = MERRA2_rmse_master_anom
    			df_summary_anom['GLDAS RMSE'] = GLDAS_rmse_master_anom

    			df_summary_anom['Naive Blend ubRMSE'] = naive_ubrmse_master_anom
    			df_summary_anom['CFSR ubRMSE'] = CFSR_ubrmse_master_anom
    			df_summary_anom['ERA-Interim ubRMSE'] = ERAI_ubrmse_master_anom
    			df_summary_anom['ERA5 ubRMSE'] = ERA5_ubrmse_master_anom
    			df_summary_anom['JRA-55 ubRMSE'] = JRA_ubrmse_master_anom
    			df_summary_anom['MERRA2 ubRMSE'] = MERRA2_ubrmse_master_anom
    			df_summary_anom['GLDAS ubRMSE'] = GLDAS_ubrmse_master_anom

    			df_summary_anom['Naive Blend corr'] = naive_corr_master_anom
    			df_summary_anom['CFSR corr'] = CFSR_corr_master_anom
    			df_summary_anom['ERA-Interim corr'] = ERAI_corr_master_anom
    			df_summary_anom['ERA5 corr'] = ERA5_corr_master_anom
    			df_summary_anom['JRA-55 corr'] = JRA_corr_master_anom
    			df_summary_anom['MERRA2 corr'] = MERRA2_corr_master_anom
    			df_summary_anom['GLDAS corr'] = GLDAS_corr_master_anom

    			print(df_summary_anom)

##### create CSV files #####

    			raw_sum_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/grid_cell_stn/raw_temp/'+str(naive_type_j)+'/'+str(remap_type)+'_'+str(olr_k)+'_top_30cm_thr'+str(thr_l)+'_summary_statistics_gridcell_stn.csv'])
    			print(raw_sum_fil)
    			path = pathlib.Path(raw_sum_fil)
    			path.parent.mkdir(parents=True, exist_ok=True)			
    			df_summary_raw.to_csv(raw_sum_fil,index=False)


    			anom_sum_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/grid_cell_stn/anom/'+str(naive_type_j)+'/'+str(remap_type)+'_'+str(olr_k)+'_top_30cm_thr'+str(thr_l)+'_summary_statistics_anom_gridcell_stn.csv'])
    			print(anom_sum_fil)
    			path2 = pathlib.Path(anom_sum_fil)
    			path2.parent.mkdir(parents=True, exist_ok=True)			
    			df_summary_anom.to_csv(anom_sum_fil,index=False)
