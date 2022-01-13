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
from scipy.stats import pearsonr
from dateutil.relativedelta import *
from matplotlib.ticker import (MultipleLocator, AutoLocator, AutoMinorLocator)

########## Define Functions ##########

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

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

    			gcell_master = []
    			lat_master = []
    			lon_master = []

    			naive_temp_master_raw = []
    			TC_temp_master_raw = []
    			CFSR_temp_master_raw = []
    			ERAI_temp_master_raw = []
    			ERA5_temp_master_raw = []
    			JRA_temp_master_raw = []
    			MERRA2_temp_master_raw = []
    			GLDAS_temp_master_raw = []

    			rnys_time_master_raw = []
    			naive_temp_master_raw = []
    			TC_temp_master_raw = []
    			CFSR_temp_master_raw = []
    			ERAI_temp_master_raw = []
    			ERA5_temp_master_raw = []
    			JRA_temp_master_raw = []
    			MERRA2_temp_master_raw = []
    			GLDAS_temp_master_raw = []

    			rnys_time_master_anom = []
    			naive_temp_master_anom = []
    			TC_temp_master_anom = []
    			CFSR_temp_master_anom = []
    			ERAI_temp_master_anom = []
    			ERA5_temp_master_anom = []
    			JRA_temp_master_anom = []
    			MERRA2_temp_master_anom = []
    			GLDAS_temp_master_anom = []

######### loop through insitu files

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
    				print("Grid Cell:",gcell)
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

######### Create Dataframe for Anomalies collocated with Station Data ##########

    				dframe_anom_master = pd.DataFrame(data=rnysis_date_master, columns=['Date'])
    				dframe_anom_master['Name'] = rnysis_name_master
    				dframe_anom_master['Raw Temp'] = rnysis_stemp_master
    				dframe_anom_master['Anom'] = rnysis_anom_master
    				dframe_anom_master.dropna(inplace=True)
    				len_dframe_anom = len(dframe_anom_master)

    				station_anom = station_anom_master
    				naive_anom_stn = dframe_anom_master[dframe_anom_master['Name'] == 'Naive Blend']
    				naive_anom_stn = naive_anom_stn['Anom'].values
    				CFSR_anom_stn = dframe_anom_master[dframe_anom_master['Name'] == 'CFSR']
    				CFSR_anom_stn = CFSR_anom_stn['Anom'].values
    				ERAI_anom_stn = dframe_anom_master[dframe_anom_master['Name'] == 'ERA-Interim']
    				ERAI_anom_stn = ERAI_anom_stn['Anom'].values
    				ERA5_anom_stn = dframe_anom_master[dframe_anom_master['Name'] == 'ERA-5']
    				ERA5_anom_stn = ERA5_anom_stn['Anom'].values
    				JRA_anom_stn = dframe_anom_master[dframe_anom_master['Name'] == 'JRA-55']
    				JRA_anom_stn = JRA_anom_stn['Anom'].values					
    				MERRA2_anom_stn = dframe_anom_master[dframe_anom_master['Name'] == 'MERRA2']
    				MERRA2_anom_stn = MERRA2_anom_stn['Anom'].values
    				GLDAS_anom_stn = dframe_anom_master[dframe_anom_master['Name'] == 'GLDAS']
    				GLDAS_anom_stn = GLDAS_anom_stn['Anom'].values


    				if (len(CFSR_anom_stn) == 0):
    					print('Grid Cell Skipped')
    					continue

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


    				dframe_temp_gcell_stn_anom = pd.DataFrame(data=date_temp_master,columns=['Date'])
    				dframe_temp_gcell_stn_anom['Grid Cell'] = gcell
    				dframe_temp_gcell_stn_anom['Lat'] = lat_cen
    				dframe_temp_gcell_stn_anom['Lon'] = lon_cen
    				dframe_temp_gcell_stn_anom['Station'] = station_anom_master
    				dframe_temp_gcell_stn_anom['Naive Blend'] = naive_anom_stn
    				dframe_temp_gcell_stn_anom['CFSR'] = CFSR_anom_stn
    				dframe_temp_gcell_stn_anom['ERA-Interim'] = ERAI_anom_stn
    				dframe_temp_gcell_stn_anom['ERA5'] = ERA5_anom_stn
    				dframe_temp_gcell_stn_anom['JRA55'] = JRA_anom_stn
    				dframe_temp_gcell_stn_anom['MERRA2'] = MERRA2_anom_stn
    				dframe_temp_gcell_stn_anom['GLDAS'] = GLDAS_anom_stn
    				dframe_temp_gcell_stn_anom = dframe_temp_gcell_stn_anom[['Grid Cell','Lat','Lon','Date','Naive Blend','Station','CFSR','ERA-Interim','ERA5','JRA55','MERRA2','GLDAS']]


    				dframe_temp_gcell_stn_raw = pd.DataFrame(data=date_temp_master,columns=['Date'])
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
    				dframe_temp_gcell_stn_raw = dframe_temp_gcell_stn_raw[['Grid Cell','Lat','Lon','Date','Naive Blend','Station','CFSR','ERA-Interim','ERA5','JRA55','MERRA2','GLDAS']]


################# Calculate Correlations ####################
    				plot_lat = lat_cen
    				plot_lon = lon_cen

    				len_dt = len(date_temp_master)

    				Dataset_Datetime = [datetime.datetime.strptime(i,'%Y-%m-%d') for i in date_temp_master]
    				Start_Year = Dataset_Datetime[0] - relativedelta(years=1)
    				End_Year = Dataset_Datetime[len_dt - 1] + relativedelta(years=1)
    				Dataset_Station_raw = station_temp_master
    				Dataset_Naive_raw = naive_temp_master
    				Dataset_CFSR_raw = CFSR_temp_master
    				Dataset_ERAI_raw = ERAI_temp_master
    				Dataset_ERA5_raw = ERA5_temp_master
    				Dataset_JRA_raw = JRA_temp_master
    				Dataset_MERRA2_raw = MERRA2_temp_master 
    				Dataset_GLDAS_raw = GLDAS_temp_master

    				Dataset_Station_anom = station_anom_master
    				Dataset_Naive_anom = naive_anom_stn
    				Dataset_CFSR_anom = CFSR_anom_stn
    				Dataset_ERAI_anom = ERAI_anom_stn
    				Dataset_ERA5_anom = ERA5_anom_stn
    				Dataset_JRA_anom = JRA_anom_stn
    				Dataset_MERRA2_anom = MERRA2_anom_stn 
    				Dataset_GLDAS_anom = GLDAS_anom_stn


    				if(len(dframe_temp_gcell_stn_raw) == 0 or len(dframe_temp_gcell_stn_anom) == 0):
    					print('Grid Cell Skipped')
    					continue


    				Correlation_Dataframe_Raw = dframe_temp_gcell_stn_raw[['Station','Naive Blend','CFSR','ERA-Interim','ERA5','JRA55','MERRA2','GLDAS']]				 
    				Correlation_Dataframe_Anom = dframe_temp_gcell_stn_anom[['Station','Naive Blend','CFSR','ERA-Interim','ERA5','JRA55','MERRA2','GLDAS']]

    				Corr_raw = Correlation_Dataframe_Raw.corr()
    				Corr_anom = Correlation_Dataframe_Anom.corr()


########### Create Figures ##############
    				ymin_raw = -40
    				ymax_raw = 30
    				ymin_anom = -15
    				ymax_anom = 15				
    				xmin = np.datetime64(Start_Year,'Y')
    				xmax = np.datetime64(End_Year,'Y')

    				fig,axs = plt.subplots(nrows = 2, ncols = 2, sharex = 'col', sharey = 'row', figsize=(20,20))

#### Raw Temp Timeseries ####
    				ax1 = plt.subplot(221)
    				ax1.plot(Dataset_Datetime,Dataset_Station_raw,label='Station',marker='o',markerfacecolor='black',markersize=2,color='black')
    				ax1.plot(Dataset_Datetime,Dataset_Naive_raw,label='Naive Blended',marker='s',markerfacecolor='red',markersize=2,color='red')
    				ax1.plot(Dataset_Datetime,Dataset_CFSR_raw,label='CFSR',marker='d',markerfacecolor='darkorchid',markersize=2,color='indigo')					
    				ax1.plot(Dataset_Datetime,Dataset_ERAI_raw,label='ERA-Interim',marker='^',markerfacecolor='chartreuse',markersize=2,color='lawngreen')
    				ax1.plot(Dataset_Datetime,Dataset_ERA5_raw,label='ERA5',marker='*',markerfacecolor='gold',markersize=2,color='goldenrod')
    				ax1.plot(Dataset_Datetime,Dataset_JRA_raw,label='JRA55',marker='x',markerfacecolor='dodgerblue',markersize=2,color='blue')
    				ax1.plot(Dataset_Datetime,Dataset_MERRA2_raw,label='MERRA2',marker='+',markerfacecolor='aqua',markersize=2,color='cyan')
    				ax1.plot(Dataset_Datetime,Dataset_GLDAS_raw,label='GLDAS',marker='1',markerfacecolor='orange',markersize=2,color='orange')								
    				ax1.xaxis.set_major_locator(mdates.YearLocator(2)) #major tick every 2 years
    				ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y')) #only show the year
    				ax1.xaxis.set_minor_locator(mdates.YearLocator(1)) #minor tick every year   					
    				ax1.yaxis.set_major_locator(MultipleLocator(5)) #every 10 degrees will be a major tick
    				ax1.yaxis.set_minor_locator(MultipleLocator(1)) #every 2 degrees will be a minor tick
    				ax1.set_xlim(xmin,xmax)
    				ax1.set_ylim(ymin_raw,ymax_raw)
    				ax1.set_title('Absolute Temperature Timeseries')
    				ax1.set_xlabel('Date',fontweight='bold')
    				ax1.set_ylabel('Soil Temp ($^\circ$C)',fontweight='bold')	
    				ax1.legend(loc='best')
#### Raw Temp Correlation ####
    				ax2 = plt.subplot(222)
    				corr1 = sn.heatmap(Corr_raw,annot=True,vmin=0,vmax=1)
    				ax2.set_title('Absolute Temperature Correlation Matrix')

#### Anomaly Timeseries ####
    				ax3 = plt.subplot(223)
    				ax3.plot(Dataset_Datetime,Dataset_Station_anom,label='Station',marker='o',markerfacecolor='black',markersize=2,color='black')
    				ax3.plot(Dataset_Datetime,Dataset_Naive_anom,label='Naive Blended',marker='s',markerfacecolor='red',markersize=2,color='red')
    				ax3.plot(Dataset_Datetime,Dataset_CFSR_anom,label='CFSR',marker='d',markerfacecolor='darkorchid',markersize=2,color='indigo')					
    				ax3.plot(Dataset_Datetime,Dataset_ERAI_anom,label='ERA-Interim',marker='^',markerfacecolor='chartreuse',markersize=2,color='lawngreen')
    				ax3.plot(Dataset_Datetime,Dataset_ERA5_anom,label='ERA5',marker='*',markerfacecolor='gold',markersize=2,color='goldenrod')
    				ax3.plot(Dataset_Datetime,Dataset_JRA_anom,label='JRA55',marker='x',markerfacecolor='dodgerblue',markersize=2,color='blue')
    				ax3.plot(Dataset_Datetime,Dataset_MERRA2_anom,label='MERRA2',marker='+',markerfacecolor='aqua',markersize=2,color='cyan')
    				ax3.plot(Dataset_Datetime,Dataset_GLDAS_anom,label='GLDAS',marker='1',markerfacecolor='orange',markersize=2,color='orange')								
    				ax3.xaxis.set_major_locator(mdates.YearLocator(2)) #major tick every 2 years
    				ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y')) #only show the year
    				ax3.xaxis.set_minor_locator(mdates.YearLocator(1)) #minor tick every year   					
    				ax3.yaxis.set_major_locator(MultipleLocator(3)) #every 3 degrees will be a major tick
    				ax3.yaxis.set_minor_locator(MultipleLocator(1)) #every 2 degrees will be a minor tick
    				ax3.set_xlim(xmin,xmax)
    				ax3.set_ylim(ymin_anom,ymax_anom)
    				ax3.set_title('Anomaly Timeseries')
    				ax3.set_xlabel('Date',fontweight='bold')
    				ax3.set_ylabel('Soil Temp Anomaly ($^\circ$C)',fontweight='bold')	
    				ax3.legend(loc='best')
#### Anomaly Correlation ####
    				ax4 = plt.subplot(224)
    				corr2 = sn.heatmap(Corr_anom,annot=True,vmin=0,vmax=1)
    				ax4.set_title('Temperature Anomaly Correlation Matrix')

    				plt.suptitle('Grid Cell: '+str(gcell)+', Lat: '+str(plot_lat)+'$^\circ$N, Lon :'+str(plot_lon)+'$^\circ$E')
    				plt_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/plots/Naive_Blend_Corr/'+str(remap_type)+'_'+str(olr_k)+'_top_30cm_thr_'+str(thr_l)+'_'+str(naive_type_j)+'_grid_'+str(gcell)+'_summary.png'])
    				plt.savefig(plt_fil)


    				print(plt_fil)
    				plt.close()

