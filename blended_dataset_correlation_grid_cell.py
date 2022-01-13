import os
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


cdo = Cdo()

########## Set directories ##########
TC_dir = ['global_triple_collocation_no_rescaling']#,'global_triple_collocationB','global_triple_collocationC','global_triple_collocationD','global_triple_collocationE']
olr = ['zscore']
lyr = ['0_9.9']
thr = ['100']
rmp_type = ['nn']
tmp_type = ['raw_temp']
gcell = [4096,7425,9719,22982,42887,53322,70671,70923,71181,72450,72451,72708,73980,76275,76276,76282,76528,76529,76783,76791,77045,77814,80366,83168,83423,92836,96434,96691,105333,110954,122932]
str_gcell = str(gcell)
CFSR_layer = "Soil_Temp_L1"
CFSR2_layer = "Soil_Temp_L1"
GLDAS_layer = "Soil_Temp_L1"
ERA5_layer = "Soil_Temp_L1"
ERAI_layer = "Soil_Temp_L1"
JRA_layer = "Soil_Temp"
MERRA2_layer = "Soil_Temp_L1"


########### Grab Reanalysis Data ##########
for i in tmp_type: #loop through data type (absolute temp, anomalies)
    tmp_type_i = i
    if (tmp_type_i == 'raw_temp'):
    	temp_type = 'Absolute Temps'
    	bldsfx = 'raw'
    	bldvar = 'TC_blended_stemp'
    	nvar = 'naive_blended_stemp'
    	s_nam = 'spatial_average'
    	stemp_nam = 'Spatial Avg Temp'
    if (tmp_type_i == 'anom'):
    	temp_type = 'Anomalies'
    	bldsfx = 'anom'
    	bldvar = 'TC_blended_anom'
    	nvar = 'naive_blended_anom'
    	s_nam = 'spatial_average_anom'
    	stemp_nam = 'Spatial Avg Anom'   	
    for j in rmp_type:
    	rmp_type_j = j
    	remap_type = ''.join(['remap'+rmp_type_j])
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


################# Grab Blended Soil Temperature Data ############### 
    	for k in TC_dir:
    		TC_dir_k = k   	
    		TC_basedir_raw = ''.join(['/mnt/data/users/herringtont/soil_temp/'+str(TC_dir_k)+'/raw_temp/'+str(remap_type)+'/blended_products/'])
    		TC_basedir_anom = ''.join(['/mnt/data/users/herringtont/soil_temp/'+str(TC_dir_k)+'/anom/'+str(remap_type)+'/blended_products/'])
    		TC_fi = ''.join([TC_basedir_raw+str(remap_type)+'_TC_blended_raw.nc']) 
    		naive_fi = ''.join([TC_basedir_raw+str(remap_type)+'_naive_blended_raw.nc'])
    		TC_fi_anom = ''.join([TC_basedir_anom+str(remap_type)+'_TC_blended_anom.nc']) 
    		naive_fi_anom = ''.join([TC_basedir_anom+str(remap_type)+'_naive_blended_anom.nc'])
    		for l in olr:
    			olr_l = l

    			for m in thr:
    				thr_m = m
    				insitu_dir =  ''.join(['/mnt/data/users/herringtont/soil_temp/In-Situ/All/spatial_average_anom/'+str(remap_type)+'/no_outliers/'+str(olr_l)+'/0_9.9/thr_'+str(thr_m)+'/'])


###### Create Master Arrays #########
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

###### Grab Grid Cell Level values #######
    				for n in gcell:

    					gcell_n = str(n)

    					print('Grid Cell:',gcell_n)

    					insitu_fil = ''.join([insitu_dir,'grid_'+str(gcell_n)+'_anom.csv'])
    
    					dframe_insitu = pd.read_csv(insitu_fil)
    					dattim = dframe_insitu['Date']
    					DateTime = [datetime.datetime.strptime(x, '%Y-%m-%d') for x in dattim]
    					lat_cen = dframe_insitu['Central Lat'].iloc[0]
    					lon_cen = dframe_insitu['Central Lon'].iloc[0]

    					CFSR_gcell_fi = "".join([rnys_gcell_dir+"CFSR_"+str(remap_type)+'_'+str(olr_l)+'_thr_'+str(thr_m)+'_'+str(TC_dir_k)+'_grid_'+str(gcell_n)+'.nc'])
    					MERRA2_gcell_fi = "".join([rnys_gcell_dir+"MERRA2_"+str(remap_type)+'_'+str(olr_l)+'_thr_'+str(thr_m)+'_'+str(TC_dir_k)+'_grid_'+str(gcell_n)+'.nc'])
    					ERA5_gcell_fi = "".join([rnys_gcell_dir,"ERA5_"+str(remap_type)+'_'+str(olr_l)+'_thr_'+str(thr_m)+'_'+str(TC_dir_k)+'_grid_'+str(gcell_n)+'.nc'])
    					ERAI_gcell_fi = "".join([rnys_gcell_dir,"ERA-Interim_"+str(remap_type)+'_'+str(olr_l)+'_thr_'+str(thr_m)+'_'+str(TC_dir_k)+'_grid_'+str(gcell_n)+'.nc'])
    					JRA_gcell_fi = "".join([rnys_gcell_dir,"JRA55_"+str(remap_type)+'_'+str(olr_l)+'_thr_'+str(thr_m)+'_'+str(TC_dir_k)+'_grid_'+str(gcell_n)+'.nc'])
    					GLDAS_gcell_fi = "".join([rnys_gcell_dir,"GLDAS_"+str(remap_type)+'_'+str(olr_l)+'_thr_'+str(thr_m)+'_'+str(TC_dir_k)+'_grid_'+str(gcell_n)+'.nc'])

    					
    					CFSR_gcell_fi_anom = "".join([rnys_gcell_dir,"CFSR_anom_"+str(remap_type)+'_'+str(olr_l)+'_thr_'+str(thr_m)+'_'+str(TC_dir_k)+'_grid_'+str(gcell_n)+'.nc'])
    					MERRA2_gcell_fi_anom = "".join([rnys_gcell_dir,"MERRA2_anom_"+str(remap_type)+'_'+str(olr_l)+'_thr_'+str(thr_m)+'_'+str(TC_dir_k)+'_grid_'+str(gcell_n)+'.nc'])
    					ERA5_gcell_fi_anom = "".join([rnys_gcell_dir,"ERA5_anom_"+str(remap_type)+'_'+str(olr_l)+'_thr_'+str(thr_m)+'_'+str(TC_dir_k)+'_grid_'+str(gcell_n)+'.nc'])
    					ERAI_gcell_fi_anom = "".join([rnys_gcell_dir,"ERA-Interim_anom_"+str(remap_type)+'_'+str(olr_l)+'_thr'+str(thr_m)+'_'+str(TC_dir_k)+'_grid_'+str(gcell_n)+'.nc'])
    					JRA_gcell_fi_anom = "".join([rnys_gcell_dir,"JRA55_anom_"+str(remap_type)+'_'+str(olr_l)+'_thr_'+str(thr_m)+'_'+str(TC_dir_k)+'_grid_'+str(gcell_n)+'.nc'])
    					GLDAS_gcell_fi_anom = "".join([rnys_gcell_dir,"GLDAS_anom_"+str(remap_type)+'_'+str(olr_l)+'_thr_'+str(thr_m)+'_'+str(TC_dir_k)+'_grid_'+str(gcell_n)+'.nc'])
					
    					TC_gcell_dir_raw = ''.join([TC_basedir_raw,'grid_cell_level/'])
    					Path(TC_gcell_dir_raw).mkdir(parents=True,exist_ok=True)

    					TC_gcell_dir_anom = ''.join([TC_basedir_anom,'grid_cell_level/'])
    					Path(TC_gcell_dir_anom).mkdir(parents=True,exist_ok=True)

    					err_var_dir_raw = ''.join(['/mnt/data/users/herringtont/soil_temp/'+str(TC_dir_k)+'/raw_temp/remap'+str(rmp_type_j)+'/'])
    					err_var_dir_anom = ''.join(['/mnt/data/users/herringtont/soil_temp/'+str(TC_dir_k)+'/anom/remap'+str(rmp_type_j)+'/'])
    					err_var_dir_raw_gcell = ''.join(['/mnt/data/users/herringtont/soil_temp/'+str(TC_dir_k)+'/raw_temp/remap'+str(rmp_type_j)+'/grid_cell_level/'])    	
    					err_var_dir_anom_gcell = ''.join(['/mnt/data/users/herringtont/soil_temp/'+str(TC_dir_k)+'/anom/remap'+str(rmp_type_j)+'/grid_cell_level'])

    					Path(err_var_dir_raw_gcell).mkdir(parents=True,exist_ok=True)
    					Path(err_var_dir_anom_gcell).mkdir(parents=True,exist_ok=True)

    					TC_gcell_fi = ''.join([TC_gcell_dir_raw,'TC_blended_'+str(remap_type)+'_'+str(olr_l)+'_thr'+str(thr_m)+'_raw_'+str(TC_dir_k)+'_grid_'+str(gcell_n)+'.nc'])
    					naive_gcell_fi = ''.join([TC_gcell_dir_raw,'naive_blended_'+str(remap_type)+'_'+str(olr_l)+'_thr_'+str(thr_m)+'_raw_'+str(TC_dir_k)+'_grid_'+str(gcell_n)+'.nc'])

    					TC_gcell_fi_anom = ''.join([TC_gcell_dir_anom,'TC_blended_anom_'+str(remap_type)+'_'+str(olr_l)+'_thr'+str(thr_m)+'_anom_'+str(TC_dir_k)+'_grid_'+str(gcell_n)+'.nc'])
    					naive_gcell_fi_anom = ''.join([TC_gcell_dir_anom,'naive_blended_anom_'+str(remap_type)+'_'+str(olr_l)+'_thr_'+str(thr_m)+'_anom_'+str(TC_dir_k)+'_grid_'+str(gcell_n)+'.nc'])


    					if (TC_dir_k == 'global_triple_collocation' or TC_dir_k == 'global_triple_collocation_no_rescaling'):
    						model1_nam = 'JRA55 err_var'
    						model2_nam = 'MERRA2 err_var'
    						model3_nam = 'GLDAS err_var'
    						model1 = 'JRA55'
    						model2 = 'MERRA2'
    						model3 = 'GLDAS'

    					if (TC_dir_k == 'global_triple_collocationB'):
    						model1_nam = 'JRA55 err_var'
    						model2_nam = 'MERRA2 err_var'
    						model3_nam = 'ERA-Interim err_var'
    						model1 = 'JRA55'
    						model2 = 'MERRA2'
    						model3 = 'ERA-Interim'

    					if (TC_dir_k == 'global_triple_collocationC'):
    						model1_nam = 'JRA55 err_var'
    						model2_nam = 'MERRA2 err_var'
    						model3_nam = 'ERA5 err_var'
    						model1 = 'JRA55'
    						model2 = 'MERRA2'
    						model3 = 'ERA5'


    					if (TC_dir_k == 'global_triple_collocationD'):
    						model1_nam = 'CFSR err_var'
    						model2_nam = 'MERRA2 err_var'
    						model3_nam = 'ERA5 err_var'
    						model1 = 'CFSR'
    						model2 = 'MERRA2'
    						model3 = 'ERA5'

    					if (TC_dir_k == 'global_triple_collocationE'):
    						model1_nam = 'JRA55 err_var'
    						model2_nam = 'ERA-Interim err_var'
    						model3_nam = 'ERA5 err_var'
    						model1 = 'JRA55'
    						model2 = 'ERA-Interim'
    						model3 = 'ERA5'

    					err_var_model1_fi = ''.join([err_var_dir_raw+'remap'+str(rmp_type_j)+'_'+str(model1)+'_err_var_cov.nc'])
    					err_var_model2_fi = ''.join([err_var_dir_raw+'remap'+str(rmp_type_j)+'_'+str(model2)+'_err_var_cov.nc'])
    					err_var_model3_fi = ''.join([err_var_dir_raw+'remap'+str(rmp_type_j)+'_'+str(model3)+'_err_var_cov.nc'])

    					err_var_model1_fi_anom = ''.join([err_var_dir_anom+'remap'+str(rmp_type_j)+'_'+str(model1)+'_err_var_cov.nc'])
    					err_var_model2_fi_anom = ''.join([err_var_dir_anom+'remap'+str(rmp_type_j)+'_'+str(model2)+'_err_var_cov.nc'])
    					err_var_model3_fi_anom = ''.join([err_var_dir_anom+'remap'+str(rmp_type_j)+'_'+str(model3)+'_err_var_cov.nc'])

    					err_var_model1_gcell_fi = ''.join([err_var_dir_raw+'remap'+str(rmp_type_j)+'_'+str(model1)+'_err_var_cov_grid_'+str(gcell_n)+'.nc'])
    					err_var_model2_gcell_fi = ''.join([err_var_dir_raw+'remap'+str(rmp_type_j)+'_'+str(model2)+'_err_var_cov_grid_'+str(gcell_n)+'.nc'])
    					err_var_model3_gcell_fi = ''.join([err_var_dir_raw+'remap'+str(rmp_type_j)+'_'+str(model3)+'_err_var_cov_grid_'+str(gcell_n)+'.nc'])

    					err_var_model1_gcell_fi_anom = ''.join([err_var_dir_anom+'remap'+str(rmp_type_j)+'_'+str(model1)+'_err_var_cov_grid_'+str(gcell_n)+'.nc'])
    					err_var_model2_gcell_fi_anom = ''.join([err_var_dir_anom+'remap'+str(rmp_type_j)+'_'+str(model2)+'_err_var_cov_grid_'+str(gcell_n)+'.nc'])
    					err_var_model3_gcell_fi_anom = ''.join([err_var_dir_anom+'remap'+str(rmp_type_j)+'_'+str(model3)+'_err_var_cov_grid_'+str(gcell_n)+'.nc'])
					
    					cdo.remapnn('lon='+str(lon_cen)+'/lat='+str(lat_cen), input=CFSR_fi, output=CFSR_gcell_fi, options = '-f nc')    					
    					cdo.remapnn('lon='+str(lon_cen)+'/lat='+str(lat_cen), input=MERRA2_fi, output=MERRA2_gcell_fi, options = '-f nc') 
    					cdo.remapnn('lon='+str(lon_cen)+'/lat='+str(lat_cen), input=ERA5_fi, output=ERA5_gcell_fi, options = '-f nc') 
    					cdo.remapnn('lon='+str(lon_cen)+'/lat='+str(lat_cen), input=ERAI_fi, output=ERAI_gcell_fi, options = '-f nc')
    					cdo.remapnn('lon='+str(lon_cen)+'/lat='+str(lat_cen), input=JRA_fi, output=JRA_gcell_fi, options = '-f nc')
    					cdo.remapnn('lon='+str(lon_cen)+'/lat='+str(lat_cen), input=GLDAS_fi, output=GLDAS_gcell_fi, options = '-f nc') 
    					cdo.remapnn('lon='+str(lon_cen)+'/lat='+str(lat_cen), input=TC_fi, output=TC_gcell_fi, options = '-f nc')
    					cdo.remapnn('lon='+str(lon_cen)+'/lat='+str(lat_cen), input=naive_fi, output=naive_gcell_fi, options = '-f nc')

    					cdo.remapnn('lon='+str(lon_cen)+'/lat='+str(lat_cen), input=CFSR_fi_anom, output=CFSR_gcell_fi_anom, options = '-f nc')    					
    					cdo.remapnn('lon='+str(lon_cen)+'/lat='+str(lat_cen), input=MERRA2_fi_anom, output=MERRA2_gcell_fi_anom, options = '-f nc') 
    					cdo.remapnn('lon='+str(lon_cen)+'/lat='+str(lat_cen), input=ERA5_fi_anom, output=ERA5_gcell_fi_anom, options = '-f nc') 
    					cdo.remapnn('lon='+str(lon_cen)+'/lat='+str(lat_cen), input=ERAI_fi_anom, output=ERAI_gcell_fi_anom, options = '-f nc')
    					cdo.remapnn('lon='+str(lon_cen)+'/lat='+str(lat_cen), input=JRA_fi_anom, output=JRA_gcell_fi_anom, options = '-f nc')
    					cdo.remapnn('lon='+str(lon_cen)+'/lat='+str(lat_cen), input=GLDAS_fi_anom, output=GLDAS_gcell_fi_anom, options = '-f nc') 
    					cdo.remapnn('lon='+str(lon_cen)+'/lat='+str(lat_cen), input=TC_fi_anom, output=TC_gcell_fi_anom, options = '-f nc')
    					cdo.remapnn('lon='+str(lon_cen)+'/lat='+str(lat_cen), input=naive_fi_anom, output=naive_gcell_fi_anom, options = '-f nc')

    					cdo.remapnn('lon='+str(lon_cen)+'/lat='+str(lat_cen), input=err_var_model1_fi, output=err_var_model1_gcell_fi, options = '-f nc')
    					cdo.remapnn('lon='+str(lon_cen)+'/lat='+str(lat_cen), input=err_var_model2_fi, output=err_var_model2_gcell_fi, options = '-f nc')
    					cdo.remapnn('lon='+str(lon_cen)+'/lat='+str(lat_cen), input=err_var_model3_fi, output=err_var_model3_gcell_fi, options = '-f nc')

    					cdo.remapnn('lon='+str(lon_cen)+'/lat='+str(lat_cen), input=err_var_model1_fi_anom, output=err_var_model1_gcell_fi_anom, options = '-f nc')
    					cdo.remapnn('lon='+str(lon_cen)+'/lat='+str(lat_cen), input=err_var_model2_fi_anom, output=err_var_model2_gcell_fi_anom, options = '-f nc')
    					cdo.remapnn('lon='+str(lon_cen)+'/lat='+str(lat_cen), input=err_var_model3_fi_anom, output=err_var_model3_gcell_fi_anom, options = '-f nc')

    					GLDAS_fil = xr.open_dataset(GLDAS_gcell_fi).isel(lat=0,lon=0,drop=True)
    					JRA_fil = xr.open_dataset(JRA_gcell_fi).isel(lat=0,lon=0,drop=True)
    					ERAI_fil = xr.open_dataset(ERAI_gcell_fi).isel(lat=0,lon=0,drop=True)
    					ERA5_fil = xr.open_dataset(ERA5_gcell_fi).isel(lat=0,lon=0,drop=True)
    					MERRA2_fil = xr.open_dataset(MERRA2_gcell_fi).isel(lat=0,lon=0,drop=True)
    					CFSR_fil = xr.open_dataset(CFSR_gcell_fi).isel(lat=0,lon=0,drop=True)
    					TC_fil = xr.open_dataset(TC_gcell_fi).isel(lat=0,lon=0,drop=True)
    					naive_fil = xr.open_dataset(naive_gcell_fi).isel(lat=0,lon=0,drop=True)

    					err_var_model1_fil = xr.open_dataset(err_var_model1_gcell_fi).isel(lat=0,lon=0,drop=True)
    					err_var_model2_fil = xr.open_dataset(err_var_model2_gcell_fi).isel(lat=0,lon=0,drop=True)
    					err_var_model3_fil = xr.open_dataset(err_var_model3_gcell_fi).isel(lat=0,lon=0,drop=True)					

    					GLDAS_fil_anom = xr.open_dataset(GLDAS_gcell_fi_anom).isel(lat=0,lon=0,drop=True)
    					JRA_fil_anom = xr.open_dataset(JRA_gcell_fi_anom).isel(lat=0,lon=0,drop=True)
    					ERAI_fil_anom = xr.open_dataset(ERAI_gcell_fi_anom).isel(lat=0,lon=0,drop=True)
    					ERA5_fil_anom = xr.open_dataset(ERA5_gcell_fi_anom).isel(lat=0,lon=0,drop=True)
    					MERRA2_fil_anom = xr.open_dataset(MERRA2_gcell_fi_anom).isel(lat=0,lon=0,drop=True)
    					CFSR_fil_anom = xr.open_dataset(CFSR_gcell_fi_anom).isel(lat=0,lon=0,drop=True)
    					TC_fil_anom = xr.open_dataset(TC_gcell_fi_anom).isel(lat=0,lon=0,drop=True)
    					naive_fil_anom = xr.open_dataset(naive_gcell_fi_anom).isel(lat=0,lon=0,drop=True)

    					err_var_model1_fil_anom = xr.open_dataset(err_var_model1_gcell_fi_anom).isel(lat=0,lon=0,drop=True)
    					err_var_model2_fil_anom = xr.open_dataset(err_var_model2_gcell_fi_anom).isel(lat=0,lon=0,drop=True)
    					err_var_model3_fil_anom = xr.open_dataset(err_var_model3_gcell_fi_anom).isel(lat=0,lon=0,drop=True)

    					GLDAS_temp = GLDAS_fil[GLDAS_layer] - 273.15
    					#GLDAS_temp = GLDAS_temp.values
    					JRA_temp = JRA_fil[JRA_layer] - 273.15
    					#JRA_temp = JRA_temp.values
    					ERAI_temp = ERAI_fil[ERAI_layer] - 273.15
    					#ERAI_temp = ERAI_temp.values
    					ERA5_temp = ERA5_fil[ERA5_layer] - 273.15
    					#ERA5_temp = ERA5_temp.values
    					MERRA2_temp = MERRA2_fil[MERRA2_layer] - 273.15 #convert from Kelvin to Celsius
    					#MERRA2_temp = MERRA2_temp.values
    					CFSR_temp = CFSR_fil[CFSR_layer] - 273.15
    					#CFSR_temp = CFSR_temp.values
    					TC_temp = TC_fil['TC_blended_stemp']
    					#TC_temp = TC_temp.values
    					naive_temp = naive_fil['naive_blended_stemp']
    					#naive_temp = naive_temp.values

    					err_var_model1 = err_var_model1_fil['err_var_cov']
    					err_var_model2 = err_var_model2_fil['err_var_cov']
    					err_var_model3 = err_var_model3_fil['err_var_cov']

    					model1_err_var = err_var_model1.values.tolist()
    					model2_err_var = err_var_model2.values.tolist()
    					model3_err_var = err_var_model3.values.tolist()

    					model1_err_var_rnd = round(model1_err_var,3)
    					model2_err_var_rnd = round(model2_err_var,3)					
    					model3_err_var_rnd = round(model3_err_var,3)

    					GLDAS_anom = GLDAS_fil_anom[GLDAS_layer].values
    					JRA_anom = JRA_fil_anom[JRA_layer].values
    					ERAI_anom = ERAI_fil_anom[ERAI_layer].values
    					ERA5_anom = ERA5_fil_anom[ERA5_layer].values
    					MERRA2_anom = MERRA2_fil_anom[MERRA2_layer].values
    					CFSR_anom = CFSR_fil_anom[CFSR_layer].values
    					TC_anom = TC_fil_anom['TC_blended_anom'].values
    					naive_anom = naive_fil_anom['naive_blended_anom'].values

    					err_var_model1_anom = err_var_model1_fil_anom['err_var_cov']
    					err_var_model2_anom = err_var_model2_fil_anom['err_var_cov']
    					err_var_model3_anom = err_var_model3_fil_anom['err_var_cov']

    					model1_err_var_anom = err_var_model1_anom.values.tolist()
    					model2_err_var_anom = err_var_model2_anom.values.tolist()
    					model3_err_var_anom = err_var_model3_anom.values.tolist()

    					model1_err_var_anom_rnd = round(model1_err_var_anom,3)
    					model2_err_var_anom_rnd = round(model2_err_var_anom,3)					
    					model3_err_var_anom_rnd = round(model3_err_var_anom,3)

    					rnys_dattim = TC_fil['time']
    					rnys_datetime = rnys_dattim.dt.strftime('%Y-%m-%d').values

    					rnys_dt = [datetime.datetime.strptime(x, '%Y-%m-%d') for x in rnys_datetime]

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

    					#dframe_insitu_idx = dframe_insitu.set_index(dframe_insitu['Date'])
    					#dframe_insitu_rdx = dframe_insitu_idx.reindex(rnys_dt,fill_value=np.nan)   					 

    					station_temp = dframe_insitu['Spatial Avg Temp'].values

    					station_anom = dframe_insitu['Spatial Avg Anom'].values

    					dframe_temp_gcell_raw = pd.DataFrame(data=rnys_datetime,columns=['Date'])
    					dframe_temp_gcell_raw['Grid Cell'] = gcell_n
    					dframe_temp_gcell_raw['Lat'] = lat_cen
    					dframe_temp_gcell_raw['Lon'] = lon_cen
    					dframe_temp_gcell_raw[model1_nam] = err_var_model1.values
    					dframe_temp_gcell_raw[model2_nam] = err_var_model2.values
    					dframe_temp_gcell_raw[model3_nam] = err_var_model3.values
    					dframe_temp_gcell_raw['TC Blend'] = TC_temp.values
    					dframe_temp_gcell_raw['Naive Blend'] = naive_temp.values
    					dframe_temp_gcell_raw['CFSR'] = CFSR_temp.values
    					dframe_temp_gcell_raw['ERA-Interim'] = ERAI_temp.values
    					dframe_temp_gcell_raw['ERA5'] = ERA5_temp.values
    					dframe_temp_gcell_raw['JRA55'] = JRA_temp.values
    					dframe_temp_gcell_raw['MERRA2'] = MERRA2_temp.values
    					dframe_temp_gcell_raw['GLDAS'] = GLDAS_temp.values
    					dframe_temp_gcell_raw = dframe_temp_gcell_raw[['Grid Cell','Lat','Lon',model1_nam,model2_nam,model3_nam,'Date','TC Blend','Naive Blend','CFSR','ERA-Interim','ERA5','JRA55','MERRA2','GLDAS']]
    					dframe_temp_gcell_raw.dropna(inplace=True)
    					#print(dframe_temp_gcell_raw)

    					dframe_temp_gcell_anom = pd.DataFrame(data=rnys_datetime,columns=['Date'])
    					dframe_temp_gcell_anom['Grid Cell'] = gcell_n
    					dframe_temp_gcell_anom['Lat'] = lat_cen
    					dframe_temp_gcell_anom['Lon'] = lon_cen
    					dframe_temp_gcell_anom[model1_nam] = err_var_model1_anom.values
    					dframe_temp_gcell_anom[model2_nam] = err_var_model2_anom.values
    					dframe_temp_gcell_anom[model3_nam] = err_var_model3_anom.values
    					dframe_temp_gcell_anom['TC Blend'] = TC_anom
    					dframe_temp_gcell_anom['Naive Blend'] = naive_anom
    					dframe_temp_gcell_anom['CFSR'] = CFSR_anom
    					dframe_temp_gcell_anom['ERA-Interim'] = ERAI_anom
    					dframe_temp_gcell_anom['ERA5'] = ERA5_anom
    					dframe_temp_gcell_anom['JRA55'] = JRA_anom
    					dframe_temp_gcell_anom['MERRA2'] = MERRA2_anom
    					dframe_temp_gcell_anom['GLDAS'] = GLDAS_anom
    					dframe_temp_gcell_anom = dframe_temp_gcell_anom[['Grid Cell','Lat','Lon',model1_nam,model2_nam,model3_nam,'Date','TC Blend','Naive Blend','CFSR','ERA-Interim','ERA5','JRA55','MERRA2','GLDAS']]
    					dframe_temp_gcell_anom.dropna(inplace=True)
    					#print(dframe_temp_gcell_anom)


    					#print(dframe_temp_gcell_stn_raw)


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
    						TC_temp_dt = TC_temp.sel(time=DateTime_m).values.tolist()
    						if(str(TC_temp_dt) == "nan"):
    							TC_temp_dt = np.nan
    						naive_temp_dt = naive_temp.sel(time=DateTime_m).values.tolist()
    						if(str(naive_temp_dt) == "nan"):
    							naive_temp_dt = np.nan  						
    						dframe_insitu_dt = dframe_insitu[dframe_insitu['Date'] == dattim_m]
    						station_temp_dt = dframe_insitu_dt[stemp_nam].values.tolist()
    						if(str(station_temp_dt) == "nan"):
    							station_temp_dt = np.nan
    						station_temp_master.append(station_temp_dt)
    						station_anom_dt = dframe_insitu_dt['Spatial Avg Anom'].tolist()
    						if(str(station_anom_dt) == "nan"):
    							station_anom_dt = np.nan
    						station_anom_master.append(station_anom_dt)									
    						CFSR_temp_dt = CFSR_temp.sel(time=DateTime_m).values.tolist()
    						if(str(CFSR_temp_dt) == "nan"):
    							CFSR_temp_dt = np.nan
    						CFSR_temp_master.append(CFSR_temp_dt)    						
    						JRA_temp_dt = JRA_temp.sel(time=DateTime_m).values.tolist()
    						if(str(JRA_temp_dt) == "nan"):
    							JRA_temp_dt = np.nan
    						JRA_temp_master.append(JRA_temp_dt)      							
    						ERAI_temp_dt = ERAI_temp.sel(time=DateTime_m).values.tolist()
    						if(str(ERAI_temp_dt) == "nan"):
    							ERAI_temp_dt = np.nan
    						ERAI_temp_master.append(ERAI_temp_dt)
    						ERA5_temp_dt = ERA5_temp.sel(time=DateTime_m).values.tolist()
    						if(str(ERA5_temp_dt) == "nan"):
    							ERA5_temp_dt = np.nan
    						ERA5_temp_master.append(ERA5_temp_dt)
    						MERRA2_temp_dt = MERRA2_temp.sel(time=DateTime_m).values.tolist()
    						if(str(MERRA2_temp_dt) == "nan"):
    							MERRA2_temp_dt = np.nan
    						MERRA2_temp_master.append(MERRA2_temp_dt)
    						GLDAS_temp_dt = GLDAS_temp.sel(time=DateTime_m).values.tolist()
    						if(str(GLDAS_temp_dt) == "nan"):
    							GLDAS_temp_dt = np.nan
    						GLDAS_temp_master.append(GLDAS_temp_dt)
    						TC_temp_master.append(TC_temp_dt)
    						date_temp_master.append(dattim_m)    						
    						naive_temp_master.append(naive_temp_dt)            							    						


    					station_temp_master = [i for sub in station_temp_master for i in sub]
    					station_anom_master = [i for sub in station_anom_master for i in sub]
    					station_temp_master = np.array(station_temp_master)
    					station_anom_master = np.array(station_anom_master)
    					date_temp_master = np.array(date_temp_master)
    					CFSR_temp_master = np.array(CFSR_temp_master)
    					ERAI_temp_master = np.array(ERAI_temp_master)
    					ERA5_temp_master = np.array(ERA5_temp_master)
    					JRA_temp_master = np.array(JRA_temp_master)
    					MERRA2_temp_master = np.array(MERRA2_temp_master)
    					GLDAS_temp_master = np.array(GLDAS_temp_master)
    					TC_temp_master = np.array(TC_temp_master)
    					naive_temp_master = np.array(naive_temp_master)
						
#################### create anomalies for reanalysis files #######################
    					rnysis_anom_master = []
    					rnysis_date_master = []
    					rnysis_name_master = []
    					rnysis_stemp_master = []
    					rnysis = [TC_temp_master,naive_temp_master,CFSR_temp_master,ERAI_temp_master,ERA5_temp_master,JRA_temp_master,MERRA2_temp_master,GLDAS_temp_master]
    					rnysis_name = ['TC Blend','Naive Blend','CFSR','ERA-Interim','ERA-5','JRA-55','MERRA2','GLDAS']
    					dat_rowlist = [datetime.datetime.strptime(x,'%Y-%m-%d') for x in date_temp_master]
    					dat_rowlist2 = date_temp_master
    					num_rows = len(dat_rowlist)

    					for m in range(0,8):
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



######### Create Dataframe for Anomalies collocated with Station Data ##########

    					dframe_anom_master = pd.DataFrame(data=rnysis_date_master, columns=['Date'])
    					dframe_anom_master['Name'] = rnysis_name_master
    					dframe_anom_master['Raw Temp'] = rnysis_stemp_master
    					dframe_anom_master['Anom'] = rnysis_anom_master
    					dframe_anom_master.dropna(inplace=True)
    					len_dframe_anom = len(dframe_anom_master)

    					station_anom = station_anom_master
    					TC_anom_stn = dframe_anom_master[dframe_anom_master['Name'] == 'TC Blend']
    					TC_anom_stn = TC_anom_stn['Anom'].values
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

    					TC_no_nan = TC_temp_master[~np.isnan(TC_temp_master)]

    					#print(naive_no_nan,TC_no_nan)

    					CFSR_no_nan = CFSR_temp_master[~np.isnan(CFSR_temp_master)]
    					#print(CFSR_no_nan)

    					if(DateTime[0]>CFSR_edate_dt or DateTime[len(DateTime) -1] < CFSR_sdate_dt): #skip if the CFSR dates and station dates do not overlap
    						print('Grid Cell Skipped')
    						continue
    					
    					if(len(naive_no_nan) == 0 or len(TC_no_nan) == 0 or len(CFSR_no_nan) == 0): #skip if there are NaN values in blended data
    						print('Grid Cell Skipped')
    						continue


    					dframe_temp_gcell_stn_anom = pd.DataFrame(data=date_temp_master,columns=['Date'])
    					dframe_temp_gcell_stn_anom['Grid Cell'] = gcell_n
    					dframe_temp_gcell_stn_anom['Lat'] = lat_cen
    					dframe_temp_gcell_stn_anom['Lon'] = lon_cen
    					dframe_temp_gcell_stn_anom[model1_nam] = err_var_model1_anom.values
    					dframe_temp_gcell_stn_anom[model2_nam] = err_var_model2_anom.values
    					dframe_temp_gcell_stn_anom[model3_nam] = err_var_model3_anom.values
    					dframe_temp_gcell_stn_anom['Station'] = station_anom_master
    					dframe_temp_gcell_stn_anom['TC Blend'] = TC_anom_stn
    					dframe_temp_gcell_stn_anom['Naive Blend'] = naive_anom_stn
    					dframe_temp_gcell_stn_anom['CFSR'] = CFSR_anom_stn
    					dframe_temp_gcell_stn_anom['ERA-Interim'] = ERAI_anom_stn
    					dframe_temp_gcell_stn_anom['ERA5'] = ERA5_anom_stn
    					dframe_temp_gcell_stn_anom['JRA55'] = JRA_anom_stn
    					dframe_temp_gcell_stn_anom['MERRA2'] = MERRA2_anom_stn
    					dframe_temp_gcell_stn_anom['GLDAS'] = GLDAS_anom_stn
    					dframe_temp_gcell_stn_anom = dframe_temp_gcell_stn_anom[['Grid Cell','Lat','Lon',model1_nam,model2_nam,model3_nam,'Date','TC Blend','Naive Blend','Station','CFSR','ERA-Interim','ERA5','JRA55','MERRA2','GLDAS']]
    					anom_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/blended_metrics/Mar29/'+str(remap_type)+'_'+str(olr_l)+'_thr_'+str(thr_m)+'_anom_'+str(TC_dir_k)+'_grid_'+str(gcell_n)+'_anom_table.csv'])
    					dframe_temp_gcell_stn_anom.to_csv(anom_fil,index=False)


    					dframe_temp_gcell_stn_raw = pd.DataFrame(data=date_temp_master,columns=['Date'])
    					dframe_temp_gcell_stn_raw['Grid Cell'] = gcell_n
    					dframe_temp_gcell_stn_raw['Lat'] = lat_cen
    					dframe_temp_gcell_stn_raw['Lon'] = lon_cen
    					dframe_temp_gcell_stn_raw[model1_nam] = err_var_model1.values
    					dframe_temp_gcell_stn_raw[model2_nam] = err_var_model2.values
    					dframe_temp_gcell_stn_raw[model3_nam] = err_var_model3.values
    					dframe_temp_gcell_stn_raw['Station'] = station_temp_master
    					dframe_temp_gcell_stn_raw['TC Blend'] = TC_temp_master
    					dframe_temp_gcell_stn_raw['Naive Blend'] = naive_temp_master
    					dframe_temp_gcell_stn_raw['CFSR'] = CFSR_temp_master
    					dframe_temp_gcell_stn_raw['ERA-Interim'] = ERAI_temp_master
    					dframe_temp_gcell_stn_raw['ERA5'] = ERA5_temp_master
    					dframe_temp_gcell_stn_raw['JRA55'] = JRA_temp_master
    					dframe_temp_gcell_stn_raw['MERRA2'] = MERRA2_temp_master
    					dframe_temp_gcell_stn_raw['GLDAS'] = GLDAS_temp_master
    					dframe_temp_gcell_stn_raw = dframe_temp_gcell_stn_raw[['Grid Cell','Lat','Lon',model1_nam,model2_nam,model3_nam,'Date','TC Blend','Naive Blend','Station','CFSR','ERA-Interim','ERA5','JRA55','MERRA2','GLDAS']]
    					raw_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/blended_metrics/Mar29/'+str(remap_type)+'_'+str(olr_l)+'_thr_'+str(thr_m)+'_anom_'+str(TC_dir_k)+'_grid_'+str(gcell_n)+'_raw_table.csv'])
    					dframe_temp_gcell_stn_anom.to_csv(raw_fil,index=False)


################# Calculate Correlations ####################
    					plot_lat = lat_cen
    					plot_lon = lon_cen




    					if(len(dframe_temp_gcell_stn_raw) == 0 or len(dframe_temp_gcell_stn_anom) == 0):
    						print('Grid Cell Skipped')
    						continue
    					#print(dframe_temp_gcell_stn_raw)
    					#print(dframe_temp_gcell_stn_anom)

    					Dataset_Datetime = [datetime.datetime.strptime(i,'%Y-%m-%d') for i in date_temp_master]
    					Dataset_A_raw = dframe_temp_gcell_stn_raw['TC Blend'].values
    					Dataset_B_raw = dframe_temp_gcell_stn_raw['Naive Blend'].values
    					Dataset_S_raw = dframe_temp_gcell_stn_raw['Station'].values
    					Dataset_A_anom = dframe_temp_gcell_stn_anom['TC Blend'].values
    					Dataset_B_anom = dframe_temp_gcell_stn_anom['Naive Blend'].values
    					Dataset_S_anom = dframe_temp_gcell_stn_anom['Station'].values

    					if(TC_dir_k == 'global_triple_collocation' or TC_dir_k == 'global_triple_collocation_no_rescaling'):
    						Dataset_C_raw = dframe_temp_gcell_stn_raw['JRA55'].values 
    						Dataset_D_raw = dframe_temp_gcell_stn_raw['MERRA2'].values
    						Dataset_E_raw = dframe_temp_gcell_stn_raw['GLDAS'].values
    						Dataset_C_anom = dframe_temp_gcell_stn_anom['JRA55'].values 
    						Dataset_D_anom = dframe_temp_gcell_stn_anom['MERRA2'].values
    						Dataset_E_anom = dframe_temp_gcell_stn_anom['GLDAS'].values

    						Dataset_C_nam = 'JRA55'
    						Dataset_D_nam = 'MERRA2'
    						Dataset_E_nam = 'GLDAS'


    					if(TC_dir_k == 'global_triple_collocationB'):
    						Dataset_C_raw = dframe_temp_gcell_stn_raw['JRA55'].values 
    						Dataset_D_raw = dframe_temp_gcell_stn_raw['MERRA2'].values
    						Dataset_E_raw = dframe_temp_gcell_stn_raw['ERA-Interim'].values
    						Dataset_C_anom = dframe_temp_gcell_stn_anom['JRA55'].values 
    						Dataset_D_anom = dframe_temp_gcell_stn_anom['MERRA2'].values
    						Dataset_E_anom = dframe_temp_gcell_stn_anom['ERA-Interim'].values

    						Dataset_C_nam = 'JRA55'
    						Dataset_D_nam = 'MERRA2'
    						Dataset_E_nam = 'ERA-Interim'

    					if(TC_dir_k == 'global_triple_collocationC'):
    						Dataset_C_raw = dframe_temp_gcell_stn_raw['JRA55'].values 
    						Dataset_D_raw = dframe_temp_gcell_stn_raw['MERRA2'].values
    						Dataset_E_raw = dframe_temp_gcell_stn_raw['ERA5'].values
    						Dataset_C_anom = dframe_temp_gcell_stn_anom['JRA55'].values 
    						Dataset_D_anom = dframe_temp_gcell_stn_anom['MERRA2'].values
    						Dataset_E_anom = dframe_temp_gcell_stn_anom['ERA5'].values

    						Dataset_C_nam = 'JRA55'
    						Dataset_D_nam = 'MERRA2'
    						Dataset_E_nam = 'ERA5'


    					if(TC_dir_k == 'global_triple_collocationD'):
    						Dataset_C_raw = dframe_temp_gcell_stn_raw['CFSR'].values 
    						Dataset_D_raw = dframe_temp_gcell_stn_raw['MERRA2'].values
    						Dataset_E_raw = dframe_temp_gcell_stn_raw['ERA5'].values
    						Dataset_C_anom = dframe_temp_gcell_stn_anom['CFSR'].values 
    						Dataset_D_anom = dframe_temp_gcell_stn_anom['MERRA2'].values
    						Dataset_E_anom = dframe_temp_gcell_stn_anom['ERA5'].values

    						Dataset_C_nam = 'JRA55'
    						Dataset_D_nam = 'MERRA2'
    						Dataset_E_nam = 'ERA5'

    					if(TC_dir_k == 'global_triple_collocationE'):
    						Dataset_C_raw = dframe_temp_gcell_stn_raw['JRA55'].values 
    						Dataset_D_raw = dframe_temp_gcell_stn_raw['ERA-Interim'].values
    						Dataset_E_raw = dframe_temp_gcell_stn_raw['ERA5'].values
    						Dataset_C_anom = dframe_temp_gcell_stn_anom['JRA55'].values 
    						Dataset_D_anom = dframe_temp_gcell_stn_anom['ERA-Interim'].values
    						Dataset_E_anom = dframe_temp_gcell_stn_anom['ERA5'].values


    					Correlation_Dataframe_raw = pd.DataFrame(Dataset_S_raw, columns=['Station'])
    					Correlation_Dataframe_raw['TC Blended'] = Dataset_A_raw
    					Correlation_Dataframe_raw['Naive Blended'] = Dataset_B_raw
    					Correlation_Dataframe_raw[Dataset_C_nam] = Dataset_C_raw
    					Correlation_Dataframe_raw[Dataset_D_nam] = Dataset_D_raw
    					Correlation_Dataframe_raw[Dataset_E_nam] = Dataset_E_raw

    					Correlation_Dataframe_anom = pd.DataFrame(Dataset_S_anom, columns=['Station'])
    					Correlation_Dataframe_anom['TC Blended'] = Dataset_A_anom
    					Correlation_Dataframe_anom['Naive Blended'] = Dataset_B_anom
    					Correlation_Dataframe_anom[Dataset_C_nam] = Dataset_C_anom
    					Correlation_Dataframe_anom[Dataset_D_nam] = Dataset_D_anom
    					Correlation_Dataframe_anom[Dataset_E_nam] = Dataset_E_anom

    					Corr_raw = Correlation_Dataframe_raw.corr()
    					Corr_anom = Correlation_Dataframe_anom.corr()

########### Create Figures ##############
    					ymin = -40
    					ymax = 30
    					xmin = np.datetime64(datetime.date(1990,1,1),'Y')
    					xmax = np.datetime64(datetime.date(2020,1,1),'Y')


    					fig,axs = plt.subplots(nrows = 2, ncols = 2, sharex = 'col', sharey = 'row', figsize=(20,20))

#### Raw Temp Timeseries ####
    					ax1 = plt.subplot(221)
    					ax1.plot(Dataset_Datetime,Dataset_S_raw,label='Station',marker='o',markerfacecolor='dodgerblue',markersize=2,color='royalblue')
    					ax1.plot(Dataset_Datetime,Dataset_A_raw,label='TC Blended',marker='s',markerfacecolor='chartreuse',markersize=2,color='lawngreen')
    					ax1.plot(Dataset_Datetime,Dataset_B_raw,label='Naive Blended',marker='s',markerfacecolor='darkorchid',markersize=2,color='indigo')					
    					ax1.plot(Dataset_Datetime,Dataset_C_raw,label=Dataset_C_nam,marker='^',markerfacecolor='orangered',markersize=2,color='red')
    					ax1.plot(Dataset_Datetime,Dataset_D_raw,label=Dataset_D_nam,marker='*',markerfacecolor='gold',markersize=2,color='goldenrod')
    					ax1.plot(Dataset_Datetime,Dataset_E_raw,label=Dataset_E_nam,marker='x',markerfacecolor='dimgrey',markersize=2,color='black')
    					ax1.xaxis.set_major_locator(mdates.YearLocator(5)) #major tick every 5 years
    					ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y')) #only show the year
    					ax1.xaxis.set_minor_locator(mdates.YearLocator(1)) #minor tick every year   					
    					ax1.yaxis.set_major_locator(MultipleLocator(5)) #every 10 degrees will be a major tick
    					ax1.yaxis.set_minor_locator(MultipleLocator(2)) #every 2 degrees will be a minor tick
    					ax1.set_xlim(xmin,xmax)
    					ax1.set_ylim(ymin,ymax)
    					ax1.set_title('Absolute Temperature Timeseries')
    					ax1.set_xlabel('Date',fontweight='bold')
    					ax1.set_ylabel('Soil Temp ($^\circ$C)',fontweight='bold')
    					annotation_string = str(model1)+' Err Var: '+str(model1_err_var_rnd)
    					annotation_string += "\n"
    					annotation_string += str(model2)+' Err Var: '+str(model2_err_var_rnd)
    					annotation_string += "\n"
    					annotation_string += str(model3)+' Err Var: '+str(model3_err_var_rnd)
    					#ax1.annotate(annotation_string,xy=(0.05,0.95), xycoords='figure fraction')
    					ax1.text(0.05,0.10, annotation_string, transform=ax1.transAxes)	
    					ax1.legend(loc='best')
#### Raw Temp Correlation ####
    					ax2 = plt.subplot(222)
    					corr1 = sn.heatmap(Corr_raw,annot=True,vmin=0,vmax=1)
    					ax2.set_title('Absolute Temperature Correlation Matrix')

#### Anomaly Timeseries ####
    					ax3 = plt.subplot(223)
    					ax3.plot(Dataset_Datetime,Dataset_S_anom,label='Station',marker='o',markerfacecolor='dodgerblue',markersize=2,color='royalblue')
    					ax3.plot(Dataset_Datetime,Dataset_A_anom,label='TC Blended',marker='s',markerfacecolor='chartreuse',markersize=2,color='lawngreen')
    					ax3.plot(Dataset_Datetime,Dataset_B_anom,label='Naive Blended',marker='s',markerfacecolor='darkorchid',markersize=2,color='indigo')					
    					ax3.plot(Dataset_Datetime,Dataset_C_anom,label=Dataset_C_nam,marker='^',markerfacecolor='orangered',markersize=2,color='red')
    					ax3.plot(Dataset_Datetime,Dataset_D_anom,label=Dataset_D_nam,marker='*',markerfacecolor='gold',markersize=2,color='goldenrod')
    					ax3.plot(Dataset_Datetime,Dataset_E_anom,label=Dataset_E_nam,marker='x',markerfacecolor='dimgrey',markersize=2,color='black')
    					ax3.xaxis.set_major_locator(mdates.YearLocator(5)) #major tick every 5 years
    					ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y')) #only show the year
    					ax3.xaxis.set_minor_locator(mdates.YearLocator(1)) #minor tick every year   					
    					ax3.yaxis.set_major_locator(MultipleLocator(5)) #every 10 degrees will be a major tick
    					ax3.yaxis.set_minor_locator(MultipleLocator(1)) #every 2 degrees will be a minor tick
    					ax3.set_xlim(xmin,xmax)
    					ax3.set_ylim(-20,20)
    					ax3.set_title('Temperature Anomaly Timeseries')
    					ax3.set_xlabel('Date',fontweight='bold')
    					ax3.set_ylabel('Soil Temp Anomaly ($^\circ$C)',fontweight='bold')
    					annotation_string_a = str(model1)+' Err Var: '+str(model1_err_var_anom_rnd)
    					annotation_string_a += "\n"
    					annotation_string_a += str(model2)+' Err Var: '+str(model2_err_var_anom_rnd)
    					annotation_string_a += "\n"
    					annotation_string_a += str(model3)+' Err Var: '+str(model3_err_var_anom_rnd)
    					#ax3.annotate(annotation_string_a,xy=(0.05,0.95), xycoords='figure fraction')
    					ax3.text(0.05,0.10, annotation_string_a, transform=ax3.transAxes)
    					ax3.legend(loc='best')
#### Anomaly Correlation ####
    					ax4 = plt.subplot(224)
    					corr2 = sn.heatmap(Corr_anom,annot=True,vmin=0,vmax=1)
    					ax4.set_title('Temperature Anomaly Correlation Matrix')

    					plt.suptitle('Grid Cell: '+str(gcell_n)+', Lat: '+str(plot_lat)+'$^\circ$N, Lon :'+str(plot_lon)+'$^\circ$E')
    					plt_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/plots/Mar26_Plots/'+str(remap_type)+'_'+str(olr_l)+'_thr_'+str(thr_m)+'_anom_'+str(TC_dir_k)+'_grid_'+str(gcell_n)+'_summary.png'])
    					plt.savefig(plt_fil)
    					print(model1_nam,'raw:',model1_err_var_rnd)
    					print(model2_nam,'raw:',model2_err_var_rnd)
    					print(model3_nam,'raw:',model3_err_var_rnd)

    					print(model1_nam,'anom:',model1_err_var_anom_rnd)
    					print(model2_nam,'anom:',model2_err_var_anom_rnd)
    					print(model3_nam,'anom:',model3_err_var_anom_rnd)

    					print(plt_fil)
    					plt.close()
