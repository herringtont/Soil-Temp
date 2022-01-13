import os
import csv
import datetime
import pathlib
from pathlib import Path
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
from calendar import isleap
from dateutil.relativedelta import *
from pathlib import Path
import seaborn as sn
from calendar import isleap
from dateutil.relativedelta import *
from pathlib import Path
from matplotlib.ticker import (MultipleLocator, AutoLocator, AutoMinorLocator)
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from scipy.stats import pearsonr
from decimal import *


############################## Reanalysis Products Coverage ################
#CFSR/CFSR2 01/1979 - 09/2020
#ERA-Interim 01/1979 - 08/2019
#ERA5 01/1979 - 12/2018
#JRA-55 01/1958 - 12/2019
#MERRA2 01/1980 - 08/2020
#GLDAS 01/1948 - 07/2020

#### Reanalysis Climatology = 1981-2010
#### Collocated Dates 01/1980 - 12/2018

############################## grab sites and grid cells for each soil layer ##############################
geom_dir = "/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/"

rmp_type = ['nn','bil']


#######################################set reanalysis soil temperature layers##########################################################

###Reanalysis Soil Layers
#CFSR 4 layers (0-10 cm, 10-40 cm, 40-100 cm, 100-200 cm)
#ERA-Interim (0-7 cm, 7-28 cm, 28-100 cm, 100-289 cm)
#ERA5 (0-7 cm, 7-28 cm, 28-100 cm, 100-289 cm)
#JRA (averaged over entire soil column)
#MERRA2 (0- 9.88 cm, 9.88-29.4 cm, 29.4-67.99cm, 67.99cm-144.25cm, 144.25-294.96 cm, 294.96-1294.96 cm) 
#GLDAS 
    #Noah (0-10 cm, 10-40 cm, 40-100 cm, 100-200 cm)  ***Noah available at higher resolution - used here
    #VIC (0-10 cm, 10 - 160cm, 160-190cm)  ***Only available at 1deg resolution
    #CLSM (0-1.8cm, 1.8-4.5cm, 4.5-9.1cm, 9.1-16.6cm, 16.6-28.9cm, 28.9-49.3cm, 49.3-82.9cm, 82.9-138.3cm, 138-229.6cm, 229.6-343.3cm)  ***only available at 1deg resolution


CFSR_layer = "Soil_Temp_L1"
CFSR2_layer = "Soil_Temp_L1"
GLDAS_layer = "Soil_Temp_L1"
ERA5_layer = "Soil_Temp_L1"
ERAI_layer = "Soil_Temp_L1"
JRA_layer = "Soil_Temp"
MERRA2_layer = "Soil_Temp_L1"


################# loop through in-situ files ###############
for h in rmp_type: #loops through remap type
    rmph = h
    if(rmph == "nn"):
    	remap_type = "remapnn"
    elif(rmph == "bil"):
    	remap_type = "remapbil"    	 

	
    
################################## grab corresponding reanalysis data ##################################
    base_dir  = "".join(["/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/common_grid/remap",rmph,"/common_date/"])
    clim_dir = ''.join(["/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/common_grid/remap",rmph,"/"])
    anom_dir = ''.join(["/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/common_grid/remap",rmph,"/common_date/"])

    CFSR_fi = "".join([base_dir,"CFSR_all.nc"])
    MERRA2_fi = "".join([base_dir,"MERRA2.nc"])
    ERA5_fi = "".join([base_dir,"ERA5.nc"])
    ERAI_fi = "".join([base_dir,"ERA-Interim.nc"])
    JRA_fi = "".join([base_dir,"JRA55.nc"])
    GLDAS_fi = "".join([base_dir,"GLDAS.nc"])
    				#print(CFSR_fi)

    CFSR_anom_fi = "".join([base_dir,"CFSR_anom.nc"])
    MERRA2_anom_fi = "".join([base_dir,"MERRA2_anom.nc"])
    ERA5_anom_fi = "".join([base_dir,"ERA5_anom.nc"])
    ERAI_anom_fi = "".join([base_dir,"ERA-Interim_anom.nc"])
    JRA_anom_fi = "".join([base_dir,"JRA55_anom.nc"])
    GLDAS_anom_fi = "".join([base_dir,"GLDAS_anom.nc"])

    GLDAS_fil = xr.open_dataset(GLDAS_fi)
    JRA_fil = xr.open_dataset(JRA_fi)
    ERAI_fil = xr.open_dataset(ERAI_fi)
    ERA5_fil = xr.open_dataset(ERA5_fi)
    MERRA2_fil = xr.open_dataset(MERRA2_fi)
    CFSR_fil = xr.open_dataset(CFSR_fi) #open NetCDF file with xarray

    GLDAS_anom_fil = xr.open_dataset(GLDAS_anom_fi)
    JRA_anom_fil = xr.open_dataset(JRA_anom_fi)
    ERAI_anom_fil = xr.open_dataset(ERAI_anom_fi)
    ERA5_anom_fil = xr.open_dataset(ERA5_anom_fi)
    MERRA2_anom_fil = xr.open_dataset(MERRA2_anom_fi)
    CFSR_anom_fil = xr.open_dataset(CFSR_anom_fi)
																				
########### extract soil temperatures and convert to celsius #######
    GLDAS_stemp = GLDAS_fil[GLDAS_layer] -273.15
    JRA_stemp = JRA_fil[JRA_layer] - 273.15
    ERAI_stemp = ERAI_fil[ERAI_layer] - 273.15
    ERA5_stemp = ERA5_fil[ERA5_layer] - 273.15
    MERRA2_stemp = MERRA2_fil[MERRA2_layer] - 273.15 #convert from Kelvin to Celsius
    CFSR_stemp = CFSR_fil[CFSR_layer] - 273.15  #convert from Kelvin to Celsius

    GLDAS_anom = GLDAS_anom_fil[GLDAS_layer]
    JRA_anom = JRA_anom_fil[JRA_layer]    				
    ERAI_anom = ERAI_anom_fil[ERAI_layer]
    ERA5_anom = ERA5_anom_fil[ERA5_layer]
    MERRA2_anom = MERRA2_anom_fil[MERRA2_layer]
    CFSR_anom = CFSR_anom_fil[CFSR_layer]				




########################### ABSOLUTE TEMPERATURES ############################
    x = JRA_stemp
    y = ERA5_stemp
    z = ERAI_stemp
    											
############### APPROACH 1 (SCALING) ############

    x_df = x - xr.DataArray(x).mean(dim='time',keep_attrs=True)    ####This is the timeseries mean 				
    y_df = y - xr.DataArray(y).mean(dim='time',keep_attrs=True)    ####This is the timeseries mean
    z_df = z - xr.DataArray(z).mean(dim='time',keep_attrs=True)   ####This is the timeseries mean				

    xdf_zdf = x_df*z_df
    xdf_ydf = x_df*y_df
    ydf_zdf = y_df*z_df
    zdf_ydf = z_df*y_df
				
    beta_ystar = xr.DataArray(xdf_zdf).mean(dim='time',keep_attrs=True)/xr.DataArray(ydf_zdf).mean(dim='time',keep_attrs=True) 
    beta_zstar = xr.DataArray(xdf_ydf).mean(dim='time',keep_attrs=True)/xr.DataArray(zdf_ydf).mean(dim='time',keep_attrs=True) 

    scaling_factor_Y = 1/beta_ystar ##rescaling factor for Y
    scaling_factor_Z = 1/beta_zstar ##rescaling factor for Z

    scaling_factor_Y = scaling_factor_Y.rename('ERA5_SF')
    scaling_factor_Z = scaling_factor_Z.rename('ERA-Interim_SF')

    x_bar = xr.DataArray(x).mean(dim='time',keep_attrs=True)
    y_bar = xr.DataArray(y).mean(dim='time',keep_attrs=True)
    z_bar = xr.DataArray(z).mean(dim='time',keep_attrs=True)
    
    y_diff = y-y_bar
    z_diff = z-z_bar

    y_rescaled = (beta_ystar*y_diff)+x_bar
    z_rescaled = (beta_zstar*z_diff)+x_bar   				

    x_scl_inside = (x-y_rescaled)*(x-z_rescaled)
    y_scl_inside = (y_rescaled-x)*(y_rescaled-z_rescaled)
    z_scl_inside = (z_rescaled-x)*(z_rescaled-y_rescaled)
    	
    err_varx_scaled = xr.DataArray(x_scl_inside).mean(dim='time',keep_attrs=True) ## error variance of x using difference notation
    err_vary_scaled = xr.DataArray(y_scl_inside).mean(dim='time',keep_attrs=True) ## error variance of y using difference notation
    err_varz_scaled = xr.DataArray(z_scl_inside).mean(dim='time',keep_attrs=True) ## error variance of z using difference notation

    err_varx_scaled = err_varx_scaled.rename('err_var')
    err_vary_scaled = err_varx_scaled.rename('err_var')
    err_varz_scaled = err_varz_scaled.rename('err_var')
				   		

################ APPROACH 2 (COVARIANCES) ##############
    x_std = xr.DataArray(x).std(dim='time',keep_attrs=True)
    y_std = xr.DataArray(y).std(dim='time',keep_attrs=True)
    z_std = xr.DataArray(z).std(dim='time',keep_attrs=True)

    x_var = xr.DataArray(x).var(dim='time',keep_attrs=True)
    y_var = xr.DataArray(y).var(dim='time',keep_attrs=True)
    z_var = xr.DataArray(z).var(dim='time',keep_attrs=True)

    x_var = x_var.rename('dataset_var')
    y_var = y_var.rename('dataset_var')
    z_var = z_var.rename('dataset_var')

    signal_varx = (xr.cov(x,y,dim="time")*xr.cov(x,z,dim="time"))/xr.cov(y,z,dim="time") ###Signal to Noise Ratio of X (soil temperature sensitivity of the data set) 
    signal_vary = (xr.cov(y,x,dim="time")*xr.cov(y,z,dim="time"))/xr.cov(x,z,dim="time") ###Signal to Noise Ratio of Y (soil temperature sensitivity of the data set) 
    signal_varz = (xr.cov(z,x,dim="time")*xr.cov(z,y,dim="time"))/xr.cov(x,y,dim="time") ###Signal to Noise Ratio of Z (soil temperature sensitivity of the data set)

    signal_varx = signal_varx.rename('signal_var')
    signal_vary = signal_vary.rename('signal_var')
    signal_varz = signal_varz.rename('signal_var')

    err_varx = xr.DataArray(x).var(dim='time',keep_attrs=True) - signal_varx ##Error variance of dataset X using covariance notation
    err_vary = xr.DataArray(y).var(dim='time',keep_attrs=True) - signal_vary ##Error variance of dataset Y using covariance notation
    err_varz = xr.DataArray(z).var(dim='time',keep_attrs=True) - signal_varz ##Error variance of dataset Z using covariance notation

    err_varx = err_varx.rename('err_var_cov')
    err_vary = err_vary.rename('err_var_cov')
    err_varz = err_varz.rename('err_var_cov')

    snrx = signal_varx/err_varx    				
    snry = signal_vary/err_vary
    snrz = signal_varz/err_varz 

    snrx = snrx.rename('SNR')
    snry = snry.rename('SNR')
    snrz = snrz.rename('SNR')
				
    nsrx = err_varx/signal_varx ##Noise to Signal Ratio of dataset x
    nsry = err_vary/signal_vary ##Noise to Signal Ratio of dataset y
    nsrz = err_varz/signal_varz ##Noise to Signal Ratio of dataset z

    Rxy = 1/xr.ufuncs.sqrt((1+nsrx)*(1+nsry)) ##Pearson correlation between dataset X and dataset Y
    Ryz = 1/xr.ufuncs.sqrt((1+nsry)*(1+nsrz)) ##Pearson correlation between dataset Y and dataset Z
    Rxz = 1/xr.ufuncs.sqrt((1+nsrx)*(1+nsrz)) ##Pearson correlation between dataset X and dataset Z

    beta_ystar_cov = xr.cov(x,z,dim="time")/xr.cov(y,z,dim="time")
    beta_zstar_cov = xr.cov(x,y,dim="time")/xr.cov(z,y,dim="time")
    scaling_factor_Y_cov = beta_ystar_cov
    scaling_factor_Z_cov = beta_zstar_cov

    scaling_factor_Y_cov = scaling_factor_Y_cov.rename('SF_cov')
    scaling_factor_Z_cov = scaling_factor_Z_cov.rename('SF_cov')


    y_beta_scaled = y * beta_ystar_cov
    z_beta_scaled = z * beta_zstar_cov

    y_rescaled_cov = (beta_ystar_cov*(y - y_bar))+x_bar
    z_rescaled_cov = (beta_zstar_cov*(z - z_bar))+x_bar


    Rx = xr.ufuncs.sqrt(snrx/(1+snrx)) ##Correlation between Dataset X and true soil temp 
    Ry = xr.ufuncs.sqrt(snry/(1+snry)) ##Correlation between Dataset Y and true soil temp 
    Rz = xr.ufuncs.sqrt(snrz/(1+snrz)) ##Correlation between Dataset Y and true soil temp 

    Rx = Rx.rename('R')
    Ry = Ry.rename('R')
    Rz = Rz.rename('R')			
    						#print(Rx, Ry, Rz)

    						#print("fMSE:")
    fMSE_x = 1/(1+snrx)
    fMSE_y = 1/(1+snry)
    fMSE_z = 1/(1+snrz)
    
    fMSE_x= fMSE_x.rename('fMSE')
    fMSE_y = fMSE_y.rename('fMSE')
    fMSE_z = fMSE_z.rename('fMSE')

    x_bar.rename('xbar')
    y_bar.rename('ybar')
    z_bar.rename('zbar')

    Path("/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/raw_temp/"+str(remap_type)+"/blended_products/").mkdir(parents=True,exist_ok=True)
            
    x_bar.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/raw_temp/"+str(remap_type)+"/"+str(remap_type)+"_JRA55_xbar.nc",mode='w')
    y_bar.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/raw_temp/"+str(remap_type)+"/"+str(remap_type)+"_ERA5_ybar.nc",mode='w')
    z_bar.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/raw_temp/"+str(remap_type)+"/"+str(remap_type)+"_ERA-Interim_zbar.nc",mode='w')    

    x_var.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/raw_temp/"+str(remap_type)+"/"+str(remap_type)+"_JRA55_var.nc",mode='w')
    y_var.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/raw_temp/"+str(remap_type)+"/"+str(remap_type)+"_ERA5_var.nc",mode='w')
    z_var.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/raw_temp/"+str(remap_type)+"/"+str(remap_type)+"_ERA-Interim_var.nc",mode='w')
    
    err_varx_scaled.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/raw_temp/"+str(remap_type)+"/"+str(remap_type)+"_JRA55_err_var.nc",mode='w')
    err_vary_scaled.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/raw_temp/"+str(remap_type)+"/"+str(remap_type)+"_ERA5_err_var.nc",mode='w')
    err_varz_scaled.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/raw_temp/"+str(remap_type)+"/"+str(remap_type)+"_ERA-Interim_err_var.nc",mode='w')

    err_varx.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/raw_temp/"+str(remap_type)+"/"+str(remap_type)+"_JRA55_err_var_cov.nc",mode='w')
    err_vary.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/raw_temp/"+str(remap_type)+"/"+str(remap_type)+"_ERA5_err_var_cov.nc",mode='w')
    err_varz.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/raw_temp/"+str(remap_type)+"/"+str(remap_type)+"_ERA-Interim_err_var_cov.nc",mode='w')

    signal_varx.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/raw_temp/"+str(remap_type)+"/"+str(remap_type)+"_JRA55_signal_var.nc",mode='w')
    signal_vary.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/raw_temp/"+str(remap_type)+"/"+str(remap_type)+"_ERA5_signal_var.nc",mode='w')
    signal_varz.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/raw_temp/"+str(remap_type)+"/"+str(remap_type)+"_ERA-Interim_signal_var.nc",mode='w')

    scaling_factor_Y.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/raw_temp/"+str(remap_type)+"/"+str(remap_type)+"_ERA5_SF.nc",mode='w')
    scaling_factor_Z.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/raw_temp/"+str(remap_type)+"/"+str(remap_type)+"_ERA-Interim_SF.nc",mode='w')

    scaling_factor_Y_cov.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/raw_temp/"+str(remap_type)+"/"+str(remap_type)+"_ERA5_SF_cov.nc",mode='w')
    scaling_factor_Z_cov.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/raw_temp/"+str(remap_type)+"/"+str(remap_type)+"_ERA-Interim_SF_cov.nc",mode='w')
    
    snrx.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/raw_temp/"+str(remap_type)+"/"+str(remap_type)+"_JRA55_SNR.nc",mode='w')
    snry.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/raw_temp/"+str(remap_type)+"/"+str(remap_type)+"_ERA5_SNR.nc",mode='w')
    snrz.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/raw_temp/"+str(remap_type)+"/"+str(remap_type)+"_ERA-Interim_SNR.nc",mode='w')
    
    Rx.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/raw_temp/"+str(remap_type)+"/"+str(remap_type)+"_JRA55_R.nc",mode='w')
    Ry.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/raw_temp/"+str(remap_type)+"/"+str(remap_type)+"_ERA5_R.nc",mode='w')
    Rz.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/raw_temp/"+str(remap_type)+"/"+str(remap_type)+"_ERA-Interim_R.nc",mode='w')

    fMSE_x.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/raw_temp/"+str(remap_type)+"/"+str(remap_type)+"_JRA55_fMSE.nc",mode='w')
    fMSE_y.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/raw_temp/"+str(remap_type)+"/"+str(remap_type)+"_ERA5_fMSE.nc",mode='w')
    fMSE_z.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/raw_temp/"+str(remap_type)+"/"+str(remap_type)+"_ERA-Interim_fMSE.nc",mode='w')


######################### Scale individual datasets ############

    x_scaled_raw = x  ## JRA55 doesn't need to be rescaled here because it is the reference
    y_scaled_raw = (scaling_factor_Y_cov*(y-y_bar))+x_bar
    z_scaled_raw = (scaling_factor_Z_cov*(z-z_bar))+x_bar
    
    #y_scaled_raw.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/raw_temp/"+str(remap_type)+"/blended_products/"+str(remap_type)+"_ERA5_scaled_raw.nc",mode='w')
    #z_scaled_raw.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/raw_temp/"+str(remap_type)+"/blended_products/"+str(remap_type)+"_ERA-Interim_scaled_raw.nc",mode='w')
				  			
####################### Calculate weighted averages and create blended product ###############

##### calculate product weights based on SNR values ####

##### Absolute (Raw) Temps ####
    wght_denom_r = snrx + snry + snrz
    x_wght_r = snrx/wght_denom_r
    y_wght_r = snry/wght_denom_r
    z_wght_r = snrz/wght_denom_r

    x_wght_r = x_wght_r.rename('TC_weight')
    y_wght_r = y_wght_r.rename('TC_weight')
    z_wght_r = z_wght_r.rename('TC_weight')

    sum_wght_r = x_wght_r + y_wght_r + z_wght_r
    sum_wght_r = sum_wght_r.rename('TC_weight')

    sum_wght_r.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/raw_temp/"+str(remap_type)+"/blended_products/"+str(remap_type)+"_SUM_weights_raw.nc",mode='w')     
    x_wght_r.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/raw_temp/"+str(remap_type)+"/blended_products/"+str(remap_type)+"_JRA55_weights_raw.nc",mode='w')
    y_wght_r.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/raw_temp/"+str(remap_type)+"/blended_products/"+str(remap_type)+"_ERA5_weights_raw.nc",mode='w')
    z_wght_r.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/raw_temp/"+str(remap_type)+"/blended_products/"+str(remap_type)+"_ERA-Interim_weights_raw.nc",mode='w')
            
##### create TC blended product ####
    x_weighted = x_scaled_raw*x_wght_r
    y_weighted = y_scaled_raw*y_wght_r
    z_weighted = z_scaled_raw*z_wght_r
    
    TC_blended = x_weighted + y_weighted + z_weighted
    TC_blended = TC_blended.rename('TC_blended_stemp')

    TC_blended.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/raw_temp/"+str(remap_type)+"/blended_products/"+str(remap_type)+"_TC_blended_raw.nc",mode='w')
##### create naive blended product ####

    naive_blended = (x+y+z)/3
    naive_blended = naive_blended.rename('naive_blended_stemp')

    naive_blended.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/raw_temp/"+str(remap_type)+"/blended_products/"+str(remap_type)+"_naive_blended_raw.nc",mode='w')




########################### ANOMALIES ############################
    x = JRA_anom
    y = ERA5_anom
    z = ERAI_anom
    											
############### APPROACH 1 (SCALING) ############

    x_df = x - xr.DataArray(x).mean(dim='time',keep_attrs=True)    ####This is the timeseries mean 				
    y_df = y - xr.DataArray(y).mean(dim='time',keep_attrs=True)    ####This is the timeseries mean
    z_df = z - xr.DataArray(z).mean(dim='time',keep_attrs=True)   ####This is the timeseries mean				

    xdf_zdf = x_df*z_df
    xdf_ydf = x_df*y_df
    ydf_zdf = y_df*z_df
    zdf_ydf = z_df*y_df
				
    beta_ystar = xr.DataArray(xdf_zdf).mean(dim='time',keep_attrs=True)/xr.DataArray(ydf_zdf).mean(dim='time',keep_attrs=True) 
    beta_zstar = xr.DataArray(xdf_ydf).mean(dim='time',keep_attrs=True)/xr.DataArray(zdf_ydf).mean(dim='time',keep_attrs=True) 

    scaling_factor_Y = 1/beta_ystar ##rescaling factor for Y
    scaling_factor_Z = 1/beta_zstar ##rescaling factor for Z

    scaling_factor_Y = scaling_factor_Y.rename('SF')
    scaling_factor_Z = scaling_factor_Z.rename('SF')

    x_bar = xr.DataArray(x).mean(dim='time',keep_attrs=True)
    y_bar = xr.DataArray(y).mean(dim='time',keep_attrs=True)
    z_bar = xr.DataArray(z).mean(dim='time',keep_attrs=True)
    
    y_diff = y-y_bar
    z_diff = z-z_bar

    y_rescaled = (beta_ystar*y_diff)+x_bar
    z_rescaled = (beta_zstar*z_diff)+x_bar   				

    x_scl_inside = (x-y_rescaled)*(x-z_rescaled)
    y_scl_inside = (y_rescaled-x)*(y_rescaled-z_rescaled)
    z_scl_inside = (z_rescaled-x)*(z_rescaled-y_rescaled)
    	
    err_varx_scaled = xr.DataArray(x_scl_inside).mean(dim='time',keep_attrs=True) ## error variance of x using difference notation
    err_vary_scaled = xr.DataArray(y_scl_inside).mean(dim='time',keep_attrs=True) ## error variance of y using difference notation
    err_varz_scaled = xr.DataArray(z_scl_inside).mean(dim='time',keep_attrs=True) ## error variance of z using difference notation

    err_varx_scaled = err_varx_scaled.rename('err_var')
    err_vary_scaled = err_varx_scaled.rename('err_var')
    err_varz_scaled = err_varz_scaled.rename('err_var')
				   		

################ APPROACH 2 (COVARIANCES) ##############
    x_std = xr.DataArray(x).std(dim='time',keep_attrs=True)
    y_std = xr.DataArray(y).std(dim='time',keep_attrs=True)
    z_std = xr.DataArray(z).std(dim='time',keep_attrs=True)

    x_var = xr.DataArray(x).var(dim='time',keep_attrs=True)
    y_var = xr.DataArray(y).var(dim='time',keep_attrs=True)
    z_var = xr.DataArray(z).var(dim='time',keep_attrs=True)

    x_var = x_var.rename('dataset_var')
    y_var = y_var.rename('dataset_var')
    z_var = z_var.rename('dataset_var')

    signal_varx = (xr.cov(x,y,dim="time")*xr.cov(x,z,dim="time"))/xr.cov(y,z,dim="time") ###Signal to Noise Ratio of X (soil temperature sensitivity of the data set) 
    signal_vary = (xr.cov(y,x,dim="time")*xr.cov(y,z,dim="time"))/xr.cov(x,z,dim="time") ###Signal to Noise Ratio of Y (soil temperature sensitivity of the data set) 
    signal_varz = (xr.cov(z,x,dim="time")*xr.cov(z,y,dim="time"))/xr.cov(x,y,dim="time") ###Signal to Noise Ratio of Z (soil temperature sensitivity of the data set)

    signal_varx = signal_varx.rename('signal_var')
    signal_vary = signal_vary.rename('signal_var')
    signal_varz = signal_varz.rename('signal_var')

    err_varx = xr.DataArray(x).var(dim='time',keep_attrs=True) - signal_varx ##Error variance of dataset X using covariance notation
    err_vary = xr.DataArray(y).var(dim='time',keep_attrs=True) - signal_vary ##Error variance of dataset Y using covariance notation
    err_varz = xr.DataArray(z).var(dim='time',keep_attrs=True) - signal_varz ##Error variance of dataset Z using covariance notation

    err_varx = err_varx.rename('err_var_cov')
    err_vary = err_vary.rename('err_var_cov')
    err_varz = err_varz.rename('err_var_cov')

    snrx = signal_varx/err_varx    				
    snry = signal_vary/err_vary
    snrz = signal_varz/err_varz 

    snrx = snrx.rename('SNR')
    snry = snry.rename('SNR')
    snrz = snrz.rename('SNR')
				
    nsrx = err_varx/signal_varx ##Noise to Signal Ratio of dataset x
    nsry = err_vary/signal_vary ##Noise to Signal Ratio of dataset y
    nsrz = err_varz/signal_varz ##Noise to Signal Ratio of dataset z

    Rxy = 1/xr.ufuncs.sqrt((1+nsrx)*(1+nsry)) ##Pearson correlation between dataset X and dataset Y
    Ryz = 1/xr.ufuncs.sqrt((1+nsry)*(1+nsrz)) ##Pearson correlation between dataset Y and dataset Z
    Rxz = 1/xr.ufuncs.sqrt((1+nsrx)*(1+nsrz)) ##Pearson correlation between dataset X and dataset Z

    beta_ystar_cov = xr.cov(x,z,dim="time")/xr.cov(y,z,dim="time")
    beta_zstar_cov = xr.cov(x,y,dim="time")/xr.cov(z,y,dim="time")
    scaling_factor_Y_cov = beta_ystar_cov
    scaling_factor_Z_cov = beta_zstar_cov

    scaling_factor_Y_cov = scaling_factor_Y_cov.rename('SF_cov')
    scaling_factor_Z_cov = scaling_factor_Z_cov.rename('SF_cov')


    y_beta_scaled = y * beta_ystar_cov
    z_beta_scaled = z * beta_zstar_cov

    y_rescaled_cov = (beta_ystar_cov*(y - y_bar))+x_bar
    z_rescaled_cov = (beta_zstar_cov*(z - z_bar))+x_bar


    Rx = xr.ufuncs.sqrt(snrx/(1+snrx)) ##Correlation between Dataset X and true soil temp 
    Ry = xr.ufuncs.sqrt(snry/(1+snry)) ##Correlation between Dataset Y and true soil temp 
    Rz = xr.ufuncs.sqrt(snrz/(1+snrz)) ##Correlation between Dataset Y and true soil temp 

    Rx = Rx.rename('R')
    Ry = Ry.rename('R')
    Rz = Rz.rename('R')			
    						#print(Rx, Ry, Rz)

    						#print("fMSE:")
    fMSE_x = 1/(1+snrx)
    fMSE_y = 1/(1+snry)
    fMSE_z = 1/(1+snrz)
    
    fMSE_x= fMSE_x.rename('fMSE')
    fMSE_y = fMSE_y.rename('fMSE')
    fMSE_z = fMSE_z.rename('fMSE')

    x_bar.rename('xbar')
    y_bar.rename('ybar')
    z_bar.rename('zbar')

    Path("/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/anom/"+str(remap_type)+"/blended_products/").mkdir(parents=True,exist_ok=True)
            
    x_bar.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/anom/"+str(remap_type)+"/"+str(remap_type)+"_JRA55_xbar.nc",mode='w')
    y_bar.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/anom/"+str(remap_type)+"/"+str(remap_type)+"_ERA5_ybar.nc",mode='w')
    z_bar.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/anom/"+str(remap_type)+"/"+str(remap_type)+"_ERA-Interim_zbar.nc",mode='w') 

    x_var.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/anom/"+str(remap_type)+"/"+str(remap_type)+"_JRA55_var.nc",mode='w')
    y_var.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/anom/"+str(remap_type)+"/"+str(remap_type)+"_ERA5_var.nc",mode='w')
    z_var.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/anom/"+str(remap_type)+"/"+str(remap_type)+"_ERA-Interim_var.nc",mode='w')
    
    err_varx_scaled.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/anom/"+str(remap_type)+"/"+str(remap_type)+"_JRA55_err_var.nc",mode='w')
    err_vary_scaled.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/anom/"+str(remap_type)+"/"+str(remap_type)+"_ERA5_err_var.nc",mode='w')
    err_varz_scaled.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/anom/"+str(remap_type)+"/"+str(remap_type)+"_ERA-Interim_err_var.nc",mode='w')

    signal_varx.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/anom/"+str(remap_type)+"/"+str(remap_type)+"_JRA55_signal_var.nc",mode='w')
    signal_vary.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/anom/"+str(remap_type)+"/"+str(remap_type)+"_ERA5_signal_var.nc",mode='w')
    signal_varz.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/anom/"+str(remap_type)+"/"+str(remap_type)+"_ERA-Interim_signal_var.nc",mode='w')

    err_varx.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/anom/"+str(remap_type)+"/"+str(remap_type)+"_JRA55_err_var_cov.nc",mode='w')
    err_vary.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/anom/"+str(remap_type)+"/"+str(remap_type)+"_ERA5_err_var_cov.nc",mode='w')
    err_varz.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/anom/"+str(remap_type)+"/"+str(remap_type)+"_ERA-Interim_err_var_cov.nc",mode='w')

    scaling_factor_Y.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/anom/"+str(remap_type)+"/"+str(remap_type)+"_ERA5_SF.nc",mode='w')
    scaling_factor_Z.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/anom/"+str(remap_type)+"/"+str(remap_type)+"_ERA-Interim_SF.nc",mode='w')

    scaling_factor_Y_cov.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/anom/"+str(remap_type)+"/"+str(remap_type)+"_ERA5_SF_cov.nc",mode='w')
    scaling_factor_Z_cov.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/anom/"+str(remap_type)+"/"+str(remap_type)+"_ERA-Interim_SF_cov.nc",mode='w')
    
    snrx.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/anom/"+str(remap_type)+"/"+str(remap_type)+"_JRA55_SNR.nc",mode='w')
    snry.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/anom/"+str(remap_type)+"/"+str(remap_type)+"_ERA5_SNR.nc",mode='w')
    snrz.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/anom/"+str(remap_type)+"/"+str(remap_type)+"_ERA-Interim_SNR.nc",mode='w')
    
    Rx.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/anom/"+str(remap_type)+"/"+str(remap_type)+"_JRA55_R.nc",mode='w')
    Ry.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/anom/"+str(remap_type)+"/"+str(remap_type)+"_ERA5_R.nc",mode='w')
    Rz.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/anom/"+str(remap_type)+"/"+str(remap_type)+"_ERA-Interim_R.nc",mode='w')

    fMSE_x.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/anom/"+str(remap_type)+"/"+str(remap_type)+"_JRA55_fMSE.nc",mode='w')
    fMSE_y.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/anom/"+str(remap_type)+"/"+str(remap_type)+"_ERA5_fMSE.nc",mode='w')
    fMSE_z.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/anom/"+str(remap_type)+"/"+str(remap_type)+"_ERA-Interim_fMSE.nc",mode='w')


######################### Scale individual datasets ############

    x_scaled_anom = x  ## JRA55 doesn't need to be rescaled here because it is the reference
    y_scaled_anom = (scaling_factor_Y_cov*(y-y_bar))+x_bar
    z_scaled_anom = (scaling_factor_Z_cov*(z-z_bar))+x_bar

    y_scaled_anom.rename('Soil_Temp_L1')
    z_scaled_anom.rename('Soil_Temp_L1')
    
    #y_scaled_anom.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/anom/"+str(remap_type)+"/blended_products/"+str(remap_type)+"_ERA5_scaled_anom.nc",mode='w')
    #z_scaled_anom.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/anom/"+str(remap_type)+"/blended_products/"+str(remap_type)+"_ERA-Interim_scaled_anom.nc",mode='w')
				  			
####################### Calculate weighted averages and create blended product ###############

##### calculate product weights based on SNR values ####

##### Anomalies ####
    wght_denom_a = snrx + snry + snrz
    x_wght_a = snrx/wght_denom_a
    y_wght_a = snry/wght_denom_a
    z_wght_a = snrz/wght_denom_a

    x_wght_a = x_wght_a.rename('TC_weight')
    y_wght_a = y_wght_a.rename('TC_weight')
    z_wght_a = z_wght_a.rename('TC_weight')
    
    sum_wght_a = x_wght_a + y_wght_a + z_wght_a
    sum_wght_a = sum_wght_a.rename('TC_weight')

    sum_wght_a.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/anom/"+str(remap_type)+"/blended_products/"+str(remap_type)+"_SUM_weights_anom.nc",mode='w')    
    x_wght_a.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/anom/"+str(remap_type)+"/blended_products/"+str(remap_type)+"_JRA55_weights_anom.nc",mode='w')
    y_wght_a.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/anom/"+str(remap_type)+"/blended_products/"+str(remap_type)+"_ERA5_weights_anom.nc",mode='w')
    z_wght_a.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/anom/"+str(remap_type)+"/blended_products/"+str(remap_type)+"_ERA-Interim_weights_anom.nc",mode='w')
            
##### create TC blended product ####
    x_weighted = x_scaled_anom*x_wght_a
    y_weighted = y_scaled_anom*y_wght_a
    z_weighted = z_scaled_anom*z_wght_a
    
    TC_blended = x_weighted + y_weighted + z_weighted
    TC_blended = TC_blended.rename('TC_blended_anom')

    TC_blended.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/anom/"+str(remap_type)+"/blended_products/"+str(remap_type)+"_TC_blended_anom.nc",mode='w')
    
##### create naive blended product ####

    naive_blended = (x+y+z)/3
    naive_blended = naive_blended.rename('naive_blended_anom')

    naive_blended.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/anom/"+str(remap_type)+"/blended_products/"+str(remap_type)+"_naive_blended_anom.nc",mode='w')











