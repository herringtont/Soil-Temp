import os
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
import shapely
import cartopy.crs as ccrs
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


CFSR_layer = "Soil_Temp_TOP30"
CFSR2_layer = "Soil_Temp_TOP30"
GLDAS_layer = "Soil_Temp_TOP30"
ERA5_layer = "Soil_Temp_TOP30"
ERAI_layer = "Soil_Temp_TOP30"
JRA_layer = "Soil_Temp"
MERRA2_layer = "Soil_Temp_TOP30"


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

########### extract lat/lon ##########

    GLDAS_lat = GLDAS_fil.coords['lat']
    GLDAS_lon = GLDAS_fil.coords['lon']
    JRA_lat = JRA_fil.coords['lat']
    JRA_lon = JRA_fil.coords['lon']
    ERAI_lat = ERAI_fil.coords['lat']
    ERAI_lon = ERAI_fil.coords['lon']
    ERA5_lat = ERA5_fil.coords['lat']
    ERA5_lon = ERA5_fil.coords['lon']
    MERRA2_lat = MERRA2_fil.coords['lat']
    MERRA2_lon = MERRA2_fil.coords['lon']
    CFSR_lat = CFSR_fil.coords['lat']
    CFSR_lon = CFSR_fil.coords['lon']					    																				
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
    Naive_stemp = (GLDAS_stemp + JRA_stemp + ERAI_stemp + ERA5_stemp + MERRA2_stemp + CFSR_stemp)/6
    Naive_stemp = Naive_stemp.rename('Soil_Temp_TOP30')
    Path("/mnt/data/users/herringtont/soil_temp/naive_blended_product/simple_average/raw_temp/"+str(remap_type)+"/").mkdir(parents=True,exist_ok=True)

    Naive_stemp.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/naive_blended_product/simple_average/raw_temp/"+str(remap_type)+"/"+str(remap_type)+"_Naive_stemp_TOP30cm.nc",mode='w')


########################### ANOMALIES ################################
    Naive_anom = (GLDAS_anom + JRA_anom + ERAI_anom + ERA5_anom + MERRA2_anom + CFSR_anom)/6   
    Naive_anom = Naive_anom.rename('Soil_Temp_TOP30')
    Path("/mnt/data/users/herringtont/soil_temp/naive_blended_product/simple_average/anom/"+str(remap_type)+"/").mkdir(parents=True,exist_ok=True)

    Naive_anom.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/naive_blended_product/simple_average/anom/"+str(remap_type)+"/"+str(remap_type)+"_Naive_anom_TOP30cm.nc",mode='w')
