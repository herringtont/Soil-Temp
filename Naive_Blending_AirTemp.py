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


CFSR_layer = "Air_Temp"
CFSR2_layer = "Air_Temp"
GLDAS_layer = "Air_Temp"
ERA5_layer = "Air_Temp"
ERAI_layer = "Air_Temp"
JRA_layer = "Air_Temp"
MERRA2_layer = "Air_Temp"


################# loop through remapping types ###############
for h in rmp_type: #loops through remap type
    rmph = h
    if(rmph == "nn"):
    	remap_type = "remapnn"
    elif(rmph == "bil"):
    	remap_type = "remapbil"   


################################## grab corresponding reanalysis data ##################################
    base_dir = "".join(['/mnt/data/users/herringtont/soil_temp/reanalysis/2m_AirTemp/rename/land_only/common_grid/remap',rmph,'/'])

    CFSR_fi = "".join([base_dir,"CFSR_all_2m_air.nc"])
    MERRA2_fi = "".join([base_dir,"MERRA2_2m_air.nc"])
    ERA5_fi = "".join([base_dir,"ERA5_2m_air.nc"])
    ERAI_fi = "".join([base_dir,"ERA-Interim_2m_air.nc"])
    JRA_fi = "".join([base_dir,"JRA55_2m_air.nc"])
    GLDAS_fi = "".join([base_dir,"GLDAS_2m_air.nc"])


    GLDAS_fil = xr.open_dataset(GLDAS_fi)
    JRA_fil = xr.open_dataset(JRA_fi)
    ERAI_fil = xr.open_dataset(ERAI_fi)
    ERA5_fil = xr.open_dataset(ERA5_fi)
    MERRA2_fil = xr.open_dataset(MERRA2_fi)
    CFSR_fil = xr.open_dataset(CFSR_fi) #open NetCDF file with xarray

					    																				
########### extract air temperatures and convert to celsius #######
    GLDAS_temp = GLDAS_fil[GLDAS_layer] -273.15
    JRA_temp = JRA_fil[JRA_layer] - 273.15
    ERAI_temp = ERAI_fil[ERAI_layer] - 273.15
    ERA5_temp = ERA5_fil[ERA5_layer] - 273.15
    MERRA2_temp = MERRA2_fil[MERRA2_layer] - 273.15 #convert from Kelvin to Celsius
    CFSR_temp = CFSR_fil[CFSR_layer] - 273.15  #convert from Kelvin to Celsius


########################### ABSOLUTE TEMPERATURES ############################
    Naive_temp = (CFSR_temp + ERAI_temp + ERA5_temp + JRA_temp + MERRA2_temp + GLDAS_temp)/6

    #print(CFSR_temp,ERAI_temp,ERA5_temp,JRA_temp,MERRA2_temp,GLDAS_temp)
    
    Naive_temp = Naive_temp.rename('Air_Temp')
    Path("/mnt/data/users/herringtont/soil_temp/reanalysis/2m_AirTemp/rename/land_only/common_grid/"+str(remap_type)+"/").mkdir(parents=True,exist_ok=True)

    Naive_temp.to_netcdf(path="/mnt/data/users/herringtont/soil_temp/reanalysis/2m_AirTemp/rename/land_only/common_grid/"+str(remap_type)+"/"+str(remap_type)+"_Naive_2m_air.nc",mode='w')







