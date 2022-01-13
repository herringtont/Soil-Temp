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
from calendar import isleap
from dateutil.relativedelta import *

################################# create master arrays for in-situ, models, and their anomalies ######################################
grid_master = []
station_master = []
CFSR_master = []
ERAI_master = []
ERA5_master = []
JRA_master = []
MERRA2_master = []
GLDAS_master = []

grid_master_a = []
station_master_a = []
CFSR_master_a = []
ERAI_master_a = []
ERA5_master_a = []
JRA_master_a = []
MERRA2_master_a = []
GLDAS_master_a = []


############################## grab sites and grid cells for each soil layer ##############################
geom_dir = "/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/"

olr = ['outliers','zscore','IQR']
lyr = ['0_9.9','10_29.9','30_99.9','100_299.9','300_deeper']
thr = ['0','25','50','75','100']
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


##### CFSR, CFSR 2 and GLDAS Noah ####
    if (0 <= sdep < 10):
    	CFSR_layer = "Soil_Temp_L1"
    	CFSR2_layer = "Soil_Temp_L1"
    	GLDAS_layer = "Soil_Temp_L1"
    elif (10 <= sdep < 40):
    	CFSR_layer = "Soil_Temp_L2"
    	CFSR2_layer = "Soil_Temp_L2"
    	GLDAS_layer = "Soil_Temp_L2"
    elif (40 <= sdep < 100):
    	CFSR_layer = "Soil_Temp_L3"
    	CFSR2_layer = "Soil_Temp_L3"
    	GLDAS_layer = "Soil_Temp_L3"
    elif (sdep >= 100):
    	CFSR_layer = "Soil_Temp_L4"
    	CFSR2_layer = "Soil_Temp_L4"
    	GLDAS_layer = "Soil_Temp_L4"

##### ERA-Interim and ERA5 ####
    if (0 <= sdep < 7):
    	ERAI_layer = "Soil_Temp_L1"
    	ERA5_layer = "Soil_Temp_L1"
    elif (7 <= sdep < 28):
    	ERAI_layer = "Soil_Temp_L2"
    	ERA5_layer = "Soil_Temp_L2"
    elif (28 <= sdep < 100):
    	ERAI_layer = "Soil_Temp_L3"
    	ERA5_layer = "Soil_Temp_L3"
    elif (sdep >= 100):
    	ERAI_layer = "Soil_Temp_L4"
    	ERA5_layer = "Soil_Temp_L4"

##### JRA55 ####
    JRA_layer = "Soil_Temp"

##### MERRA2 ####
    if (0 <= sdep < 9.88):
    	MERRA2_layer = "Soil_Temp_L1"
    elif (9.88 <= sdep < 29.4):
    	MERRA2_layer = "Soil_Temp_L2"
    elif (29.4 <= sdep < 67.99):
    	MERRA2_layer = "Soil_Temp_L3"
    elif (67.99 <= sdep < 144.25):
    	MERRA2_layer = "Soil_Temp_L4"
    elif (144.25 <= sdep < 294.96):
    	MERRA2_layer = "Soil_Temp_L5"
    elif (sdep >= 294.96):
    	MERRA2_layer = "Soil_Temp_L6" 
