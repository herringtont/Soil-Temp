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
from pandas.tseries.offsets import MonthBegin

############ Set Directories and Variables ##############
path_005_degree = '/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/005_degree/'
path_sellonlatbox_layers = '/soil_temp/sellonlatbox/layers/'

asr_stl1 = 'ASR_stl1.nc'
asr_stl2 = 'ASR_stl2.nc'
asr_stl3 = 'ASR_stl3.nc'
asr_stl4 = 'ASR_stl4.nc'
era5_land_stl1 = 'ERA5_Land_stl1.nc'
era5_land_stl2 = 'ERA5_Land_stl2.nc'
era5_land_stl3 = 'ERA5_Land_stl3.nc'
era5_land_stl4 = 'ERA5_Land_stl4.nc'
fldas_stl1 = 'FLDAS_stl1.nc'
fldas_stl2 = 'FLDAS_stl2.nc'
fldas_stl3 = 'FLDAS_stl3.nc'
fldas_stl4 = 'FLDAS_stl4.nc'

common_date_index = pd.date_range(start='1970-01-01',end='2021-09-01', freq='MS')

######## Grab Data ########

def get_remap(style):
    wkdir = path_005_degree+style+path_sellonlatbox_layers
    
    #stl1
    sublayer1_ASR = xr.open_dataset(wkdir+asr_stl1)
    sublayer1_ERA5_Land = xr.open_dataset(wkdir+era5_land_stl1)
    sublayer1_FLDAS = xr.open_dataset(wkdir+fldas_stl1)
    
    #stl2
    sublayer2_ASR = xr.open_dataset(wkdir+asr_stl2)
    sublayer2_ERA5_Land = xr.open_dataset(wkdir+era5_land_stl2)
    sublayer2_FLDAS = xr.open_dataset(wkdir+fldas_stl2)
    
    #stl3
    layer2_ASR = xr.open_dataset(wkdir+asr_stl3)
    layer2_ERA5_Land = xr.open_dataset(wkdir+era5_land_stl3)
    layer2_FLDAS = xr.open_dataset(wkdir+fldas_stl3)
    
    #stl4
    layer3_ASR = xr.open_dataset(wkdir+asr_stl4)
    layer3_ERA5_Land = xr.open_dataset(wkdir+era5_land_stl4)
    layer3_FLDAS = xr.open_dataset(wkdir+fldas_stl4)

    print(layer2_ASR)

get_remap('remapbil')
#get_remap('remapnn')
