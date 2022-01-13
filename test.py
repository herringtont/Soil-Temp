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
from calendar import isleap
from dateutil.relativedelta import *

#set model variable name

temp_depth = "Soil_Temp_L1"

###grab corresponding reanalysis data
CFSR_fi = "/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/site_level/CFSR/CFSR_site_109.nc"
CFSR2_fi = "/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/site_level/CFSR2/CFSR2_site_109.nc"

CFSR_fil = xr.open_dataset(CFSR_fi) #open NetCDF file with xarray
CFSR2_fil = xr.open_dataset(CFSR2_fi) #open NetCDF file with xarray

#extract soil temperatures and convert to celsius

CFSR_stemp = CFSR_fil[temp_depth] - 273.15  #convert from Kelvin to Celsius
CFSR2_stemp = CFSR2_fil.Soil_Temp_L1 - 273.15  #convert from Kelvin to Celsius

print(CFSR_stemp)
