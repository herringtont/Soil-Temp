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
import geopandas as gpd
import xarray as xr
import seaborn as sns
import shapely
import math
import cftime
import re
import cdms2
import cartopy.crs as ccrs
from decimal import *
from calendar import isleap
from shapely.geometry import Polygon, Point, GeometryCollection
from dateutil.relativedelta import *
from mpl_toolkits.basemap import Basemap



############### Grab Grid Cell and Spatial Info #################
TC_metrics_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/global_triple_collocation/compare_metrics/raw_temp/remapnn_compare_metrics.csv'])
TC_metrics_dframe = pd.read_csv(TC_metrics_fil)

gcells = TC_metrics_dframe['Grid Cell'].values
lat = TC_metrics_dframe['Central Lat'].values
lon = TC_metrics_dframe['Central Lon'].values


dframe_7425 = TC_metrics_dframe[TC_metrics_dframe['Grid Cell'] == 7425]
lat_7425 = dframe_7425['Central Lat'].values
lon_7425 = dframe_7425['Central Lon'].values

print(dframe_7425)
############### Grab data from remapped land cover grids #################

CFSR_land_fi = '/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/land_sea_mask/common_grid/remapnn/CFSR_land_mask_remapnn.nc'
CFSR2_land_fi = '/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/land_sea_mask/common_grid/remapnn/CFSR2_land_mask_remapnn.nc'
ERAI_land_fi = '/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/land_sea_mask/common_grid/remapnn/ERA-Interim_land_mask_remapnn.nc'
ERA5_land_fi = '/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/land_sea_mask/common_grid/remapnn/ERA5_land_mask_remapnn.nc'
JRA_land_fi = '/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/land_sea_mask/common_grid/remapnn/JRA55_land_mask_remapnn.nc'
MERRA2_land_fi = '/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/land_sea_mask/common_grid/remapnn/MERRA2_land_mask_remapnn.nc'
GLDAS_land_fi = '/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/land_sea_mask/common_grid/remapnn/GLDAS_land_mask_remapnn.nc'

CFSR_land_fil = xr.open_dataset(CFSR_land_fi)
CFSR2_land_fil = xr.open_dataset(CFSR2_land_fi)
ERAI_land_fil = xr.open_dataset(ERAI_land_fi)
ERA5_land_fil = xr.open_dataset(ERA5_land_fi)
JRA_land_fil = xr.open_dataset(JRA_land_fi)
MERRA2_land_fil = xr.open_dataset(MERRA2_land_fi)
GLDAS_land_fil = xr.open_dataset(GLDAS_land_fi)

CFSR_land_cover = CFSR_land_fil['LSM']
CFSR2_land_cover = CFSR2_land_fil['LSM']
ERAI_land_cover = ERAI_land_fil['LSM']
ERA5_land_cover = ERA5_land_fil['LSM']
JRA_land_cover = JRA_land_fil['LSM']
MERRA2_land_cover = MERRA2_land_fil['LSM']
GLDAS_land_cover = GLDAS_land_fil['LSM']


####################### Create Geospatial Plot #####################
#fig=plt.figure(figsize=(16,16))
#m = Basemap(projection='npstere',boundinglat=50,lon_0=180,resolution='l')
#
#x,y = m(lon,lat)
#x2,y2 = m(lon_7425,lat_7425)
#
#ax = fig.add_subplot(111,projection=ccrs.NorthPolarStereo(central_longitude=180)) 
#JRA_land_cover.plot(ax=ax,transform=ccrs.PlateCarree())
#ax.set_extent([0,360,50,90])
##All_Grids = plt.scatter(x,y,20,marker='s', color='Blue')
##Grid_7425 = plt.scatter(x2,y2,20,marker='s',color='Red')
#
#m.drawcoastlines()
##m.drawmapboundary(fill_color='white')
#plt.tight_layout()
#plt.show()


