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


################### define functions ################
def str_to_datetime(column, date_fmt):

    date_list = []

    for dt_str in column:
        new_dt = datetime.datetime.strptime(dt_str, date_fmt)
        date_list.append(new_dt)
			

    return date_list
    
def remove_trailing_zeros(x):
    return str(x).rstrip('0').rstrip('.')


################## grab lat/lon grid from reanalysis ############

ERAI_fic = "/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/obs_depths/common_grid/remapcon/common_date/ERA-Interim.nc"
ERAI_filc = xr.open_dataset(ERAI_fic)
latRc = ERAI_filc.lat
lonRc = ERAI_filc.lon
latRc = np.array(latRc.values)
lonRc = np.array(lonRc.values)


################# define crid cell coordinates ##############

grid_cellsc = []

for f in range(0,len(lonRc)-1):
    for g in  range(0,len(latRc)-1):  ### loop through latitude values
    	x1 = lonRc[f] #leftmost x-coordinate of grid cell
    	x2 = lonRc[f+1] #rightmost x-coordinate of grid cell
    	y1 = latRc[g] #topmost y-coordinate of grid cell 
    	y2 = latRc[g+1] #bottommost y-coordinate of grid cell
    	grid = Polygon([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])
    	#print(grid)
    	grid_cellsc.append(grid)

grid_cellsc = np.array(grid_cellsc)
grid_numc = np.arange(1,len(grid_cellsc)+1,1)

#print(grid_cellsc)

################# grab central lat/lon and num of sites of grid cells collocated with stations ###############

fil_top30 = "/mnt/data/users/herringtont/soil_temp/naive_blend_timeseries/summary_fil/new_data/remapcon_simple_average_zscore_top_30cm_thr100_gcell_summary.csv"
fil_30_100 = "/mnt/data/users/herringtont/soil_temp/naive_blend_timeseries/summary_fil/new_data/remapcon_simple_average_zscore_30cm_100cm_thr100_gcell_summary.csv"
fil_100_300 = "/mnt/data/users/herringtont/soil_temp/naive_blend_timeseries/summary_fil/new_data/remapcon_simple_average_zscore_100cm_300cm_thr100_gcell_summary.csv"


dframe_top30 = pd.read_csv(fil_top30)
dframe_30_100 = pd.read_csv(fil_30_100)
dframe_100_300 = pd.read_csv(fil_100_300)

grid_top30 = dframe_top30['Grid Cell'].values
lat_top30 = dframe_top30['Central Lat'].values
lon_top30 = dframe_top30['Central Lon'].values
sites_top30 = dframe_top30['Avg Sites Incl'].values

grid_30_100 = dframe_30_100['Grid Cell'].values
lat_30_100 = dframe_30_100['Central Lat'].values
lon_30_100 = dframe_30_100['Central Lon'].values
sites_30_100 = dframe_30_100['Avg Sites Incl'].values

grid_100_300 = dframe_100_300['Grid Cell'].values
lat_100_300 = dframe_100_300['Central Lat'].values
lon_100_300 = dframe_100_300['Central Lon'].values
sites_100_300 = dframe_100_300['Avg Sites Incl'].values


################## create geopandas dataframes #######################

#dframe_gridc = pd.DataFrame({'Grid Cell':grid_numc})
#dframe_grid_geoc = gpd.GeoDataFrame(dframe_gridc, geometry = grid_cellsc)

gcells = grid_numc


##### loop through grid cells and extract num of sites ######

### top 30cm layer ###

grid_cell_master_top30 = []
num_sites_master_top30 = []
for gridcell in gcells:
    dframe_top30_gcell = dframe_top30[dframe_top30['Grid Cell'] == gridcell]
    if (len(dframe_top30_gcell) == 0): # if grid cell not in dataframe, num of sites included = 0
    	num_sites = 0
    elif (len(dframe_top30_gcell) > 0):
    	num_sites = dframe_top30_gcell['Avg Sites Incl'].values

    grid_cell_master_top30.append(gridcell)
    num_sites_master_top30.append(num_sites)

dframe_grid_top30 = pd.DataFrame({'Grid Cell':grid_cell_master_top30,'Number of Sites':num_sites})
dframe_grid_geo_top30 = gpd.GeoDataFrame(dframe_grid_top30, geometry = grid_cellsc)
#print(dframe_grid_geo_top30)

num_sites_top30 = dframe_grid_top30['Number of Sites'].values

#print(dframe_grid_top30.iloc[1])

### 30cm - 100cm layer ###

grid_cell_master_30_100 = []
num_sites_master_30_100 = []
for gridcell in gcells:
    dframe_30_100_gcell = dframe_30_100[dframe_30_100['Grid Cell'] == gridcell]
    if (len(dframe_30_100_gcell) == 0): # if grid cell not in dataframe, num of sites included = 0
    	num_sites = 0
    elif (len(dframe_30_100_gcell) > 0):
    	num_sites = dframe_30_100_gcell['Avg Sites Incl'].values

    grid_cell_master_30_100.append(gridcell)
    num_sites_master_30_100.append(num_sites)

dframe_grid_30_100 = pd.DataFrame({'Grid Cell':grid_cell_master_30_100,'Number of Sites':num_sites})
dframe_grid_geo_30_100 = gpd.GeoDataFrame(dframe_grid_30_100, geometry = grid_cellsc)
#print(dframe_grid_geo_30_100)

num_sites_30_100 = dframe_grid_30_100['Number of Sites'].values 


### 100cm - 300cm layer ###

grid_cell_master_100_300 = []
num_sites_master_100_300 = []
for gridcell in gcells:
    dframe_100_300_gcell = dframe_100_300[dframe_100_300['Grid Cell'] == gridcell]
    if (len(dframe_100_300_gcell) == 0): # if grid cell not in dataframe, num of sites included = 0
    	num_sites = 0
    elif (len(dframe_100_300_gcell) > 0):
    	num_sites = dframe_100_300_gcell['Avg Sites Incl'].values

    grid_cell_master_100_300.append(gridcell)
    num_sites_master_100_300.append(num_sites)

dframe_grid_100_300 = pd.DataFrame({'Grid Cell':grid_cell_master_100_300,'Number of Sites':num_sites})
dframe_grid_geo_100_300 = gpd.GeoDataFrame(dframe_grid_100_300, geometry = grid_cellsc)
#print(dframe_grid_geo_100_300)




################# create plot ############

fig = plt.figure(figsize = (15,15))


### top 30cm ###
ax1 = fig.add_subplot(111, projection=ccrs.NorthPolarStereo())
ax1.coastlines()
for i in range(0,len(dframe_grid_top30)):
    dframe_i = dframe_grid_top30.iloc[i]
    print(dframe_i)
    #geometry_i = dframe_i.geometry
    ax1.add_geometries(dframe_i.geometry, crs=ccrs.PlateCarree(), facecolor='none', edgecolor='k')
ax1.set_extent([0,360,40,90], crs=ccrs.PlateCarree())
plt.show()













