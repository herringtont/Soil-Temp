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
from decimal import *
from calendar import isleap
from shapely.geometry import Polygon, Point, GeometryCollection
from dateutil.relativedelta import *
from mpl_toolkits.basemap import Basemap


Rnysis_f = "/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/common_grid/remapnn/ERA-Interim.nc"
multisite_grids = "/mnt/data/users/herringtont/soil_temp/In-Situ/grid_cell_subsets/multiple_sites/remapbil_zscore_300_deeper_thr_100_multiple_sites.csv"
singlesite_grids = "/mnt/data/users/herringtont/soil_temp/In-Situ/grid_cell_subsets/single_site/remapbil_zscore_300_deeper_thr_100_single_site.csv"
site_geometry = "/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/geometry_L5_bil.csv"
############## Grab Grid Data ###############
dframe_multi_grid = pd.read_csv(multisite_grids)
grid_multi = dframe_multi_grid['Grid Cell'].values
len_grid_multi = len(grid_multi)

cen_lat_multi = dframe_multi_grid['Lat'].values
cen_lon_multi = dframe_multi_grid['Lon'].values

dframe_single_grid = pd.read_csv(singlesite_grids)
grid_single = dframe_single_grid['Grid Cell'].values
len_grid_single = len(grid_single)
#sites_grid_single = dframe_single_grid['Sites'].values

cen_lat_single = dframe_single_grid['Lat'].values
cen_lon_single = dframe_single_grid['Lon'].values

print(grid_multi)
#print(grid_single)

############## Grab Site Locations ############
dframe_geometry = pd.read_csv(site_geometry)

############## subset sites by multi-site grid cells and single-site grid-cells ############

######## multi-site grid-cells ########

lat_site_multi_master = []
lon_site_multi_master = []

for i in range(0,len_grid_multi):
    grid_multi_i = grid_multi[i]
    #print(grid_multi_i)
    dframe_geometry_multi = dframe_geometry[dframe_geometry['Grid Cell'] == grid_multi_i]
    #print(dframe_geometry_multi)
    site_geometry_multi = dframe_geometry_multi['site'].tolist()
    #print(site_geometry_multi)
    for j in range(0,len(site_geometry_multi)):
    	site_i = site_geometry_multi[j]
    	#print(site_i)
    	dframe_geometry_multi_site = dframe_geometry[dframe_geometry['site'] == site_i]
    	#print(dframe_geometry_multi_site)
### store site locations associated with multi-site grid cells ###
    lat_site_multi = dframe_geometry_multi_site['lat'].tolist()
    lat_site_multi_master.append(lat_site_multi)
    lon_site_multi = dframe_geometry_multi_site['lon'].tolist()
    lon_site_multi_master.append(lon_site_multi)
    #print(lat_site_multi, lon_site_multi)
######## single-site grid-cells ########

lat_site_single_master = []
lon_site_single_master = []

for i in range(0,len_grid_single):
    grid_single_i = grid_single[i]
    #print(grid_single_i)
    dframe_geometry_single = dframe_geometry[dframe_geometry['Grid Cell'] == grid_single_i]
    #print(dframe_geometry_multi)
    site_geometry_single = dframe_geometry_single['site'].tolist()
    #print(site_geometry_multi)
    for j in range(0,len(site_geometry_single)):
    	site_i = site_geometry_single[j]
    	#print(site_i)
    	dframe_geometry_single_site = dframe_geometry[dframe_geometry['site'] == site_i]
    	#print(dframe_geometry_single_site)
### store site locations associated with single-site grid cells ###
    lat_site_single = dframe_geometry_single_site['lat'].tolist()
    lat_site_single_master.append(lat_site_single)
    lon_site_single = dframe_geometry_single_site['lon'].tolist()
    lon_site_single_master.append(lon_site_single) 


lat_site_multi_master = [j for sub in lat_site_multi_master for j in sub]
lon_site_multi_master = [j for sub in lon_site_multi_master for j in sub]

lat_site_single_master = [j for sub in lat_site_single_master for j in sub]
lon_site_single_master = [j for sub in lon_site_single_master for j in sub]

#print("multi-site")
#print(lat_site_multi_master)

#print("single-site")
#print(lat_site_single_master)
    

############# create Basemap ###############
plt.figure(figsize=(16,16))
m = Basemap(projection='npstere',boundinglat=50,lon_0=180,resolution='l')
m.bluemarble()
m.drawcoastlines()
m.drawmapboundary(fill_color='white')
x,y = m(cen_lon_single,cen_lat_single)
x2,y2 = m(lon_site_single_master,lat_site_single_master)
x3,y3 = m(cen_lon_multi,cen_lat_multi)
x4,y4 = m(lon_site_multi_master,lat_site_multi_master)
Grid_Single = plt.scatter(x,y,20,marker='s', color='Red') ## plot single-site grid cells as yellow squares
#Site_Single = plt.scatter(x2,y2, 20,marker='^', color = 'Yellow') ## plot sites associated with single-site grid cells as yellow triangles
Grid_Multi = plt.scatter(x3,y3,20,marker='s', color='Red') ## plot multi-site grid cells as red squares
#Site_Multi = plt.scatter(x4,y4, 20,marker='^', color = 'Red') ## plot sites associated with multi-site grid cells as red triangles
#plt.legend((Site_Single,Site_Multi),('Single-Site Locations','Multi-Site Locations'), loc = 'best')
plt.tight_layout()
plt_name = ''.join(["/mnt/data/users/herringtont/soil_temp/plots/Site_Locations/All/Grid_Locations_L5.png"])
print(plt_name)
plt.savefig(plt_name)
plt.close()
