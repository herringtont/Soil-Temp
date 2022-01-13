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

stn_metrics_filr = "/mnt/data/users/herringtont/soil_temp/blended_metrics/grid_cell_stn/raw_temp/remapbil_zscore_0_9.9_thr100_summary_statistics_gridcell_stn.csv"
stn_metrics_fila = "/mnt/data/users/herringtont/soil_temp/blended_metrics/grid_cell_stn/anom/remapbil_zscore_0_9.9_thr100_summary_statistics_anom_gridcell_stn.csv"

metrics_filr = "/mnt/data/users/herringtont/soil_temp/blended_metrics/grid_cell/raw_temp/remapbil_zscore_0_9.9_thr100_summary_statistics_gridcell.csv"
metrics_fila = "/mnt/data/users/herringtont/soil_temp/blended_metrics/grid_cell/anom/remapbil_zscore_0_9.9_thr100_summary_statistics_anom_gridcell.csv"
site_geometry = "/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/geometry_L5_bil.csv"

############## Grab Grid Data ###############
dframe_stn_r = pd.read_csv(stn_metrics_filr)
dframe_stn_a = pd.read_csv(stn_metrics_fila)
dframe_r = pd.read_csv(metrics_filr)
dframe_a = pd.read_csv(metrics_fila)

cen_lat_stn_r = dframe_stn_r['Central Lat'].values
cen_lon_stn_r = dframe_stn_r['Central Lon'].values
cen_lat_stn_a = dframe_stn_a['Central Lat'].values
cen_lon_stn_a = dframe_stn_a['Central Lon'].values

cen_lat_r = dframe_r['Central Lat'].values
cen_lon_r = dframe_r['Central Lon'].values
cen_lat_a = dframe_a['Central Lat'].values
cen_lon_a = dframe_a['Central Lon'].values

corr_stn_r = dframe_stn_r['delta corr'].values
corr_stn_a = dframe_stn_a['delta corr'].values
corr_r = dframe_r['delta corr'].values
corr_a = dframe_a['delta corr'].values

print(corr_stn_a)
print(corr_stn_r) 
print(corr_r)
print(corr_a)
############# create Basemap ###############
fig, [[ax1, ax2],[ax3,ax4]] = plt.subplots(2,2,figsize=(16,16))


##### Stn Metrics (Raw Temp Data)
plt.subplot(221)
m = Basemap(projection='npstere',boundinglat=50,lon_0=180,resolution='l')
#m.bluemarble()
m.drawcoastlines()
m.drawmapboundary(fill_color='white')
x1,y1 = m(cen_lon_stn_r,cen_lat_stn_r)

plt.scatter(x1,y1, s=30 ,marker='s', c=corr_stn_r, cmap='seismic', vmin =-0.25, vmax=0.25) 
plt.title('$\delta$ Corr (Relative to Station Temp)')
plt.colorbar()

plt.subplot(222)
m = Basemap(projection='npstere',boundinglat=50,lon_0=180,resolution='l')
#m.bluemarble()
m.drawcoastlines()
m.drawmapboundary(fill_color='white')
x2,y2 = m(cen_lon_stn_a,cen_lat_stn_a)

plt.scatter(x2,y2, s=30,marker='s', c=corr_stn_a, cmap='seismic', vmin =-0.25, vmax=0.25) 
plt.title('$\delta$ Corr (Relative to Station Anom)')
plt.colorbar()

plt.subplot(223)
m = Basemap(projection='npstere',boundinglat=50,lon_0=180,resolution='l')
#m.bluemarble()
m.drawcoastlines()
m.drawmapboundary(fill_color='white')
x3,y3 = m(cen_lon_r,cen_lat_r)

plt.scatter(x3,y3, s=30,marker='s', c=corr_r, cmap='seismic', vmin =-0.25, vmax=0.25) 
plt.title('$\delta$ Corr (Relative to ERA5 Temp)')
plt.colorbar()

plt.subplot(224)
m = Basemap(projection='npstere',boundinglat=50,lon_0=180,resolution='l')
#m.bluemarble()
m.drawcoastlines()
m.drawmapboundary(fill_color='white')
x4,y4 = m(cen_lon_a,cen_lat_a)

plt.scatter(x4,y4, s=30,marker='s', c=corr_a, cmap='seismic', vmin =-0.25, vmax=0.25) 
plt.title('$\delta$ Corr (Relative to ERA5 Anom)')
plt.colorbar()

plt.tight_layout()
plt_name = ''.join(["/mnt/data/users/herringtont/soil_temp/plots/blended_metrics/correlation_map/Blended_performance_corr_L1.png"])
print(plt_name)
plt.savefig(plt_name)
plt.close()
