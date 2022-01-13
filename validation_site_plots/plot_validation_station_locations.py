import os
import csv
import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import matplotlib.cm as cm
import matplotlib.patches as mpl_patches
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



########### set directories and files ##############
#geometry_fil = "/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/geometry_L1_nn.csv"
val_stn_fil_top30 = "/mnt/data/users/herringtont/soil_temp/naive_blend_timeseries/summary_fil/new_data/remapcon_simple_average_zscore_top_30cm_thr100_gcell_summary.csv"
val_stn_fil_30_100 = "/mnt/data/users/herringtont/soil_temp/naive_blend_timeseries/summary_fil/new_data/remapcon_simple_average_zscore_30cm_100cm_thr100_gcell_summary.csv"
val_stn_fil_100_300 = "/mnt/data/users/herringtont/soil_temp/naive_blend_timeseries/summary_fil/new_data/remapcon_simple_average_zscore_100cm_300cm_thr100_gcell_summary.csv"

dframe_val_stn_top30 = pd.read_csv(val_stn_fil_top30)
gcell_top30 = dframe_val_stn_top30['Grid Cell'].values
cen_lat_top30 = dframe_val_stn_top30['Central Lat'].values
cen_lon_top30 = dframe_val_stn_top30['Central Lon'].values
gcell_uq_top30 = np.unique(gcell_top30)


dframe_val_stn_30_100 = pd.read_csv(val_stn_fil_30_100)
gcell_30_100 = dframe_val_stn_30_100['Grid Cell'].values
cen_lat_30_100 = dframe_val_stn_30_100['Central Lat'].values
cen_lon_30_100 = dframe_val_stn_30_100['Central Lon'].values
gcell_uq_30_100 = np.unique(gcell_30_100)

dframe_val_stn_100_300 = pd.read_csv(val_stn_fil_100_300)
#print(dframe_val_stn_top30)
#print(dframe_val_stn_30_100)
#print(dframe_val_stn_100_300)
gcell_100_300 = dframe_val_stn_100_300['Grid Cell'].values
cen_lat_100_300 = dframe_val_stn_100_300['Central Lat'].values
cen_lon_100_300 = dframe_val_stn_100_300['Central Lon'].values
gcell_uq_100_300 = np.unique(gcell_100_300)

lat_master_top30 = []
lon_master_top30 = []
gcell_master_top30 = []
sites_master_top30 = []
for i in gcell_uq_top30:
    gcell_i = i
    dframe_gcell = dframe_val_stn_top30[dframe_val_stn_top30['Grid Cell'] == gcell_i]
    print(dframe_gcell)
    gcell_lat = dframe_gcell['Central Lat'].iloc[0]
    lat_master_top30.append(gcell_lat)
    gcell_lon = dframe_gcell['Central Lon'].iloc[0]
    lon_master_top30.append(gcell_lon)
    gcell_master_top30.append(gcell_i)    
    gcell_sites = dframe_gcell['Avg Sites Incl'].iloc[0]
    sites_master_top30.append(gcell_sites)


lat_master_30_100 = []
lon_master_30_100 = []
gcell_master_30_100 = []
sites_master_30_100 = []
for i in gcell_uq_30_100:
    gcell_i = i
    dframe_gcell = dframe_val_stn_30_100[dframe_val_stn_30_100['Grid Cell'] == gcell_i]
    gcell_lat = dframe_gcell['Central Lat'].iloc[0]
    lat_master_30_100.append(gcell_lat)
    gcell_lon = dframe_gcell['Central Lon'].iloc[0]
    lon_master_30_100.append(gcell_lon)
    gcell_master_30_100.append(gcell_i)
    gcell_sites = dframe_gcell['Avg Sites Incl'].iloc[0]
    sites_master_30_100.append(gcell_sites)

lat_master_100_300 = []
lon_master_100_300 = []
gcell_master_100_300 = []
sites_master_100_300 = []
for i in gcell_uq_100_300:
    gcell_i = i
    dframe_gcell = dframe_val_stn_100_300[dframe_val_stn_100_300['Grid Cell'] == gcell_i]
    gcell_lat = dframe_gcell['Central Lat'].iloc[0]
    lat_master_100_300.append(gcell_lat)
    gcell_lon = dframe_gcell['Central Lon'].iloc[0]
    lon_master_100_300.append(gcell_lon)
    gcell_master_100_300.append(gcell_i)
    gcell_sites = dframe_gcell['Avg Sites Incl'].iloc[0]
    sites_master_100_300.append(gcell_sites)
#lat_master = [i for sub in lat_master for i in sub]
#lon_master = [i for sub in lon_master for i in sub]
#gcell_master = [i for sub in gcell_master for i in sub]

print(sites_master_top30)

size_top30 = np.array(sites_master_top30)
max_sites_top30 = max(size_top30)
size_30_100 = np.array(sites_master_30_100)
max_sites_30_100 = max(size_30_100)
size_100_300 = np.array(sites_master_100_300)
max_sites_100_300 = max(size_100_300)    
print("Number of Grid Cells in top 30cm layer:",len(gcell_master_top30))
print(max_sites_top30)
print("Number of Grid Cells in 30cm - 100cm layer:",len(gcell_master_30_100))
print(max_sites_30_100)
print("Number of Grid Cells in 100cm - 300cm layer:",len(gcell_master_100_300))
print(max_sites_100_300)
############ create Plot ###############
fig = plt.figure(figsize = (15,15))

#create array for parallels
parallels = np.arange(40.,81.,10.)

#create subplot for top 30cm layer
ax1 = fig.add_subplot(221)
map1 = Basemap(projection='npstere',boundinglat=40,lon_0=180,resolution='h')
#map1.drawcoastlines(zorder=2)
map1.drawmapboundary(fill_color='aqua',zorder=0) # fill to edge
map1.fillcontinents(color='palegoldenrod',lake_color='aqua',zorder=1)

#draw parallels on map
map1.drawparallels(parallels,labels=[False,True,True,False])

#labels = [Left, Top, Right, Bottom]


#add text to Subplot 1:
handles1 = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)]
ax1text = []
ax1text.append('Number of Sites: '+str(len(gcell_master_top30)))
#ax1.legend(handles1, ax1text, loc='best', fontsize=18, fancybox=False, framealpha=0, handlelength=0, handletextpad=0)

#plot scatterplot
x1,y1 = map1(lon_master_top30,lat_master_top30)
GridCell1 = ax1.scatter(x1,y1,alpha=0.6,s=size_top30*30,marker='.',color='firebrick',zorder=3)
ax1.set_title('a) Top 30cm Layer',fontsize=20)

# make legend with dummy points
for a in [1,3,5,7,9]:
    ax1.scatter([], [], c='firebrick', alpha=0.6, s=a*7, label=str(a) + ' sites')

ax1.legend(scatterpoints=1, frameon=False, labelspacing = 1, loc = 'upper right')

#create subplot for 30cm - 100cm layer
ax2 = fig.add_subplot(222)
map2 = Basemap(projection='npstere',boundinglat=40,lon_0=180,resolution='h')
#map2.drawcoastlines(zorder=2)
map2.drawmapboundary(fill_color='aqua',zorder=0) # fill to edge
map2.fillcontinents(color='palegoldenrod',lake_color='aqua',zorder=1)

#draw parallels on map
map2.drawparallels(parallels,labels=[False,True,True,False])

#labels = [Left, Top, Right, Bottom]

#add text to Subplot 2:
handles2 = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)]
ax2text = []
ax2text.append('Number of Sites: '+str(len(gcell_master_30_100)))
#ax2.legend(handles2, ax2text, loc='best', fontsize=18, fancybox=False, framealpha=0, handlelength=0, handletextpad=0)

#plot scatterplot
x2,y2 = map1(lon_master_30_100,lat_master_30_100)
GridCell2 = ax2.scatter(x2,y2,alpha=0.6,s=size_30_100*30,marker='.',color='firebrick',zorder=3)
ax2.set_title('b) 30cm - 100cm Layer',fontsize=20)

# make legend with dummy points
for a in [1,3,5,7,9]:
    ax2.scatter([], [], c='firebrick', alpha=0.6, s=a*7, label=str(a) + ' sites')

ax2.legend(scatterpoints=1, frameon=False, labelspacing = 1, loc = 'upper right')

#create subplot for 100cm - 300cm layer
ax3 = fig.add_subplot(223)
map3 = Basemap(projection='npstere',boundinglat=40,lon_0=180,resolution='h')
#map3.drawcoastlines(zorder=2)
map3.drawmapboundary(fill_color='aqua',zorder=0) # fill to edge
map3.fillcontinents(color='palegoldenrod',lake_color='aqua',zorder=1)

#draw parallels on map
map3.drawparallels(parallels,labels=[False,True,True,False])

#labels = [Left, Top, Right, Bottom]

#add text to Subplot 2:
handles3 = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)]
ax3text = []
ax3text.append('Number of Sites: '+str(len(gcell_master_100_300)))
#ax3.legend(handles3, ax3text, loc='best', fontsize=18, fancybox=False, framealpha=0, handlelength=0, handletextpad=0)

#plot scatterplot
x3,y3 = map3(lon_master_100_300,lat_master_100_300)
GridCell3 = ax3.scatter(x3,y3,alpha=0.6,s=size_100_300*30,marker='.',color='firebrick',zorder=3)
ax3.set_title('c) 100cm - 300cm Layer',fontsize=20)

# make legend with dummy points
for a in [1,3,5,7,9]:
    ax3.scatter([], [], c='firebrick', alpha=0.6, s=a*7, label=str(a) + ' sites')

ax3.legend(scatterpoints=1, frameon=False, labelspacing = 1, loc = 'upper right')

plt.tight_layout()
plt_fil = "/mnt/data/users/herringtont/soil_temp/plots/validation_sites/new_data/validation_sites_remapcon_zscore_all_layers_thr100.png"
fig.savefig(plt_fil)
plt.close()
