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


########### set variables ##############
permafrost_type = ['RS_2002_continuous','Brown_1970_continuous','RS_2002_discontinuous','Brown_1970_discontinuous','RS_2002_none','Brown_1970_none']


########### set directories and files ##############

val_stn_fil_top30_con = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_timeseries/summary_fil/new_data/CLSM_res/remapcon_top_30cm_thr100_gcell_summary_CLSM_RS_2002_continuous_BEST_Sep2021_NWT.csv'])
val_stn_fil_30_300_con = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_timeseries/summary_fil/new_data/CLSM_res/remapcon_30cm_300cm_thr100_gcell_summary_CLSM_RS_2002_continuous_BEST_Sep2021_NWT.csv'])
val_stn_fil_top30_dis = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_timeseries/summary_fil/new_data/CLSM_res/remapcon_top_30cm_thr100_gcell_summary_CLSM_RS_2002_discontinuous_BEST_Sep2021_NWT.csv'])
val_stn_fil_30_300_dis = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_timeseries/summary_fil/new_data/CLSM_res/remapcon_30cm_300cm_thr100_gcell_summary_CLSM_RS_2002_discontinuous_BEST_Sep2021_NWT.csv'])
val_stn_fil_top30 = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_timeseries/summary_fil/new_data/CLSM_res/remapcon_top_30cm_thr100_gcell_summary_CLSM_RS_2002_none_BEST_Sep2021_NWT.csv'])
val_stn_fil_30_300 = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_timeseries/summary_fil/new_data/CLSM_res/remapcon_30cm_300cm_thr100_gcell_summary_CLSM_RS_2002_none_BEST_Sep2021_NWT.csv'])

dframe_val_stn_top30 = pd.read_csv(val_stn_fil_top30)
gcell_top30 = dframe_val_stn_top30['Grid Cell'].values
cen_lat_top30 = dframe_val_stn_top30['Central Lat'].values
cen_lon_top30 = dframe_val_stn_top30['Central Lon'].values
gcell_uq_top30 = np.unique(gcell_top30)

dframe_val_stn_top30_dis = pd.read_csv(val_stn_fil_top30_dis)
gcell_top30_dis = dframe_val_stn_top30_dis['Grid Cell'].values
cen_lat_top30_dis = dframe_val_stn_top30_dis['Central Lat'].values
cen_lon_top30_dis = dframe_val_stn_top30_dis['Central Lon'].values
gcell_uq_top30_dis = np.unique(gcell_top30_dis)

dframe_val_stn_top30_con = pd.read_csv(val_stn_fil_top30_con)
gcell_top30_con = dframe_val_stn_top30_con['Grid Cell'].values
cen_lat_top30_con = dframe_val_stn_top30_con['Central Lat'].values
cen_lon_top30_con = dframe_val_stn_top30_con['Central Lon'].values
gcell_uq_top30_con = np.unique(gcell_top30_con)

dframe_val_stn_30_300 = pd.read_csv(val_stn_fil_30_300)
gcell_30_300 = dframe_val_stn_30_300['Grid Cell'].values
cen_lat_30_300 = dframe_val_stn_30_300['Central Lat'].values
cen_lon_30_300 = dframe_val_stn_30_300['Central Lon'].values
gcell_uq_30_300 = np.unique(gcell_30_300)

dframe_val_stn_30_300_dis = pd.read_csv(val_stn_fil_30_300_dis)
gcell_30_300_dis = dframe_val_stn_30_300_dis['Grid Cell'].values
cen_lat_30_300_dis = dframe_val_stn_30_300_dis['Central Lat'].values
cen_lon_30_300_dis = dframe_val_stn_30_300_dis['Central Lon'].values
gcell_uq_30_300_dis = np.unique(gcell_30_300_dis)

dframe_val_stn_30_300_con = pd.read_csv(val_stn_fil_30_300_con)
gcell_30_300_con = dframe_val_stn_30_300_con['Grid Cell'].values
cen_lat_30_300_con = dframe_val_stn_30_300_con['Central Lat'].values
cen_lon_30_300_con = dframe_val_stn_30_300_con['Central Lon'].values
gcell_uq_30_300_con = np.unique(gcell_30_300_con)
    
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
    #gcell_sites = dframe_gcell['Avg Sites Incl'].iloc[0]
    #sites_master_top30.append(gcell_sites)

lat_master_top30_dis = []
lon_master_top30_dis = []
gcell_master_top30_dis = []
sites_master_top30_dis = []
for i in gcell_uq_top30_dis:
    gcell_i = i
    dframe_gcell = dframe_val_stn_top30_dis[dframe_val_stn_top30_dis['Grid Cell'] == gcell_i]
    print(dframe_gcell)
    gcell_lat = dframe_gcell['Central Lat'].iloc[0]
    lat_master_top30_dis.append(gcell_lat)
    gcell_lon = dframe_gcell['Central Lon'].iloc[0]
    lon_master_top30_dis.append(gcell_lon)
    gcell_master_top30_dis.append(gcell_i)    
    #gcell_sites = dframe_gcell['Avg Sites Incl'].iloc[0]
    #sites_master_top30_dis.append(gcell_sites)

lat_master_top30_con = []
lon_master_top30_con = []
gcell_master_top30_con = []
sites_master_top30_con = []
for i in gcell_uq_top30_con:
    gcell_i = i
    dframe_gcell = dframe_val_stn_top30_con[dframe_val_stn_top30_con['Grid Cell'] == gcell_i]
    print(dframe_gcell)
    gcell_lat = dframe_gcell['Central Lat'].iloc[0]
    lat_master_top30_con.append(gcell_lat)
    gcell_lon = dframe_gcell['Central Lon'].iloc[0]
    lon_master_top30_con.append(gcell_lon)
    gcell_master_top30_con.append(gcell_i)    
    #gcell_sites = dframe_gcell['Avg Sites Incl'].iloc[0]
    #sites_master_top30_con.append(gcell_sites)

lat_master_30_300 = []
lon_master_30_300 = []
gcell_master_30_300 = []
sites_master_30_300 = []
for i in gcell_uq_30_300:
    gcell_i = i
    dframe_gcell = dframe_val_stn_30_300[dframe_val_stn_30_300['Grid Cell'] == gcell_i]
    print(dframe_gcell)
    gcell_lat = dframe_gcell['Central Lat'].iloc[0]
    lat_master_30_300.append(gcell_lat)
    gcell_lon = dframe_gcell['Central Lon'].iloc[0]
    lon_master_30_300.append(gcell_lon)
    gcell_master_30_300.append(gcell_i)    
    #gcell_sites = dframe_gcell['Avg Sites Incl'].iloc[0]
    #sites_master_30_300.append(gcell_sites)

lat_master_30_300_dis = []
lon_master_30_300_dis = []
gcell_master_30_300_dis = []
sites_master_30_300_dis = []
for i in gcell_uq_30_300_dis:
    gcell_i = i
    dframe_gcell = dframe_val_stn_30_300_dis[dframe_val_stn_30_300_dis['Grid Cell'] == gcell_i]
    print(dframe_gcell)
    gcell_lat = dframe_gcell['Central Lat'].iloc[0]
    lat_master_30_300_dis.append(gcell_lat)
    gcell_lon = dframe_gcell['Central Lon'].iloc[0]
    lon_master_30_300_dis.append(gcell_lon)
    gcell_master_30_300_dis.append(gcell_i)    
    #gcell_sites = dframe_gcell['Avg Sites Incl'].iloc[0]
    #sites_master_30_300_dis.append(gcell_sites)

lat_master_30_300_con = []
lon_master_30_300_con = []
gcell_master_30_300_con = []
sites_master_30_300_con = []
for i in gcell_uq_30_300_con:
    gcell_i = i
    dframe_gcell = dframe_val_stn_30_300_con[dframe_val_stn_30_300_con['Grid Cell'] == gcell_i]
    print(dframe_gcell)
    gcell_lat = dframe_gcell['Central Lat'].iloc[0]
    lat_master_30_300_con.append(gcell_lat)
    gcell_lon = dframe_gcell['Central Lon'].iloc[0]
    lon_master_30_300_con.append(gcell_lon)
    gcell_master_30_300_con.append(gcell_i)    
    #gcell_sites = dframe_gcell['Avg Sites Incl'].iloc[0]
    #sites_master_30_300_con.append(gcell_sites)

    #lat_master = [i for sub in lat_master for i in sub]
    #lon_master = [i for sub in lon_master for i in sub]
    #gcell_master = [i for sub in gcell_master for i in sub]

print(sites_master_top30)

print("Grid Cells in the Top 30cm:")
print("Number of Grid Cells (continuous permafrost):",len(gcell_master_top30_con))
print("Number of Grid Cells (discontinuos permafrost):",len(gcell_master_top30_dis))
print("Number of Grid Cells (little to no permafrost):",len(gcell_master_top30))
total_gcell_top30 = len(gcell_master_top30_con)+len(gcell_top30_dis)+len(gcell_top30)
print(total_gcell_top30)

print("Grid Cells in the 30cm - 300cm Layer:")
print("Number of Grid Cells (continuous permafrost):",len(gcell_master_30_300_con))
print("Number of Grid Cells (discontinuos permafrost):",len(gcell_master_30_300_dis))
print("Number of Grid Cells (little to no permafrost):",len(gcell_master_30_300))
total_gcell_30_300 = len(gcell_master_30_300_con)+len(gcell_30_300_dis)+len(gcell_30_300)
print(total_gcell_30_300)

############ create Plot ###############
fig = plt.figure(figsize = (8,8))

#create array for parallels
parallels = np.arange(50.,81.,10.)
meridians = np.arange(0.,351.,10)
#create subplot for top 30cm layer
ax1 = fig.add_subplot(111)
map1 = Basemap(projection='npstere',boundinglat=45,lon_0=0,resolution='h')
#map1.drawcoastlines(zorder=2)
map1.drawmapboundary(fill_color='aqua',zorder=0) # fill to edge
map1.fillcontinents(color='darkgrey',lake_color='aqua',zorder=1)

#draw parallels on map
map1.drawparallels(parallels,labels=[False,True,True,False])
map1.drawmeridians(meridians,labels=[False,False,False,False])
#labels = [Left, Top, Right, Bottom]


#add text to Subplot 1:
handles1 = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)]
ax1text = []
ax1text.append('Number of Sites: '+str(len(gcell_master_top30)))
#ax1.legend(handles1, ax1text, loc='best', fontsize=18, fancybox=False, framealpha=0, handlelength=0, handletextpad=0)

#plot scatterplot
x1,y1 = map1(lon_master_top30,lat_master_top30) #little to no permafrost
GridCell1 = ax1.scatter(x1,y1,alpha=0.6,marker='.',s=100,color='blue',zorder=3) #little to no permafrost
x2,y2 = map1(lon_master_top30_dis,lat_master_top30_dis) #little to no permafrost
GridCell2 = ax1.scatter(x2,y2,alpha=0.6,marker='.',s=100,color='red',zorder=4) #discontinuous permafrost
x3,y3 = map1(lon_master_top30_con,lat_master_top30_con) #little to no permafrost
GridCell3 = ax1.scatter(x3,y3,alpha=0.6,marker='.',s=100,color='green',zorder=5) #continuous permafrost

# make legend with dummy points
for a in ['continuous', 'discontinuous', 'little to no']:
    if (a == 'continuous'):
    	colour = 'green'
    	b = ('(MAAT $\leq$ -7$^\circ$C)')	
    elif (a == 'discontinuous'):
    	colour = 'red'
    	b = ('(-2$^\circ$C $\leq$ MAAT > -7$^\circ$C)')
    elif (a == 'little to no'):
    	colour = 'blue'
    	b = ('(MAAT > -2$^\circ$C)')	
    ax1.scatter([], [], c=colour, alpha=0.6, s=40, label=str(a) + ' permafrost ' + str(b))

## make legend with dummy points
#for a in [1,3,5,7,9]:
#    ax1.scatter([], [], c='firebrick', alpha=0.6, s=a*14, label=str(a) + ' sites')
#
#ax1.legend(scatterpoints=1, frameon=False, labelspacing = 1, loc = 'right',bbox_to_anchor=(1.5,0.5),ncol = 1, fontsize = 20)

##create subplot for 30cm - 300cm layer
#ax3 = fig.add_subplot(212)
#map3 = Basemap(projection='npstere',boundinglat=40,lon_0=180,resolution='h')
##map3.drawcoastlines(zorder=2)
#map3.drawmapboundary(fill_color='aqua',zorder=0) # fill to edge
#map3.fillcontinents(color='palegoldenrod',lake_color='aqua',zorder=1)
#
##draw parallels on map
#map3.drawparallels(parallels,labels=[False,True,True,False])
#
##labels = [Left, Top, Right, Bottom]
#
##add text to Subplot 2:
#handles3 = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)]
#ax3text = []
#ax3text.append('Number of Sites: '+str(len(gcell_master_30_300)))
##ax3.legend(handles3, ax3text, loc='best', fontsize=18, fancybox=False, framealpha=0, handlelength=0, handletextpad=0)
#
##plot scatterplot
#x4,y4 = map3(lon_master_30_300,lat_master_30_300) #little to no permafrost
#GridCell4 = ax3.scatter(x4,y4,alpha=0.6,s=size_30_300*60,marker='.',color='red',zorder=3) #little to no permafrost
#x5,y5 = map3(lon_master_30_300_dis,lat_master_30_300_dis) #little to no permafrost
#GridCell5 = ax3.scatter(x5,y5,alpha=0.6,s=size_30_300_dis*60,marker='.',color='purple',zorder=4) #discontinuous permafrost
#x6,y6 = map3(lon_master_30_300_con,lat_master_30_300_con) #little to no permafrost
#GridCell6 = ax3.scatter(x6,y6,alpha=0.6,s=size_30_300_con*60,marker='.',color='blue',zorder=5) #continuous permafrost
#
## make legend with dummy points
#for a in [1,3,5,7,9]:
#    ax3.scatter([], [], c='firebrick', alpha=0.6, s=a*14, label=str(a) + ' sites')
#
#ax3.legend(scatterpoints=1, frameon=False, labelspacing = 1, loc = 'right', bbox_to_anchor=(1.5,0.5), ncol = 1, fontsize = 20)

plt.legend()
plt.tight_layout()
plt_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/plots/validation_sites/new_data/CLSM_res/validation_sites_remapcon_zscore_all_layers_thr100_CMOS_CLSM_RS_2002_all_permafrost_types_top30_BEST_Sep2021_NWT.png'])
fig.savefig(plt_fil)
plt.close()
