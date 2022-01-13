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


######### Grab Data ########

top_30_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/sites_2yr/sites_2yr_zscore_top_30cm_100.csv'])
depth_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/sites_2yr/sites_2yr_zscore_30_299.9_100.csv'])

dframe_top30 = pd.read_csv(top_30_fil)
dframe_depth = pd.read_csv(depth_fil)

sites_top30 = dframe_top30['Sites'].values
sites_depth = dframe_depth['Sites'].values


 
dframe_both_master_sites = []
dframe_both_master_lat = []
dframe_both_master_lon = []

dframe_top30_master_sites = []
dframe_top30_master_lat = []
dframe_top30_master_lon = []

dframe_depth_master_sites = []
dframe_depth_master_lat = []
dframe_depth_master_lon = []

for i in sites_top30: #loop through sites in top_30cm
    dframe_top30_site = dframe_top30[dframe_top30['Sites'] == i]
    dframe_depth_site = dframe_depth[dframe_depth['Sites'] == i]

    if (len(dframe_depth_site) == 0):  #check to see if site is also present in depth data, and store site details in dframe_both_master if this is true
    	site_number = dframe_top30_site['Sites'].iloc[0]
    	site_lat = dframe_top30_site['Lat'].iloc[0]
    	site_lon = dframe_top30_site['Long'].iloc[0]
    	dframe_top30_master_sites.append(site_number)
    	dframe_top30_master_lat.append(site_lat)
    	dframe_top30_master_lon.append(site_lon)

    else: #else store in dataframe for top-30cm only
    	site_number = dframe_depth_site['Sites'].iloc[0]
    	site_lat = dframe_depth_site['Lat'].iloc[0]
    	site_lon = dframe_depth_site['Long'].iloc[0]
    	dframe_both_master_sites.append(site_number)
    	dframe_both_master_lat.append(site_lat)
    	dframe_both_master_lon.append(site_lon)
	


dframe_both = pd.DataFrame(data=dframe_both_master_sites, columns=['Site'])
dframe_both['Lat'] = dframe_both_master_lat
dframe_both['Lon'] = dframe_both_master_lon
both_site = dframe_both['Site'].values


for j in sites_depth:
    dframe_top30_site = dframe_top30[dframe_top30['Sites'] == j]
    dframe_depth_site = dframe_depth[dframe_depth['Sites'] == j]

    if (len(dframe_top30_site) == 0):  #check to see if dframe_top30cm is empty. If it is empty, store site details in dframe_depth_master
    	site_number = dframe_depth_site['Sites'].iloc[0]
    	site_lat = dframe_depth_site['Lat'].iloc[0]
    	site_lon = dframe_depth_site['Long'].iloc[0]
    	dframe_depth_master_sites.append(site_number)
    	dframe_depth_master_lat.append(site_lat)
    	dframe_depth_master_lon.append(site_lon)
    	
    
    else:
    	check_dframe_both = dframe_both[dframe_both['Site'] == j]
    	check_sites = check_dframe_both['Site'].iloc[0]
    	if (check_sites == ""): #check to see if site already exists in dframe_both_master. If not, store site details in dframe_both
    		site_number = dframe_depth_site['Sites'].iloc[0]
    		site_lat = dframe_depth_site['Lat'].iloc[0]
    		site_lon = dframe_depth_site['Long'].iloc[0]
    		dframe_both_master_sites.append(site_number)
    		dframe_both_master_lat.append(site_lat)
    		dframe_both_master_lon.append(site_lon)
    		
    	else:
    		continue

dframe_depth_final = pd.DataFrame(data=dframe_depth_master_sites, columns=['Site'])
dframe_depth_final['Lat'] = dframe_depth_master_lat
dframe_depth_final['Lon'] = dframe_depth_master_lon

dframe_top30_final = pd.DataFrame(data=dframe_top30_master_sites, columns=['Site'])
dframe_top30_final['Lat'] = dframe_top30_master_lat
dframe_top30_final['Lon'] = dframe_top30_master_lon

dframe_both_final = pd.DataFrame(data=dframe_both_master_sites, columns=['Site'])
dframe_both_final['Lat'] = dframe_both_master_lat
dframe_both_final['Lon'] = dframe_both_master_lon


print(dframe_depth_final)
print(dframe_top30_final)
print(dframe_both_final)


dframe_top30_final_lat = dframe_top30_final['Lat'].values
dframe_top30_final_lon = dframe_top30_final['Lon'].values

dframe_depth_final_lat = dframe_depth_final['Lat'].values
dframe_depth_final_lon = dframe_depth_final['Lon'].values

dframe_both_final_lat = dframe_both_final['Lat'].values
dframe_both_final_lon = dframe_both_final['Lon'].values

########### Create Plot #############
fig = plt.figure(figsize = (20,10))

#create array for parallels
parallels = np.arange(40.,81.,10.)

map1 = Basemap(projection='npstere',boundinglat=40,lon_0=180,resolution='h')
#map1.drawcoastlines(zorder=2)
map1.drawmapboundary(fill_color='aqua',zorder=0) # fill to edge
map1.fillcontinents(color='palegoldenrod',lake_color='aqua',zorder=1)

#draw parallels on map
map1.drawparallels(parallels,labels=[False,True,True,False])

#plot scatterplot
x1,y1 = map1(dframe_top30_final_lon,dframe_top30_final_lat)
Site_top30 = plt.scatter(x1,y1,alpha=0.6,marker='.',s=100,color='red',zorder=3,label='Near Surface Only') 
x2,y2 = map1(dframe_depth_final_lon,dframe_depth_final_lat)
Site_depth = plt.scatter(x2,y2,alpha=0.6,marker='.',s=100,color='green',zorder=4,label='Depth Only')
x3,y3 = map1(dframe_both_final_lon,dframe_both_final_lat)
Site_both = plt.scatter(x3,y3,alpha=0.6,marker='.',s=100,color='blue',zorder=5,label='Both Depths')

plt.legend(loc='best')
plt.tight_layout()
plt_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/plots/validation_sites/new_data/CLSM_res/Site_Locations_ForBrendan.png'])
fig.savefig(plt_fil)
plt.close()
