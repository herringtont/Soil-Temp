import os
import glob
import netCDF4
import csv
import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import matplotlib.patches as mpl_patches
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
from scipy.stats import pearsonr
from matplotlib.ticker import (MultipleLocator, AutoLocator, AutoMinorLocator)


val_stn_fil_top30 = "/mnt/data/users/herringtont/soil_temp/naive_blend_timeseries/summary_fil/new_data/remapcon_simple_average_zscore_top_30cm_thr100_gcell_summary_newdepth.csv"
val_stn_fil_30_300 = "/mnt/data/users/herringtont/soil_temp/naive_blend_timeseries/summary_fil/new_data/remapcon_simple_average_zscore_30cm_300cm_thr100_gcell_summary_newdepth.csv"


geom_fil_top30 = "/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/geometry_top30_con.csv"
geom_fil_30_300 = "/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/geometry_L7_con.csv"

sites_2yr_fil_top30 = "/mnt/data/users/herringtont/soil_temp/sites_2yr/sites_2yr_zscore_top_30cm_100.csv"
sites_2yr_fil_30_300 = "/mnt/data/users/herringtont/soil_temp/sites_2yr/sites_2yr_zscore_30_299.9_100.csv"

dframe_gcell_top30 = pd.read_csv(val_stn_fil_top30)
dframe_gcell_30_300 = pd.read_csv(val_stn_fil_30_300)

dframe_sites_2yr_top30 = pd.read_csv(sites_2yr_fil_top30)
dframe_sites_2yr_30_300 = pd.read_csv(sites_2yr_fil_30_300)

dframe_geom_top30 = pd.read_csv(geom_fil_top30)
dframe_geom_30_300 = pd.read_csv(geom_fil_30_300)

gcell_top30 = dframe_gcell_top30['Grid Cell'].values
sites_2yr_top30 = dframe_sites_2yr_top30['Sites'].values

gcell_30_300 = dframe_gcell_30_300['Grid Cell'].values
sites_2yr_30_300 = dframe_sites_2yr_30_300['Sites'].values


######## top 30cm layer #######

site_master_top30 = []
lat_master_top30 = []
lon_master_top30 = []
cen_lat_master_top30 = []
cen_lon_master_top30 = []
grid_master_top30 = []
dtst_master_top30 = []

for gcell in gcell_top30:
    dframe_geom_gcell_top30 = dframe_geom_top30[dframe_geom_top30['Grid Cell'] == gcell]
    #print(dframe_geom_gcell_top30)

    for sites in sites_2yr_top30:
    	dframe_geom_sites_top30 = dframe_geom_gcell_top30[dframe_geom_gcell_top30['site'] == sites]
    	if (len(dframe_geom_sites_top30) > 0):
    		site_i = dframe_geom_sites_top30['site'].values
    		#print(site_i)
    		site_master_top30.append(site_i)
    		lat_i = dframe_geom_sites_top30['lat'].values
    		lat_master_top30.append(lat_i)
    		lon_i = dframe_geom_sites_top30['lon'].values
    		lon_master_top30.append(lon_i)		
    		grid_i = dframe_geom_sites_top30['Grid Cell'].values
    		grid_master_top30.append(grid_i)
    		cen_lat_i = dframe_geom_sites_top30['Lat Cen'].values
    		cen_lat_master_top30.append(cen_lat_i)
    		cen_lon_i = dframe_geom_sites_top30['Lon Cen'].values
    		cen_lon_master_top30.append(cen_lon_i)
    		if (int(site_i) < 70):
    			dtst_i = 'GTN-P'
    		elif (70 <= int(site_i) < 300):
    			dtst_i = 'Kropp'
    		elif (300 <= int(site_i) < 758):
    			dtst_i = 'Russian'
    		elif (int(site_i) >= 758):
    			dtst_i = 'Nordicana'
    		dtst_master_top30.append(dtst_i)
    		#print(dtst_i)
site_master_top30 = [i for sub in site_master_top30 for i in sub]
lat_master_top30 = [i for sub in lat_master_top30 for i in sub]
lon_master_top30 = [i for sub in lon_master_top30 for i in sub]
cen_lat_master_top30 = [i for sub in cen_lat_master_top30 for i in sub]
cen_lon_master_top30 = [i for sub in cen_lon_master_top30 for i in sub]
grid_master_top30 = [i for sub in grid_master_top30 for i in sub]
#dtst_master_top30 = [i for sub in dtst_master_top30 for i in sub]

dframe_master_top30 = pd.DataFrame(data=site_master_top30, columns=['Site'])
dframe_master_top30['Lat'] = lat_master_top30
dframe_master_top30['Lon'] = lon_master_top30
dframe_master_top30['Grid Cell'] = grid_master_top30
dframe_master_top30['Central Lat'] = cen_lat_master_top30
dframe_master_top30['Central Lon'] = cen_lon_master_top30
dframe_master_top30['Dataset'] = dtst_master_top30

print(dframe_master_top30)
top30_ofil = "/mnt/data/users/herringtont/soil_temp/naive_blend_timeseries/summary_fil/new_data/extended_summary/remapcon_simple_average_zscore_top_30cm_thr100_gcell_extended_summary.csv"
print(top30_ofil)
dframe_master_top30.to_csv(top30_ofil,index=False)



site_master_30_300 = []
lat_master_30_300 = []
lon_master_30_300 = []
cen_lat_master_30_300 = []
cen_lon_master_30_300 = []
grid_master_30_300 = []
dtst_master_30_300 = []

for gcell in gcell_30_300:
    dframe_geom_gcell_30_300 = dframe_geom_30_300[dframe_geom_30_300['Grid Cell'] == gcell]
    #print(dframe_geom_gcell_30_100)

    for sites in sites_2yr_30_300:
    	dframe_geom_sites_30_300 = dframe_geom_gcell_30_300[dframe_geom_gcell_30_300['site'] == sites]
    	if (len(dframe_geom_sites_30_300) > 0):
    		site_i = dframe_geom_sites_30_300['site'].values
    		#print(site_i)
    		site_master_30_300.append(site_i)
    		lat_i = dframe_geom_sites_30_300['lat'].values
    		lat_master_30_300.append(lat_i)
    		lon_i = dframe_geom_sites_30_300['lon'].values
    		lon_master_30_300.append(lon_i)		
    		grid_i = dframe_geom_sites_30_300['Grid Cell'].values
    		grid_master_30_300.append(grid_i)
    		cen_lat_i = dframe_geom_sites_30_300['Lat Cen'].values
    		cen_lat_master_30_300.append(cen_lat_i)
    		cen_lon_i = dframe_geom_sites_30_300['Lon Cen'].values
    		cen_lon_master_30_300.append(cen_lon_i)
    		if (int(site_i) < 70):
    			dtst_i = 'GTN-P'
    		elif (70 <= int(site_i) < 300):
    			dtst_i = 'Kropp'
    		elif (300 <= int(site_i) < 758):
    			dtst_i = 'Russian'
    		elif (int(site_i) >= 758):
    			dtst_i = 'Nordicana'
    		dtst_master_30_300.append(dtst_i)
    		#print(dtst_i)
site_master_30_300 = [i for sub in site_master_30_300 for i in sub]
lat_master_30_300 = [i for sub in lat_master_30_300 for i in sub]
lon_master_30_300 = [i for sub in lon_master_30_300 for i in sub]
cen_lat_master_30_300 = [i for sub in cen_lat_master_30_300 for i in sub]
cen_lon_master_30_300 = [i for sub in cen_lon_master_30_300 for i in sub]
grid_master_30_300 = [i for sub in grid_master_30_300 for i in sub]
#dtst_master_30_100 = [i for sub in dtst_master_30_100 for i in sub]

dframe_master_30_300 = pd.DataFrame(data=site_master_30_300, columns=['Site'])
dframe_master_30_300['Lat'] = lat_master_30_300
dframe_master_30_300['Lon'] = lon_master_30_300
dframe_master_30_300['Grid Cell'] = grid_master_30_300
dframe_master_30_300['Central Lat'] = cen_lat_master_30_300
dframe_master_30_300['Central Lon'] = cen_lon_master_30_300
dframe_master_30_300['Dataset'] = dtst_master_30_300

print(dframe_master_30_300)

ofil_30_300 = "/mnt/data/users/herringtont/soil_temp/naive_blend_timeseries/summary_fil/new_data/extended_summary/remapcon_simple_average_zscore_30cm_100cm_thr100_gcell_extended_summary_newdepth.csv"
print(ofil_30_300)

dframe_master_30_300.to_csv(ofil_30_300,index=False)



