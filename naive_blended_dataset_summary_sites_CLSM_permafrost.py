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



#################### Create Summary File ##############

############## Set Directories #################

two_year_dir = '/mnt/data/users/herringtont/soil_temp/sites_2yr/CLSM/'
geom_dir = '/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/CLSM/'
timeseries_dir = '/mnt/data/users/herringtont/soil_temp/naive_blend_timeseries/new_data/CLSM_res/subset/'

naive_type = ['simple_average']
olr = ['zscore']#['outliers','zscore','IQR']
lyr = ['top_30cm','30cm_300cm']
thr = ['100']#['0','25','50','75','100']
rmp_type = ['con']#['nn','bil']
tmp_type = ['raw_temp']
permafrost_type = ['RS_2002_continuous','Brown_1970_continuous','RS_2002_discontinuous','Brown_1970_discontinuous','RS_2002_none','Brown_1970_none']
############## Loop through files ###############

for i in rmp_type:
    rmp_type_i = i
    remap_type = ''.join(['remap'+rmp_type_i])
    
    for j in naive_type:
    	naive_type_j = j

    	for k in olr:
    		olr_k = k

    		for l in lyr:
    			lyr_l = l

    			if (lyr_l == "top_30cm"):
    				geom_layer = 'top30'
    				two_year_layer = 'top_30cm'
				
    			elif (lyr_l == "30cm_300cm"):
    				geom_layer = 'L3'
    				two_year_layer = '30_299.9'


    			for m in thr:
    				thr_m = m

    				two_year_fil = ''.join([two_year_dir+'sites_2yr_'+str(olr_k)+'_'+two_year_layer+'_'+str(thr_m)+'.csv'])
    				#print(two_year_fil)
    				dframe_two_year = pd.read_csv(two_year_fil)

    				geom_fil = ''.join([geom_dir+'geometry_'+geom_layer+'_'+rmp_type_i+'_CLSM.csv'])
    				#print(geom_fil)
    				dframe_geom = pd.read_csv(geom_fil)



  
    				timeseries_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/CLSM_res/subset/'+str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_dframe_scatterplot_BEST_'+str(temp_thr_o)+'_CMOS_CLSM_subset_permafrost.csv'])
    				print(timeseries_fil)
    				dframe_timeseries = pd.read_csv(timeseries_fil)

    				for o in permafrost_type:
    					permafrost_type_o = o
    					if (permafrost_type_o == 'RS_2002_continuous'):
    						dframe_timeseries_permafrost = dframe_timeseries[dframe_timeseries['RS 2002 Permafrost'] == 'continuous']

    					elif (permafrost_type_o == 'Brown_1970_continuous'):
    						dframe_timeseries_permafrost = dframe_timeseries[dframe_timeseries['Brown 1970 Permafrost'] == 'continuous']

    					elif (permafrost_type_o == 'RS_2002_discontinuous'):
    						dframe_timeseries_permafrost = dframe_timeseries[dframe_timeseries['RS 2002 Permafrost'] == 'discontinuous']

    					elif (permafrost_type_o == 'Brown_1970_discontinuous'):
    						dframe_timeseries_permafrost = dframe_timeseries[dframe_timeseries['Brown 1970 Permafrost'] == 'discontinuous']

    					elif (permafrost_type_o == 'Brown_1970_none'):
    						dframe_timeseries_permafrost = dframe_timeseries[dframe_timeseries['Brown 1970 Permafrost'] == 'none']

    					elif (permafrost_type_o == 'RS_2002_none'):
    						dframe_timeseries_permafrost = dframe_timeseries[dframe_timeseries['RS 2002 Permafrost'] == 'none']

    					gcell = dframe_timeseries_permafrost['Grid Cell'].values

    					gcell_uq = np.unique(gcell)


    					gcell_master = []
    					lat_master = []
    					lon_master = []
    					sites_master = []
    					for n in gcell_uq:
    						gcell_n = n
    						gcell_master.append(gcell_n)
    						dframe_gcell = dframe_timeseries_permafrost[dframe_timeseries_permafrost['Grid Cell'] == gcell_n]
    						cen_lat = dframe_gcell['Central Lat'].iloc[0]
    						lat_master.append(cen_lat)
    						cen_lon = dframe_gcell['Central Lon'].iloc[0]
    						lon_master.append(cen_lon)
    						avg_sites = dframe_gcell['Sites Incl'].iloc[0]
    						sites_master.append(avg_sites)

    					dframe_master = pd.DataFrame(data=gcell_master, columns=['Grid Cell'])
    					dframe_master['Central Lat'] = lat_master
    					dframe_master['Central Lon'] = lon_master
    					dframe_master['Avg Sites Incl'] = sites_master

    					print(dframe_master)

    					summary_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_timeseries/summary_fil/new_data/CLSM_res/'+str(remap_type)+'_'+str(lyr_l)+'_thr'+str(thr_m)+'_gcell_summary_CLSM_'+str(permafrost_type_o)+'.csv'])
    					print(summary_fil)
    					dframe_master.to_csv(summary_fil,index=False)  


#################### Create Extended Summary File ###################

for z in permafrost_type:
    permafrost_type_z = z

    val_stn_fil_top30 = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_timeseries/summary_fil/new_data/CLSM_res/remapcon_top_30cm_thr100_gcell_summary_CLSM_'+str(permafrost_type_o)+'.csv'])
    val_stn_fil_30_300 =''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_timeseries/summary_fil/new_data/CLSM_res/remapcon_30cm_300cm_thr100_gcell_summary_CLSM_'+str(permafrost_type_o)+'.csv'])

    geom_fil_top30 = "/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/CLSM/geometry_top30_con_CLSM.csv"
    geom_fil_30_300 = "/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/CLSM/geometry_L7_con_CLSM.csv"

    sites_2yr_fil_top30 = "/mnt/data/users/herringtont/soil_temp/sites_2yr/CLSM/sites_2yr_zscore_top_30cm_100.csv"
    sites_2yr_fil_30_300 = "/mnt/data/users/herringtont/soil_temp/sites_2yr/CLSM/sites_2yr_zscore_30_299.9_100.csv"

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
    top30_ofil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_timeseries/summary_fil/new_data/extended_summary/CLSM_res/remapcon_top_30cm_thr100_gcell_extended_summary_'+str(permafrost_type_z)+'.csv'])
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
    	#print(dframe_geom_gcell_30_300)

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
    #dtst_master_30_300 = [i for sub in dtst_master_30_300 for i in sub]

    dframe_master_30_300 = pd.DataFrame(data=site_master_30_300, columns=['Site'])
    dframe_master_30_300['Lat'] = lat_master_30_300
    dframe_master_30_300['Lon'] = lon_master_30_300
    dframe_master_30_300['Grid Cell'] = grid_master_30_300
    dframe_master_30_300['Central Lat'] = cen_lat_master_30_300
    dframe_master_30_300['Central Lon'] = cen_lon_master_30_300
    dframe_master_30_300['Dataset'] = dtst_master_30_300


    print(dframe_master_30_300)

    ofil_30_300 = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_timeseries/summary_fil/new_data/extended_summary/CLSM_res/remapcon_30cm_300cm_thr100_gcell_extended_summary_'+str(permafrost_type_z)+'.csv'])

    print(ofil_30_300)
    dframe_master_30_300.to_csv(ofil_30_300,index=False)

