import os
import glob
import netCDF4
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
import pytesmo
import math
import cftime
import pathlib
import re
import cdo
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
from dateutil.relativedelta import *


######################## set missing threshold ##############################
val_thresh = 0 #percent missing allowed (100, 75, 50, 25, 0)

miss_thr = 100 - val_thresh #percent valid required in order to be included in monthly average


######################## define functions ###################################
def str_to_datetime(column, date_fmt):

    date_list = []

    for dt_str in column:
        new_dt = datetime.datetime.strptime(dt_str, date_fmt)
        date_list.append(new_dt)
			

    return date_list


def load_pandas(file_name):
    print("Loading in-situ file: ", file_name)
    dframe = pd.read_csv(file_name)
    


    

############################## grab sites and grid cells for each soil layer ##############################
geom_dir = "/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/CLSM/"

olr = ['outliers','zscore','IQR']
lyr = ['0_9.9','10_29.9','30_99.9','100_299.9','300_deeper','top_30cm','30_299.9']
thr = ['0','25','50','75','100']
rmp_type = ['nn','bil','con']

sit_master = []
grid_master = []
olr_master = []
lyr_master = []
thr_master = []
lat_master = []
lon_master = []
rmp_master = []

############ loop through remap style ##############
for h in rmp_type:

############ loop through outlier type #############
    for i in olr:
    	olri = i

############ loop through soil layer ##############    
    	for j in lyr:
    		lyrj = j
    		if (j == '0_9.9'):
    			if(h == "nn"):
    				geom_fil = "".join([geom_dir,"geometry_L1_nn_CLSM.csv"])
    			if(h == "bil"):
    				geom_fil = "".join([geom_dir,"geometry_L1_bil_CLSM.csv"])
    			if(h == "con"):
    				geom_fil = "".join([geom_dir,"geometry_L1_con_CLSM.csv"])
    		if (j == '10_29.9'):
    			if(h == "nn"):
    				geom_fil = "".join([geom_dir,"geometry_L2_nn_CLSM.csv"])
    			if(h == "bil"):
    				geom_fil = "".join([geom_dir,"geometry_L2_bil_CLSM.csv"])
    			if(h == "con"):
    				geom_fil = "".join([geom_dir,"geometry_L2_con_CLSM.csv"])
    		if (j == '30_99.9'):
    			if(h == "nn"):
    				geom_fil = "".join([geom_dir,"geometry_L3_nn_CLSM.csv"])
    			if(h == "bil"):
    				geom_fil = "".join([geom_dir,"geometry_L3_bil_CLSM.csv"])
    			if(h == "con"):
    				geom_fil = "".join([geom_dir,"geometry_L3_con_CLSM.csv"])
    		if (j == '100_299.9'):
    			if(h == "nn"):
    				geom_fil = "".join([geom_dir,"geometry_L4_nn_CLSM.csv"])
    			if(h == "bil"):
    				geom_fil = "".join([geom_dir,"geometry_L4_bil_CLSM.csv"])
    			if(h == "con"):
    				geom_fil = "".join([geom_dir,"geometry_L4_con_CLSM.csv"])
    		if (j == '300_deeper'):
    			if(h == "nn"):
    				geom_fil = "".join([geom_dir,"geometry_L5_nn_CLSM.csv"])
    			if(h == "bil"):
    				geom_fil = "".join([geom_dir,"geometry_L5_bil_CLSM.csv"])
    			if(h == "con"):
    				geom_fil = "".join([geom_dir,"geometry_L5_con_CLSM.csv"])
    		if (j == 'top_30cm'):
    			if(h == "nn"):
    				geom_fil = "".join([geom_dir,"geometry_top30_nn_CLSM.csv"])
    			if(h == "bil"):
    				geom_fil = "".join([geom_dir,"geometry_top30_bil_CLSM.csv"])
    			if(h == "con"):
    				geom_fil = "".join([geom_dir,"geometry_top30_con_CLSM.csv"])
    		if (j == '30_299.9'):
    			if(h == "nn"):
    				geom_fil = "".join([geom_dir,"geometry_L7_nn_CLSM.csv"])
    			if(h == "bil"):
    				geom_fil = "".join([geom_dir,"geometry_L7_bil_CLSM.csv"])
    			if(h == "con"):
    				geom_fil = "".join([geom_dir,"geometry_L7_con_CLSM.csv"])
    		dframe_geom = pd.read_csv(geom_fil)
    		sit_geom = dframe_geom['site']
    		#print(dframe_geom)		

############# loop through missing threshold ##########    	
    		for k in thr:
    			thrk = k
    			two_year_fil = "".join(["/mnt/data/users/herringtont/soil_temp/sites_2yr/CLSM/sites_2yr_",olri,"_",lyrj,"_",thrk,".csv"])
    			#print(two_year_fil)
    			dframe_2yr = pd.read_csv(two_year_fil)
    			sites_2yr = dframe_2yr['Sites'].values	

    			for l in range(0,len(sites_2yr)):
    				sitid = sites_2yr[l]
    				if (sitid > 788):
    					continue
    				#print(sitid)
    				site_grid = dframe_geom[dframe_geom['site'] == int(sitid)]
    				grid_cell = site_grid['Grid Cell'].values
    				lat = site_grid['Lat Cen'].values
    				lon = site_grid['Lon Cen'].values
    				#if (len(grid_cell) > 1):
    				grid_cell = grid_cell[0]
    				lat = lat[0]
    				lon = lon[0]
    				#print("Grid Cell:",grid_cell)
    				#print("Lat:",lat)
    				#print("Lon:",lon)
    				sit_master.append(sitid)
    				grid_master.append(grid_cell)
    				olr_master.append(olri)
    				lyr_master.append(lyrj)
    				thr_master.append(thrk)
    				lat_master.append(lat)
    				lon_master.append(lon)
    				rmp_master.append(h)

#print(sit_master)
#print(grid_master)
#print(olr_master)
#print(lyr_master)
#print(thr_master)
#print(lat_master)
#print(lon_master)
#print(rmp_master)

#sit_master = [m for sub in sit_master for m in sub]
#grid_master = [m for sub in grid_master for m in sub]
#lat_master = [m for sub in lat_master for m in sub]
#lon_master = [m for sub in lon_master for m in sub]
#
sit_master2 = np.array(sit_master).flatten()
grid_master2 = np.array(grid_master).flatten()
lat_master2 = np.array(lat_master).flatten()
lon_master2 = np.array(lon_master).flatten()
olr_master2 = np.array(olr_master).flatten()
lyr_master2 = np.array(lyr_master).flatten()
thr_master2 = np.array(thr_master).flatten()
rmp_master2 = np.array(rmp_master).flatten()

#print(sit_master)
#print(grid_master)
#print(olr_master)
#print(lyr_master)
#print(thr_master)
#print(lat_master)
#print(lon_master)
#print(rmp_master)

#print(len(sit_master2))
#print(len(grid_master2))
#print(len(lat_master2))
#print(len(lon_master2))
#print(len(olr_master2))
#print(len(lyr_master2))
#print(len(thr_master2))
#print(len(rmp_master2))

############## create master dataframe with sites from all combinations of remap style, outlier type, soil layer and missing threshold ###########

data_master = {'Site':sit_master2,'Lat':lat_master2,'Lon':lon_master2,'Grid Cell':grid_master2,'Remap':rmp_master2,'Outlier':olr_master2,'Layer':lyr_master2,'Threshold':thr_master}

dframe_master = pd.DataFrame(data_master)
#print(dframe_master)

############# grab sites within a particular grid cell for each combination of remap style, outlier type, soil layer, and missing threshold ############
for i in rmp_type:
    rmpi = i 
    for j in olr:
    	olrj = j
    	for k in lyr:
    		lyrk = k
    		for l in thr:
    			thrl = l
    			dframe_master_new = dframe_master[(dframe_master['Remap'] == rmpi) & (dframe_master['Outlier'] == olrj) & (dframe_master['Layer'] == lyrk) & (dframe_master['Threshold'] == thrl)]
    			#print(dframe_master_new)
    			grid_new = dframe_master_new['Grid Cell']
    			grid_new_uq = np.unique(grid_new)

    			#print(grid_new_uq)
    						
    			for m in grid_new_uq:
    				date_grid = []
    				dframe_gcell = "None"
    				dframe_gcella = "None"			
    				dframe_grid = dframe_master_new[dframe_master_new['Grid Cell'] == m]
    				print(dframe_grid)
    				gcell = m
    				print("the grid cell is:",gcell)
    				sites = dframe_grid['Site']
    				print("Sites:",sites)
    				lat_g = dframe_grid['Lat']
    				lat_g = lat_g.iloc[0]
    				lon_g = dframe_grid['Lon']
    				lon_g = lon_g.iloc[0]   				
################# get unique dates to reindex soil temperature against ##########################
    				for n in sites:
    					#print("the threshold is:",thrl)
    					sitid = n
    					if (sitid == 6):
    						continue
    					if (sitid > 788): #skip all NWT and Yukon sites
    						continue	
    					stemp_fil = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/",olrj,"/",lyrk,"/monthly_average/thr_",thrl,"/CLSM/site_",str(sitid),".csv"])
    					#print(stemp_fil)
    					dframe_stemp = pd.read_csv(stemp_fil)
    					#print(dframe_stemp)			
    					dates = dframe_stemp['Date']
    					date_grid.append(dates)
    					#print(len(dates))					
    				date_grid2 = [p for sub in date_grid for p in sub]
    				date_grid3 = np.asarray(date_grid2)
    				date_grid_uq = np.unique(date_grid3)
    				#print(len(date_grid_uq))
    				#print(len(date_grid_uq))

################# reindex all files within a grid cell onto a common time axis ######################

    				for o in sites:
    					sitid2 = o
    					if (sitid2 == 6):
    						continue
    					if (sitid2 > 788): #skip all NWT and Yukon sites
    						continue
    					stemp_fil2 = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/",olrj,"/",lyrk,"/monthly_average/thr_",thrl,"/CLSM/site_",str(sitid2),".csv"])
    					#print(stemp_fil2)    					
    					dframe_stemp2 = pd.read_csv(stemp_fil2)	
    					dates2 = dframe_stemp2['Date']
    					dframe_stemp3 = dframe_stemp2.set_index(dates2)
    					dframe_rdx = dframe_stemp3.reindex(date_grid_uq,fill_value=np.nan)
    					#print(dframe_rdx)
    					stemp_sit = dframe_rdx['Layer_Avg']
    					#print("the site ID is:", sitid2)
    					#print(stemp_sit)
    					stemp_anom_sit = dframe_rdx['Layer_Anom']
    					stemp_anom_sit2 = stemp_anom_sit.values
    					stemp_sit2 = stemp_sit.values
    					#print(stemp_sit)					
################## store temperature data of all sites within a grid cell in a common dataframe #####################
    					if(len(dframe_gcell) == 4):  #if pandas dataframe does not exist, create it
    						dframe_gcell = pd.DataFrame(data=stemp_sit2, columns=[sitid2])
    					elif(len(dframe_gcell) > 4):
    						dframe_gcell[sitid2] = stemp_sit2
    					#print(dframe_gcell)
################# store temperature anomalies of all sites within a grid cell in a common dataframe #################
    					if(len(dframe_gcella) == 4):
    						dframe_gcella = pd.DataFrame(data=stemp_anom_sit2, columns=[sitid2])
    					elif(len(dframe_gcella) > 4):
    						dframe_gcella[sitid2] = stemp_anom_sit2    						
    					#print(dframe_gcella)
    				#print("Anomaly Length:",len(stemp_anom_sit2))
    				#print("Raw Temp Length:",len(stemp_sit2))

################## do the spatial averaging ###################
    				if(len(dframe_gcell) > 4):
    					sit_count = dframe_gcell.count(axis=1)
    					dframe_gcell['Spatial Avg'] = dframe_gcell.mean(axis=1)
    					temp_raw_avg = dframe_gcell['Spatial Avg'].values
    					dframe_gcell['Sites Incl'] = sit_count
    					dframe_gcell.insert(0,'Date',date_grid_uq)
    					dframe_gcell.insert(1,'Grid Cell',gcell)
    					dframe_gcell.insert(2,'Central Lat',lat_g)
    					dframe_gcell.insert(3,'Central Lon',lon_g)
    					dframe_gcell.drop(dframe_gcell[dframe_gcell['Sites Incl'] == 0].index, inplace=True)
    					ofil = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/spatial_average/remap",rmpi,"/no_outliers/",olrj,"/",lyrk,"/thr_",thrl,"/CLSM/Sep2021/grid_",str(gcell),"_Sep2021.csv"])
    					path = pathlib.Path(ofil)
    					path.parent.mkdir(parents=True,exist_ok=True)
    					print(ofil)
    					print(dframe_gcell)
    					dframe_gcell.to_csv(ofil,index=False,na_rep="NaN") 				    					 
    									
    				if(len(dframe_gcella) > 4):
    					sit_counta = dframe_gcella.count(axis=1)
    					dframe_gcella['Spatial Avg Anom'] = dframe_gcella.mean(axis=1)
    					dframe_gcella['Spatial Avg Temp'] = temp_raw_avg
    					dframe_gcella['Sites Incl'] = sit_counta
    					dframe_gcella.insert(0,'Date',date_grid_uq)
    					dframe_gcella.insert(1,'Grid Cell',gcell)
    					dframe_gcella.insert(2,'Central Lat',lat_g)
    					dframe_gcella.insert(3,'Central Lon',lon_g)
    					dframe_gcella.drop(dframe_gcella[dframe_gcella['Sites Incl'] == 0].index, inplace=True)
    					ofila = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/spatial_average_anom/remap",rmpi,"/no_outliers/",olrj,"/",lyrk,"/thr_",thrl,"/CLSM/Sep2021/grid_",str(gcell),"_anom_Sep2021.csv"])
    					patha = pathlib.Path(ofila)
    					patha.parent.mkdir(parents=True,exist_ok=True)
    					print(ofila)
    					print(dframe_gcella)
    					dframe_gcella.to_csv(ofila,index=False,na_rep="NaN")				
