import os
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
from calendar import isleap
from dateutil.relativedelta import *
from pathlib import Path
from matplotlib.ticker import (MultipleLocator, AutoLocator, AutoMinorLocator)

################### define global functions ################
def str_to_datetime(column, date_fmt):

    date_list = []

    for dt_str in column:
        new_dt = datetime.datetime.strptime(dt_str, date_fmt)
        date_list.append(new_dt)
			

    return date_list
    
def remove_trailing_zeros(x):
    return str(x).rstrip('0').rstrip('.')


############################## grab sites and grid cells for each soil layer ##############################
geom_dir = "/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/"

olr = ['outliers','zscore','IQR']
lyr = ['0_9.9','10_29.9','30_99.9','100_299.9','300_deeper']
thr = ['0','25','50','75','100']
rmp_type = ['nn','bil']


####################### Raw Temperatures #################################
################# loop through in-situ files ###############
for h in rmp_type: #loops through remap type
    rmph = h
    if(rmph == "nn"):
    	remap_type = "remapnn"
    elif(rmph == "bil"):
    	remap_type = "remapbil"    	 
    for i in olr: #loops throuh outlier type
    	olri = i
    	for j in lyr: #loops through layer
    		lyrj = j
    		for k in thr: #loops through missing threshold
    			thr_dir = 'thr_'+str(k)
################################# create master arrays ########################
    			multiple_site_grid = []
    			multiple_site_num = []
    			multiple_site_lat = []
    			multiple_site_lon = []
    			multiple_site_sites = []

    			single_site_grid = []
    			single_site_num = []
    			single_site_lat = []
    			single_site_lon = []
    			single_site_sites = []
			
    			thrk = k
    			thrshld = "".join(["thr_"+str(thrk)])
    			indir = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/spatial_average/remap",rmph,"/no_outliers/",olri,"/",lyrj,"/thr_",thrk,"/"])
    			indira = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/spatial_average_anom/remap",rmph,"/no_outliers/",olri,"/",lyrj,"/thr_",thrk,"/"])
    			pathlist = Path(indir).glob('*.csv')
    			pathlista = Path(indira).glob('*anom.csv')
    			#print(indir)		

################################## loop through files within a threshold directory ##################################
    			for path in sorted(pathlista, key=lambda path: int(path.stem.split("_")[1].split("*.csv")[0])):
    				fil = str(path)
    				dframe = pd.read_csv(fil)
    				Num_Sites = dframe['Sites Incl']
    				mean_sites = round(np.mean(Num_Sites.values),2)
    				#print("mean number of sites:", mean_sites)
    				gcell = dframe['Grid Cell'].iloc[0]
    				lat_cen = dframe['Central Lat'].iloc[0]
    				lon_cen = dframe['Central Lon'].iloc[0]
    				column_names = list(dframe.columns)
    				#print(column_names)
    				len_column_names = len(column_names)
    				#print(len_column_names)
    				site_names = []
    				for x in range(0,len_column_names):   
    					column_i = column_names[x]
    					column_int_check = column_i.isdigit() ####check if column has numeric digits (all column names with numerical digits are site names)
    					#print(column_int_check)
    					if(column_int_check == True):
    						site_names.append(column_i)



################################# sort into single or multiple site arrays ##########################################

    				if (mean_sites >= 1.5): #if multiple sites for 50% or more of grid cells
    					multiple_site_grid.append(gcell)
    					multiple_site_num.append(mean_sites)
    					multiple_site_lat.append(lat_cen)
    					multiple_site_lon.append(lon_cen)
    					multiple_site_sites.append(site_names)
    				elif (mean_sites < 1.5): # if single site present for more than half of grid cells 
    					single_site_grid.append(gcell)
    					single_site_num.append(mean_sites)
    					single_site_lat.append(lat_cen)
    					single_site_lon.append(lon_cen)
    					single_site_sites.append(site_names)

    			#print(multiple_site_sites)

    			dframe_multiple = pd.DataFrame(data=multiple_site_grid, columns=['Grid Cell'])
    			#dframe_multiple['Sites'] = multiple_site_sites
    			dframe_multiple['Avg Num of Sites'] = multiple_site_num
    			dframe_multiple['Lat'] = multiple_site_lat
    			dframe_multiple['Lon'] = multiple_site_lon
    			dframe_multiple['Remap Type'] = remap_type
    			dframe_multiple['Outlier Type'] = olri
    			dframe_multiple['Layer'] = lyrj
    			dframe_multiple['Threshold'] = thr_dir

    			dframe_single = pd.DataFrame(data=single_site_grid, columns=['Grid Cell'])
    			#dframe_single['Sites'] = single_site_sites
    			dframe_single['Avg Num of Sites'] = single_site_num
    			dframe_single['Lat'] = single_site_lat
    			dframe_single['Lon'] = single_site_lon
    			dframe_single['Remap Type'] = remap_type
    			dframe_single['Outlier Type'] = olri
    			dframe_single['Layer'] = lyrj
    			dframe_single['Threshold'] = thr_dir

    			#print(dframe_multiple)
    			#print(dframe_single)    			

    			mult_fil = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/grid_cell_subsets/anom/multiple_sites/"+str(remap_type)+"_"+str(olri)+"_"+str(lyrj)+"_"+str(thr_dir)+"_multiple_sites_anom.csv"])
    			print(mult_fil)
    			dframe_multiple.to_csv(mult_fil,index=False)

    			single_fil = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/grid_cell_subsets/anom/single_site/"+str(remap_type)+"_"+str(olri)+"_"+str(lyrj)+"_"+str(thr_dir)+"_single_site_anom.csv"])
    			print(single_fil)
    			dframe_single.to_csv(single_fil,index=False)

