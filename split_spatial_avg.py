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
import shutil
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


################ set working directories ###################
subset_dir = "/mnt/data/users/herringtont/soil_temp/In-Situ/grid_cell_subsets/"
subset_dir_anom = "/mnt/data/users/herringtont/soil_temp/In-Situ/grid_cell_subsets/anom/"

mult_dir = "".join([str(subset_dir)+"multiple_sites/"])
mult_dir_anom = "".join([str(subset_dir_anom)+"multiple_sites/"])
single_dir = "".join([str(subset_dir)+"single_site/"])
single_dir_anom = "".join([str(subset_dir_anom)+"single_site/"]) 

olr = ['outliers','zscore','IQR']
lyr = ['0_9.9','10_29.9','30_99.9','100_299.9','300_deeper']
thr = ['0','25','50','75','100']
rmp_type = ['nn','bil']

############################## grab grid cell subsets ##############################
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

############################ multi-site grid cells ##################################
    			multi_fil = "".join([str(mult_dir)+str(remap_type)+"_"+olri+"_"+lyrj+"_"+thr_dir+"_multiple_sites.csv"])
    			#print(multi_fil)
    			multi_fil_anom =  "".join([str(mult_dir_anom)+str(remap_type)+"_"+olri+"_"+lyrj+"_"+thr_dir+"_multiple_sites_anom.csv"]) 

    			dframe_multi = pd.read_csv(multi_fil)
    			dframe_multi_anom = pd.read_csv(multi_fil_anom)
    			grid_multi = dframe_multi['Grid Cell'].values
    			grid_multi_anom = dframe_multi_anom['Grid Cell'].values
    			len_multi = len(grid_multi)
    			len_multi_anom = len(grid_multi_anom)
    			#print(len(grid_multi))
########################## loop through grid cells ##################################
    			for l in range (0,len_multi):
    				grid_multi_i = grid_multi[l]
    				grid_multi_fil = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/spatial_average/"+str(remap_type)+"/no_outliers/"+str(olri)+"/"+str(lyrj)+"/"+str(thr_dir)+"/grid_"+str(grid_multi_i)+".csv"])
    				multi_odir = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/spatial_average/subset_multiple/"+str(remap_type)+"/"+str(olri)+"/"+str(lyrj)+"/"+str(thr_dir)+"/"])
    				path = Path(multi_odir)
    				path.mkdir(parents=True, exist_ok=True)    				
				#print(grid_multi_fil)
    				#print(multi_odir)
    				shutil.copy2(grid_multi_fil,multi_odir)    				

    			for m in range (0,len_multi_anom):
    				grid_multi_anom_i = grid_multi_anom[m]
    				grid_multi_fil_anom = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/spatial_average_anom/"+str(remap_type)+"/no_outliers/"+str(olri)+"/"+str(lyrj)+"/"+str(thr_dir)+"/grid_"+str(grid_multi_anom_i)+"_anom.csv"])
    				multi_odir_anom = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/spatial_average_anom/subset_multiple/"+str(remap_type)+"/"+str(olri)+"/"+str(lyrj)+"/"+str(thr_dir)+"/"])
    				path2 = Path(multi_odir_anom)
    				path2.mkdir(parents=True, exist_ok=True)
    				shutil.copy2(grid_multi_fil_anom,multi_odir_anom)
				


############################ single-site grid cells ##################################
    			single_fil = "".join([str(single_dir)+str(remap_type)+"_"+olri+"_"+lyrj+"_"+thr_dir+"_single_site.csv"])
    			#print(multi_fil)
    			single_fil_anom =  "".join([str(single_dir_anom)+str(remap_type)+"_"+olri+"_"+lyrj+"_"+thr_dir+"_single_site_anom.csv"]) 

    			dframe_single = pd.read_csv(single_fil)
    			dframe_single_anom = pd.read_csv(single_fil_anom)
    			grid_single = dframe_single['Grid Cell'].values
    			grid_single_anom = dframe_single_anom['Grid Cell'].values
    			len_single = len(grid_single)
    			len_single_anom = len(grid_single_anom)
    			#print(len(grid_multi))

########################## loop through grid cells ##################################
    			for n in range (0,len_single):
    				grid_single_i = grid_single[n]
    				grid_single_fil = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/spatial_average/"+str(remap_type)+"/no_outliers/"+str(olri)+"/"+str(lyrj)+"/"+str(thr_dir)+"/grid_"+str(grid_single_i)+".csv"])
    				#print(grid_single_fil)
    				single_odir = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/spatial_average/subset_single/"+str(remap_type)+"/"+str(olri)+"/"+str(lyrj)+"/"+str(thr_dir)+"/"])
    				#print(single_odir)
    				path3 = Path(single_odir)
    				path3.mkdir(parents=True, exist_ok=True)
    				shutil.copy2(grid_single_fil,single_odir)    				

    			for o in range (0,len_single_anom):
    				grid_single_anom_i = grid_single_anom[o]
    				grid_single_fil_anom = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/spatial_average_anom/"+str(remap_type)+"/no_outliers/"+str(olri)+"/"+str(lyrj)+"/"+str(thr_dir)+"/grid_"+str(grid_single_anom_i)+"_anom.csv"])
    				single_odir_anom = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/spatial_average_anom/subset_single/"+str(remap_type)+"/"+str(olri)+"/"+str(lyrj)+"/"+str(thr_dir)+"/"])
    				path4 = Path(single_odir_anom)
    				path4.mkdir(parents=True, exist_ok=True)
    				#print(grid_single_fil_anom)
    				#print(single_odir_anom)
    				shutil.copy2(grid_single_fil_anom,single_odir_anom)
