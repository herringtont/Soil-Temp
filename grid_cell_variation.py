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
import xarray as xr
import seaborn as sns
import pytesmo
import math
import cftime
import re
from decimal import *
from calendar import isleap
from dateutil.relativedelta import *


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
geom_dir = "/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/"
multi_dir = "/mnt/data/users/herringtont/soil_temp/In-Situ/grid_cell_subsets/multiple_sites/"
multi_dir_anom = "/mnt/data/users/herringtont/soil_temp/In-Situ/grid_cell_subsets/anom/multiple_sites/"
raw_temp_dir = "/mnt/data/users/herringtont/soil_temp/In-Situ/All/spatial_average/subset_multiple/"
anom_dir = "/mnt/data/users/herringtont/soil_temp/In-Situ/All/spatial_average_anom/subset_multiple/"
olr = ['outliers','zscore','IQR']
lyr = ['0_9.9','10_29.9','30_99.9','100_299.9','300_deeper']
thr = ['0','25','50','75','100']
rmp_type = ['nn','bil']

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
    rmph = h
    remap_type = ''.join(["remap"+rmph])
############ loop through outlier type #############
    for i in olr:
    	olri = i

############ loop through soil layer ##############    
    	for j in lyr:
    		lyrj = j

############# loop through missing threshold ##########    	
    		for k in thr:
    			thrk = k
    			multi_fil = ''.join([multi_dir+str(remap_type)+"_"+str(olri)+"_"+str(lyrj)+"_thr_"+str(thrk)+"_multiple_sites.csv"]) ##### for raw temperature data
    			multi_fil_anom = ''.join([multi_dir+str(remap_type)+"_"+str(olri)+"_"+str(lyrj)+"_thr_"+str(thrk)+"_multiple_sites.csv"]) ##### for anomaly data		
    			dframe_multi_site = pd.read_csv(multi_fil)
    			gcell = dframe_multi_site['Grid Cell'].values
    			#print(gcell)


############# loop through grid cells #################
    			for l in gcell:
    				gcell_l = l
    				raw_temp_fil = ''.join([raw_temp_dir+str(remap_type)+"/"+str(olri)+"/"+str(lyrj)+"/thr_"+str(thrk)+"/grid_"+str(gcell_l)+".csv"])
    				#print(raw_temp_fil)
    				dframe_raw_temp = pd.read_csv(raw_temp_fil)
    				#print(dframe_raw_temp)
    				date_rt = dframe_raw_temp['Date']
    				cent_lat = dframe_raw_temp['Central Lat'].iloc[0]
    				cent_lon = dframe_raw_temp['Central Lon'].iloc[0]
    				col_nam = dframe_raw_temp.columns
    				#print(col_nam)
    				sit_num = [s for s in col_nam if s.isdigit()] ######## check which elements of list are digits (as these are the site numbers)
    				#print(sit_num)

    				for m in sit_num:
			
    					sitid = m
    					#print(sitid)
    					site_temp = dframe_raw_temp[str(sitid)].values
    					#print(site_temp)
				
################### create the final dataframes ###################

    					dframe_gcell = pd.DataFrame(data=site_temp, columns=["Soil Temp"])
    					dframe_gcell.insert(0,'Date',date_rt)
    					dframe_gcell.insert(1,'Grid Cell',gcell_l)
    					dframe_gcell.insert(2,'Central Lat',cent_lat)
    					dframe_gcell.insert(3,'Central Lon',cent_lon)
    					dframe_gcell = dframe_gcell.dropna()
    					#dframe_gcell = dframe_gcell.reset_index(drop = True)
    					odir = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/grid_cell_variation/remap",rmph,"/",olri,"/",lyrj,"/thr_",thrk,"/grid_",str(gcell_l),"/"])
    					ofil = "".join([odir,"grid_",str(gcell_l),"_site_",str(sitid),"_variation.csv"])
    					import pathlib
    					path = pathlib.Path(ofil)
    					path.parent.mkdir(parents=True, exist_ok=True)
    					print(ofil)
    					print(dframe_gcell)
    					dframe_gcell.to_csv(ofil,index=False,na_rep="NaN") 				    					 
    					#print(dframe_gcell)			
