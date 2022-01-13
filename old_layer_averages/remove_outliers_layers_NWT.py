# -*- coding: utf-8 -*-
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
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from dateutil.relativedelta import *
from natsort import natsorted
from natsort import os_sorted



################### define functions ################
def str_to_datetime(column, date_fmt):

    date_list = []

    for dt_str in column:
        new_dt = datetime.datetime.strptime(dt_str, date_fmt)
        date_list.append(new_dt)
			

    return date_list
    
def remove_trailing_zeros(x):
    return str(x).rstrip('0').rstrip('.')

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)



################## define directories ################

wkdir = '/mnt/data/users/herringtont/soil_temp/In-Situ/NWT_All/'

pathlist = os.listdir(wkdir)
pathlist_sorted = natural_sort(pathlist)


################## loop through files ###############

for path in pathlist_sorted:
    wfil = ''.join([wkdir+path])
    print(path)
    dframe_insitu = pd.read_csv(wfil)
    print(dframe_insitu)
    col_depths = dframe_insitu.columns
    sit_n = path.split("_")[1]
    sit_n2 = sit_n.split('.')[0]
    sit_num = int(sit_n2)

    if (sit_num < 896):
    	dtst = "YZ_ibuttons"
    	lat = dframe_insitu['Latitude'].iloc[0]
    	lon = dframe_insitu['Longitude'].iloc[0]
    	dates = dframe_insitu['Date'].values
    	dframe_depths = dframe_insitu.drop(['Date','Latitude','Longitude'], axis=1)
    elif (896 <= sit_num <= 968):
    	dtst = "NWT_2017_009"
    	lat = dframe_insitu['latitude'].iloc[0]
    	lon = dframe_insitu['longitude'].iloc[0]
    	dates = dframe_insitu['datetime'].values
    	dframe_depths = dframe_insitu.drop(['latitude','longitude','datetime'], axis=1)
    elif (969 <= sit_num <= 1182):
    	dtst = "NWT_2018_009"
    	lat = dframe_insitu['latitude'].iloc[0]
    	lon = dframe_insitu['longitude'].iloc[0]
    	dates = dframe_insitu['datetime'].values
    	dframe_depths = dframe_insitu.drop(['latitude','longitude','datetime'], axis=1)
    elif (1183 <= sit_num <= 1191):
    	dtst = "NWT_2019_004"
    	lat = dframe_insitu['latitude'].iloc[0]
    	lon = dframe_insitu['longitude'].iloc[0]
    	dates = dframe_insitu['datetime'].values
    	dframe_depths = dframe_insitu.drop(['latitude','longitude','datetime'], axis=1)
    elif (1183 <= sit_num <= 1222):
    	dtst = "NWT_2019_017"
    	lat = dframe_insitu['latitude'].iloc[0]
    	lon = dframe_insitu['longitude'].iloc[0]
    	dates = dframe_insitu['datetime'].values
    	dframe_depths = dframe_insitu.drop(['latitude','longitude','datetime'], axis=1)

    elif (1223 <= sit_num <= 1227):
    	dtst = "Street_2016"
    	lat = dframe_insitu['latitude'].iloc[0]
    	lon = dframe_insitu['longitude'].iloc[0]
    	dates = dframe_insitu['datetime'].values
    	dframe_depths = dframe_insitu.drop(['latitude','longitude','datetime'], axis=1)

    #print(dates)
    #datetime = [datetime.datetime.strptime(i, date_fmt) for i in dates]    
    dframe_stemp = dframe_depths
    col_depths = np.array(dframe_depths.columns)
    col_float = col_depths.astype(float)

    total_col = len(col_depths)




    dframe_stemp_layer1_m1 = "None"
    dframe_stemp_layer2_m1 = "None" 
    dframe_stemp_layer3_m1 = "None"
    dframe_stemp_layer4_m1 = "None"
    dframe_stemp_layer5_m1 = "None"
    dframe_stemp_top30_m1 = "None"
    dframe_stemp_layer7_m1 = "None"

    dframe_stemp_layer1_m2 = "None"
    dframe_stemp_layer2_m2 = "None" 
    dframe_stemp_layer3_m2 = "None"
    dframe_stemp_layer4_m2 = "None"
    dframe_stemp_layer5_m2 = "None"
    dframe_stemp_top30_m2 = "None"
    dframe_stemp_layer7_m2 = "None"

    if (dtst == "YZ_ibuttons"):
    	date_fmt = '%Y-%m-%d'

    else:
    	date_fmt = '%Y-%m-%d %H:%M%:%S'

######## loop through soil depths ##########     	
    for j in range (0,total_col):
    	col_flt = col_float[j]
    	col_int = int(col_flt)


    	dframe_stemp_1 = dframe_stemp.iloc[:,j]
#    	#print(dframe_stemp_1)
#
    	stemp_new_m1 = []
    	dat_new_m1 = []
    	stemp_new_m2 = []
    	dat_new_m2 = []
#
#
#### Test for outliers ######
#######1 - Standard Deviation Method #####
#    	threshold = 3.5 #set outlier threshold to this many standard deviations away from mean
#    	mean_value = dframe_stemp_1.mean(axis = 0, skipna = True)
#    	stdev = dframe_stemp_1.std()
#	    	
#    	
#    	for k in range(0,len(dframe_stemp_1)):
#    		stmp = dframe_stemp_1.iloc[k]
#    		dat2 = dates[k]
#   		
#    		z = (stmp - mean_value)/stdev
#    		z_abs = abs(z)
#    		#print(stmp)
#    		#print(z)
#		    		
#    		if (z_abs > threshold or stmp == np.nan):
#    			sval = np.nan
#    			dval = dat2
#    		else:
#    			sval = stmp
#    			dval = dat2
#    		stemp_new_m1.append(sval)
#    		dat_new_m1.append(dval)
#    	
#    	stemp_new_m1n = np.array(stemp_new_m1)
#    	dat_new_m1n = np.array(dat_new_m1)
#    	#print(stemp_new_m1n)
#
###### if col name < 10cm then store in layer 1 dataframe ######
#    	if(0 <= col_flt < 10):
#    		if(len(dframe_stemp_layer1_m1) == 4):  #if layer1 does not exist then create it
#    			dframe_stemp_layer1_m1 = pd.DataFrame(stemp_new_m1n, columns = [col_int])
#    		elif(len(dframe_stemp_layer1_m1) > 4):  #else append column to existing dataframe
#    			dframe_stemp2 = stemp_new_m1n
#    			dframe_stemp_layer1_m1[str(col_int)] = dframe_stemp2
#
#
###### if 10cm <= col name < 30cm then store in layer 2 dataframe ######
#    	elif(10 <= col_flt < 30):
#    		if(len(dframe_stemp_layer2_m1) == 4):  #if layer1 does not exist then create it
#    			dframe_stemp_layer2_m1 = pd.DataFrame(stemp_new_m1n, columns=[col_int])
#    		elif(len(dframe_stemp_layer2_m1) > 4):  #else append column to existing dataframe
#    			dframe_stemp2 = stemp_new_m1n
#    			dframe_stemp_layer2_m1[str(col_int)] = dframe_stemp2
#
###### if 30cm <= col name < 100cm then store in layer 3 dataframe ######
#    	elif(30 <= col_flt < 100):
#    		if(len(dframe_stemp_layer3_m1) == 4):  #if layer1 does not exist then create it
#    			dframe_stemp_layer3_m1 = pd.DataFrame(stemp_new_m1n, columns=[col_int])
#    		elif(len(dframe_stemp_layer3_m1) > 4):  #else append column to existing dataframe
#    			dframe_stemp2 = stemp_new_m1n
#    			dframe_stemp_layer3_m1[str(col_int)] = dframe_stemp2
#
###### if 100cm < col name < 300cm then store in layer 4 dataframe ######
#    	elif(100 <= col_flt < 300):
#    		if(len(dframe_stemp_layer4_m1) == 4):  #if layer1 does not exist then create it
#    			dframe_stemp_layer4_m1 = pd.DataFrame(stemp_new_m1n, columns=[col_int])
#    		elif(len(dframe_stemp_layer4_m1) > 4):  #else append column to existing dataframe
#    			dframe_stemp2 = stemp_new_m1n
#    			dframe_stemp_layer4_m1[str(col_int)] = dframe_stemp2
#	
###### if col name > 300cm then store in layer 5 dataframe ######                		
#    	elif(col_flt >= 300):
#    		if(len(dframe_stemp_layer5_m1) == 4):  #if layer1 does not exist then create it
#    			dframe_stemp_layer5_m1 = pd.DataFrame(stemp_new_m1n, columns=[col_int])
#    		elif(len(dframe_stemp_layer5_m1) > 4):  #else append column to existing dataframe
#    			dframe_stemp2 = stemp_new_m1n
#    			dframe_stemp_layer5_m1[str(col_int)] = dframe_stemp2
#
###### if col name <= 30cm then store in top30 dataframe ######
#    	if(0 <= col_flt <= 30):
#    		if(len(dframe_stemp_top30_m1) == 4):  #if layer1 does not exist then create it
#    			dframe_stemp_top30_m1 = pd.DataFrame(stemp_new_m1n, columns = [col_int])
#    		elif(len(dframe_stemp_top30_m1) > 4):  #else append column to existing dataframe
#    			dframe_stemp2 = stemp_new_m1n
#    			dframe_stemp_top30_m1[str(col_int)] = dframe_stemp2
#
###### if 100cm < col name < 300cm then store in layer 4 dataframe ######
#    	if(30 <= col_flt < 300):
#    		if(len(dframe_stemp_layer7_m1) == 4):  #if layer1 does not exist then create it
#    			dframe_stemp_layer7_m1 = pd.DataFrame(stemp_new_m1n, columns=[col_int])
#    		elif(len(dframe_stemp_layer7_m1) > 4):  #else append column to existing dataframe
#    			dframe_stemp2 = stemp_new_m1n
#    			dframe_stemp_layer7_m1[str(col_int)] = dframe_stemp2
#
################# do the averaging ####################### 
#    if(len(dframe_stemp_layer1_m1) > 4): 
#    	layer1_count = dframe_stemp_layer1_m1.count(axis=1)
#    	dframe_stemp_layer1_m1['Layer_Avg'] = dframe_stemp_layer1_m1.mean(axis=1)
#    	dframe_stemp_layer1_m1['Depths_Incl'] = layer1_count
#    	dframe_stemp_layer1_m1.insert(0,'Date',dates)
#    	dframe_stemp_layer1_m1.insert(1,'Dataset',dtst)
#    	dframe_stemp_layer1_m1.insert(2,'Lat',lat)
#    	dframe_stemp_layer1_m1.insert(3,'Long',lon)
#    	dframe_stemp_layer1_m1.drop(dframe_stemp_layer1_m1[dframe_stemp_layer1_m1['Depths_Incl'] == 0].index, inplace=True)
#    	if(len(dframe_stemp_layer1_m1) > 10):
#    		#print(dframe_stemp_layer1_m1)
#    		ofil1Z = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/zscore/0_9.9/site_",str(sit_num),".csv"])
#    		print(ofil1Z)
#    		dframe_stemp_layer1_m1.to_csv(ofil1Z,na_rep="NaN",index=False)
#						
#    if(len(dframe_stemp_layer2_m1) > 4):
#    	layer2_count = dframe_stemp_layer2_m1.count(axis=1)
#    	dframe_stemp_layer2_m1['Layer_Avg'] = dframe_stemp_layer2_m1.mean(axis=1)
#    	dframe_stemp_layer2_m1['Depths_Incl'] = layer2_count
#    	dframe_stemp_layer2_m1.insert(0,'Date',dates)
#    	dframe_stemp_layer2_m1.insert(1,'Dataset',dtst)
#    	dframe_stemp_layer2_m1.insert(2,'Lat',lat)
#    	dframe_stemp_layer2_m1.insert(3,'Long',lon)
#    	dframe_stemp_layer2_m1.drop(dframe_stemp_layer2_m1[dframe_stemp_layer2_m1['Depths_Incl'] == 0].index, inplace=True)
#    	if(len(dframe_stemp_layer2_m1) > 10):
#    		#print(dframe_stemp_layer2_m1)    		
#    		ofil2Z = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/zscore/10_29.9/site_",str(sit_num),".csv"])
#    		print(ofil2Z)
#    		dframe_stemp_layer2_m1.to_csv(ofil2Z,na_rep="NaN",index=False)
#				
#    if(len(dframe_stemp_layer3_m1) > 4):
#    	layer3_count = dframe_stemp_layer3_m1.count(axis=1)
#    	dframe_stemp_layer3_m1['Layer_Avg'] = dframe_stemp_layer3_m1.mean(axis=1)
#    	dframe_stemp_layer3_m1['Depths_Incl'] = layer3_count
#    	dframe_stemp_layer3_m1.insert(0,'Date',dates)
#    	dframe_stemp_layer3_m1.insert(1,'Dataset',dtst)
#    	dframe_stemp_layer3_m1.insert(2,'Lat',lat)
#    	dframe_stemp_layer3_m1.insert(3,'Long',lon)
#    	dframe_stemp_layer3_m1.drop(dframe_stemp_layer3_m1[dframe_stemp_layer3_m1['Depths_Incl'] == 0].index, inplace=True)
#    	if(len(dframe_stemp_layer3_m1) > 10):	   		
#    		ofil3Z = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/zscore/30_99.9/site_",str(sit_num),".csv"])
#    		#print(dframe_stemp_layer3_m1)
#    		print(ofil3Z)
#    		dframe_stemp_layer3_m1.to_csv(ofil3Z,na_rep="NaN",index=False)
#
#    if(len(dframe_stemp_layer4_m1) > 4):
#    	layer4_count = dframe_stemp_layer4_m1.count(axis=1)
#    	dframe_stemp_layer4_m1['Layer_Avg'] = dframe_stemp_layer4_m1.mean(axis=1)
#    	dframe_stemp_layer4_m1['Depths_Incl'] = layer4_count
#    	dframe_stemp_layer4_m1.insert(0,'Date',dates)
#    	dframe_stemp_layer4_m1.insert(1,'Dataset',dtst)
#    	dframe_stemp_layer4_m1.insert(2,'Lat',lat)
#    	dframe_stemp_layer4_m1.insert(3,'Long',lon)
#    	dframe_stemp_layer4_m1.drop(dframe_stemp_layer4_m1[dframe_stemp_layer4_m1['Depths_Incl'] == 0].index, inplace=True)
#    	if(len(dframe_stemp_layer4_m1) > 10):
#    		#print(dframe_stemp_layer4_m1)
#    		ofil4Z = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/zscore/100_299.9/site_",str(sit_num),".csv"])
#    		print(ofil4Z)
#    		dframe_stemp_layer4_m1.to_csv(ofil4Z,na_rep="NaN",index=False)
#				
#    if(len(dframe_stemp_layer5_m1) > 4):
#    	layer5_count = dframe_stemp_layer5_m1.count(axis=1)
#    	dframe_stemp_layer5_m1['Layer_Avg'] = dframe_stemp_layer5_m1.mean(axis=1)
#    	dframe_stemp_layer5_m1['Depths_Incl'] = layer5_count
#    	dframe_stemp_layer5_m1.insert(0,'Date',dates)
#    	dframe_stemp_layer5_m1.insert(1,'Dataset',dtst)
#    	dframe_stemp_layer5_m1.insert(2,'Lat',lat)
#    	dframe_stemp_layer5_m1.insert(3,'Long',lon)
#    	dframe_stemp_layer5_m1.drop(dframe_stemp_layer5_m1[dframe_stemp_layer5_m1['Depths_Incl'] == 0].index, inplace=True)
#    	if(len(dframe_stemp_layer5_m1) > 10):
#    		#print(dframe_stemp_layer5_m1)
#    		ofil5Z = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/zscore/300_deeper/site_",str(sit_num),".csv"])
#    		print(ofil5Z)
#    		dframe_stemp_layer5_m1.to_csv(ofil5Z,na_rep="NaN",index=False)
#
#    if(len(dframe_stemp_top30_m1) > 4):
#    	top30_count = dframe_stemp_top30_m1.count(axis=1)
#    	dframe_stemp_top30_m1['Layer_Avg'] = dframe_stemp_top30_m1.mean(axis=1)
#    	dframe_stemp_top30_m1['Depths_Incl'] = top30_count
#    	dframe_stemp_top30_m1.insert(0,'Date',dates)
#    	dframe_stemp_top30_m1.insert(1,'Dataset',dtst)
#    	dframe_stemp_top30_m1.insert(2,'Lat',lat)
#    	dframe_stemp_top30_m1.insert(3,'Long',lon)
#    	dframe_stemp_top30_m1.drop(dframe_stemp_top30_m1[dframe_stemp_top30_m1['Depths_Incl'] == 0].index, inplace=True)
#    	if(len(dframe_stemp_top30_m1) > 10):
#    		#print(dframe_stemp_top30_m1)
#    		ofil6Z = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/zscore/top_30cm/site_",str(sit_num),".csv"])
#    		path = pathlib.Path(ofil6Z)
#    		path.parent.mkdir(parents=True, exist_ok=True)
#    		print(ofil6Z)
#    		dframe_stemp_top30_m1.to_csv(ofil6Z,na_rep="NaN",index=False)
#    if(len(dframe_stemp_layer7_m1) > 4):
#    	layer7_count = dframe_stemp_layer7_m1.count(axis=1)
#    	dframe_stemp_layer7_m1['Layer_Avg'] = dframe_stemp_layer7_m1.mean(axis=1)
#    	dframe_stemp_layer7_m1['Depths_Incl'] = layer7_count
#    	dframe_stemp_layer7_m1.insert(0,'Date',dates)
#    	dframe_stemp_layer7_m1.insert(1,'Dataset',dtst)
#    	dframe_stemp_layer7_m1.insert(2,'Lat',lat)
#    	dframe_stemp_layer7_m1.insert(3,'Long',lon)
#    	dframe_stemp_layer7_m1.drop(dframe_stemp_layer7_m1[dframe_stemp_layer7_m1['Depths_Incl'] == 0].index, inplace=True)
#    	#print(dframe_stemp_layer7_m1)
#    	if(len(dframe_stemp_layer7_m1) > 10):
#    		ofil7Z = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/zscore/30_299.9/site_",str(sit_num),".csv"])
#    		path = pathlib.Path(ofil7Z)
#    		path.parent.mkdir(parents=True, exist_ok=True)
#    		print(ofil7Z)
#    		dframe_stemp_layer7_m1.to_csv(ofil7Z,na_rep="NaN",index=False)	

#####2 - IQR Method #####		
    	Q1 = dframe_stemp_1.quantile(0.25)
    	Q3 = dframe_stemp_1.quantile(0.75)
    	IQR = Q3-Q1
    	fence = IQR*1.5
    	for k in range(0,len(dframe_stemp_1)):
    		stmp = dframe_stemp_1.iloc[k]
    		dat2 = dates[k]
				    		   								
    		if(stmp < (Q1 - fence) or stmp > (Q3 + fence) or stmp == np.nan):
    			sval = np.nan
    			dval = dat2
    		else:
    			sval = stmp
    			dval = dat2
    		stemp_new_m2.append(sval)
    		dat_new_m2.append(dval)

    	stemp_new_m2n = np.array(stemp_new_m2)
    	dat_new_m2n = np.array(dat_new_m2)


        	
##### if col name < 10cm then store in layer 1 dataframe ######
    	if(0 <= col_flt < 10):
    		if(len(dframe_stemp_layer1_m2) == 4):  #if layer1 does not exist then create it
    			dframe_stemp_layer1_m2 = pd.DataFrame(stemp_new_m2n, columns = [col_int])
    		elif(len(dframe_stemp_layer1_m2) > 4):  #else append column to existing dataframe
    			dframe_stemp2 = stemp_new_m2n
    			dframe_stemp_layer1_m2[str(col_int)] = dframe_stemp2


##### if 10cm <= col name < 30cm then store in layer 2 dataframe ######
    	elif(10 <= col_flt < 30):
    		if(len(dframe_stemp_layer2_m2) == 4):  #if layer1 does not exist then create it
    			dframe_stemp_layer2_m2 = pd.DataFrame(stemp_new_m2n, columns=[col_int])
    		elif(len(dframe_stemp_layer2_m2) > 4):  #else append column to existing dataframe
    			dframe_stemp2 = stemp_new_m2n
    			dframe_stemp_layer2_m2[str(col_int)] = dframe_stemp2

##### if 30cm <= col name < 100cm then store in layer 3 dataframe ######
    	elif(30 <= col_flt < 100):
    		if(len(dframe_stemp_layer3_m2) == 4):  #if layer1 does not exist then create it
    			dframe_stemp_layer3_m2 = pd.DataFrame(stemp_new_m2n, columns=[col_int])
    		elif(len(dframe_stemp_layer3_m2) > 4):  #else append column to existing dataframe
    			dframe_stemp2 = stemp_new_m2n
    			dframe_stemp_layer3_m2[str(col_int)] = dframe_stemp2

##### if 100cm < col name < 300cm then store in layer 4 dataframe ######
    	elif(100 <= col_flt < 300):
    		if(len(dframe_stemp_layer4_m2) == 4):  #if layer1 does not exist then create it
    			dframe_stemp_layer4_m2 = pd.DataFrame(stemp_new_m2n, columns=[col_int])
    		elif(len(dframe_stemp_layer4_m2) > 4):  #else append column to existing dataframe
    			dframe_stemp2 = stemp_new_m2n
    			dframe_stemp_layer4_m2[str(col_int)] = dframe_stemp2
	
##### if col name > 300cm then store in layer 5 dataframe ######                		
    	elif(col_flt >= 300):
    		if(len(dframe_stemp_layer5_m2) == 4):  #if layer1 does not exist then create it
    			dframe_stemp_layer5_m2 = pd.DataFrame(stemp_new_m2n, columns=[col_int])
    		elif(len(dframe_stemp_layer5_m2) > 4):  #else append column to existing dataframe
    			dframe_stemp2 = stemp_new_m2n
    			dframe_stemp_layer5_m2[str(col_int)] = dframe_stemp2

##### if col name <= 30cm then store in top30 dataframe ######
    	if(0 <= col_flt <= 30):
    		if(len(dframe_stemp_top30_m2) == 4):  #if layer1 does not exist then create it
    			dframe_stemp_top30_m2 = pd.DataFrame(stemp_new_m2n, columns = [col_int])
    		elif(len(dframe_stemp_top30_m2) > 4):  #else append column to existing dataframe
    			dframe_stemp2 = stemp_new_m2n
    			dframe_stemp_top30_m2[str(col_int)] = dframe_stemp2
##### if 30cm <= col name < 100cm then store in layer 3 dataframe ######
    	elif(30 <= col_flt < 300):
    		if(len(dframe_stemp_layer7_m2) == 4):  #if layer1 does not exist then create it
    			dframe_stemp_layer7_m2 = pd.DataFrame(stemp_new_m2n, columns=[col_int])
    		elif(len(dframe_stemp_layer7_m2) > 4):  #else append column to existing dataframe
    			dframe_stemp2 = stemp_new_m2n
    			dframe_stemp_layer7_m2[str(col_int)] = dframe_stemp2
#
################# do the averaging ####################### 
#    if(len(dframe_stemp_layer1_m2) > 4): 
#    	layer1_count = dframe_stemp_layer1_m2.count(axis=1)
#    	dframe_stemp_layer1_m2['Layer_Avg'] = dframe_stemp_layer1_m2.mean(axis=1)
#    	dframe_stemp_layer1_m2['Depths_Incl'] = layer1_count
#    	dframe_stemp_layer1_m2.insert(0,'Date',dates)
#    	dframe_stemp_layer1_m2.insert(1,'Dataset',dtst)
#    	dframe_stemp_layer1_m2.insert(2,'Lat',lat)
#    	dframe_stemp_layer1_m2.insert(3,'Long',lon)
#    	dframe_stemp_layer1_m2.drop(dframe_stemp_layer1_m2[dframe_stemp_layer1_m2['Depths_Incl'] == 0].index, inplace=True)
#    	#print(dframe_stemp_layer1_m2)
#    	if(len(dframe_stemp_layer1_m2) > 10):
#    		ofil1 = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/IQR/0_9.9/site_",str(sit_num),".csv"])
#    		print(ofil1)
#    		dframe_stemp_layer1_m2.to_csv(ofil1,na_rep="NaN",index=False)
#						
#    if(len(dframe_stemp_layer2_m2) > 4):
#    	layer2_count = dframe_stemp_layer2_m2.count(axis=1)
#    	dframe_stemp_layer2_m2['Layer_Avg'] = dframe_stemp_layer2_m2.mean(axis=1)
#    	dframe_stemp_layer2_m2['Depths_Incl'] = layer2_count
#    	dframe_stemp_layer2_m2.insert(0,'Date',dates)
#    	dframe_stemp_layer2_m2.insert(1,'Dataset',dtst)
#    	dframe_stemp_layer2_m2.insert(2,'Lat',lat)
#    	dframe_stemp_layer2_m2.insert(3,'Long',lon)
#    	dframe_stemp_layer2_m2.drop(dframe_stemp_layer2_m2[dframe_stemp_layer2_m2['Depths_Incl'] == 0].index, inplace=True)
#    	#print(dframe_stemp_layer2_m2)
#    	if(len(dframe_stemp_layer2_m2) > 10):    		
#    		ofil2 = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/IQR/10_29.9/site_",str(sit_num),".csv"])
#    		print(ofil2)
#    		dframe_stemp_layer2_m2.to_csv(ofil2,na_rep="NaN",index=False)
#				
#    if(len(dframe_stemp_layer3_m2) > 4):
#    	layer3_count = dframe_stemp_layer3_m2.count(axis=1)
#    	dframe_stemp_layer3_m2['Layer_Avg'] = dframe_stemp_layer3_m2.mean(axis=1)
#    	dframe_stemp_layer3_m2['Depths_Incl'] = layer3_count
#    	dframe_stemp_layer3_m2.insert(0,'Date',dates)
#    	dframe_stemp_layer3_m2.insert(1,'Dataset',dtst)
#    	dframe_stemp_layer3_m2.insert(2,'Lat',lat)
#    	dframe_stemp_layer3_m2.insert(3,'Long',lon)
#    	dframe_stemp_layer3_m2.drop(dframe_stemp_layer3_m2[dframe_stemp_layer3_m2['Depths_Incl'] == 0].index, inplace=True)
#    	if(len(dframe_stemp_layer3_m2) > 10):	   		
#    		ofil3 = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/IQR/30_99.9/site_",str(sit_num),".csv"])
#    		#print(dframe_stemp_layer3_m2)
#    		print(ofil3)
#    		dframe_stemp_layer3_m2.to_csv(ofil3,na_rep="NaN",index=False)
#
#    if(len(dframe_stemp_layer4_m2) > 4):
#    	layer4_count = dframe_stemp_layer4_m2.count(axis=1)
#    	dframe_stemp_layer4_m2['Layer_Avg'] = dframe_stemp_layer4_m2.mean(axis=1)
#    	dframe_stemp_layer4_m2['Depths_Incl'] = layer4_count
#    	dframe_stemp_layer4_m2.insert(0,'Date',dates)
#    	dframe_stemp_layer4_m2.insert(1,'Dataset',dtst)
#    	dframe_stemp_layer4_m2.insert(2,'Lat',lat)
#    	dframe_stemp_layer4_m2.insert(3,'Long',lon)
#    	dframe_stemp_layer4_m2.drop(dframe_stemp_layer4_m2[dframe_stemp_layer4_m2['Depths_Incl'] == 0].index, inplace=True)
#    	#print(dframe_stemp_layer4_m2)
#    	if(len(dframe_stemp_layer4_m2) > 10):
#    		ofil4 = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/IQR/100_299.9/site_",str(sit_num),".csv"])
#    		print(ofil4)
#    		dframe_stemp_layer4_m2.to_csv(ofil4,na_rep="NaN",index=False)
#				
#    if(len(dframe_stemp_layer5_m2) > 4):
#    	layer5_count = dframe_stemp_layer5_m2.count(axis=1)
#    	dframe_stemp_layer5_m2['Layer_Avg'] = dframe_stemp_layer5_m2.mean(axis=1)
#    	dframe_stemp_layer5_m2['Depths_Incl'] = layer5_count
#    	dframe_stemp_layer5_m2.insert(0,'Date',dates)
#    	dframe_stemp_layer5_m2.insert(1,'Dataset',dtst)
#    	dframe_stemp_layer5_m2.insert(2,'Lat',lat)
#    	dframe_stemp_layer5_m2.insert(3,'Long',lon)
#    	dframe_stemp_layer5_m2.drop(dframe_stemp_layer5_m2[dframe_stemp_layer5_m2['Depths_Incl'] == 0].index, inplace=True)
#    	if(len(dframe_stemp_layer5_m2) > 10):
#    		#print(dframe_stemp_layer5_m2)
#    		ofil5 = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/IQR/300_deeper/site_",str(sit_num),".csv"])
#    		print(ofil5)
#    		dframe_stemp_layer5_m2.to_csv(ofil5,na_rep="NaN",index=False)
#
#    if(len(dframe_stemp_top30_m2) > 4):
#    	top30_count = dframe_stemp_top30_m2.count(axis=1)
#    	dframe_stemp_top30_m2['Layer_Avg'] = dframe_stemp_top30_m2.mean(axis=1)
#    	dframe_stemp_top30_m2['Depths_Incl'] = top30_count
#    	dframe_stemp_top30_m2.insert(0,'Date',dates)
#    	dframe_stemp_top30_m2.insert(1,'Dataset',dtst)
#    	dframe_stemp_top30_m2.insert(2,'Lat',lat)
#    	dframe_stemp_top30_m2.insert(3,'Long',lon)
#    	dframe_stemp_top30_m2.drop(dframe_stemp_top30_m2[dframe_stemp_top30_m2['Depths_Incl'] == 0].index, inplace=True)
#    	#print(dframe_stemp_top30_m2)
#    	if(len(dframe_stemp_top30_m2) > 10):
#    		ofil6 = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/IQR/top_30cm/site_",str(sit_num),".csv"])
#    		path = pathlib.Path(ofil6)
#    		path.parent.mkdir(parents=True, exist_ok=True)
#    		print(ofil6)
#    		dframe_stemp_top30_m2.to_csv(ofil6,na_rep="NaN",index=False)
#	
    if(len(dframe_stemp_layer7_m2) > 4):
    	layer7_count = dframe_stemp_layer7_m2.count(axis=1)
    	dframe_stemp_layer7_m2['Layer_Avg'] = dframe_stemp_layer7_m2.mean(axis=1)
    	dframe_stemp_layer7_m2['Depths_Incl'] = layer7_count
    	dframe_stemp_layer7_m2.insert(0,'Date',dates)
    	dframe_stemp_layer7_m2.insert(1,'Dataset',dtst)
    	dframe_stemp_layer7_m2.insert(2,'Lat',lat)
    	dframe_stemp_layer7_m2.insert(3,'Long',lon)
    	dframe_stemp_layer7_m2.drop(dframe_stemp_layer7_m2[dframe_stemp_layer7_m2['Depths_Incl'] == 0].index, inplace=True)
    	#print(dframe_stemp_layer7_m2)
    	if(len(dframe_stemp_layer7_m2) > 10):
    		ofil7 = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/IQR/30_299.9/site_",str(sit_num),".csv"])
    		path = pathlib.Path(ofil7)
    		path.parent.mkdir(parents=True, exist_ok=True)
    		print(ofil7)
    		dframe_stemp_layer7_m2.to_csv(ofil7,na_rep="NaN",index=False)
