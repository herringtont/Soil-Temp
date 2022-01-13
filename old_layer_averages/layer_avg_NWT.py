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


    dframe_stemp_layer1 = "None"
    dframe_stemp_layer2 = "None" 
    dframe_stemp_layer3 = "None"
    dframe_stemp_layer4 = "None"
    dframe_stemp_layer5 = "None"
    dframe_stemp_top30 = "None"
    dframe_stemp_layer7 = "None"
    #print(len(dframe_stemp_layer1))	   	
######## loop through soil depths ##########     	
    for j in range (0,total_col):
    	col_flt = col_float[j]
    	col_int = int(col_flt)		
    	#print(col_int,col_nam)    	
##### if col name < 10cm then store in layer 1 dataframe ######
    	if(0 <= col_flt < 10):
    		if(len(dframe_stemp_layer1) == 4):  #if layer1 does not exist then create it
    			dframe_stemp_layer1 = pd.DataFrame(dframe_stemp.iloc[:,j])
    			dframe_stemp_layer1.rename(columns={dframe_stemp_layer1.columns[0]: col_int},inplace=True)
    		elif(len(dframe_stemp_layer1) > 4):  #else append column to existing dataframe
    			dframe_stemp2 = dframe_stemp.iloc[:,j]
    			dframe_stemp_layer1[str(col_int)] = dframe_stemp2


##### if 10cm <= col name < 30cm then store in layer 2 dataframe ######
    	if(10 <= col_flt < 30):
    		if(len(dframe_stemp_layer2) == 4):  #if layer1 does not exist then create it
    			dframe_stemp_layer2 = pd.DataFrame(dframe_stemp.iloc[:,j])
    			dframe_stemp_layer2.rename(columns={dframe_stemp_layer2.columns[0]: col_int},inplace=True)
    		elif(len(dframe_stemp_layer2) > 4):  #else append column to existing dataframe
    			dframe_stemp2 = dframe_stemp.iloc[:,j]
    			dframe_stemp_layer2[str(col_int)] = dframe_stemp2

##### if 30cm <= col name < 100cm then store in layer 3 dataframe ######
    	if(30 <= col_flt < 100):
    		if(len(dframe_stemp_layer3) == 4):  #if layer1 does not exist then create it
    			dframe_stemp_layer3 = pd.DataFrame(dframe_stemp.iloc[:,j])
    			dframe_stemp_layer3.rename(columns={dframe_stemp_layer3.columns[0]: col_int},inplace=True)
    		elif(len(dframe_stemp_layer3) > 4):  #else append column to existing dataframe
    			dframe_stemp2 = dframe_stemp.iloc[:,j]
    			dframe_stemp_layer3[str(col_int)] = dframe_stemp2

##### if 100cm < col name < 300cm then store in layer 4 dataframe ######
    	if(100 <= col_flt < 300):
    		if(len(dframe_stemp_layer4) == 4):  #if layer1 does not exist then create it
    			dframe_stemp_layer4 = pd.DataFrame(dframe_stemp.iloc[:,j])
    			dframe_stemp_layer4.rename(columns={dframe_stemp_layer4.columns[0]: col_int},inplace=True)
    		elif(len(dframe_stemp_layer4) > 4):  #else append column to existing dataframe
    			dframe_stemp2 = dframe_stemp.iloc[:,j]
    			dframe_stemp_layer4[str(col_int)] = dframe_stemp2
	
##### if col name > 300cm then store in layer 5 dataframe ######                		
    	if(col_flt >= 300):
    		if(len(dframe_stemp_layer5) == 4):  #if layer1 does not exist then create it
    			dframe_stemp_layer5 = pd.DataFrame(dframe_stemp.iloc[:,j])
    			dframe_stemp_layer5.rename(columns={dframe_stemp_layer5.columns[0]: col_int},inplace=True)
    		elif(len(dframe_stemp_layer5) > 4):  #else append column to existing dataframe
    			dframe_stemp2 = dframe_stemp.iloc[:,j]
    			dframe_stemp_layer5[str(col_int)] = dframe_stemp2

##### if col name >=30cm then store in top30 dataframe #####
    	if(0 <= col_flt <= 30):
    		if(len(dframe_stemp_top30) == 4):  #if layer1 does not exist then create it
    			dframe_stemp_top30 = pd.DataFrame(dframe_stemp.iloc[:,j])
    			dframe_stemp_top30.rename(columns={dframe_stemp_top30.columns[0]: col_int},inplace=True)
    		elif(len(dframe_stemp_top30) > 4):  #else append column to existing dataframe
    			dframe_stemp2 = dframe_stemp.iloc[:,j]
    			dframe_stemp_top30[str(col_int)] = dframe_stemp2


##### if 30cm < col name < 300cm then store in layer 4 dataframe ######
    	if(30 < col_flt < 300):
    		if(len(dframe_stemp_layer7) == 4):  #if layer1 does not exist then create it
    			dframe_stemp_layer7 = pd.DataFrame(dframe_stemp.iloc[:,j])
    			dframe_stemp_layer7.rename(columns={dframe_stemp_layer7.columns[0]: col_int},inplace=True)
    		elif(len(dframe_stemp_layer7) > 4):  #else append column to existing dataframe
    			dframe_stemp2 = dframe_stemp.iloc[:,j]
    			dframe_stemp_layer7[str(col_int)] = dframe_stemp2

############### do the layer averaging ##################

################ do the averaging ####################### 
    print(dframe_stemp_layer1)
    print(dframe_stemp_layer2)
    print(dframe_stemp_layer3)
    print(dframe_stemp_layer4)
    print(dframe_stemp_layer5)
    print(dframe_stemp_top30)
    print(dframe_stemp_layer7)
    if(len(dframe_stemp_layer1) > 4): 
    	layer1_count = dframe_stemp_layer1.count(axis=1)
    	dframe_stemp_layer1['Layer_Avg'] = dframe_stemp_layer1.mean(axis=1)
    	dframe_stemp_layer1['Depths_Incl'] = layer1_count
    	dframe_stemp_layer1.insert(0,'Date',dates)
    	dframe_stemp_layer1.insert(1,'Dataset',dtst)
    	dframe_stemp_layer1.insert(2,'Lat',lat)
    	dframe_stemp_layer1.insert(3,'Long',lon)
    	dframe_stemp_layer1.drop(dframe_stemp_layer1[dframe_stemp_layer1['Depths_Incl'] == 0].index, inplace=True)
    	if (len(dframe_stemp_layer1) > 10):
    		print(dframe_stemp_layer1)
    		ofil1 = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/outliers/0_9.9/site_",str(sit_num),".csv"])
    		print(ofil1)
    		print(dframe_stemp_layer1)
    		dframe_stemp_layer1.to_csv(ofil1,na_rep="NaN",index=False)
						
    if(len(dframe_stemp_layer2) > 4):
    	layer2_count = dframe_stemp_layer2.count(axis=1)
    	dframe_stemp_layer2['Layer_Avg'] = dframe_stemp_layer2.mean(axis=1)
    	dframe_stemp_layer2['Depths_Incl'] = layer2_count
    	dframe_stemp_layer2.insert(0,'Date',dates)
    	dframe_stemp_layer2.insert(1,'Dataset',dtst)
    	dframe_stemp_layer2.insert(2,'Lat',lat)
    	dframe_stemp_layer2.insert(3,'Long',lon)
    	dframe_stemp_layer2.drop(dframe_stemp_layer2[dframe_stemp_layer2['Depths_Incl'] == 0].index, inplace=True)
    	if (len(dframe_stemp_layer2) > 10):	
    	#print(dframe_stemp_layer2)    		
    		ofil2 = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/outliers/10_29.9/site_",str(sit_num),".csv"])
    		print(ofil2)
    		print(dframe_stemp_layer2)
    		dframe_stemp_layer2.to_csv(ofil2,na_rep="NaN",index=False)
				
    if(len(dframe_stemp_layer3) > 4):
    	layer3_count = dframe_stemp_layer3.count(axis=1)
    	dframe_stemp_layer3['Layer_Avg'] = dframe_stemp_layer3.mean(axis=1)
    	dframe_stemp_layer3['Depths_Incl'] = layer3_count
    	dframe_stemp_layer3.insert(0,'Date',dates)
    	dframe_stemp_layer3.insert(1,'Dataset',dtst)
    	dframe_stemp_layer3.insert(2,'Lat',lat)
    	dframe_stemp_layer3.insert(3,'Long',lon)
    	dframe_stemp_layer3.drop(dframe_stemp_layer3[dframe_stemp_layer3['Depths_Incl'] == 0].index, inplace=True)
    	if (len(dframe_stemp_layer3) > 10):	   		
    		ofil3 = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/outliers/30_99.9/site_",str(sit_num),".csv"])
    		#print(dframe_stemp_layer3)
    		print(ofil3)
    		print(dframe_stemp_layer3)
    		dframe_stemp_layer3.to_csv(ofil3,na_rep="NaN",index=False)

    if(len(dframe_stemp_layer4) > 4):
    	layer4_count = dframe_stemp_layer4.count(axis=1)
    	dframe_stemp_layer4['Layer_Avg'] = dframe_stemp_layer4.mean(axis=1)
    	dframe_stemp_layer4['Depths_Incl'] = layer4_count
    	dframe_stemp_layer4.insert(0,'Date',dates)
    	dframe_stemp_layer4.insert(1,'Dataset',dtst)
    	dframe_stemp_layer4.insert(2,'Lat',lat)
    	dframe_stemp_layer4.insert(3,'Long',lon)
    	dframe_stemp_layer4.drop(dframe_stemp_layer4[dframe_stemp_layer4['Depths_Incl'] == 0].index, inplace=True)
    	#print(dframe_stemp_layer4)
    	if (len(dframe_stemp_layer4) > 10):
    		ofil4 = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/outliers/100_299.9/site_",str(sit_num),".csv"])
    		print(ofil4)
    		print(dframe_stemp_layer4)
    		dframe_stemp_layer4.to_csv(ofil4,na_rep="NaN",index=False)
				
    if(len(dframe_stemp_layer5) > 4):
    	layer5_count = dframe_stemp_layer5.count(axis=1)
    	dframe_stemp_layer5['Layer_Avg'] = dframe_stemp_layer5.mean(axis=1)
    	dframe_stemp_layer5['Depths_Incl'] = layer5_count
    	dframe_stemp_layer5.insert(0,'Date',dates)
    	dframe_stemp_layer5.insert(1,'Dataset',dtst)
    	dframe_stemp_layer5.insert(2,'Lat',lat)
    	dframe_stemp_layer5.insert(3,'Long',lon)
    	dframe_stemp_layer5.drop(dframe_stemp_layer5[dframe_stemp_layer5['Depths_Incl'] == 0].index, inplace=True)
    	#print(dframe_stemp_layer5)
    	if (len(dframe_stemp_layer5) > 10):
    		ofil5 = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/outliers/300_deeper/site_",str(sit_num),".csv"])
    		print(ofil5)
    		print(dframe_stemp_layer5)
    		dframe_stemp_layer5.to_csv(ofil5,na_rep="NaN",index=False)

    if(len(dframe_stemp_top30) > 4):
    	top30_count = dframe_stemp_top30.count(axis=1)
    	dframe_stemp_top30['Layer_Avg'] = dframe_stemp_top30.mean(axis=1)
    	dframe_stemp_top30['Depths_Incl'] = top30_count
    	dframe_stemp_top30.insert(0,'Date',dates)
    	dframe_stemp_top30.insert(1,'Dataset',dtst)
    	dframe_stemp_top30.insert(2,'Lat',lat)
    	dframe_stemp_top30.insert(3,'Long',lon)
    	dframe_stemp_top30.drop(dframe_stemp_top30[dframe_stemp_top30['Depths_Incl'] == 0].index, inplace=True)
    	#print(dframe_stemp_layer5)
    	ofil6 = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/outliers/top_30cm/site_",str(sit_num),".csv"])
    	if (len(dframe_stemp_top30) > 10):
    		path = pathlib.Path(ofil6)
    		path.parent.mkdir(parents=True, exist_ok=True)
    		print(ofil6)
    		print(dframe_stemp_top30)
    		dframe_stemp_top30.to_csv(ofil6,na_rep="NaN",index=False)


    if(len(dframe_stemp_layer7) > 4):
    	layer7_count = dframe_stemp_layer7.count(axis=1)
    	dframe_stemp_layer7['Layer_Avg'] = dframe_stemp_layer7.mean(axis=1)
    	dframe_stemp_layer7['Depths_Incl'] = layer7_count
    	dframe_stemp_layer7.insert(0,'Date',dates)
    	dframe_stemp_layer7.insert(1,'Dataset',dtst)
    	dframe_stemp_layer7.insert(2,'Lat',lat)
    	dframe_stemp_layer7.insert(3,'Long',lon)
    	dframe_stemp_layer7.drop(dframe_stemp_layer7[dframe_stemp_layer7['Depths_Incl'] == 0].index, inplace=True)
    	#print(dframe_stemp_layer5)
    	ofil7 = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/outliers/30_299.9/site_",str(sit_num),".csv"])
    	if (len(dframe_stemp_layer7) > 10):
    		path = pathlib.Path(ofil7)
    		path.parent.mkdir(parents=True, exist_ok=True)
    		print(ofil7)
    		print(dframe_stemp_layer7)
    		dframe_stemp_layer7.to_csv(ofil7,na_rep="NaN",index=False)





















