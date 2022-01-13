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



################### define functions ################
def str_to_datetime(column, date_fmt):

    date_list = []

    for dt_str in column:
        new_dt = datetime.datetime.strptime(dt_str, date_fmt)
        date_list.append(new_dt)
			

    return date_list
    
def remove_trailing_zeros(x):
    return str(x).rstrip('0').rstrip('.')



############################ if dataset is GTN-P###########################

############### loop through files ##############
for i in range (1,69):
    wdir = "/mnt/data/users/herringtont/soil_temp/In-Situ/GTN-P/site_level/"
    wfil = "".join([wdir,"site_",str(i),".csv"])
    print(wfil)  	   	
    dframe = pd.read_csv(wfil)
    dframe.replace(-999,np.nan,inplace=True)
    levels = dframe.columns.values.tolist()
    dtst = "GTN-P"
    date = dframe['Date/Depth']
    lat = dframe['Lat']		
    lon = dframe['Long']
    col1 = dframe[levels[0]]
    date_fmt = "%Y-%m-%d %H:%M:%S"
    datetime_column = str_to_datetime(col1, date_fmt)
    dframe_stemp = dframe.drop(['Date/Depth','Lat','Long'],axis=1)
    col_val = np.array(dframe_stemp.columns) #store column names
    col_float = col_val.astype(float)
    #print(col_val)
    col_cm = col_float*100
    #print(col_cm)
    total_col = len(col_cm) #count number of columns



    dframe_stemp_layer1 = "None"
    dframe_stemp_layer2 = "None" 
    dframe_stemp_layer3 = "None"
    dframe_stemp_layer4 = "None"
    dframe_stemp_layer5 = "None"
    dframe_stemp_top30 = "None"
    dframe_stemp_layer7 = "None"
    dframe_stemp_0_50 = "None"
    dframe_stemp_50_100 = "None"
    dframe_stemp_100_200 = "None"

    dframe_stemp_layer1_m1 = "None"
    dframe_stemp_layer2_m1 = "None" 
    dframe_stemp_layer3_m1 = "None"
    dframe_stemp_layer4_m1 = "None"
    dframe_stemp_layer5_m1 = "None"
    dframe_stemp_top30_m1 = "None"
    dframe_stemp_layer7_m1 = "None"
    dframe_stemp_0_50_m1 = "None"
    dframe_stemp_50_100_m1 = "None"
    dframe_stemp_100_200_m1 = "None"

    dframe_stemp_layer1_m2 = "None"
    dframe_stemp_layer2_m2 = "None" 
    dframe_stemp_layer3_m2 = "None"
    dframe_stemp_layer4_m2 = "None"
    dframe_stemp_layer5_m2 = "None"
    dframe_stemp_top30_m2 = "None"
    dframe_stemp_layer7_m2 = "None"
    dframe_stemp_0_50_m2 = "None"
    dframe_stemp_50_100_m2 = "None"
    dframe_stemp_100_200_m2 = "None"

#### loop through soil depths #########     	
    for j in range (0,total_col):
    	col_nam = col_val[j]
    	col_flt = float(col_nam)*100
    	col_int = int(col_flt)		
    	#print("The depth is: ",col_nam)
    	dframe_stemp = dframe_stemp.iloc[:,j]
    	#print(dframe_stemp_1)

###### if col name < 10cm then store in layer 1 dataframe ######
#    	if(0 <= col_flt < 10):
#    		if(len(dframe_stemp_layer1) == 4):  #if layer1 does not exist then create it
#    			dframe_stemp_layer1 = pd.DataFrame(dframe_stemp.iloc[:,j])
#    			dframe_stemp_layer1.rename(columns={dframe_stemp_layer1.columns[0]: col_int},inplace=True)
#    		elif(len(dframe_stemp_layer1) > 4):  #else append column to existing dataframe
#    			dframe_stemp1 = dframe_stemp.iloc[:,j]
#    			dframe_stemp_layer1[str(col_int)] = dframe_stemp1
#
#
###### if 10cm <= col name < 30cm then store in layer 2 dataframe ######
#    	if(10 <= col_flt < 30):
#    		if(len(dframe_stemp_layer2) == 4):  #if layer1 does not exist then create it
#    			dframe_stemp_layer2 = pd.DataFrame(dframe_stemp.iloc[:,j])
#    			dframe_stemp_layer2.rename(columns={dframe_stemp_layer2.columns[0]: col_int},inplace=True)
#    		elif(len(dframe_stemp_layer2) > 4):  #else append column to existing dataframe
#    			dframe_stemp2 = dframe_stemp.iloc[:,j]
#    			dframe_stemp_layer2[str(col_int)] = dframe_stemp2
#
###### if 30cm <= col name < 100cm then store in layer 3 dataframe ######
#    	if(30 <= col_flt < 100):
#    		if(len(dframe_stemp_layer3) == 4):  #if layer1 does not exist then create it
#    			dframe_stemp_layer3 = pd.DataFrame(dframe_stemp.iloc[:,j])
#    			dframe_stemp_layer3.rename(columns={dframe_stemp_layer3.columns[0]: col_int},inplace=True)
#    		elif(len(dframe_stemp_layer3) > 4):  #else append column to existing dataframe
#    			dframe_stemp3 = dframe_stemp.iloc[:,j]
#    			dframe_stemp_layer3[str(col_int)] = dframe_stemp3
#
###### if 100cm < col name < 300cm then store in layer 4 dataframe ######
#    	if(100 <= col_flt < 300):
#    		if(len(dframe_stemp_layer4) == 4):  #if layer1 does not exist then create it
#    			dframe_stemp_layer4 = pd.DataFrame(dframe_stemp.iloc[:,j])
#    			dframe_stemp_layer4.rename(columns={dframe_stemp_layer4.columns[0]: col_int},inplace=True)
#    		elif(len(dframe_stemp_layer4) > 4):  #else append column to existing dataframe
#    			dframe_stemp4 = dframe_stemp.iloc[:,j]
#    			dframe_stemp_layer4[str(col_int)] = dframe_stemp4
#	
###### if col name > 300cm then store in layer 5 dataframe ######                		
#    	if(col_flt >= 300):
#    		if(len(dframe_stemp_layer5) == 4):  #if layer1 does not exist then create it
#    			dframe_stemp_layer5 = pd.DataFrame(dframe_stemp.iloc[:,j])
#    			dframe_stemp_layer5.rename(columns={dframe_stemp_layer5.columns[0]: col_int},inplace=True)
#    		elif(len(dframe_stemp_layer5) > 4):  #else append column to existing dataframe
#    			dframe_stemp5 = dframe_stemp.iloc[:,j]
#    			dframe_stemp_layer5[str(col_int)] = dframe_stemp5
#
###### if col name <=30cm then store in top30 dataframe #####
#    	if(0 <= col_flt <= 30):
#    		if(len(dframe_stemp_top30) == 4):  #if layer1 does not exist then create it
#    			dframe_stemp_top30 = pd.DataFrame(dframe_stemp.iloc[:,j])
#    			dframe_stemp_top30.rename(columns={dframe_stemp_top30.columns[0]: col_int},inplace=True)
#    		elif(len(dframe_stemp_top30) > 4):  #else append column to existing dataframe
#    			dframe_stemp6 = dframe_stemp.iloc[:,j]
#    			dframe_stemp_top30[str(col_int)] = dframe_stemp6
#
###### if 30cm <= col name < 300cm then store in layer 7 dataframe ######
#    	if(30 < col_flt < 300):
#    		if(len(dframe_stemp_layer7) == 4):  #if layer1 does not exist then create it
#    			dframe_stemp_layer7 = pd.DataFrame(dframe_stemp.iloc[:,j])
#    			dframe_stemp_layer7.rename(columns={dframe_stemp_layer7.columns[0]: col_int},inplace=True)
#    		elif(len(dframe_stemp_layer7) > 4):  #else append column to existing dataframe
#    			dframe_stemp7 = dframe_stemp.iloc[:,j]
#    			dframe_stemp_layer7[str(col_int)] = dframe_stemp7

##### if 0cm <= col name <= 50cm then store in 0_50 dataframe
    	if(0 < col_flt < 50):
    		if(len(dframe_stemp_layer_0_50) == 4):  #if layer1 does not exist then create it
    			dframe_stemp_layer_0_50) = pd.DataFrame(dframe_stemp.iloc[:,j])
    			dframe_stemp_layer_0_50.rename(columns={dframe_stemp_layer_0_50.columns[0]: col_int},inplace=True)
    		elif(len(dframe_stemp_layer_0_50) > 4):  #else append column to existing dataframe
    			dframe_stemp_0_50 = dframe_stemp.iloc[:,j]
    			dframe_stemp_layer_0_50[str(col_int)] = dframe_stemp_0_50

##### if 0cm <= col name <= 50cm then store in 0_50 dataframe
    	if(50 <= col_flt < 100):
    		if(len(dframe_stemp_layer_50_100) == 4):  #if layer1 does not exist then create it
    			dframe_stemp_layer_50_100) = pd.DataFrame(dframe_stemp.iloc[:,j])
    			dframe_stemp_layer_50_100.rename(columns={dframe_stemp_layer_50_100.columns[0]: col_int},inplace=True)
    		elif(len(dframe_stemp_layer_50_100) > 4):  #else append column to existing dataframe
    			dframe_stemp_50_100 = dframe_stemp.iloc[:,j]
    			dframe_stemp_layer_50_100[str(col_int)] = dframe_stemp_50_100

############### do the averaging ####################### 
#    print(dframe_stemp_top30)
#    if(len(dframe_stemp_layer1) > 4): 
#    	layer1_count = dframe_stemp_layer1.count(axis=1)
#    	dframe_stemp_layer1['Layer_Avg'] = dframe_stemp_layer1.mean(axis=1)
#    	dframe_stemp_layer1['Depths_Incl'] = layer1_count
#    	dframe_stemp_layer1.insert(0,'Date',date)
#    	dframe_stemp_layer1.insert(1,'Dataset',dtst)
#    	dframe_stemp_layer1.insert(2,'Lat',lat)
#    	dframe_stemp_layer1.insert(3,'Long',lon)
#    	dframe_stemp_layer1.drop(dframe_stemp_layer1[dframe_stemp_layer1['Depths_Incl'] == 0].index, inplace=True)
#    	#print(dframe_stemp_layer1)
#    	ofil1 = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/outliers/0_9.9/site_",str(i),".csv"])
#    	print(ofil1)
#    	#print(dframe_stemp_layer1)
#    	dframe_stemp_layer1.to_csv(ofil1,na_rep="NaN",index=False)
#						
#    if(len(dframe_stemp_layer2) > 4):
#    	layer2_count = dframe_stemp_layer2.count(axis=1)
#    	dframe_stemp_layer2['Layer_Avg'] = dframe_stemp_layer2.mean(axis=1)
#    	dframe_stemp_layer2['Depths_Incl'] = layer2_count
#    	dframe_stemp_layer2.insert(0,'Date',date)
#    	dframe_stemp_layer2.insert(1,'Dataset',dtst)
#    	dframe_stemp_layer2.insert(2,'Lat',lat)
#    	dframe_stemp_layer2.insert(3,'Long',lon)
#    	dframe_stemp_layer2.drop(dframe_stemp_layer2[dframe_stemp_layer2['Depths_Incl'] == 0].index, inplace=True)
#    	#print(dframe_stemp_layer2)    		
#    	ofil2 = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/outliers/10_29.9/site_",str(i),".csv"])
#    	print(ofil2)
#    	#print(dframe_stemp_layer2)
#    	dframe_stemp_layer2.to_csv(ofil2,na_rep="NaN",index=False)
#				
#    if(len(dframe_stemp_layer3) > 4):
#    	layer3_count = dframe_stemp_layer3.count(axis=1)
#    	dframe_stemp_layer3['Layer_Avg'] = dframe_stemp_layer3.mean(axis=1)
#    	dframe_stemp_layer3['Depths_Incl'] = layer3_count
#    	dframe_stemp_layer3.insert(0,'Date',date)
#    	dframe_stemp_layer3.insert(1,'Dataset',dtst)
#    	dframe_stemp_layer3.insert(2,'Lat',lat)
#    	dframe_stemp_layer3.insert(3,'Long',lon)
#    	dframe_stemp_layer3.drop(dframe_stemp_layer3[dframe_stemp_layer3['Depths_Incl'] == 0].index, inplace=True)	   		
#    	ofil3 = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/outliers/30_99.9/site_",str(i),".csv"])
#    	#print(dframe_stemp_layer3)
#    	print(ofil3)
#    	#print(dframe_stemp_layer3)
#    	dframe_stemp_layer3.to_csv(ofil3,na_rep="NaN",index=False)
#
#    if(len(dframe_stemp_layer4) > 4):
#    	layer4_count = dframe_stemp_layer4.count(axis=1)
#    	dframe_stemp_layer4['Layer_Avg'] = dframe_stemp_layer4.mean(axis=1)
#    	dframe_stemp_layer4['Depths_Incl'] = layer4_count
#    	dframe_stemp_layer4.insert(0,'Date',date)
#    	dframe_stemp_layer4.insert(1,'Dataset',dtst)
#    	dframe_stemp_layer4.insert(2,'Lat',lat)
#    	dframe_stemp_layer4.insert(3,'Long',lon)
#    	dframe_stemp_layer4.drop(dframe_stemp_layer4[dframe_stemp_layer4['Depths_Incl'] == 0].index, inplace=True)
#    	#print(dframe_stemp_layer4)
#    	ofil4 = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/outliers/100_299.9/site_",str(i),".csv"])
#    	print(ofil4)
#    	#print(dframe_stemp_layer4)
#    	dframe_stemp_layer4.to_csv(ofil4,na_rep="NaN",index=False)
#				
#    if(len(dframe_stemp_layer5) > 4):
#    	layer5_count = dframe_stemp_layer5.count(axis=1)
#    	dframe_stemp_layer5['Layer_Avg'] = dframe_stemp_layer5.mean(axis=1)
#    	dframe_stemp_layer5['Depths_Incl'] = layer5_count
#    	dframe_stemp_layer5.insert(0,'Date',date)
#    	dframe_stemp_layer5.insert(1,'Dataset',dtst)
#    	dframe_stemp_layer5.insert(2,'Lat',lat)
#    	dframe_stemp_layer5.insert(3,'Long',lon)
#    	dframe_stemp_layer5.drop(dframe_stemp_layer5[dframe_stemp_layer5['Depths_Incl'] == 0].index, inplace=True)
#    	#print(dframe_stemp_layer5)
#    	ofil5 = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/outliers/300_deeper/site_",str(i),".csv"])
#    	print(ofil5)
#    	#print(dframe_stemp_layer5)
#    	dframe_stemp_layer5.to_csv(ofil5,na_rep="NaN",index=False)
#
#    if(len(dframe_stemp_top30) > 4):
#    	top30_count = dframe_stemp_top30.count(axis=1)
#    	dframe_stemp_top30['Layer_Avg'] = dframe_stemp_top30.mean(axis=1)
#    	dframe_stemp_top30['Depths_Incl'] = top30_count
#    	dframe_stemp_top30.insert(0,'Date',date)
#    	dframe_stemp_top30.insert(1,'Dataset',dtst)
#    	dframe_stemp_top30.insert(2,'Lat',lat)
#    	dframe_stemp_top30.insert(3,'Long',lon)
#    	dframe_stemp_top30.drop(dframe_stemp_top30[dframe_stemp_top30['Depths_Incl'] == 0].index, inplace=True)
#    	#print(dframe_stemp_layer5)
#    	ofil6 = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/outliers/top_30cm/site_",str(i),".csv"])
#    	path = pathlib.Path(ofil6)
#    	path.parent.mkdir(parents=True, exist_ok=True)
#    	print(ofil6)
#    	#print(dframe_stemp_layer5)
#    	dframe_stemp_top30.to_csv(ofil6,na_rep="NaN",index=False)
#
#    if(len(dframe_stemp_layer7) > 4):
#    	layer7_count = dframe_stemp_layer7.count(axis=1)
#    	dframe_stemp_layer7['Layer_Avg'] = dframe_stemp_layer7.mean(axis=1)
#    	dframe_stemp_layer7['Depths_Incl'] = layer7_count
#    	dframe_stemp_layer7.insert(0,'Date',date)
#    	dframe_stemp_layer7.insert(1,'Dataset',dtst)
#    	dframe_stemp_layer7.insert(2,'Lat',lat)
#    	dframe_stemp_layer7.insert(3,'Long',lon)
#    	dframe_stemp_layer7.drop(dframe_stemp_layer7[dframe_stemp_layer7['Depths_Incl'] == 0].index, inplace=True)	   		
#    	ofil7 = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/outliers/30_299.9/site_",str(i),".csv"])
#    	path = pathlib.Path(ofil7)
#    	path.parent.mkdir(parents=True, exist_ok=True)
#    	print(ofil7)
#    	#print(dframe_stemp_layer7)
#    	dframe_stemp_layer7.to_csv(ofil7,na_rep="NaN",index=False)


    if(len(dframe_stemp_layer_0_50) > 4):
    	layer_0_50_count = dframe_stemp_layer_0_50.count(axis=1)
    	dframe_stemp_layer_0_50['Layer_Avg'] = dframe_stemp_layer_0_50.mean(axis=1)
    	dframe_stemp_layer_0_50['Depths_Incl'] = layer_0_50_count
    	dframe_stemp_layer_0_50.insert(0,'Date',date)
    	dframe_stemp_layer_0_50.insert(1,'Dataset',dtst)
    	dframe_stemp_layer_0_50.insert(2,'Lat',lat)
    	dframe_stemp_layer_0_50.insert(3,'Long',lon)
    	dframe_stemp_layer_0_50.drop(dframe_stemp_layer_0_50[dframe_stemp_layer_0_50['Depths_Incl'] == 0].index, inplace=True)	   		
    	ofil8 = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/outliers/0_49.9/site_",str(i),".csv"])
    	path = pathlib.Path(ofil7)
    	path.parent.mkdir(parents=True, exist_ok=True)
    	print(ofil8)
    	#print(dframe_stemp_layer_0_50)
    	dframe_stemp_layer_0_50.to_csv(ofil7,na_rep="NaN",index=False)

    if(len(dframe_stemp_layer_50_100) > 4):
    	layer_50_100_count = dframe_stemp_layer_50_100.count(axis=1)
    	dframe_stemp_layer_50_100['Layer_Avg'] = dframe_stemp_layer_50_100.mean(axis=1)
    	dframe_stemp_layer_50_100['Depths_Incl'] = layer_50_100_count
    	dframe_stemp_layer_50_100.insert(0,'Date',date)
    	dframe_stemp_layer_50_100.insert(1,'Dataset',dtst)
    	dframe_stemp_layer_50_100.insert(2,'Lat',lat)
    	dframe_stemp_layer_50_100.insert(3,'Long',lon)
    	dframe_stemp_layer_50_100.drop(dframe_stemp_layer_50_100[dframe_stemp_layer_50_100['Depths_Incl'] == 0].index, inplace=True)	   		
    	ofil9 = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/outliers/50_99.9/site_",str(i),".csv"])
    	path = pathlib.Path(ofil7)
    	path.parent.mkdir(parents=True, exist_ok=True)
    	print(ofil9)
    	#print(dframe_stemp_layer_50_100)
    	dframe_stemp_layer_50_100.to_csv(ofil7,na_rep="NaN",index=False)

#### remove outliers ####
## z-score method ##

#### loop through soil depths #########     	
    for j in range (0,total_col):
    	col_nam = col_val[j]
    	col_flt = float(col_nam)*100
    	col_int = int(col_flt)		
    	#print("The depth is: ",col_nam)
    	dframe_stemp = dframe_stemp.iloc[:,j]
    	#print(dframe_stemp_1)

    	threshold = 3.5 #set outlier threshold to this many standard deviations away from mean
    	mean_value = dframe_stemp_1.mean(axis = 0, skipna = True)
    	stdev = dframe_stemp_1.std()
	    	
    	
    	for k in range(0,len(dframe_stemp_1)):
    		stmp = dframe_stemp_1.iloc[k]
    		dat2 = date.iloc[k]
   		
    		z = (stmp - mean_value)/stdev
    		z_abs = abs(z)
    		#print(stmp)
    		#print(z)
		    		
    		if (z_abs > threshold or stmp == np.nan):
    			sval = np.nan
    			dval = dat2
    		else:
    			sval = stmp
    			dval = dat2
    		stemp_new_m1.append(sval)
    		dat_new_m1.append(dval)
    	
    	stemp_new_m1n = np.array(stemp_new_m1)
    	dat_new_m1n = np.array(dat_new_m1)
    	#print(stemp_new_m1n)

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
###### if 30cm < col name < 300cm then store in layer 7 dataframe ######
#    	elif(30 <= col_flt < 300):
#    		if(len(dframe_stemp_layer7_m1) == 4):  #if layer1 does not exist then create it
#    			dframe_stemp_layer7_m1 = pd.DataFrame(stemp_new_m1n, columns=[col_int])
#    		elif(len(dframe_stemp_layer7_m1) > 4):  #else append column to existing dataframe
#    			dframe_stemp2 = stemp_new_m1n
#    			dframe_stemp_layer7_m1[str(col_int)] = dframe_stemp2
#
###### if col name <= 30cm then store in top30 dataframe ######
#    	if(0 <= col_flt <= 30):
#    		if(len(dframe_stemp_top30_m1) == 4):  #if layer1 does not exist then create it
#    			dframe_stemp_top30_m1 = pd.DataFrame(stemp_new_m1n, columns = [col_int])
#    		elif(len(dframe_stemp_top30_m1) > 4):  #else append column to existing dataframe
#    			dframe_stemp2 = stemp_new_m1n
#    			dframe_stemp_top30_m1[str(col_int)] = dframe_stemp2


##### if 0cm <= col name <= 50cm then store in 0_50 dataframe
    	if(0 < col_flt < 50):
    		if(len(dframe_stemp_layer_0_50_m1) == 4):  #if layer1 does not exist then create it
    			dframe_stemp_layer_0_50_m1) = pd.DataFrame(dframe_stemp.iloc[:,j])
    			dframe_stemp_layer_0_50_m1.rename(columns={dframe_stemp_layer_0_50_m1.columns[0]: col_int},inplace=True)
    		elif(len(dframe_stemp_layer_0_50_m1) > 4):  #else append column to existing dataframe
    			dframe_stemp_0_50_m1 = dframe_stemp.iloc[:,j]
    			dframe_stemp_layer_0_50_m1[str(col_int)] = dframe_stemp_0_50_m1

##### if 0cm <= col name <= 50cm then store in 0_50 dataframe
    	if(50 <= col_flt < 100):
    		if(len(dframe_stemp_layer_50_100_m1) == 4):  #if layer1 does not exist then create it
    			dframe_stemp_layer_50_100_m1) = pd.DataFrame(dframe_stemp.iloc[:,j])
    			dframe_stemp_layer_50_100_m1.rename(columns={dframe_stemp_layer_50_100_m1.columns[0]: col_int},inplace=True)
    		elif(len(dframe_stemp_layer_50_100_m1) > 4):  #else append column to existing dataframe
    			dframe_stemp_50_100_m1 = dframe_stemp.iloc[:,j]
    			dframe_stemp_layer_50_100_m1[str(col_int)] = dframe_stemp_50_100_m1


################# do the averaging ####################### 
#    if(len(dframe_stemp_layer1_m1) > 4): 
#    	layer1_count = dframe_stemp_layer1_m1.count(axis=1)
#    	dframe_stemp_layer1_m1['Layer_Avg'] = dframe_stemp_layer1_m1.mean(axis=1)
#    	dframe_stemp_layer1_m1['Depths_Incl'] = layer1_count
#    	dframe_stemp_layer1_m1.insert(0,'Date',date)
#    	dframe_stemp_layer1_m1.insert(1,'Dataset',dtst)
#    	dframe_stemp_layer1_m1.insert(2,'Lat',lat)
#    	dframe_stemp_layer1_m1.insert(3,'Long',lon)
#    	dframe_stemp_layer1_m1.drop(dframe_stemp_layer1_m1[dframe_stemp_layer1_m1['Depths_Incl'] == 0].index, inplace=True)
#    	#print(dframe_stemp_layer1_m1)
#    	ofil1Z = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/zscore/0_9.9/site_",str(i),".csv"])
#    	print(ofil1Z)
#    	dframe_stemp_layer1_m1.to_csv(ofil1Z,na_rep="NaN",index=False)
#						
#    if(len(dframe_stemp_layer2_m1) > 4):
#    	layer2_count = dframe_stemp_layer2_m1.count(axis=1)
#    	dframe_stemp_layer2_m1['Layer_Avg'] = dframe_stemp_layer2_m1.mean(axis=1)
#    	dframe_stemp_layer2_m1['Depths_Incl'] = layer2_count
#    	dframe_stemp_layer2_m1.insert(0,'Date',date)
#    	dframe_stemp_layer2_m1.insert(1,'Dataset',dtst)
#    	dframe_stemp_layer2_m1.insert(2,'Lat',lat)
#    	dframe_stemp_layer2_m1.insert(3,'Long',lon)
#    	dframe_stemp_layer2_m1.drop(dframe_stemp_layer2_m1[dframe_stemp_layer2_m1['Depths_Incl'] == 0].index, inplace=True)
#    	#print(dframe_stemp_layer2_m1)    		
#    	ofil2Z = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/zscore/10_29.9/site_",str(i),".csv"])
#    	print(ofil2Z)
#    	dframe_stemp_layer2_m1.to_csv(ofil2Z,na_rep="NaN",index=False)
#				
#    if(len(dframe_stemp_layer3_m1) > 4):
#    	layer3_count = dframe_stemp_layer3_m1.count(axis=1)
#    	dframe_stemp_layer3_m1['Layer_Avg'] = dframe_stemp_layer3_m1.mean(axis=1)
#    	dframe_stemp_layer3_m1['Depths_Incl'] = layer3_count
#    	dframe_stemp_layer3_m1.insert(0,'Date',date)
#    	dframe_stemp_layer3_m1.insert(1,'Dataset',dtst)
#    	dframe_stemp_layer3_m1.insert(2,'Lat',lat)
#    	dframe_stemp_layer3_m1.insert(3,'Long',lon)
#    	dframe_stemp_layer3_m1.drop(dframe_stemp_layer3_m1[dframe_stemp_layer3_m1['Depths_Incl'] == 0].index, inplace=True)	   		
#    	ofil3Z = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/zscore/30_99.9/site_",str(i),".csv"])
#    	#print(dframe_stemp_layer3_m1)
#    	print(ofil3Z)
#    	dframe_stemp_layer3_m1.to_csv(ofil3Z,na_rep="NaN",index=False)
#
#    if(len(dframe_stemp_layer4_m1) > 4):
#    	layer4_count = dframe_stemp_layer4_m1.count(axis=1)
#    	dframe_stemp_layer4_m1['Layer_Avg'] = dframe_stemp_layer4_m1.mean(axis=1)
#    	dframe_stemp_layer4_m1['Depths_Incl'] = layer4_count
#    	dframe_stemp_layer4_m1.insert(0,'Date',date)
#    	dframe_stemp_layer4_m1.insert(1,'Dataset',dtst)
#    	dframe_stemp_layer4_m1.insert(2,'Lat',lat)
#    	dframe_stemp_layer4_m1.insert(3,'Long',lon)
#    	dframe_stemp_layer4_m1.drop(dframe_stemp_layer4_m1[dframe_stemp_layer4_m1['Depths_Incl'] == 0].index, inplace=True)
#    	#print(dframe_stemp_layer4_m1)
#    	ofil4Z = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/zscore/100_299.9/site_",str(i),".csv"])
#    	print(ofil4Z)
#    	dframe_stemp_layer4_m1.to_csv(ofil4Z,na_rep="NaN",index=False)
#				
#    if(len(dframe_stemp_layer5_m1) > 4):
#    	layer5_count = dframe_stemp_layer5_m1.count(axis=1)
#    	dframe_stemp_layer5_m1['Layer_Avg'] = dframe_stemp_layer5_m1.mean(axis=1)
#    	dframe_stemp_layer5_m1['Depths_Incl'] = layer5_count
#    	dframe_stemp_layer5_m1.insert(0,'Date',date)
#    	dframe_stemp_layer5_m1.insert(1,'Dataset',dtst)
#    	dframe_stemp_layer5_m1.insert(2,'Lat',lat)
#    	dframe_stemp_layer5_m1.insert(3,'Long',lon)
#    	dframe_stemp_layer5_m1.drop(dframe_stemp_layer5_m1[dframe_stemp_layer5_m1['Depths_Incl'] == 0].index, inplace=True)
#    	#print(dframe_stemp_layer5_m1)
#    	ofil5Z = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/zscore/300_deeper/site_",str(i),".csv"])
#    	print(ofil5Z)
#    	dframe_stemp_layer5_m1.to_csv(ofil5Z,na_rep="NaN",index=False)
#
#    if(len(dframe_stemp_top30_m1) > 4):
#    	top30_count = dframe_stemp_top30_m1.count(axis=1)
#    	dframe_stemp_top30_m1['Layer_Avg'] = dframe_stemp_top30_m1.mean(axis=1)
#    	dframe_stemp_top30_m1['Depths_Incl'] = top30_count
#    	dframe_stemp_top30_m1.insert(0,'Date',date)
#    	dframe_stemp_top30_m1.insert(1,'Dataset',dtst)
#    	dframe_stemp_top30_m1.insert(2,'Lat',lat)
#    	dframe_stemp_top30_m1.insert(3,'Long',lon)
#    	dframe_stemp_top30_m1.drop(dframe_stemp_top30_m1[dframe_stemp_top30_m1['Depths_Incl'] == 0].index, inplace=True)
#    	#print(dframe_stemp_top30_m1)
#    	ofil6Z = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/zscore/top_30cm/site_",str(i),".csv"])
#    	path = pathlib.Path(ofil6Z)
#    	path.parent.mkdir(parents=True, exist_ok=True)
#    	print(ofil6Z)
#    	dframe_stemp_top30_m1.to_csv(ofil6Z,na_rep="NaN",index=False)
#
    if(len(dframe_stemp_layer_0_50_m1) > 4):
    	layer_0_50_count = dframe_stemp_layer_0_50_m1.count(axis=1)
    	dframe_stemp_layer_0_50_m1['Layer_Avg'] = dframe_stemp_layer_0_50_m1.mean(axis=1)
    	dframe_stemp_layer_0_50_m1['Depths_Incl'] = layer_0_50_count
    	dframe_stemp_layer_0_50_m1.insert(0,'Date',date)
    	dframe_stemp_layer_0_50_m1.insert(1,'Dataset',dtst)
    	dframe_stemp_layer_0_50_m1.insert(2,'Lat',lat)
    	dframe_stemp_layer_0_50_m1.insert(3,'Long',lon)
    	dframe_stemp_layer_0_50_m1.drop(dframe_stemp_layer_0_50_m1[dframe_stemp_layer_0_50_m1['Depths_Incl'] == 0].index, inplace=True)
    	#print(dframe_stemp_layer_0_50_m1)
    	ofil7Z = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/zscore/0_49.9/site_",str(i),".csv"])
    	path = pathlib.Path(ofil7Z)
    	path.parent.mkdir(parents=True, exist_ok=True)
    	print(ofil7Z)
    	dframe_stemp_layer_0_50_m1.to_csv(ofil7Z,na_rep="NaN",index=False)

    if(len(dframe_stemp_layer_50_100_m1) > 4):
    	layer_50_100_count = dframe_stemp_layer_50_100_m1.count(axis=1)
    	dframe_stemp_layer_50_100_m1['Layer_Avg'] = dframe_stemp_layer_50_100_m1.mean(axis=1)
    	dframe_stemp_layer_50_100_m1['Depths_Incl'] = layer_50_100_count
    	dframe_stemp_layer_50_100_m1.insert(0,'Date',date)
    	dframe_stemp_layer_50_100_m1.insert(1,'Dataset',dtst)
    	dframe_stemp_layer_50_100_m1.insert(2,'Lat',lat)
    	dframe_stemp_layer_50_100_m1.insert(3,'Long',lon)
    	dframe_stemp_layer_50_100_m1.drop(dframe_stemp_layer_50_100_m1[dframe_stemp_layer_50_100_m1['Depths_Incl'] == 0].index, inplace=True)
    	#print(dframe_stemp_layer_50_100_m1)
    	ofil7Z = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/zscore/50_99.9/site_",str(i),".csv"])
    	path = pathlib.Path(ofil7Z)
    	path.parent.mkdir(parents=True, exist_ok=True)
    	print(ofil7Z)
    	dframe_stemp_layer_50_100_m1.to_csv(ofil7Z,na_rep="NaN",index=False)


## IQR method ##

#### loop through soil depths #########     	
    for j in range (0,total_col):
    	col_nam = col_val[j]
    	col_flt = float(col_nam)*100
    	col_int = int(col_flt)		
    	#print("The depth is: ",col_nam)
    	dframe_stemp = dframe_stemp.iloc[:,j]
    	#print(dframe_stemp_1)



    	Q1 = dframe_stemp_1.quantile(0.25)
    	Q3 = dframe_stemp_1.quantile(0.75)
    	IQR = Q3-Q1
    	fence = IQR*1.5
    	for k in range(0,len(dframe_stemp_1)):
    		stmp = dframe_stemp_1.iloc[k]
    		dat2 = date.iloc[k]
				    		   								
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


        	
###### if col name < 10cm then store in layer 1 dataframe ######
#    	if(0 <= col_flt < 10):
#    		if(len(dframe_stemp_layer1_m2) == 4):  #if layer1 does not exist then create it
#    			dframe_stemp_layer1_m2 = pd.DataFrame(stemp_new_m2n, columns = [col_int])
#    		elif(len(dframe_stemp_layer1_m2) > 4):  #else append column to existing dataframe
#    			dframe_stemp2 = stemp_new_m2n
#    			dframe_stemp_layer1_m2[str(col_int)] = dframe_stemp2
#
#
###### if 10cm <= col name < 30cm then store in layer 2 dataframe ######
#    	elif(10 <= col_flt < 30):
#    		if(len(dframe_stemp_layer2_m2) == 4):  #if layer1 does not exist then create it
#    			dframe_stemp_layer2_m2 = pd.DataFrame(stemp_new_m2n, columns=[col_int])
#    		elif(len(dframe_stemp_layer2_m2) > 4):  #else append column to existing dataframe
#    			dframe_stemp2 = stemp_new_m2n
#    			dframe_stemp_layer2_m2[str(col_int)] = dframe_stemp2
#
###### if 30cm <= col name < 100cm then store in layer 3 dataframe ######
#    	elif(30 <= col_flt < 100):
#    		if(len(dframe_stemp_layer3_m2) == 4):  #if layer1 does not exist then create it
#    			dframe_stemp_layer3_m2 = pd.DataFrame(stemp_new_m2n, columns=[col_int])
#    		elif(len(dframe_stemp_layer3_m2) > 4):  #else append column to existing dataframe
#    			dframe_stemp2 = stemp_new_m2n
#    			dframe_stemp_layer3_m2[str(col_int)] = dframe_stemp2
#
###### if 100cm < col name < 300cm then store in layer 4 dataframe ######
#    	elif(100 <= col_flt < 300):
#    		if(len(dframe_stemp_layer4_m2) == 4):  #if layer1 does not exist then create it
#    			dframe_stemp_layer4_m2 = pd.DataFrame(stemp_new_m2n, columns=[col_int])
#    		elif(len(dframe_stemp_layer4_m2) > 4):  #else append column to existing dataframe
#    			dframe_stemp2 = stemp_new_m2n
#    			dframe_stemp_layer4_m2[str(col_int)] = dframe_stemp2
#	
###### if col name > 300cm then store in layer 5 dataframe ######                		
#    	elif(col_flt >= 300):
#    		if(len(dframe_stemp_layer5_m2) == 4):  #if layer1 does not exist then create it
#    			dframe_stemp_layer5_m2 = pd.DataFrame(stemp_new_m2n, columns=[col_int])
#    		elif(len(dframe_stemp_layer5_m2) > 4):  #else append column to existing dataframe
#    			dframe_stemp2 = stemp_new_m2n
#    			dframe_stemp_layer5_m2[str(col_int)] = dframe_stemp2
#
###### if col name <= 30cm then store in top30 dataframe ######
#    	if(0 <= col_flt <= 30):
#    		if(len(dframe_stemp_top30_m2) == 4):  #if layer1 does not exist then create it
#    			dframe_stemp_top30_m2 = pd.DataFrame(stemp_new_m2n, columns = [col_int])
#    		elif(len(dframe_stemp_top30_m2) > 4):  #else append column to existing dataframe
#    			dframe_stemp2 = stemp_new_m2n
#    			dframe_stemp_top30_m2[str(col_int)] = dframe_stemp2
#
###### if 30cm < col name < 300cm then store in layer 4 dataframe ######
#    	elif(30 <= col_flt < 300):
#    		if(len(dframe_stemp_layer7_m2) == 4):  #if layer1 does not exist then create it
#    			dframe_stemp_layer7_m2 = pd.DataFrame(stemp_new_m2n, columns=[col_int])
#    		elif(len(dframe_stemp_layer7_m2) > 4):  #else append column to existing dataframe
#    			dframe_stemp2 = stemp_new_m2n
#    			dframe_stemp_layer7_m2[str(col_int)] = dframe_stemp2

##### if 0cm <= col name <= 50cm then store in 0_50 dataframe
    	if(0 < col_flt < 50):
    		if(len(dframe_stemp_layer_0_50_m2) == 4):  #if layer1 does not exist then create it
    			dframe_stemp_layer_0_50_m2) = pd.DataFrame(dframe_stemp.iloc[:,j])
    			dframe_stemp_layer_0_50_m2.rename(columns={dframe_stemp_layer_0_50_m2.columns[0]: col_int},inplace=True)
    		elif(len(dframe_stemp_layer_0_50_m2) > 4):  #else append column to existing dataframe
    			dframe_stemp_0_50_m2 = dframe_stemp.iloc[:,j]
    			dframe_stemp_layer_0_50_m2[str(col_int)] = dframe_stemp_0_50_m2

##### if 0cm <= col name <= 50cm then store in 0_50 dataframe
    	if(50 <= col_flt < 100):
    		if(len(dframe_stemp_layer_50_100_m2) == 4):  #if layer1 does not exist then create it
    			dframe_stemp_layer_50_100_m2) = pd.DataFrame(dframe_stemp.iloc[:,j])
    			dframe_stemp_layer_50_100_m2.rename(columns={dframe_stemp_layer_50_100_m2.columns[0]: col_int},inplace=True)
    		elif(len(dframe_stemp_layer_50_100_m2) > 4):  #else append column to existing dataframe
    			dframe_stemp_50_100_m2 = dframe_stemp.iloc[:,j]
    			dframe_stemp_layer_50_100_m2[str(col_int)] = dframe_stemp_50_100_m2	



################# do the averaging ####################### 
#    if(len(dframe_stemp_layer1_m2) > 4): 
#    	layer1_count = dframe_stemp_layer1_m2.count(axis=1)
#    	dframe_stemp_layer1_m2['Layer_Avg'] = dframe_stemp_layer1_m2.mean(axis=1)
#    	dframe_stemp_layer1_m2['Depths_Incl'] = layer1_count
#    	dframe_stemp_layer1_m2.insert(0,'Date',date)
#    	dframe_stemp_layer1_m2.insert(1,'Dataset',dtst)
#    	dframe_stemp_layer1_m2.insert(2,'Lat',lat)
#    	dframe_stemp_layer1_m2.insert(3,'Long',lon)
#    	dframe_stemp_layer1_m2.drop(dframe_stemp_layer1_m2[dframe_stemp_layer1_m2['Depths_Incl'] == 0].index, inplace=True)
#    	#print(dframe_stemp_layer1_m2)
#    	ofil1 = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/IQR/0_9.9/site_",str(i),".csv"])
#    	print(ofil1)
#    	dframe_stemp_layer1_m2.to_csv(ofil1,na_rep="NaN",index=False)
#						
#    if(len(dframe_stemp_layer2_m2) > 4):
#    	layer2_count = dframe_stemp_layer2_m2.count(axis=1)
#    	dframe_stemp_layer2_m2['Layer_Avg'] = dframe_stemp_layer2_m2.mean(axis=1)
#    	dframe_stemp_layer2_m2['Depths_Incl'] = layer2_count
#    	dframe_stemp_layer2_m2.insert(0,'Date',date)
#    	dframe_stemp_layer2_m2.insert(1,'Dataset',dtst)
#    	dframe_stemp_layer2_m2.insert(2,'Lat',lat)
#    	dframe_stemp_layer2_m2.insert(3,'Long',lon)
#    	dframe_stemp_layer2_m2.drop(dframe_stemp_layer2_m2[dframe_stemp_layer2_m2['Depths_Incl'] == 0].index, inplace=True)
#    	#print(dframe_stemp_layer2_m2)    		
#    	ofil2 = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/IQR/10_29.9/site_",str(i),".csv"])
#    	print(ofil2)
#    	dframe_stemp_layer2_m2.to_csv(ofil2,na_rep="NaN",index=False)
#				
#    if(len(dframe_stemp_layer3_m2) > 4):
#    	layer3_count = dframe_stemp_layer3_m2.count(axis=1)
#    	dframe_stemp_layer3_m2['Layer_Avg'] = dframe_stemp_layer3_m2.mean(axis=1)
#    	dframe_stemp_layer3_m2['Depths_Incl'] = layer3_count
#    	dframe_stemp_layer3_m2.insert(0,'Date',date)
#    	dframe_stemp_layer3_m2.insert(1,'Dataset',dtst)
#    	dframe_stemp_layer3_m2.insert(2,'Lat',lat)
#    	dframe_stemp_layer3_m2.insert(3,'Long',lon)
#    	dframe_stemp_layer3_m2.drop(dframe_stemp_layer3_m2[dframe_stemp_layer3_m2['Depths_Incl'] == 0].index, inplace=True)	   		
#    	ofil3 = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/IQR/30_99.9/site_",str(i),".csv"])
#    	#print(dframe_stemp_layer3_m2)
#    	print(ofil3)
#    	dframe_stemp_layer3_m2.to_csv(ofil3,na_rep="NaN",index=False)
#
#    if(len(dframe_stemp_layer4_m2) > 4):
#    	layer4_count = dframe_stemp_layer4_m2.count(axis=1)
#    	dframe_stemp_layer4_m2['Layer_Avg'] = dframe_stemp_layer4_m2.mean(axis=1)
#    	dframe_stemp_layer4_m2['Depths_Incl'] = layer4_count
#    	dframe_stemp_layer4_m2.insert(0,'Date',date)
#    	dframe_stemp_layer4_m2.insert(1,'Dataset',dtst)
#    	dframe_stemp_layer4_m2.insert(2,'Lat',lat)
#    	dframe_stemp_layer4_m2.insert(3,'Long',lon)
#    	dframe_stemp_layer4_m2.drop(dframe_stemp_layer4_m2[dframe_stemp_layer4_m2['Depths_Incl'] == 0].index, inplace=True)
#    	#print(dframe_stemp_layer4_m2)
#    	ofil4 = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/IQR/100_299.9/site_",str(i),".csv"])
#    	print(ofil4)
#    	dframe_stemp_layer4_m2.to_csv(ofil4,na_rep="NaN",index=False)
#				
#    if(len(dframe_stemp_layer5_m2) > 4):
#    	layer5_count = dframe_stemp_layer5_m2.count(axis=1)
#    	dframe_stemp_layer5_m2['Layer_Avg'] = dframe_stemp_layer5_m2.mean(axis=1)
#    	dframe_stemp_layer5_m2['Depths_Incl'] = layer5_count
#    	dframe_stemp_layer5_m2.insert(0,'Date',date)
#    	dframe_stemp_layer5_m2.insert(1,'Dataset',dtst)
#    	dframe_stemp_layer5_m2.insert(2,'Lat',lat)
#    	dframe_stemp_layer5_m2.insert(3,'Long',lon)
#    	dframe_stemp_layer5_m2.drop(dframe_stemp_layer5_m2[dframe_stemp_layer5_m2['Depths_Incl'] == 0].index, inplace=True)
#    	#print(dframe_stemp_layer5_m2)
#    	ofil5 = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/IQR/300_deeper/site_",str(i),".csv"])
#    	print(ofil5)
#    	dframe_stemp_layer5_m2.to_csv(ofil5,na_rep="NaN",index=False)
#
#    if(len(dframe_stemp_top30_m2) > 4):
#    	top30_count = dframe_stemp_top30_m2.count(axis=1)
#    	dframe_stemp_top30_m2['Layer_Avg'] = dframe_stemp_top30_m2.mean(axis=1)
#    	dframe_stemp_top30_m2['Depths_Incl'] = top30_count
#    	dframe_stemp_top30_m2.insert(0,'Date',date)
#    	dframe_stemp_top30_m2.insert(1,'Dataset',dtst)
#    	dframe_stemp_top30_m2.insert(2,'Lat',lat)
#    	dframe_stemp_top30_m2.insert(3,'Long',lon)
#    	dframe_stemp_top30_m2.drop(dframe_stemp_top30_m2[dframe_stemp_top30_m2['Depths_Incl'] == 0].index, inplace=True)
#    	#print(dframe_stemp_top30_m2)
#    	ofil6 = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/IQR/top_30cm/site_",str(i),".csv"])
#    	path = pathlib.Path(ofil6)
#    	path.parent.mkdir(parents=True, exist_ok=True)
#    	print(ofil6)
#    	dframe_stemp_top30_m2.to_csv(ofil6,na_rep="NaN",index=False)

#    if(len(dframe_stemp_layer7_m2) > 4):
#    	layer7_count = dframe_stemp_layer7_m2.count(axis=1)
#    	dframe_stemp_layer7_m2['Layer_Avg'] = dframe_stemp_layer7_m2.mean(axis=1)
#    	dframe_stemp_layer7_m2['Depths_Incl'] = layer7_count
#    	dframe_stemp_layer7_m2.insert(0,'Date',date)
#    	dframe_stemp_layer7_m2.insert(1,'Dataset',dtst)
#    	dframe_stemp_layer7_m2.insert(2,'Lat',lat)
#    	dframe_stemp_layer7_m2.insert(3,'Long',lon)
#    	dframe_stemp_layer7_m2.drop(dframe_stemp_layer7_m2[dframe_stemp_layer7_m2['Depths_Incl'] == 0].index, inplace=True)
#    	#print(dframe_stemp_layer7_m2)
#    	ofil7 = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/IQR/30_299.9/site_",str(i),".csv"])
#    	path = pathlib.Path(ofil7)
#    	path.parent.mkdir(parents=True, exist_ok=True)
#    	print(ofil7)
#    	dframe_stemp_layer7_m2.to_csv(ofil7,na_rep="NaN",index=False)


    if(len(dframe_stemp_layer_0_50_m2) > 4):
    	layer_0_50_count = dframe_stemp_layer_0_50_m2.count(axis=1)
    	dframe_stemp_layer_0_50_m2['Layer_Avg'] = dframe_stemp_layer_0_50_m2.mean(axis=1)
    	dframe_stemp_layer_0_50_m2['Depths_Incl'] = layer_0_50_count
    	dframe_stemp_layer_0_50_m2.insert(0,'Date',date)
    	dframe_stemp_layer_0_50_m2.insert(1,'Dataset',dtst)
    	dframe_stemp_layer_0_50_m2.insert(2,'Lat',lat)
    	dframe_stemp_layer_0_50_m2.insert(3,'Long',lon)
    	dframe_stemp_layer_0_50_m2.drop(dframe_stemp_layer_0_50_m2[dframe_stemp_layer_0_50_m2['Depths_Incl'] == 0].index, inplace=True)
    	#print(dframe_stemp_layer_0_50_m2)
    	ofil7 = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/IQR/0_49.9/site_",str(i),".csv"])
    	path = pathlib.Path(ofil7)
    	path.parent.mkdir(parents=True, exist_ok=True)
    	print(ofil7)
    	dframe_stemp_layer_0_50_m2.to_csv(ofil7,na_rep="NaN",index=False)

    if(len(dframe_stemp_layer_50_100_m2) > 4):
    	layer_50_100_count = dframe_stemp_layer_50_100_m2.count(axis=1)
    	dframe_stemp_layer_50_100_m2['Layer_Avg'] = dframe_stemp_layer_50_100_m2.mean(axis=1)
    	dframe_stemp_layer_50_100_m2['Depths_Incl'] = layer_50_100_count
    	dframe_stemp_layer_50_100_m2.insert(0,'Date',date)
    	dframe_stemp_layer_50_100_m2.insert(1,'Dataset',dtst)
    	dframe_stemp_layer_50_100_m2.insert(2,'Lat',lat)
    	dframe_stemp_layer_50_100_m2.insert(3,'Long',lon)
    	dframe_stemp_layer_50_100_m2.drop(dframe_stemp_layer_50_100_m2[dframe_stemp_layer_50_100_m2['Depths_Incl'] == 0].index, inplace=True)
    	#print(dframe_stemp_layer_50_100_m2)
    	ofil7 = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/IQR/50_99.9/site_",str(i),".csv"])
    	path = pathlib.Path(ofil7)
    	path.parent.mkdir(parents=True, exist_ok=True)
    	print(ofil7)
    	dframe_stemp_layer_50_100_m2.to_csv(ofil7,na_rep="NaN",index=False)






