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
    #print(len(dframe_stemp_layer1))	   	
######## loop through soil depths ##########     	
    for j in range (0,total_col):
    	col_nam = col_val[j]
    	col_flt = float(col_nam)*100
    	col_int = int(col_flt)		
    	#print(col_int,col_nam)    	
##### if col name < 10cm then store in layer 1 dataframe ######
    	if(0 <= col_flt < 10):
    		if(len(dframe_stemp_layer1) == 4):  #if layer1 does not exist then create it
    			dframe_stemp_layer1 = pd.DataFrame(dframe_stemp.iloc[:,j])
    			dframe_stemp_layer1.rename(columns={dframe_stemp_layer1.columns[0]: col_int},inplace=True)
    		elif(len(dframe_stemp_layer1) > 4):  #else append column to existing dataframe
    			dframe_stemp1 = dframe_stemp.iloc[:,j]
    			dframe_stemp_layer1[str(col_int)] = dframe_stemp1


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
    			dframe_stemp3 = dframe_stemp.iloc[:,j]
    			dframe_stemp_layer3[str(col_int)] = dframe_stemp3

##### if 100cm < col name < 300cm then store in layer 4 dataframe ######
    	if(100 <= col_flt < 300):
    		if(len(dframe_stemp_layer4) == 4):  #if layer1 does not exist then create it
    			dframe_stemp_layer4 = pd.DataFrame(dframe_stemp.iloc[:,j])
    			dframe_stemp_layer4.rename(columns={dframe_stemp_layer4.columns[0]: col_int},inplace=True)
    		elif(len(dframe_stemp_layer4) > 4):  #else append column to existing dataframe
    			dframe_stemp4 = dframe_stemp.iloc[:,j]
    			dframe_stemp_layer4[str(col_int)] = dframe_stemp4
	
##### if col name > 300cm then store in layer 5 dataframe ######                		
    	if(col_flt >= 300):
    		if(len(dframe_stemp_layer5) == 4):  #if layer1 does not exist then create it
    			dframe_stemp_layer5 = pd.DataFrame(dframe_stemp.iloc[:,j])
    			dframe_stemp_layer5.rename(columns={dframe_stemp_layer5.columns[0]: col_int},inplace=True)
    		elif(len(dframe_stemp_layer5) > 4):  #else append column to existing dataframe
    			dframe_stemp5 = dframe_stemp.iloc[:,j]
    			dframe_stemp_layer5[str(col_int)] = dframe_stemp5

##### if col name <=30cm then store in top30 dataframe #####
    	if(0 <= col_flt <= 30):
    		if(len(dframe_stemp_top30) == 4):  #if layer1 does not exist then create it
    			dframe_stemp_top30 = pd.DataFrame(dframe_stemp.iloc[:,j])
    			dframe_stemp_top30.rename(columns={dframe_stemp_top30.columns[0]: col_int},inplace=True)
    		elif(len(dframe_stemp_top30) > 4):  #else append column to existing dataframe
    			dframe_stemp6 = dframe_stemp.iloc[:,j]
    			dframe_stemp_top30[str(col_int)] = dframe_stemp6

##### if 30cm <= col name < 300cm then store in layer 7 dataframe ######
    	if(30 < col_flt < 300):
    		if(len(dframe_stemp_layer7) == 4):  #if layer1 does not exist then create it
    			dframe_stemp_layer7 = pd.DataFrame(dframe_stemp.iloc[:,j])
    			dframe_stemp_layer7.rename(columns={dframe_stemp_layer7.columns[0]: col_int},inplace=True)
    		elif(len(dframe_stemp_layer7) > 4):  #else append column to existing dataframe
    			dframe_stemp7 = dframe_stemp.iloc[:,j]
    			dframe_stemp_layer7[str(col_int)] = dframe_stemp7

##### if 0cm <= col name <= 50cm then store in 0_50 dataframe
    	if(0 < col_flt < 50):
    		if(len(dframe_stemp_layer_0_50) == 4):  #if layer1 does not exist then create it
    			dframe_stemp_layer_0_50) = pd.DataFrame(dframe_stemp.iloc[:,j])
    			dframe_stemp_layer_0_50.rename(columns={dframe_stemp_layer_0_50.columns[0]: col_int},inplace=True)
    		elif(len(dframe_stemp_layer_0_50) > 4):  #else append column to existing dataframe
    			dframe_stemp_0_50 = dframe_stemp.iloc[:,j]
    			dframe_stemp_layer_0_50[str(col_int)] = dframe_stemp_0_50

##### if 0cm <= col name <= 50cm then store in 0_50 dataframe
    	if(50 < col_flt =< 100):
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
    	ofil8 = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/outliers/0_50/site_",str(i),".csv"])
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
    	ofil9 = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/outliers/50_100/site_",str(i),".csv"])
    	path = pathlib.Path(ofil7)
    	path.parent.mkdir(parents=True, exist_ok=True)
    	print(ofil9)
    	#print(dframe_stemp_layer_50_100)
    	dframe_stemp_layer_50_100.to_csv(ofil7,na_rep="NaN",index=False)


############################## if dataset is Kropp ##################################
wfil = str("/mnt/data/users/herringtont/soil_temp/In-Situ/Kropp/data/soil_temp_date_coord2.csv")
dframe = pd.read_csv(wfil)
dframe.replace(-999, np.nan, inplace=True)
sid = np.unique(dframe['site_id'])
sitid = sid[~np.isnan(sid)]
col1 = dframe['Date']
date_fmt = "%Y-%m-%d"
datetime_column = str_to_datetime(col1, date_fmt)
    	
####### group by site #######
for j in sitid:
    dframe_siteid = dframe[dframe['site_id'] == j]
    sdepth = np.unique(dframe_siteid['st_depth'])
    sdep = sdepth[~np.isnan(sdepth)]
    site_date = dframe_siteid['Date']
    dt_col = str_to_datetime(site_date,date_fmt)
    date_uq = np.unique(site_date)
    dt_col2 = str_to_datetime(date_uq,date_fmt)
    dframe_siteid = dframe_siteid.set_index(site_date)
    if (j <= 41):
    	j2 = j + 68
    elif (44 <= j <= 46):
    	j2 = j + 66
    elif (j >= 47):
    	j2 = j + 64
    	sint = str(j2)
    print("the site is: ",j2)
    #print(dframe_siteid)
    #print(date_uq)
    dframe_stemp_layer1 = "None"
    dframe_stemp_layer2 = "None" 
    dframe_stemp_layer3 = "None"
    dframe_stemp_layer4 = "None"
    dframe_stemp_layer5 = "None"
    dframe_stemp_top30 = "None"
    dframe_stemp_layer7 = "None"
    dframe_stemp_0_50 = "None"
    dframe_stemp_50_100 = "None"
######## loop through depths ######
    for k in sdep:
    	wdep = int(k)
    	strdep = str(wdep)
    	dframe_sdep = dframe_siteid[dframe_siteid['st_depth'] == k]
    	soil_dep = dframe_sdep.iloc[1,3]
    	soil_dep2 = int(soil_dep)
    	sdept = str(soil_dep2)
    	lat = dframe_sdep.iloc[1,5]
    	lon = dframe_sdep.iloc[1,6]
    	dtst = "Kropp"
    	#print(dframe_sdep)	
    	dframe_sdep = dframe_sdep.reindex(date_uq,fill_value = np.nan)
    	dframe_sdep.drop(['Date','stemp_id','site_id','lat','long','st_depth'], axis = 1,inplace=True)
    	dframe_sdep.insert(0,'Dataset',dtst)
    	dframe_sdep.insert(1,'Date',dframe_sdep.index)
    	dframe_sdep.insert(3,'st_depth',soil_dep)
    	dframe_sdep.insert(4,'lat',lat)
    	dframe_sdep.insert(5,'long',lon)    			
    	sdep_soilt = np.array(dframe_sdep['soil_t'].values)
    	sdep_date = dframe_sdep['Date']
    	#print(sdep_date)
    	
    	
######## if depth < 10cm then store in layer 1 dataframe ######
    	if (0 <= soil_dep2 < 10):
    		if(len(dframe_stemp_layer1) == 4):  #if layer1 does not exist then create it
    			dframe_stemp_layer1 = pd.DataFrame(data=sdep_soilt, columns=[sdept])
    		elif(len(dframe_stemp_layer1) > 4):		   			
    			dframe_stemp1 = sdep_soilt
    			dframe_stemp_layer1[sdept] = dframe_stemp1
    					
####### if 10cm <= depth < 30cm then store in layer 2 dataframe ######
    	if (10 <= soil_dep2 < 30):
    		if(len(dframe_stemp_layer2) == 4):  #if layer1 does not exist then create it
    			dframe_stemp_layer2 = pd.DataFrame(data=sdep_soilt, columns=[sdept])
    		elif(len(dframe_stemp_layer2) > 4):		   			
    			dframe_stemp2 = sdep_soilt
    			dframe_stemp_layer2[sdept] = dframe_stemp2

####### if 30cm <= depth < 100cm then store in layer 3 dataframe ######
    	if (30 <= soil_dep2 < 100):
    		if(len(dframe_stemp_layer3) == 4):  #if layer1 does not exist then create it
    			dframe_stemp_layer3 = pd.DataFrame(data=sdep_soilt, columns=[sdept])
    		elif(len(dframe_stemp_layer3) > 4):		   			
    			dframe_stemp3 = sdep_soilt
    			dframe_stemp_layer3[sdept] = dframe_stem3

####### if 100cm <= depth < 300cm then store in layer 4 dataframe ######
    	if (100 <= soil_dep2 < 300):
    		if(len(dframe_stemp_layer4) == 4):  #if layer1 does not exist then create it
    			dframe_stemp_layer4 = pd.DataFrame(data=sdep_soilt, columns=[sdept])
    		elif(len(dframe_stemp_layer4) > 4):		   			
    			dframe_stemp4= sdep_soilt
    			dframe_stemp_layer4[sdept] = dframe_stemp4

####### if depth >= 300cm then store in layer 5 dataframe ######
    	if (soil_dep2 >= 300):
    		if(len(dframe_stemp_layer5) == 4):  #if layer1 does not exist then create it
    			dframe_stemp_layer5 = pd.DataFrame(data=sdep_soilt, columns=[sdept])
    		elif(len(dframe_stemp_layer5) > 4):		   			
    			dframe_stemp5 = sdep_soilt
    			dframe_stemp_layer5[sdept] = dframe_stemp5

####### if depth <= 30cm then store in top30 dataframe ######
    	if (0 <= soil_dep2 <= 30):
    		if(len(dframe_stemp_top30) == 4):  #if layer1 does not exist then create it
    			dframe_stemp_top30 = pd.DataFrame(data=sdep_soilt, columns=[sdept])
    		elif(len(dframe_stemp_top30) > 4):		   			
    			dframe_stemp_top30 = sdep_soilt
    			dframe_stemp_top30[sdept] = dframe_top30

####### if 30cm <= depth < 300cm then store in layer 7 dataframe ######
    	if (30 <= soil_dep2 < 300):
    		if(len(dframe_stemp_layer7) == 4):  #if layer1 does not exist then create it
    			dframe_stemp_layer7 = pd.DataFrame(data=sdep_soilt, columns=[sdept])
    		elif(len(dframe_stemp_layer7) > 4):		   			
    			dframe_stemp7 = sdep_soilt
    			dframe_stemp_layer7[sdept] = dframe_stemp7

####### if 0cm <= depth =< 50cm then store in layer 7 dataframe ######
    	if (0 <= soil_dep2 =< 50):
    		if(len(dframe_stemp_layer_0_50) == 4):  #if layer1 does not exist then create it
    			dframe_stemp_layer_0_50 = pd.DataFrame(data=sdep_soilt, columns=[sdept])
    		elif(len(dframe_stemp_layer_0_50) > 4):		   			
    			dframe_stemp_0_50 = sdep_soilt
    			dframe_stemp_layer_0_50[sdept] = dframe_stemp_0_50

####### if 50cm <= depth =< 100cm then store in layer 7 dataframe ######
    	if (50 < soil_dep2 =< 100):
    		if(len(dframe_stemp_layer_50_100) == 4):  #if layer1 does not exist then create it
    			dframe_stemp_layer_50_100 = pd.DataFrame(data=sdep_soilt, columns=[sdept])
    		elif(len(dframe_stemp_layer_50_100) > 4):		   			
    			dframe_stemp_50_100 = sdep_soilt
    			dframe_stemp_layer_50_100[sdept] = dframe_stemp_50_100

################### do the averaging ######################

#    if(len(dframe_stemp_layer1) > 4): 
#    	layer1_count = dframe_stemp_layer1.count(axis=1)
#    	dframe_stemp_layer1['Layer_Avg'] = dframe_stemp_layer1.mean(axis=1)
#    	dframe_stemp_layer1['Depths_Incl'] = layer1_count
#    	dframe_stemp_layer1.insert(0,'Date',date_uq)
#    	dframe_stemp_layer1.insert(1,'Dataset',dtst)
#    	dframe_stemp_layer1.insert(2,'Lat',lat)
#    	dframe_stemp_layer1.insert(3,'Long',lon)
#    	#print(dframe_stemp_layer1)
#    	dframe_stemp_layer1.drop(dframe_stemp_layer1[dframe_stemp_layer1['Depths_Incl'] == 0].index, inplace=True)
#    	print(dframe_stemp_layer1)
#    	ofil1 = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/outliers/0_9.9/site_",str(j2),".csv"])
#    	print(ofil1)
#    	dframe_stemp_layer1.to_csv(ofil1,na_rep="NaN",index=False)
#						
#    if(len(dframe_stemp_layer2) > 4):
#    	layer2_count = dframe_stemp_layer2.count(axis=1)
#    	dframe_stemp_layer2['Layer_Avg'] = dframe_stemp_layer2.mean(axis=1)
#    	dframe_stemp_layer2['Depths_Incl'] = layer2_count
#    	dframe_stemp_layer2.insert(0,'Date',date_uq)
#    	dframe_stemp_layer2.insert(1,'Dataset',dtst)
#    	dframe_stemp_layer2.insert(2,'Lat',lat)
#    	dframe_stemp_layer2.insert(3,'Long',lon)
#    	#print(dframe_stemp_layer2)    		
#    	dframe_stemp_layer2.drop(dframe_stemp_layer2[dframe_stemp_layer2['Depths_Incl'] == 0].index, inplace=True)
#    	print(dframe_stemp_layer2)
#    	ofil2 = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/outliers/10_29.9/site_",str(j2),".csv"])
#    	print(ofil2)
#    	dframe_stemp_layer2.to_csv(ofil2,na_rep="NaN",index=False)
#				
#    if(len(dframe_stemp_layer3) > 4):
#    	layer3_count = dframe_stemp_layer3.count(axis=1)
#    	dframe_stemp_layer3['Layer_Avg'] = dframe_stemp_layer3.mean(axis=1)
#    	dframe_stemp_layer3['Depths_Incl'] = layer3_count
#    	dframe_stemp_layer3.insert(0,'Date',date_uq)
#    	dframe_stemp_layer3.insert(1,'Dataset',dtst)
#    	dframe_stemp_layer3.insert(2,'Lat',lat)
#    	dframe_stemp_layer3.insert(3,'Long',lon)   		
#    	ofil3 = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/outliers/30_99.9/site_",str(j2),".csv"])
#    	#print(dframe_stemp_layer3)
#    	dframe_stemp_layer3.drop(dframe_stemp_layer3[dframe_stemp_layer3['Depths_Incl'] == 0].index, inplace=True)
#    	print(dframe_stemp_layer3)	
#    	print(ofil3)	
#    	dframe_stemp_layer3.to_csv(ofil3,na_rep="NaN",index=False)
#
#    if(len(dframe_stemp_layer4) > 4):
#    	layer4_count = dframe_stemp_layer4.count(axis=1)
#    	dframe_stemp_layer4['Layer_Avg'] = dframe_stemp_layer4.mean(axis=1)
#    	dframe_stemp_layer4['Depths_Incl'] = layer4_count
#    	dframe_stemp_layer4.insert(0,'Date',date_uq)
#    	dframe_stemp_layer4.insert(1,'Dataset',dtst)
#    	dframe_stemp_layer4.insert(2,'Lat',lat)
#    	dframe_stemp_layer4.insert(3,'Long',lon)
#    	#print(dframe_stemp_layer4)
#    	dframe_stemp_layer4.drop(dframe_stemp_layer4[dframe_stemp_layer4['Depths_Incl'] == 0].index, inplace=True)
#    	print(dframe_stemp_layer4)
#    	ofil4 = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/outliers/100_299.9/site_",str(j2),".csv"])
#    	print(ofil4)
#    	dframe_stemp_layer4.to_csv(ofil4,na_rep="NaN",index=False)
#				
#    if(len(dframe_stemp_layer5) > 4):
#    	layer5_count = dframe_stemp_layer5.count(axis=1)
#    	dframe_stemp_layer5['Layer_Avg'] = dframe_stemp_layer5.mean(axis=1)
#    	dframe_stemp_layer5['Depths_Incl'] = layer5_count
#    	dframe_stemp_layer5.insert(0,'Date',date_uq)
#    	dframe_stemp_layer5.insert(1,'Dataset',dtst)
#    	dframe_stemp_layer5.insert(2,'Lat',lat)
#    	dframe_stemp_layer5.insert(3,'Long',lon)
#    	#print(dframe_stemp_layer5)
#    	dframe_stemp_layer5.drop(dframe_stemp_layer5[dframe_stemp_layer5['Depths_Incl'] == 0].index, inplace=True)
#    	print(dframe_stemp_layer5)
#    	ofil5 = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/outliers/300_deeper/site_",str(j2),".csv"])
#    	print(ofil5)
#    	dframe_stemp_layer5.to_csv(ofil5,na_rep="NaN",index=False)	
#
#
#
#    if(len(dframe_stemp_top30) > 4):
#    	top30_count = dframe_stemp_top30.count(axis=1)
#    	dframe_stemp_top30['Layer_Avg'] = dframe_stemp_top30.mean(axis=1)
#    	dframe_stemp_top30['Depths_Incl'] = top30_count
#    	dframe_stemp_top30.insert(0,'Date',date_uq)
#    	dframe_stemp_top30.insert(1,'Dataset',dtst)
#    	dframe_stemp_top30.insert(2,'Lat',lat)
#    	dframe_stemp_top30.insert(3,'Long',lon)
#    	#print(dframe_stemp_top30)
#    	dframe_stemp_top30.drop(dframe_stemp_top30[dframe_stemp_top30['Depths_Incl'] == 0].index, inplace=True)
#    	print(dframe_stemp_top30)
#    	ofil6 = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/outliers/top_30cm/site_",str(j2),".csv"])
#    	path = pathlib.Path(ofil6)
#    	path.parent.mkdir(parents=True, exist_ok=True)
#    	print(ofil6)
#    	dframe_stemp_top30.to_csv(ofil6,na_rep="NaN",index=False)


    if(len(dframe_stemp_layer7) > 4):
    	layer7_count = dframe_stemp_layer7.count(axis=1)
    	dframe_stemp_layer7['Layer_Avg'] = dframe_stemp_layer7.mean(axis=1)
    	dframe_stemp_layer7['Depths_Incl'] = layer7_count
    	dframe_stemp_layer7.insert(0,'Date',date_uq)
    	dframe_stemp_layer7.insert(1,'Dataset',dtst)
    	dframe_stemp_layer7.insert(2,'Lat',lat)
    	dframe_stemp_layer7.insert(3,'Long',lon)
    	#print(dframe_stemp_top30)
    	dframe_stemp_layer7.drop(dframe_stemp_layer7[dframe_stemp_layer7['Depths_Incl'] == 0].index, inplace=True)
    	print(dframe_stemp_layer7)
    	ofil7 = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/outliers/30_299.9/site_",str(j2),".csv"])
    	path = pathlib.Path(ofil7)
    	path.parent.mkdir(parents=True, exist_ok=True)
    	print(ofil7)
    	dframe_stemp_layer7.to_csv(ofil7,na_rep="NaN",index=False)
