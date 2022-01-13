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

################### define variables ##################


#dataset = ['Kropp', 'GTN-P', 'Nordicana_Russia','NWT','Yukon']
dataset = ['NWT','Yukon']
#dataset = ['GTN-P', 'Nordicana_Russia','NWT','Yukon']
#layers = ['0_9.9','10_29.9','30_99.9','100_299.9','300_deeper','top_30cm','30_299.9','0_49.9','50_99.9','100_200']
layers = [(0,50),(50,100),(100,200.1)]

dframe_stemp_layer = "None"
dframe_stemp_layer_m1 = "None"
dframe_stemp_layer_m2 = "None"

#### define dataset specific variables ####

if (dataset == 'GTN-P'):

    wdir = "/mnt/data/users/herringtont/soil_temp/In-Situ/GTN-P/site_level/"
    site_begin = 1
    site_end =  69

    date_fmt = "%Y-%m-%d %H:%M:%S"
    
elif (dataset == 'Kropp'):

    wdir = "/mnt/data/users/herringtont/soil_temp/In-Situ/Kropp"

elif (dataset == 'Nordicana_Russia'):

    wdir = "/mnt/data/users/herringtont/soil_temp/In-Situ/Russia_Nordicana/"
    site_begin = 300
    site_end = 788

## Note that if site is between 300 and 757, it is from Russia_Hydromet
## Else if site is between 758 and 788, it is from Nordicana

elif (dataset == 'NWT'):

    site_begin = 789
    site_end = 1227

elif (dataset == 'Yukon'):

    site_begin = 1228
    site_end = 1339


for i in dataset:

    dataset_i = i
    print("The datset is:",i)

    if (dataset_i == 'Kropp'):
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
    		#print("the site is: ",j2)
    		#print(sdep)

    		dframe_outlier = "None"
    		dframe_zscore = "None"
    		dframe_IQR = "None"
######## loop through depths ######
    		for k in sdep:
    			wdep = int(k)
    			strdep = str(wdep)
    			dframe_sdep = dframe_siteid[dframe_siteid['st_depth'] == k]
    			soil_dep = dframe_sdep.iloc[1,3]
    			soil_dep2 = float(soil_dep)
    			sdept = str(soil_dep2)
    			lat = dframe_sdep.iloc[1,5]
    			lon = dframe_sdep.iloc[1,6]
    			dtst = "Kropp"	
    			dframe_sdep = dframe_sdep.reindex(date_uq,fill_value = np.nan)
    			dframe_sdep.drop(['Date','stemp_id','site_id','lat','long','st_depth'], axis = 1,inplace=True)
    			dframe_sdep.insert(0,'Dataset',dtst)
    			dframe_sdep.insert(1,'Date',dframe_sdep.index)
    			dframe_sdep.insert(3,'st_depth',soil_dep)
    			dframe_sdep.insert(4,'lat',lat)
    			dframe_sdep.insert(5,'long',lon)    			
    			sdep_soilt = np.array(dframe_sdep['soil_t'].values)
    			sdep_date = dframe_sdep['Date']

    			stemp_new_m1 = []
    			dat_new_m1 = []
    			stemp_new_m2 = []
    			dat_new_m2 = []

    		#### Remove Outliers (Z-Score Method) ####

    			threshold = 3.5 #set outlier threshold to this many standard deviations away from mean
    			mean_value = np.nanmean(sdep_soilt)
    			stdev = np.nanstd(sdep_soilt)    	
    			for l in range(0,len(sdep_soilt)):
    				stmp = sdep_soilt[l]
    				dat2 = sdep_date.iloc[l]
   		
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

		#### Remove Outliers - IQR Method ####		
    			Q1 = np.nanquantile(sdep_soilt,0.25)
    			Q3 = np.nanquantile(sdep_soilt,0.75)
    			IQR = Q3-Q1
    			fence = IQR*1.5
    			for l in range(0,len(sdep_soilt)):
    				stmp = sdep_soilt[l]
    				dat2 = sdep_date[l]
				    		   								
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


##### Create Reorganized Dataframe #####

    			if (len(dframe_outlier) == 4): 
    				dframe_outlier = pd.DataFrame(data=sdep_soilt, columns=[soil_dep2])
    			elif (len (dframe_outlier) > 4):
    				dframe_temp = sdep_soilt
    				dframe_outlier[soil_dep2] = dframe_temp

    			if (len(dframe_zscore) == 4): 
    				dframe_zscore = pd.DataFrame(data=stemp_new_m1n, columns=[soil_dep2])
    			elif (len (dframe_zscore) > 4):
    				dframe_temp = stemp_new_m1n
    				dframe_zscore[soil_dep2] = dframe_temp

    			if (len(dframe_IQR) == 4): 
    				dframe_IQR = pd.DataFrame(data=stemp_new_m2n, columns=[soil_dep2])
    			elif (len (dframe_IQR) > 4):
    				dframe_temp = stemp_new_m2n
    				dframe_IQR[soil_dep2] = dframe_temp


    		depth_names_o = np.array(dframe_outlier.columns)
    		depth_names_o = depth_names_o.astype(float)
    		depth_names_z = np.array(dframe_zscore.columns)
    		depth_names_z = depth_names_z.astype(float)
    		depth_names_I = np.array(dframe_IQR.columns)
    		depth_names_I = depth_names_I.astype(float)


#### Outliers Included ####


    		#### Separate into Layers ####


    		for k in layers:
    			dframe_layer = "None"
    			top_boundary = float(k[0])
    			bottom_boundary = float(k[1])
    			btm_bdy2 = bottom_boundary-0.1

    			layer_name = ''.join([str(int(top_boundary))+'_'+str(btm_bdy2)])
    			#print(layer_name)

    			depths_in_layer = depth_names_o[(depth_names_o >= top_boundary) & (depth_names_o < bottom_boundary)]
    			if (len(depths_in_layer) == 0):
    				continue
    			#print(depths_in_layer)

    			#### only include depths in layer ####

    			for l in depths_in_layer:
    				if(len(dframe_layer) == 4):
    					dframe_temp = dframe_outlier[l].values
    					dframe_layer = pd.DataFrame(data=dframe_temp,columns=[l])

    				elif(len(dframe_outlier) > 4):
    					dframe_temp = dframe_outlier[l].values
    					dframe_layer[l] = dframe_temp

    			#print(dframe_layer) 

   				   					
    			if(len(dframe_layer) > 4): 
    				layer_count = dframe_layer.count(axis=1)
    				dframe_layer['Layer_Avg'] = dframe_layer.mean(axis=1)
    				dframe_layer['Depths_Incl'] = layer_count
    				dframe_layer.insert(0,'Date',date_uq)
    				dframe_layer.insert(1,'Dataset',dtst)
    				dframe_layer.insert(2,'Lat',lat)
    				dframe_layer.insert(3,'Long',lon)
    				#print(dframe_layer)
    				dframe_layer.drop(dframe_layer[dframe_layer['Depths_Incl'] == 0].index, inplace=True)
    				print(dframe_layer)
    				if (len(dframe_layer) == 0):
    					continue
    				ofil = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/outliers/"+str(layer_name)+"/site_",str(j2),".csv"])
    				path2 = pathlib.Path(ofil)
    				path2.parent.mkdir(parents=True, exist_ok=True)
    				print(ofil)
    				dframe_layer.to_csv(ofil,na_rep="NaN",index=False)
    			
#### Z-Score ####


    		#### Separate into Layers ####


    		for k in layers:
    			dframe_layer = "None"
    			top_boundary = float(k[0])
    			bottom_boundary = float(k[1])
    			btm_bdy2 = bottom_boundary-0.1

    			layer_name = ''.join([str(int(top_boundary))+'_'+str(btm_bdy2)])
    			#print(layer_name)

    			depths_in_layer = depth_names_z[(depth_names_z >= top_boundary) & (depth_names_z < bottom_boundary)]
    			if (len(depths_in_layer) == 0):
    				continue
    			#print(depths_in_layer)

    			#### only include depths in layer ####

    			for l in depths_in_layer:
    				if(len(dframe_layer) == 4):
    					dframe_temp = dframe_zscore[l].values
    					dframe_layer = pd.DataFrame(data=dframe_temp,columns=[l])

    				elif(len(dframe_zscore) > 4):
    					dframe_temp = dframe_zscore[l].values
    					dframe_layer[l] = dframe_temp

    			#print(dframe_layer) 

   				   					
    			if(len(dframe_layer) > 4): 
    				layer_count = dframe_layer.count(axis=1)
    				dframe_layer['Layer_Avg'] = dframe_layer.mean(axis=1)
    				dframe_layer['Depths_Incl'] = layer_count
    				dframe_layer.insert(0,'Date',dat_new_m1n)
    				dframe_layer.insert(1,'Dataset',dtst)
    				dframe_layer.insert(2,'Lat',lat)
    				dframe_layer.insert(3,'Long',lon)
    				#print(dframe_layer)
    				dframe_layer.drop(dframe_layer[dframe_layer['Depths_Incl'] == 0].index, inplace=True)
    				print(dframe_layer)
    				if (len(dframe_layer) == 0):
    					continue
    				ofil = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/zscore/"+str(layer_name)+"/site_",str(j2),".csv"])
    				path2 = pathlib.Path(ofil)
    				path2.parent.mkdir(parents=True, exist_ok=True)
    				print(ofil)
    				dframe_layer.to_csv(ofil,na_rep="NaN",index=False)

#### IQR ####


    		#### Separate into Layers ####


    		for k in layers:
    			dframe_layer = "None"
    			top_boundary = float(k[0])
    			bottom_boundary = float(k[1])
    			btm_bdy2 = bottom_boundary-0.1

    			layer_name = ''.join([str(int(top_boundary))+'_'+str(btm_bdy2)])
    			#print(layer_name)

    			depths_in_layer = depth_names_I[(depth_names_I >= top_boundary) & (depth_names_I < bottom_boundary)]
    			if (len(depths_in_layer) == 0):
    				continue
    			#print(depths_in_layer)

    			#### only include depths in layer ####

    			for l in depths_in_layer:
    				if(len(dframe_layer) == 4):
    					dframe_temp = dframe_IQR[l].values
    					dframe_layer = pd.DataFrame(data=dframe_temp,columns=[l])

    				elif(len(dframe_IQR) > 4):
    					dframe_temp = dframe_IQR[l].values
    					dframe_layer[l] = dframe_temp

    			#print(dframe_layer) 

   				   					
    			if(len(dframe_layer) > 4): 
    				layer_count = dframe_layer.count(axis=1)
    				dframe_layer['Layer_Avg'] = dframe_layer.mean(axis=1)
    				dframe_layer['Depths_Incl'] = layer_count
    				dframe_layer.insert(0,'Date',dat_new_m2n)
    				dframe_layer.insert(1,'Dataset',dtst)
    				dframe_layer.insert(2,'Lat',lat)
    				dframe_layer.insert(3,'Long',lon)
    				#print(dframe_layer)
    				dframe_layer.drop(dframe_layer[dframe_layer['Depths_Incl'] == 0].index, inplace=True)
    				print(dframe_layer)
    				if (len(dframe_layer) == 0):
    					continue
    				ofil = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/IQR/"+str(layer_name)+"/site_",str(j2),".csv"])
    				path2 = pathlib.Path(ofil)
    				path2.parent.mkdir(parents=True, exist_ok=True)
    				print(ofil)
    				dframe_layer.to_csv(ofil,na_rep="NaN",index=False)







###### For all other datasets ########
    
    elif (dataset_i == 'GTN-P' or dataset_i == 'Nordicana_Russia' or dataset_i =='NWT' or dataset_i == 'Yukon'):
 
    	if (dataset_i == 'GTN-P'):

    		wdir = "/mnt/data/users/herringtont/soil_temp/In-Situ/GTN-P/site_level/"
    		site_begin = 1
    		site_end =  69

    		date_fmt = "%Y-%m-%d %H:%M:%S"
    


    	elif (dataset_i == 'Nordicana_Russia'):

    		wdir = "/mnt/data/users/herringtont/soil_temp/In-Situ/Russia_Nordicana/"
    		site_begin = 300
    		site_end = 789

## Note that if site is between 300 and 757, it is from Russia_Hydromet
## Else if site is between 758 and 788, it is from Nordicana

    	elif (dataset_i == 'NWT'):
    		wdir = "/mnt/data/users/herringtont/soil_temp/In-Situ/NWT_All/"
    		site_begin = 789
    		site_end = 1228

    	elif (dataset_i == 'Yukon'):
    		wdir = "/mnt/data/users/herringtont/soil_temp/In-Situ/Yukon/site_level/"
    		site_begin = 1228
    		site_end = 1339

    	print(dataset_i)
    	#print(site_begin,site_end)

    	pathlist = os.listdir(wdir)
    	pathlist_sorted = natural_sort(pathlist)
 
    	for path in pathlist_sorted:
    		wfil = ''.join([wdir+path])
    		#print(wfil)
    		dframe = pd.read_csv(wfil)


    		if (dataset_i == 'GTN-P'):
    			dframe_insitu = dframe.replace(-999,np.nan)
    			sit_n = path.split("_")[1]
    			sit_n2 = sit_n.split('.')[0]
    			sit_num = int(sit_n2) 
    			#print(dframe)
    			lat = dframe_insitu['Lat']		
    			lon = dframe_insitu['Long']
    			#col1 = dframe[levels[0]]
    			#datetime_column = str_to_datetime(col1,date_fmt)
    			dates = dframe_insitu['Date/Depth'].values
    			dframe_depths = dframe_insitu.drop(['Date/Depth','Lat','Long'], axis=1)    			
    			col_val = np.array(dframe_depths.columns) #store column names
    			col_float = col_val.astype(float)
    			#print(col_val)
    			col_cm = col_float*100
    			#print(col_cm)
    			total_col = len(col_cm) #count number of columns

    			dtst = 'GTN-P'

    		elif (dataset_i == 'Nordicana_Russia'):

    			dframe_insitu = dframe
    			col_depths = dframe_insitu.columns
    			sit_num = int(dframe_insitu['Site Number'].iloc[0])
    			lat = dframe_insitu['Latitude'].iloc[0]
    			lon = dframe_insitu['Longitude'].iloc[0]
    			dates = dframe_insitu['Date'].values    
    			dframe_depths = dframe_insitu.drop(['Site Number','Latitude','Longitude','Date'], axis=1)
    			dframe_stemp = dframe_depths
    			col_depths = np.array(dframe_depths.columns)
    			col_float = col_depths.astype(float)

    			#print(col_depths)

    			total_col = len(col_depths)

    			if (300 <= sit_num < 758):
    				dtst = "RussiaHydromet"
    			elif(758 <= sit_num < 789):
    				dtst = "Nordicana"

    		elif (dataset_i == 'NWT'):
    			dframe_insitu = dframe
    			col_depths = dframe_insitu.columns
    			sit_n = path.split("_")[1]
    			sit_n2 = sit_n.split('.')[0]
    			sit_num = int(sit_n2)

    			#print(sit_num)

    			if (789 <= sit_num < 896):
    				dtst = "YZ_ibuttons"
    				lat = dframe_insitu['Latitude'].iloc[0]
    				lon = dframe_insitu['Longitude'].iloc[0]
    				dates = dframe_insitu['Date'].values
    				dframe_depths = dframe_insitu.drop(['Date','Latitude','Longitude'], axis=1)


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

    		elif (dataset_i == 'Yukon'):
    			dframe_insitu = dframe
    			col_depths = dframe_insitu.columns
    			sit_n = path.split("_")[1]
    			sit_n2 = sit_n.split('.')[0]
    			sit_num = int(sit_n2)

    			if (1228 <= sit_num <= 1339):
    				dtst = "Yukon"
    				lat = dframe_insitu['latitude'].iloc[0]
    				lon = dframe_insitu['longitude'].iloc[0]
    				dates = dframe_insitu['datetime'].values
    				dframe_depths = dframe_insitu.drop(['Site Number','Location ID','latitude','longitude','datetime'],axis=1)

    		dframe_stemp = dframe_depths

    		col_depths = np.array(dframe_depths.columns)

    		col_float = col_depths.astype(float)

    		total_col = len(col_depths)

#    		print(dframe)
#    		print(total_col)


    		dframe_outlier = "None"
    		dframe_zscore = "None"
    		dframe_IQR = "None"
    		for k in range(0,total_col):

    			dframe_stemp_1 = dframe_stemp.iloc[:,k]
    			if (dtst == 'GTN-P'):
    				col_nam = col_val[k]
    				col_flt = float(col_nam)*100
    				col_int = int(col_flt)

    			else:
    				col_flt = col_float[k]
    				col_int = int(col_flt)


    	#### Remove Outliers (Z-score Method) ####

    			stemp_new_m1 = []
    			dat_new_m1 = []
    			dframe_stemp_1 = dframe_stemp.iloc[:,k]    			    				    			
    			threshold = 3.5 #set outlier threshold to this many standard deviations away from mean
    			mean_value = dframe_stemp_1.mean(axis = 0, skipna = True)
    			stdev = dframe_stemp_1.std()

    			for l in range(0,len(dframe_stemp_1)):
    				stmp = dframe_stemp_1.iloc[l]
    				dat2 = dates[l]    		
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

    	#### Remove Outliers (IQR Method) ####


    			stemp_new_m2 = []
    			dat_new_m2 = []

		    	Q1 = dframe_stemp_1.quantile(0.25)
    			Q3 = dframe_stemp_1.quantile(0.75)
    			IQR = Q3-Q1
    			fence = IQR*1.5
    			for l in range(0,len(dframe_stemp_1)):
    				stmp = dframe_stemp_1.iloc[l]
    				dat2 = dates[l]
				    		   								
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



##### Create Reorganized Dataframe #####

    			if (len(dframe_outlier) == 4): 
    				dframe_outlier = pd.DataFrame(data=dframe_stemp_1, columns=[col_int])
    			elif (len (dframe_outlier) > 4):
    				dframe_temp = dframe_stemp_1
    				dframe_outlier[col_int] = dframe_temp

    			if (len(dframe_zscore) == 4): 
    				dframe_zscore = pd.DataFrame(data=stemp_new_m1n, columns=[col_int])
    			elif (len (dframe_zscore) > 4):
    				dframe_temp = stemp_new_m1n
    				dframe_zscore[col_int] = dframe_temp

    			if (len(dframe_IQR) == 4): 
    				dframe_IQR = pd.DataFrame(data=stemp_new_m2n, columns=[col_int])
    			elif (len (dframe_IQR) > 4):
    				dframe_temp = stemp_new_m2n
    				dframe_IQR[col_int] = dframe_temp


    		depth_names_o = np.array(dframe_outlier.columns)
    		depth_names_o = depth_names_o.astype(float)
    		depth_names_z = np.array(dframe_zscore.columns)
    		depth_names_z = depth_names_z.astype(float)
    		depth_names_I = np.array(dframe_IQR.columns)
    		depth_names_I = depth_names_I.astype(float)




#### Outliers Included ####


    		#### Separate into Layers ####


    		for k in layers:
    			dframe_layer = "None"
    			top_boundary = float(k[0])
    			bottom_boundary = float(k[1])
    			btm_bdy2 = bottom_boundary-0.1

    			layer_name = ''.join([str(int(top_boundary))+'_'+str(btm_bdy2)])
    			#print(layer_name)

    			depths_in_layer = depth_names_o[(depth_names_o >= top_boundary) & (depth_names_o < bottom_boundary)]
    			if (len(depths_in_layer) == 0):
    				continue
    			#print(depths_in_layer)

    			#### only include depths in layer ####

    			for l in depths_in_layer:
    				if(len(dframe_layer) == 4):
    					dframe_temp = dframe_outlier[l].values
    					dframe_layer = pd.DataFrame(data=dframe_temp,columns=[l])

    				elif(len(dframe_layer) > 4):
    					dframe_temp = dframe_outlier[l].values
    					dframe_layer[l] = dframe_temp

    			#print(dframe_layer) 

   				   					
    			if(len(dframe_layer) > 4): 
    				layer_count = dframe_layer.count(axis=1)
    				dframe_layer['Layer_Avg'] = dframe_layer.mean(axis=1)
    				dframe_layer['Depths_Incl'] = layer_count
    				dframe_layer.insert(0,'Date',dates)
    				dframe_layer.insert(1,'Dataset',dtst)
    				dframe_layer.insert(2,'Lat',lat)
    				dframe_layer.insert(3,'Long',lon)
    				#print(dframe_layer)
    				dframe_layer.drop(dframe_layer[dframe_layer['Depths_Incl'] == 0].index, inplace=True)
    				print(dframe_layer)
    				if (len(dframe_layer) == 0):
    					continue
    				ofil = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/outliers/"+str(layer_name)+"/site_",str(sit_num),".csv"])
    				path2 = pathlib.Path(ofil)
    				path2.parent.mkdir(parents=True, exist_ok=True)
    				print(ofil)
    				dframe_layer.to_csv(ofil,na_rep="NaN",index=False)
    			
#### Z-Score ####


    		#### Separate into Layers ####


    		for k in layers:
    			dframe_layer = "None"
    			top_boundary = float(k[0])
    			bottom_boundary = float(k[1])
    			btm_bdy2 = bottom_boundary-0.1

    			layer_name = ''.join([str(int(top_boundary))+'_'+str(btm_bdy2)])
    			#print(layer_name)

    			depths_in_layer = depth_names_z[(depth_names_z >= top_boundary) & (depth_names_z < bottom_boundary)]
    			if (len(depths_in_layer) == 0):
    				continue
    			#print(depths_in_layer)

    			#### only include depths in layer ####

    			for l in depths_in_layer:
    				if(len(dframe_layer) == 4):
    					dframe_temp = dframe_zscore[l].values
    					dframe_layer = pd.DataFrame(data=dframe_temp,columns=[l])

    				elif(len(dframe_layer) > 4):
    					dframe_temp = dframe_zscore[l].values
    					dframe_layer[l] = dframe_temp

    			#print(dframe_layer) 

   				   					
    			if(len(dframe_layer) > 4): 
    				layer_count = dframe_layer.count(axis=1)
    				dframe_layer['Layer_Avg'] = dframe_layer.mean(axis=1)
    				dframe_layer['Depths_Incl'] = layer_count
    				dframe_layer.insert(0,'Date',dat_new_m1n)
    				dframe_layer.insert(1,'Dataset',dtst)
    				dframe_layer.insert(2,'Lat',lat)
    				dframe_layer.insert(3,'Long',lon)
    				#print(dframe_layer)
    				dframe_layer.drop(dframe_layer[dframe_layer['Depths_Incl'] == 0].index, inplace=True)
    				print(dframe_layer)
    				if (len(dframe_layer) == 0):
    					continue
    				ofil = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/zscore/"+str(layer_name)+"/site_",str(sit_num),".csv"])
    				path2 = pathlib.Path(ofil)
    				path2.parent.mkdir(parents=True, exist_ok=True)
    				print(ofil)
    				dframe_layer.to_csv(ofil,na_rep="NaN",index=False)

#### IQR ####


    		#### Separate into Layers ####


    		for k in layers:
    			dframe_layer = "None"
    			top_boundary = float(k[0])
    			bottom_boundary = float(k[1])
    			btm_bdy2 = bottom_boundary-0.1

    			layer_name = ''.join([str(int(top_boundary))+'_'+str(btm_bdy2)])
    			#print(layer_name)

    			depths_in_layer = depth_names_I[(depth_names_I >= top_boundary) & (depth_names_I < bottom_boundary)]
    			if (len(depths_in_layer) == 0):
    				continue
    			#print(depths_in_layer)

    			#### only include depths in layer ####

    			for l in depths_in_layer:
    				if(len(dframe_layer) == 4):
    					dframe_temp = dframe_IQR[l].values
    					dframe_layer = pd.DataFrame(data=dframe_temp,columns=[l])

    				elif(len(dframe_layer) > 4):
    					dframe_temp = dframe_IQR[l].values
    					dframe_layer[l] = dframe_temp

    			#print(dframe_layer) 

   				   					
    			if(len(dframe_layer) > 4): 
    				layer_count = dframe_layer.count(axis=1)
    				dframe_layer['Layer_Avg'] = dframe_layer.mean(axis=1)
    				dframe_layer['Depths_Incl'] = layer_count
    				dframe_layer.insert(0,'Date',dat_new_m2n)
    				dframe_layer.insert(1,'Dataset',dtst)
    				dframe_layer.insert(2,'Lat',lat)
    				dframe_layer.insert(3,'Long',lon)
    				#print(dframe_layer)
    				dframe_layer.drop(dframe_layer[dframe_layer['Depths_Incl'] == 0].index, inplace=True)
    				print(dframe_layer)
    				if (len(dframe_layer) == 0):
    					continue
    				ofil = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/IQR/"+str(layer_name)+"/site_",str(sit_num),".csv"])
    				path2 = pathlib.Path(ofil)
    				path2.parent.mkdir(parents=True, exist_ok=True)
    				print(ofil)
    				dframe_layer.to_csv(ofil,na_rep="NaN",index=False)
