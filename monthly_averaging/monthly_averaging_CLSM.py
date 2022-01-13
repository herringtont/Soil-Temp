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
val_thresh = ['100','75','50','25','0'] #percent missing allowed (100, 75, 50, 25, 0)



######################## define functions ###################################
def str_to_datetime(column, date_fmt):

    date_list = []

    for dt_str in column:
        new_dt = datetime.datetime.strptime(dt_str, date_fmt)
        date_list.append(new_dt)
			

    return date_list


def load_pandas(file_name):
    #print("Loading in-situ file: ", file_name)
    dframe = pd.read_csv(file_name)

############################## grab sites and grid cells for each soil layer ##############################
geom_dir = "/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/CLSM/"
L1_fi = "".join([geom_dir,"geometry_L1_con_CLSM.csv"])
L2_fi = "".join([geom_dir,"geometry_L2_con_CLSM.csv"])
L3_fi = "".join([geom_dir,"geometry_L3_con_CLSM.csv"])
L4_fi = "".join([geom_dir,"geometry_L4_con_CLSM.csv"])
L5_fi = "".join([geom_dir,"geometry_L5_con_CLSM.csv"])
top30_fi = "".join([geom_dir,"geometry_top30_con_CLSM.csv"])
L7_fi = "".join([geom_dir,"geometry_L7_con_CLSM.csv"])

dframe_L1 = pd.read_csv(L1_fi)
dframe_L2 = pd.read_csv(L2_fi)
dframe_L3 = pd.read_csv(L3_fi)
dframe_L4 = pd.read_csv(L4_fi)
dframe_L5 = pd.read_csv(L5_fi)
dframe_top30 = pd.read_csv(top30_fi)
dframe_L7 = pd.read_csv(L7_fi)

L1_grid = np.array(dframe_L1["Grid Cell"].values)
L1_grid_uq = np.unique(L1_grid)
L1_id = "0_9.9"

L2_grid = np.array(dframe_L2["Grid Cell"].values)
L2_grid_uq = np.unique(L2_grid)
L2_id = "10_29.9"

L3_grid = np.array(dframe_L3["Grid Cell"].values)
L3_grid_uq = np.unique(L3_grid)
L3_id = "30_99.9"

L4_grid = np.array(dframe_L4["Grid Cell"].values)
L4_grid_uq = np.unique(L4_grid)
L4_id = "100_299.9"

L5_grid = np.array(dframe_L5["Grid Cell"].values)
L5_grid_uq = np.unique(L5_grid)
L5_id = "300_deeper"

top30_grid = np.array(dframe_top30["Grid Cell"].values)
top30_grid_uq = np.unique(top30_grid)
top30_id = "top_30cm"

L7_grid = np.array(dframe_L7["Grid Cell"].values)
L7_grid_uq = np.unique(L7_grid)
L7_id = "30_299.9"

########################### set variables for temperature data ##################################
t_dir = "/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/"
O_id = "outliers"
Z_id = "zscore"
I_id = "IQR"



############### loop through layers #############
layr = ['0_9.9','10_29.9','30_99.9','100_299.9','300_deeper','top_30cm','30_299.9']
for a in layr:
    #print("Layer:",a)
    if (a == '0_9.9'):
    	dframe_L = dframe_L1
    	L_id = L1_id
    	L_grid_uq = L1_grid_uq
    if (a == '10_29.9'):
    	dframe_L = dframe_L2
    	L_id = L2_id
    	L_grid_uq = L2_grid_uq
    if (a == '30_99.9'):
    	dframe_L = dframe_L3
    	L_id = L3_id
    	L_grid_uq = L3_grid_uq
    if (a == '100_299.9'):
    	dframe_L = dframe_L4
    	L_id = L4_id
    	L_grid_uq = L4_grid_uq
    if (a == '300_deeper'):
    	dframe_L = dframe_L5
    	L_id = L5_id
    	L_grid_uq = L5_grid_uq
    if (a == 'top_30cm'):
    	dframe_L = dframe_top30
    	L_id = top30_id
    	L_grid_uq = top30_grid_uq
    if (a == '30_299.9'):
    	dframe_L = dframe_L7
    	L_id = L7_id
    	L_grid_uq = L7_grid_uq	
################# loop through thresholds #########################
    for thrsh in val_thresh:
    	miss_thr = 100 - int(thrsh) #percent valid required in order to be included in monthly average
    	print("Threshold:",miss_thr)
########## group together stations within same grid cell ##########

    	#print(len(L_grid_uq))
    	#print(L_grid_uq)
    	for cell in L_grid_uq:
    		dframe_L_grp = dframe_L[dframe_L['Grid Cell'] == cell]
    		L_sit = dframe_L_grp['site']   
    		L_grd = dframe_L_grp['Grid Cell']
    		#print("the grid cell is:", cell)
    		#print(dframe_L_grp)
########### create master arrays for stations within a grid cell ########
    		cell_master = []
    		site_master = []
    		st_dt_master = []
    		ed_dt_master = []
      

######### loop through stations within a grid cell ###########
    		for i in range(0,len(dframe_L_grp)):
    			sitid = L_sit.iloc[i]
    			#print("the site is:",sitid)
    			gridid = L_grd.iloc[i]
    			O_fil = "".join([t_dir,O_id,"/",L_id,"/","site_",str(sitid),".csv"])
    			Z_fil = "".join([t_dir,Z_id,"/",L_id,"/","site_",str(sitid),".csv"])     
    			I_fil = "".join([t_dir,I_id,"/",L_id,"/","site_",str(sitid),".csv"])
    			#print(O_fil)
    			#print(Z_fil)
    			#print(I_fil)
    			if not os.path.exists(O_fil):
    				continue
    			if not os.path.exists(Z_fil):
    				continue
    			if not os.path.exists(I_fil):
    				continue
    			#print("the site is:",sitid)    
    			dframe_O = pd.read_csv(O_fil)
    			dframe_Z = pd.read_csv(Z_fil)
    			dframe_I = pd.read_csv(I_fil)
    			dtset = dframe_O.iloc[1,1]
    			#print(dtset)

	
    			if (dtset == 'GTN-P' or dtset == 'NWT_2017_009' or dtset == 'NWT_2019_004' or dtset == 'NWT_2019_017'):
    				dfmt = '%Y-%m-%d %H:%M:%S'

    			else:
    				dfmt = '%Y-%m-%d'
    			#print(dframe_O)


########### set datetime index ################

    			dframe_newO = dframe_O.set_index(pd.DatetimeIndex(dframe_O['Date']))
    			dframe_newZ = dframe_Z.set_index(pd.DatetimeIndex(dframe_Z['Date']))
    			dframe_newI = dframe_I.set_index(pd.DatetimeIndex(dframe_I['Date']))

    			latO = dframe_newO['Lat'].iloc[0]
    			lonO = dframe_newO['Long'].iloc[0]
    			#print(dframe_O)


########### Calculate Daily Average ############

    			dframe_newO = dframe_newO.groupby(pd.Grouper(freq='D')).mean()
    			dframe_newZ = dframe_newZ.groupby(pd.Grouper(freq='D')).mean()
    			dframe_newI = dframe_newI.groupby(pd.Grouper(freq='D')).mean()

    			dframe_newO = dframe_newO[['Layer_Avg','Depths_Incl']]
    			dframe_newO.insert(0,'Lat',latO)
    			dframe_newO.insert(1,'Long',lonO)
    			dframe_newZ = dframe_newZ[['Layer_Avg','Depths_Incl']]
    			dframe_newZ.insert(0,'Lat',latO)
    			dframe_newZ.insert(1,'Long',lonO)
    			dframe_newI = dframe_newI[['Layer_Avg','Depths_Incl']]
    			dframe_newI.insert(0,'Lat',latO)
    			dframe_newI.insert(1,'Long',lonO)
    			#print(dframe_newO)


    			ofilOday = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/",O_id,"/",L_id,"/","daily_average/CLSM/","site_",str(sitid),".csv"])
    			path_O = pathlib.Path(ofilOday)
    			path_O.parent.mkdir(parents=True,exist_ok=True)
    			ofilZday = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/",Z_id,"/",L_id,"/","daily_average/CLSM/","site_",str(sitid),".csv"])
    			path_Z = pathlib.Path(ofilZday)
    			path_Z.parent.mkdir(parents=True,exist_ok=True)
    			ofilIday = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/",I_id,"/",L_id,"/","daily_average/CLSM/","site_",str(sitid),".csv"])
    			path_I = pathlib.Path(ofilIday)
    			path_I.parent.mkdir(parents=True,exist_ok=True)

    			dframe_newO.to_csv(ofilOday)
    			dframe_newI.to_csv(ofilIday)
    			dframe_newZ.to_csv(ofilZday)	
####################### read the start and end year/month from file ###########################

######### grab datetime information ##############
    			date_O = dframe_newO.index
    			date_Z = dframe_newZ.index
    			date_I = dframe_newI.index

    			date_timeO = date_O
    			date_timeZ = date_Z
    			date_timeI = date_I
			 
    			st_dtO = date_O[0]
    			time_lenO = len(date_O)
    			ed_dtO = date_O[time_lenO-1]

    			st_dtZ = date_Z[0]
    			time_lenZ = len(date_Z)
    			ed_dtZ = date_Z[time_lenZ-1]
	
    			st_dtI = date_I[0]
    			time_lenI = len(date_I)
    			ed_dtI = date_I[time_lenI-1]	
   
    			srt_yrO = st_dtO.year
    			end_yrO = ed_dtO.year
    			beg_monO = st_dtO.month
    			beg_yrO = st_dtO.year
    			fin_monO = ed_dtO.month
    			fin_yrO = ed_dtO.year

    			srt_yrZ = st_dtO.year
    			end_yrZ = ed_dtO.year
    			beg_monZ = st_dtZ.month
    			beg_yrZ = st_dtZ.year
    			fin_monZ = ed_dtZ.month
    			fin_yrZ = ed_dtZ.year

    			srt_yrI = st_dtI.year
    			end_yrI = ed_dtI.year
    			beg_monI = st_dtI.month
    			beg_yrI = st_dtI.year
    			fin_monI = ed_dtI.month
    			fin_yrI = ed_dtI.year
    			dformat = '%Y_%m'


    			stemp_O = dframe_newO['Layer_Avg']
    			dt_stemp_O = stemp_O

    			stemp_Z = dframe_newZ['Layer_Avg']
    			dt_stemp_Z = stemp_Z

    			stemp_I = dframe_newI['Layer_Avg']
    			dt_stemp_I = stemp_I

#### determine the total possible number of datapoints per day, assuming no missing data ####
    			mon_fac = 1 #daily data, so one value per day


############################################## outliers ################################
############################################## create dictionary to store the total possible number of datapoints ###############################################################
    			tot_mon_entriesO = dict()

    			for x in range(srt_yrO, end_yrO+1):
    				if (x == srt_yrO):  #if first year, begin at starting month
    					for y in range(beg_monO,13):
    						month_d = datetime.datetime(x,y,1,0,0)
    						mon_key = month_d.strftime(dformat)
    		
    						dat_mon = month_d.month
    						dat_yr = month_d.year
    						if (dat_mon == 1 or dat_mon == 3 or dat_mon == 5 or dat_mon == 7 or dat_mon == 8 or dat_mon == 10 or dat_mon == 12): #if Jan, Mar, May, Jul, Aug, Oct, Dec then month has 31 days		
    							mon_day = 31
    						elif (dat_mon == 2):
    							if (isleap(dat_yr)): #if Feb, then check if leap year
    								mon_day = 29  #if leap year, then Feb has 29 days
    							else:
        							mon_day = 28   #else Feb has 28 days
    						elif (dat_mon == 4 or dat_mon == 6 or dat_mon == 9 or dat_mon == 11): #if Apr, Jun, Sept, or Nov, then month has 30 days
    							mon_day = 30
    						if(mon_fac == 0):							
    							tot_mon_entriesO[mon_key] = 1  #these sites have monthly data, so each month only has 1 datapoint maximum
    						else:
    							tot_mon_entriesO[mon_key] = mon_day*mon_fac   #maximum # of datapoints = # of days in month * mon_fac (the number of datapoints per day)
    	
    				elif (x == end_yrO):  #if final year, end at final month in dataset
    					for y in range(1,fin_monO+1):
    						month_d = datetime.datetime(x,y,1,0,0)
    						mon_key = month_d.strftime(dformat)
    						dat_mon = month_d.month
    						dat_yr = month_d.year
    						if (dat_mon == 1 or dat_mon == 3 or dat_mon == 5 or dat_mon == 7 or dat_mon == 8 or dat_mon == 10 or dat_mon == 12): #if Jan, Mar, May, Jul, Aug, Oct, Dec then month has 31 day		
    							mon_day = 31
    						elif (dat_mon == 2):
    							if (isleap(dat_yr)):
    								mon_day = 29
    							else:
        							mon_day = 28
    						elif (dat_mon == 4 or dat_mon == 6 or dat_mon == 9 or dat_mon == 11): #if Apr, Jun, Sept, or Nov, then month has 30 days
    							mon_day = 30
    						if(sitid == 35 or sitid == 36 or sitid == 52):							
    							tot_mon_entriesO[mon_key] = 1
    						else:
    							tot_mon_entriesO[mon_key] = mon_day*mon_fac
    				else:
    					for y in range(1,13): #else iterate over all 12 months
    						month_d = datetime.datetime(x,y,1,0,0)
    						mon_key = month_d.strftime(dformat)
    						dat_mon = month_d.month
    						dat_yr = month_d.year
    						if (dat_mon == 1 or dat_mon == 3 or dat_mon == 5 or dat_mon == 7 or dat_mon == 8 or dat_mon == 10 or dat_mon == 12):		
    							mon_day = 31
    						elif (dat_mon == 2):
    							if (isleap(dat_yr)):
    								mon_day = 29
    							else:
        							mon_day = 28
    						elif (dat_mon == 4 or dat_mon == 6 or dat_mon == 9 or dat_mon == 11):
    							mon_day = 30
    						if(sitid == 35 or sitid == 36 or sitid == 52):							
    							tot_mon_entriesO[mon_key] = 1
    						else:
    							tot_mon_entriesO[mon_key] = mon_day*mon_fac



############################# create a dictionary which keeps track of how many data entries you have per month #######################
    			monthly_entriesO = dict()

# building dictionary for datetimes in site csv file

    			for x in range(srt_yrO, end_yrO+1):
    				if (x == srt_yrO):  #if first year, begin at starting month
    					for y in range(beg_monO,13):
    						month_dt = datetime.datetime(x,y,1,0,0)
    						month_key = month_dt.strftime(dformat)
    						monthly_entriesO[month_key] = 0
    				elif (x == end_yrO):  #if final year, end at final month in dataset
    					for y in range(1,fin_monO+1):
    						month_dt = datetime.datetime(x,y,1,0,0)
    						month_key = month_dt.strftime(dformat)
    						monthly_entriesO[month_key] = 0
    				else:
    					for y in range(1,13): #else iterate over all 12 months
    						month_dt = datetime.datetime(x,y,1,0,0)
    						month_key = month_dt.strftime(dformat)
    						monthly_entriesO[month_key] = 0  			

    
################################## add in a second check where we check if the soil temperature data is valid for that particular day ####################################
    			dt_stemp_O2 = dt_stemp_O.notna()
    			stemp_newO = []
    			dat_newO = []

###Now you have a dictionary which keeps track of how many data entries you have per month. 
###Then you can iterate over all of your data rows, and just increment the monthly_entries[month_key] counter by one for each entry.
###In order to generate the month_key from a datetime object, you'd just run:

    			dt_idxO = 0 #index of date and soil temperature value

    			for dt in date_timeO:
    				dt2O = date_timeO[dt_idxO]
    				stemp_O2 = dt_stemp_O.iloc[dt_idxO]
    				stemp_O3 = dt_stemp_O2.iloc[dt_idxO]
    				#print(type(dt))
    				key = dt.strftime(dformat)
    				if (stemp_O3 == True):  #if soil temp is not missing, add it to dat_new (else do not)
    					stemp_newO.append(stemp_O2)
    					dat_newO.append(dt2O)
    					#print(dat_new)		
    				dt_idxO += 1    

    			for dt in dat_newO:
    				#print(type(dt))
    				keyO = dt.strftime(dformat) 		
    				monthly_entriesO[keyO] += 1
    

  


############################################## zscore ################################
############################################## create dictionary to store the total possible number of datapoints ###############################################################
    			tot_mon_entriesZ = dict()

    			for x in range(srt_yrZ, end_yrZ+1):
    				if (x == srt_yrZ):  #if first year, begin at starting month
    					for y in range(beg_monZ,13):
    						month_d = datetime.datetime(x,y,1,0,0)
    						mon_key = month_d.strftime(dformat)
    		
    						dat_mon = month_d.month
    						dat_yr = month_d.year
    						if (dat_mon == 1 or dat_mon == 3 or dat_mon == 5 or dat_mon == 7 or dat_mon == 8 or dat_mon == 10 or dat_mon == 12): #if Jan, Mar, May, Jul, Aug, Oct, Dec then month has 31 days		
    							mon_day = 31
    						elif (dat_mon == 2):
    							if (isleap(dat_yr)): #if Feb, then check if leap year
    								mon_day = 29  #if leap year, then Feb has 29 days
    							else:
        							mon_day = 28   #else Feb has 28 days
    						elif (dat_mon == 4 or dat_mon == 6 or dat_mon == 9 or dat_mon == 11): #if Apr, Jun, Sept, or Nov, then month has 30 days
    							mon_day = 30
    						if(sitid == 35 or sitid == 36 or sitid == 52):							
    							tot_mon_entriesZ[mon_key] = 1  #these sites have monthly data, so each month only has 1 datapoint maximum
    						else:
    							tot_mon_entriesZ[mon_key] = mon_day*mon_fac   #maximum # of datapoints = # of days in month * mon_fac (the number of datapoints per day)
    	
    				elif (x == end_yrZ):  #if final year, end at final month in dataset
    					for y in range(1,fin_monZ+1):
    						month_d = datetime.datetime(x,y,1,0,0)
    						mon_key = month_d.strftime(dformat)
    						dat_mon = month_d.month
    						dat_yr = month_d.year
    						if (dat_mon == 1 or dat_mon == 3 or dat_mon == 5 or dat_mon == 7 or dat_mon == 8 or dat_mon == 10 or dat_mon == 12): #if Jan, Mar, May, Jul, Aug, Oct, Dec then month has 31 day		
    							mon_day = 31
    						elif (dat_mon == 2):
    							if (isleap(dat_yr)):
    								mon_day = 29
    							else:
        							mon_day = 28
    						elif (dat_mon == 4 or dat_mon == 6 or dat_mon == 9 or dat_mon == 11): #if Apr, Jun, Sept, or Nov, then month has 30 days
    							mon_day = 30
    						if(sitid == 35 or sitid == 36 or sitid == 52):							
    							tot_mon_entriesZ[mon_key] = 1
    						else:
    							tot_mon_entriesZ[mon_key] = mon_day*mon_fac
    				else:
    					for y in range(1,13): #else iterate over all 12 months
    						month_d = datetime.datetime(x,y,1,0,0)
    						mon_key = month_d.strftime(dformat)
    						dat_mon = month_d.month
    						dat_yr = month_d.year
    						if (dat_mon == 1 or dat_mon == 3 or dat_mon == 5 or dat_mon == 7 or dat_mon == 8 or dat_mon == 10 or dat_mon == 12):		
    							mon_day = 31
    						elif (dat_mon == 2):
    							if (isleap(dat_yr)):
    								mon_day = 29
    							else:
        							mon_day = 28
    						elif (dat_mon == 4 or dat_mon == 6 or dat_mon == 9 or dat_mon == 11):
    							mon_day = 30
    						if(sitid == 35 or sitid == 36 or sitid == 52):							
    							tot_mon_entriesZ[mon_key] = 1
    						else:
    							tot_mon_entriesZ[mon_key] = mon_day*mon_fac



############################# create a dictionary which keeps track of how many data entries you have per month #######################
    			monthly_entriesZ = dict()

# building dictionary for datetimes in site csv file

    			for x in range(srt_yrZ, end_yrZ+1):
    				if (x == srt_yrZ):  #if first year, begin at starting month
    					for y in range(beg_monZ,13):
    						month_dt = datetime.datetime(x,y,1,0,0)
    						month_key = month_dt.strftime(dformat)
    						monthly_entriesZ[month_key] = 0
    				elif (x == end_yrZ):  #if final year, end at final month in dataset
    					for y in range(1,fin_monZ+1):
    						month_dt = datetime.datetime(x,y,1,0,0)
    						month_key = month_dt.strftime(dformat)
    						monthly_entriesZ[month_key] = 0
    				else:
    					for y in range(1,13): #else iterate over all 12 months
    						month_dt = datetime.datetime(x,y,1,0,0)
    						month_key = month_dt.strftime(dformat)
    						monthly_entriesZ[month_key] = 0  			

    
################################## add in a second check where we check if the soil temperature data is valid for that particular day ####################################
    			dt_stemp_Z2 = dt_stemp_Z.notna()
    			stemp_newZ = []
    			dat_newZ = []

###Now you have a dictionary which keeps track of how many data entries you have per month. 
###Then you can iterate over all of your data rows, and just increment the monthly_entries[month_key] counter by one for each entry.
###In order to generate the month_key from a datetime object, you'd just run:

    			dt_idxZ = 0 #index of date and soil temperature value

    			for dt in date_timeZ:
    				dt2Z = date_timeZ[dt_idxZ]
    				stemp_Z2 = dt_stemp_Z.iloc[dt_idxZ]
    				stemp_Z3 = dt_stemp_Z2.iloc[dt_idxZ]
    				#print(type(dt))
    				key = dt.strftime(dformat)
    				if (stemp_Z3 == True):  #if soil temp is not missing, add it to dat_new (else do not)
    					stemp_newZ.append(stemp_Z2)
    					dat_newZ.append(dt2Z)
    					#print(dat_new)		
    				dt_idxZ += 1    

    			for dt in dat_newZ:
    				#print(type(dt))
    				keyZ = dt.strftime(dformat) 		
    				monthly_entriesZ[keyZ] += 1


############################################## IQR ################################
############################################## create dictionary to store the total possible number of datapoints ###############################################################
    			tot_mon_entriesI = dict()

    			for x in range(srt_yrI, end_yrI+1):
    				if (x == srt_yrI):  #if first year, begin at starting month
    					for y in range(beg_monI,13):
    						month_d = datetime.datetime(x,y,1,0,0)
    						mon_key = month_d.strftime(dformat)
    		
    						dat_mon = month_d.month
    						dat_yr = month_d.year
    						if (dat_mon == 1 or dat_mon == 3 or dat_mon == 5 or dat_mon == 7 or dat_mon == 8 or dat_mon == 10 or dat_mon == 12): #if Jan, Mar, May, Jul, Aug, Oct, Dec then month has 31 days		
    							mon_day = 31
    						elif (dat_mon == 2):
    							if (isleap(dat_yr)): #if Feb, then check if leap year
    								mon_day = 29  #if leap year, then Feb has 29 days
    							else:
        							mon_day = 28   #else Feb has 28 days
    						elif (dat_mon == 4 or dat_mon == 6 or dat_mon == 9 or dat_mon == 11): #if Apr, Jun, Sept, or Nov, then month has 30 days
    							mon_day = 30
    						if(sitid == 35 or sitid == 36 or sitid == 52):							
    							tot_mon_entriesI[mon_key] = 1  #these sites have monthly data, so each month only has 1 datapoint maximum
    						else:
    							tot_mon_entriesI[mon_key] = mon_day*mon_fac   #maximum # of datapoints = # of days in month * mon_fac (the number of datapoints per day)
    	
    				elif (x == end_yrI):  #if final year, end at final month in dataset
    					for y in range(1,fin_monI+1):
    						month_d = datetime.datetime(x,y,1,0,0)
    						mon_key = month_d.strftime(dformat)
    						dat_mon = month_d.month
    						dat_yr = month_d.year
    						if (dat_mon == 1 or dat_mon == 3 or dat_mon == 5 or dat_mon == 7 or dat_mon == 8 or dat_mon == 10 or dat_mon == 12): #if Jan, Mar, May, Jul, Aug, Oct, Dec then month has 31 day		
    							mon_day = 31
    						elif (dat_mon == 2):
    							if (isleap(dat_yr)):
    								mon_day = 29
    							else:
        							mon_day = 28
    						elif (dat_mon == 4 or dat_mon == 6 or dat_mon == 9 or dat_mon == 11): #if Apr, Jun, Sept, or Nov, then month has 30 days
    							mon_day = 30
    						if(sitid == 35 or sitid == 36 or sitid == 52):							
    							tot_mon_entriesI[mon_key] = 1
    						else:
    							tot_mon_entriesI[mon_key] = mon_day*mon_fac
    				else:
    					for y in range(1,13): #else iterate over all 12 months
    						month_d = datetime.datetime(x,y,1,0,0)
    						mon_key = month_d.strftime(dformat)
    						dat_mon = month_d.month
    						dat_yr = month_d.year
    						if (dat_mon == 1 or dat_mon == 3 or dat_mon == 5 or dat_mon == 7 or dat_mon == 8 or dat_mon == 10 or dat_mon == 12):		
    							mon_day = 31
    						elif (dat_mon == 2):
    							if (isleap(dat_yr)):
    								mon_day = 29
    							else:
        							mon_day = 28
    						elif (dat_mon == 4 or dat_mon == 6 or dat_mon == 9 or dat_mon == 11):
    							mon_day = 30
    						if(sitid == 35 or sitid == 36 or sitid == 52):							
    							tot_mon_entriesI[mon_key] = 1
    						else:
    							tot_mon_entriesI[mon_key] = mon_day*mon_fac



############################# create a dictionary which keeps track of how many data entries you have per month #######################
    			monthly_entriesI = dict()

# building dictionary for datetimes in site csv file

    			for x in range(srt_yrI, end_yrI+1):
    				if (x == srt_yrI):  #if first year, begin at starting month
    					for y in range(beg_monI,13):
    						month_dt = datetime.datetime(x,y,1,0,0)
    						month_key = month_dt.strftime(dformat)
    						monthly_entriesI[month_key] = 0
    				elif (x == end_yrI):  #if final year, end at final month in dataset
    					for y in range(1,fin_monI+1):
    						month_dt = datetime.datetime(x,y,1,0,0)
    						month_key = month_dt.strftime(dformat)
    						monthly_entriesI[month_key] = 0
    				else:
    					for y in range(1,13): #else iterate over all 12 months
    						month_dt = datetime.datetime(x,y,1,0,0)
    						month_key = month_dt.strftime(dformat)
    						monthly_entriesI[month_key] = 0  			

    
################################## add in a second check where we check if the soil temperature data is valid for that particular day ####################################
    			dt_stemp_I2 = dt_stemp_I.notna()
    			stemp_newI = []
    			dat_newI = []

###Now you have a dictionary which keeps track of how many data entries you have per month. 
###Then you can iterate over all of your data rows, and just increment the monthly_entries[month_key] counter by one for each entry.
###In order to generate the month_key from a datetime object, you'd just run:

    			dt_idxI = 0 #index of date and soil temperature value

    			for dt in date_timeI:
    				dt2I = date_timeI[dt_idxI]
    				stemp_I2 = dt_stemp_I.iloc[dt_idxI]
    				stemp_I3 = dt_stemp_I2.iloc[dt_idxI]
    				#print(type(dt))
    				key = dt.strftime(dformat)
    				if (stemp_I3 == True):  #if soil temp is not missing, add it to dat_new (else do not)
    					stemp_newI.append(stemp_I2)
    					dat_newI.append(dt2I)
    					#print(dat_new)		
    				dt_idxI += 1    

    			for dt in dat_newI:
    				#print(type(dt))
    				keyI = dt.strftime(dformat) 		
    				monthly_entriesI[keyI] += 1



########################## calculate monthly average soil temperatures ##############################
    			dframe_monO = dframe_newO.resample('M', convention ='start').mean()
    			dframe_monO['Date'] = dframe_monO.index.values.astype('datetime64[M]') #change datetime index 

    			dframe_monZ = dframe_newZ.resample('M', convention ='start').mean()
    			dframe_monZ['Date'] = dframe_monZ.index.values.astype('datetime64[M]') #change datetime index

    			dframe_monI = dframe_newI.resample('M', convention ='start').mean()
    			dframe_monI['Date'] = dframe_monI.index.values.astype('datetime64[M]') #change datetime index
	
######################### reindex monthly averages ############################
    			dframe_monO2 = dframe_monO.set_index(pd.DatetimeIndex(dframe_monO['Date'])) #reindex with new datetime index
    			dframe_monZ2 = dframe_monZ.set_index(pd.DatetimeIndex(dframe_monZ['Date'])) #reindex with new datetime index	
    			dframe_monI2 = dframe_monI.set_index(pd.DatetimeIndex(dframe_monI['Date'])) #reindex with new datetime index


######## split dictionaries into values and keys, compare values #######

    			mon_valO = list(monthly_entriesO.values())
    			mon_kyO = list(monthly_entriesO.keys())

    			tot_valO = list(tot_mon_entriesO.values())
    			tot_kyO = list(tot_mon_entriesO.keys())

    			dat_new2O = []
    			dat_totO = []
    			val_newO = []
    			val_totO = []

    			for i in range(0,len(mon_valO)):
    				str_kyO = mon_kyO[i]
    				str_totO = tot_kyO[i]
    				new_dtO = datetime.datetime.strptime(str_kyO,'%Y_%m')
    				new_vlO = mon_valO[i]
    				tot_vlO = tot_valO[i]
    				dt_totO = datetime.datetime.strptime(str_totO,'%Y_%m')
    				dat_new2O.append(new_dtO)
    				dat_totO.append(dt_totO)
    				val_newO.append(new_vlO)
    				val_totO.append(tot_vlO) 

######## calculate percent valid values #########
    			val_new2O = np.array(val_newO)
    			val_tot2O = np.array(val_totO)

    			val_pctO = (val_new2O/val_tot2O)*100

	
    			val_pct2O = []

###### remove months with no data at all #######  
    			dframe_monO2 = dframe_monO2[dframe_monO2['Layer_Avg'].notna()]


###### calculate anomalies from monthly climatologies #######   			    			

    			climatologyO = dict()
    			clim_averagesO = dict()
    			num_rowsO = len(dframe_monO2)
   			
    			dat_rowlistO = dframe_monO2.index.tolist()
    			stemp_anom_masterO = []
    			for month in range(1,13):
    				month_key = f"{month:02}"
    				climatologyO[month_key] = list()
    			

    			for i in range(0, num_rowsO):
    			####add month data to list based on key
    				#row2 = i
    				#print("this is row:",row2)
    				dat_rowO = dat_rowlistO[i]
    				stemp_rowO = dframe_monO2['Layer_Avg'].iloc[i]
  				
    			# assuming you have current_date_string and current_soil_temp
    				#dt_row = datetime.datetime.strptime(dat_row, "%Y-%m-%d")
    				dt_rowO = dat_rowO
    				month_key = dt_rowO.strftime("%m")
    				climatologyO[month_key].append(stemp_rowO)
     			

    			climatology_keysO = list(climatologyO.keys())
    			climatology_keysO2 = np.array(climatology_keysO).flatten()  			   			

    			for key in climatologyO:
    			# take averages and write to averages dict
    				current_totalO = 0
    				len_current_listO = 0
    				current_listO = climatologyO[key]
    				for temp in current_listO:
    					if (temp == np.nan):
    						current_totalO = current_totalO + 0
    						len_current_listO = len_current_listO + 0
    					else:
    						current_totalO = current_totalO + temp
    						len_current_listO = len_current_listO + 1					
    				if (len_current_listO == 0):
    					averageO = np.nan
    				else:
    					averageO = current_totalO/len_current_listO
    				clim_averagesO[key] = averageO
    			clim_avgO = list(clim_averagesO.values())
    			#print(clim_avg)
    			
    			for j in range(0, len(dframe_monO2)):
    				stemp_rwO = dframe_monO2['Layer_Avg'].iloc[j]
    				dat_rwO = dat_rowlistO[j]
    				dat_rw_monO = dat_rwO.month
    				dat_rw_monOs = f"{dat_rw_monO:02}"
    				#print(dat_rw_mon)
    				stemp_anomO = stemp_rwO - clim_averagesO[dat_rw_monOs]
    				stemp_anom_masterO.append(stemp_anomO)

    			stemp_anom_masterOn = np.array(stemp_anom_masterO).flatten()
    			#print(stemp_anom_masternI)

####### add anomalies to dframe_mon #################
    			dframe_monO2['Layer_Anom'] = stemp_anom_masterOn

###### extract percent valid values for months with data #####
    			for i in range(0,len(mon_valO)):
    				val_ptO = val_pctO[i]
    				if(val_ptO > 0):
    					val_pct2O.append(val_ptO)
    
###### append percent valid data column to monthly dataframe ######    
    			dframe_monO2['Percent_Valid'] = val_pct2O
    			dframe_monO2['Date'] = dframe_monO2.index
    			dframe_monO2 = dframe_monO2[['Date','Layer_Avg','Layer_Anom','Percent_Valid']]

##### remove months where percent valid < miss_threshold ########
    			dframe_monO2 = dframe_monO2.loc[dframe_monO2['Percent_Valid'] >= miss_thr]

########## zscore ############################
    			mon_valZ = list(monthly_entriesO.values())
    			mon_kyZ = list(monthly_entriesO.keys())

    			tot_valZ = list(tot_mon_entriesO.values())
    			tot_kyZ = list(tot_mon_entriesO.keys())
    	
    			dat_new2Z = []
    			dat_totZ = []
	
    			val_newZ = []
    			val_totZ = []

######## split dictionaries into values and keys, compare values #######
    			mon_valZ = list(monthly_entriesZ.values())
    			mon_kyZ = list(monthly_entriesZ.keys())

    			tot_valZ = list(tot_mon_entriesZ.values())
    			tot_kyZ = list(tot_mon_entriesZ.keys())

    			dat_new2Z = []
    			dat_totZ = []
    			val_newZ = []
    			val_totZ = []

    			for d in range(0,len(mon_valZ)):
    				str_kyZ = mon_kyZ[d]
    				str_totZ = tot_kyZ[d]
    				new_dtZ = datetime.datetime.strptime(str_kyZ,'%Y_%m')
    				new_vlZ = mon_valZ[d]
    				tot_vlZ = tot_valZ[d]
    				dt_totZ = datetime.datetime.strptime(str_totZ,'%Y_%m')
    				dat_new2Z.append(new_dtZ)
    				dat_totZ.append(dt_totZ)
    				val_newZ.append(new_vlZ)
    				val_totZ.append(tot_vlZ) 

######## calculate percent valid values #########
    			val_new2Z = np.array(val_newZ)
    			val_tot2Z = np.array(val_totZ)

    			val_pctZ = (val_new2Z/val_tot2Z)*100

    			val_pct2Z =[]

###### remove months with no data at all #######  
    			dframe_monZ2 = dframe_monZ2[dframe_monZ2['Layer_Avg'].notna()]

###### calculate anomalies from monthly climatologies #######   			    			

    			climatologyZ = dict()
    			clim_averagesZ = dict()
    			num_rowsZ = len(dframe_monZ2)
   			
    			dat_rowlistZ = dframe_monZ2.index.tolist()
    			stemp_anom_masterZ = []
    			for month in range(1,13):
    				month_key = f"{month:02}"
    				climatologyZ[month_key] = list()
    			

    			for i in range(0, num_rowsZ):
    			####add month data to list based on key
    				#row2 = i
    				#print("this is row:",row2)
    				dat_rowZ = dat_rowlistZ[i]
    				stemp_rowZ = dframe_monZ2['Layer_Avg'].iloc[i]
  				
    			# assuming you have current_date_string and current_soil_temp
    				#dt_row = datetime.datetime.strptime(dat_row, "%Y-%m-%d")
    				dt_rowZ = dat_rowZ
    				month_key = dt_rowZ.strftime("%m")
    				climatologyZ[month_key].append(stemp_rowZ)
     			

    			climatology_keysZ = list(climatologyZ.keys())
    			climatology_keysZ2 = np.array(climatology_keysZ).flatten()  			   			

    			for key in climatologyZ:
    			# take averages and write to averages dict
    				current_totalZ = 0
    				len_current_listZ = 0
    				current_listZ = climatologyZ[key]
    				for temp in current_listZ:
    					if (temp == np.nan):
    						current_totalZ = current_totalZ + 0
    						len_current_listZ = len_current_listZ + 0
    					else:
    						current_totalZ = current_totalZ + temp
    						len_current_listZ = len_current_listZ + 1					
    				if (len_current_listZ == 0):
    					averageZ = np.nan
    				else:
    					averageZ = current_totalZ/len_current_listZ
    				clim_averagesZ[key] = averageZ
    			clim_avgZ = list(clim_averagesZ.values())
    			#print(clim_avg)
    			
    			for j in range(0, len(dframe_monZ2)):
    				stemp_rwZ = dframe_monZ2['Layer_Avg'].iloc[j]
    				dat_rwZ = dat_rowlistZ[j]
    				dat_rw_monZ = dat_rwZ.month
    				dat_rw_monZs = f"{dat_rw_monZ:02}"
    				#print(dat_rw_mon)
    				stemp_anomZ = stemp_rwZ - clim_averagesZ[dat_rw_monZs]
    				stemp_anom_masterZ.append(stemp_anomZ)

    			stemp_anom_masterZn = np.array(stemp_anom_masterZ).flatten()
    			#print(stemp_anom_masternI)

####### add anomalies to dframe_mon #################
    			dframe_monZ2['Layer_Anom'] = stemp_anom_masterZn

###### extract percent valid values for months with data #####
    			for e in range(0,len(mon_valZ)):
    				val_ptZ = val_pctZ[e]
    				if(val_ptZ > 0):
    					val_pct2Z.append(val_ptZ)
    
###### append percent valid data column to monthly dataframe ######    
    			dframe_monZ2['Percent_Valid'] = val_pct2Z
    			dframe_monZ2['Date'] = dframe_monZ2.index
    			dframe_monZ2 = dframe_monZ2[['Date','Layer_Avg','Layer_Anom','Percent_Valid']]

##### remove months where percent valid < miss_threshold ########
    			dframe_monZ2 = dframe_monZ2.loc[dframe_monZ2['Percent_Valid'] >= miss_thr]



################### IQR ###################
    			mon_valI = list(monthly_entriesI.values())
    			mon_kyI = list(monthly_entriesI.keys())

    			tot_valI = list(tot_mon_entriesI.values())
    			tot_kyI = list(tot_mon_entriesI.keys())

    			dat_new2I = []
    			dat_totI = []
	
    			val_newI = []
    			val_totI = []

######## split dictionaries into values and keys, compare values #######
    			mon_valI = list(monthly_entriesI.values())
    			mon_kyI = list(monthly_entriesI.keys())

    			tot_valI = list(tot_mon_entriesI.values())
    			tot_kyI = list(tot_mon_entriesI.keys())

    			dat_new2I = []
    			dat_totI = []
    			val_newI = []
    			val_totI = []

    			for f in range(0,len(mon_valI)):
    				str_kyI = mon_kyI[f]
    				str_totI = tot_kyI[f]
    				new_dtI = datetime.datetime.strptime(str_kyI,'%Y_%m')
    				new_vlI = mon_valI[f]
    				tot_vlI = tot_valI[f]
    				dt_totI = datetime.datetime.strptime(str_totI,'%Y_%m')
    				dat_new2I.append(new_dtI)
    				dat_totI.append(dt_totI)
    				val_newI.append(new_vlI)
    				val_totI.append(tot_vlI) 

######## calculate percent valid values #########
    			val_new2I = np.array(val_newI)
    			val_tot2I = np.array(val_totI)

    			val_pctI = (val_new2I/val_tot2I)*100

    			val_pct2I =[]

###### remove months with no data at all #######  
    			dframe_monI2 = dframe_monI2[dframe_monI2['Layer_Avg'].notna()]

###### calculate anomalies from monthly climatologies #######
    			#print(dframe_monI2)    			    			

    			climatologyI = dict()
    			clim_averagesI = dict()
    			num_rowsI = len(dframe_monI2)
    			#print(dframe_monI2)    			
    			dat_rowlistI = dframe_monI2.index.tolist()
    			stemp_anom_masterI = []
    			for month in range(1,13):
    				month_key = f"{month:02}"
    				climatologyI[month_key] = list()
    				#print(climatology)
    			

    			for i in range(0, num_rowsI):
    			####add month data to list based on key
    				#row2 = i
    				#print("this is row:",row2)
    				dat_rowI = dat_rowlistI[i]
    				stemp_rowI = dframe_monI2['Layer_Avg'].iloc[i]
  				
    			# assuming you have current_date_string and current_soil_temp
    				#dt_row = datetime.datetime.strptime(dat_row, "%Y-%m-%d")
    				dt_rowI = dat_rowI
    				month_key = dt_rowI.strftime("%m")
    				climatologyI[month_key].append(stemp_rowI)
    			#print(climatology)
     			

    			climatology_keysI = list(climatologyI.keys())
    			climatology_keysI2 = np.array(climatology_keysI).flatten() 
    			#print(climatology_keys2)  			   			

    			for key in climatologyI:
    			# take averages and write to averages dict
    				current_totalI = 0
    				len_current_listI = 0
    				current_listI = climatologyI[key]
    				for temp in current_listI:
    					if (temp == np.nan):
    						current_totalI = current_totalI + 0
    						len_current_listI = len_current_listI + 0
    					else:
    						current_totalI = current_totalI + temp
    						len_current_listI = len_current_listI + 1					
    				if (len_current_listI == 0):
    					averageI = np.nan
    				else:
    					averageI = current_totalI/len_current_listI
    				clim_averagesI[key] = averageI
    			clim_avgI = list(clim_averagesI.values())
    			#print(clim_avg)
    			
    			for j in range(0, len(dframe_monI2)):
    				stemp_rwI = dframe_monI2['Layer_Avg'].iloc[j]
    				dat_rwI = dat_rowlistI[j]
    				dat_rw_monI = dat_rwI.month
    				dat_rw_monIs = f"{dat_rw_monI:02}"
    				#print(dat_rw_mon)
    				stemp_anomI = stemp_rwI - clim_averagesI[dat_rw_monIs]
    				stemp_anom_masterI.append(stemp_anomI)

    			stemp_anom_masterIn = np.array(stemp_anom_masterI).flatten()
    			#print(stemp_anom_masternI)

####### add anomalies to dframe_mon #################
    			dframe_monI2['Layer_Anom'] = stemp_anom_masterIn    			    
    							

###### extract percent valid values for months with data #####
    			for g in range(0,len(mon_valI)):
    				val_ptI = val_pctI[g]
    				if(val_ptI > 0):
    					val_pct2I.append(val_ptI)
    
###### append percent valid data column to monthly dataframe ######    
    			dframe_monI2['Percent_Valid'] = val_pct2I
    			dframe_monI2['Date'] = dframe_monI2.index
    			dframe_monI2 = dframe_monI2[['Date','Layer_Avg','Layer_Anom','Percent_Valid']]

##### remove months where percent valid < miss_threshold ########
    			dframe_monI2 = dframe_monI2.loc[dframe_monI2['Percent_Valid'] >= miss_thr]
    			#print(dframe_monO2)

    			if (miss_thr == 100):
    				thr_fld = "thr_100"
    			elif (miss_thr == 75):
    				thr_fld = "thr_75"
    			elif (miss_thr == 50):
    				thr_fld = "thr_50"
    			elif (miss_thr == 25):
    				thr_fld = "thr_25"
    			elif (miss_thr == 0):
    				thr_fld = "thr_0"
######################### create temporary monthly file #######################
    			ofilO = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/",O_id,"/",L_id,"/","monthly_average/",thr_fld,"/CLSM/","site_",str(sitid),".csv"])
    			path_O = pathlib.Path(ofilO)
    			path_O.parent.mkdir(parents=True,exist_ok=True)
    			ofilZ = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/",Z_id,"/",L_id,"/","monthly_average/",thr_fld,"/CLSM/","site_",str(sitid),".csv"])
    			path_Z = pathlib.Path(ofilZ)
    			path_Z.parent.mkdir(parents=True,exist_ok=True)
    			ofilI = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/",I_id,"/",L_id,"/","monthly_average/",thr_fld,"/CLSM/","site_",str(sitid),".csv"])
    			path_I = pathlib.Path(ofilI)
    			path_I.parent.mkdir(parents=True,exist_ok=True)
    			if (len(dframe_monO2) > 0):
    				print(dframe_monO2)
    				print(ofilO)
    				dframe_monO2.to_csv(ofilO,index=False)
    			if (len(dframe_monZ2) > 0):
    				print(dframe_monZ2)
    				print(ofilZ)
    				dframe_monZ2.to_csv(ofilZ,index=False)
    			if(len(dframe_monI2) > 0):
    				print(dframe_monI2)
    				print(ofilI)
    				dframe_monI2.to_csv(ofilI,index=False)
#	
