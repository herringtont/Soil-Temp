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
from calendar import isleap
from dateutil.relativedelta import *


################################# create master arrays for in-situ, models, and their anomalies ######################################
site_master = []
station_master = []
CFSR_master = []
ERAI_master = []
ERA5_master = []
JRA_master = []
MERRA2_master = []
GLDAS_master =[]

site_master_a = []
station_master_a = []
CFSR_master_a = []
ERAI_master_a = []
ERA5_master_a = []
JRA_master_a = []
MERRA2_master_a = []
GLDAS_master_a = []

################################## define outlier removal method and soil ####################################
mthd = 'IQR' ### IQR or z_score
lyr = '0_9.9' #### '0_9.9','10_29.9','30_99.9','100_299.9','300_deeper'

####################################### set thresholds ###########################################################
val_thresh = 100 #percent missing allowed (100, 75, 50, 25, 0)

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
       
############################## grab site and depth values from file name ################################
    sitid = file_name.split("site_")[1].split("_depth")[0] #locate site id within filename  
    sdepth = file_name.split("_depth_")[1].split("_")[0]
    sdep = float(sdepth) 

    #print(dframe)

##############################################separate files into bins by depth#########################################################
    if (sdep == "0"):
    	bins = "0_9.9"
    elif (0 <= sdep < 10):
    	bins = "0_9.9"
    elif (10 <= sdep < 30):
    	bins = "10_29.9"
    elif (30 <= sdep < 100):
    	bins = "30_99.9"
    elif (100 <= sdep < 300):
    	bins = "100_299.9"
    elif (sdep >= 300):
    	bins = "300_deeper"

#######################################set reanalysis soil temperature layers##########################################################

###Reanalysis Soil Layers
#CFSR 4 layers (0-10 cm, 10-40 cm, 40-100 cm, 100-200 cm)
#ERA-Interim (0-7 cm, 7-28 cm, 28-100 cm, 100-289 cm)
#ERA5 (0-7 cm, 7-28 cm, 28-100 cm, 100-289 cm)
#JRA (averaged over entire soil column)
#MERRA2 (0- 9.88 cm, 9.88-29.4 cm, 29.4-67.99cm, 67.99cm-144.25cm, 144.25-294.96 cm, 294.96-1294.96 cm) 
#GLDAS 
    #Noah (0-10 cm, 10-40 cm, 40-100 cm, 100-200 cm)  ***Noah available at higher resolution - used here
    #VIC (0-10 cm, 10 - 160cm, 160-190cm)  ***Only available at 1deg resolution
    #CLSM (0-1.8cm, 1.8-4.5cm, 4.5-9.1cm, 9.1-16.6cm, 16.6-28.9cm, 28.9-49.3cm, 49.3-82.9cm, 82.9-138.3cm, 138-229.6cm, 229.6-343.3cm)  ***only available at 1deg resolution

##### CFSR, CFSR 2 and GLDAS Noah ####
    if (0 <= sdep < 10):
    	CFSR_layer = "Soil_Temp_L1"
    	CFSR2_layer = "Soil_Temp_L1"
    	GLDAS_layer = "Soil_Temp_L1"
    elif (10 <= sdep < 40):
    	CFSR_layer = "Soil_Temp_L2"
    	CFSR2_layer = "Soil_Temp_L2"
    	GLDAS_layer = "Soil_Temp_L2"
    elif (40 <= sdep < 100):
    	CFSR_layer = "Soil_Temp_L3"
    	CFSR2_layer = "Soil_Temp_L3"
    	GLDAS_layer = "Soil_Temp_L3"
    elif (sdep >= 100):
    	CFSR_layer = "Soil_Temp_L4"
    	CFSR2_layer = "Soil_Temp_L4"
    	GLDAS_layer = "Soil_Temp_L4"

##### ERA-Interim and ERA5 ####
    if (0 <= sdep < 7):
    	ERAI_layer = "Soil_Temp_L1"
    	ERA5_layer = "Soil_Temp_L1"
    elif (7 <= sdep < 28):
    	ERAI_layer = "Soil_Temp_L2"
    	ERA5_layer = "Soil_Temp_L2"
    elif (28 <= sdep < 100):
    	ERAI_layer = "Soil_Temp_L3"
    	ERA5_layer = "Soil_Temp_L3"
    elif (sdep >= 100):
    	ERAI_layer = "Soil_Temp_L4"
    	ERA5_layer = "Soil_Temp_L4"

##### JRA55 ####
    JRA_layer = "Soil_Temp"

##### MERRA2 ####
    if (0 <= sdep < 9.88):
    	MERRA2_layer = "Soil_Temp_L1"
    elif (9.88 <= sdep < 29.4):
    	MERRA2_layer = "Soil_Temp_L2"
    elif (29.4 <= sdep < 67.99):
    	MERRA2_layer = "Soil_Temp_L3"
    elif (67.99 <= sdep < 144.25):
    	MERRA2_layer = "Soil_Temp_L4"
    elif (144.25 <= sdep < 294.96):
    	MERRA2_layer = "Soil_Temp_L5"
    elif (sdep >= 294.96):
    	MERRA2_layer = "Soil_Temp_L6" 

########################################## grab data from csv ###########################################
    dframe.replace('NaN',np.nan,inplace = True) #store NaN values as missing
    dframe_datst = dframe['Dataset']
    dframe_time = dframe['Date']
    dframe_stemp = dframe['Soil_Temp']
    dframe_dt_stemp = dframe[['Date','Soil_Temp']]

######## set date format depending on dataset ######

    datst = dframe_datst.iloc[1]

    if(datst == 'GTN-P'):
    	datfmt = "%Y-%m-%d %H:%M:%S"
    elif(datst == 'Kropp'):
    	datfmt = "%Y-%m-%d"


######## set datetimeindex #######
    dframe['Datetime'] = pd.to_datetime(dframe['Date'], format=datfmt)
    dframe_new = dframe.set_index(pd.DatetimeIndex(dframe['Date']))

    time_len = len(dframe_time)
    st_yr = datetime.datetime.strptime(dframe_time.iloc[0,],datfmt)
    ed_yr = datetime.datetime.strptime(dframe_time.iloc[time_len-1],datfmt)
    srt_yr = st_yr.year
    end_yr = ed_yr.year

    date_time = str_to_datetime(dframe_time,datfmt)
 
####################### read the start and end year/month from file ###########################

    beg_mon = st_yr.month
    beg_yr = st_yr.year
    fin_mon = ed_yr.month
    fin_yr = ed_yr.year

    dformat = '%Y_%m'

#### determine the total possible number of datapoints per day, assuming no missing data ####
    if (sitid == 29): 
    	mon_fac = 4  #6hr data
    elif (sitid == 30 or sitid == 35 or sitid == 36 or sitid == 37 or sitid == 38 or sitid == 39): 
    	mon_fac = 24  #1hr data
    elif (sitid == 40 or sitid == 41 or sitid == 42 or sitid == 43 or sitid == 56 or sitid == 57):
    	mon_fac = 24  #1hr data
    elif (sitid == 62 or sitid == 63 or sitid == 64 or sitid == 65 or sitid == 66):
    	mon_fac = 12  #2hr data
    elif (sitid == 35 or sitid == 36 or sitid == 52):
    	mon_fac = 0  #monthly data (note here we ignore the mon_fac and the number of datapoints in the month will be 1)
    else:
    	mon_fac = 1 #daily data


############################################## create dictionary to store the total possible number of datapoints ###############################################################
    tot_mon_entries = dict()

    for x in range(srt_yr, end_yr+1):
    	if (x == srt_yr):  #if first year, begin at starting month
    		for y in range(beg_mon,13):
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
    				tot_mon_entries[mon_key] = 1  #these sites have monthly data, so each month only has 1 datapoint maximum
    			else:
    				tot_mon_entries[mon_key] = mon_day*mon_fac   #maximum # of datapoints = # of days in month * mon_fac (the number of datapoints per day)
    	
    	elif (x == end_yr):  #if final year, end at final month in dataset
    		for y in range(1,fin_mon+1):
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
    				tot_mon_entries[mon_key] = 1
    			else:
    				tot_mon_entries[mon_key] = mon_day*mon_fac
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
    				tot_mon_entries[mon_key] = 1
    			else:
    				tot_mon_entries[mon_key] = mon_day*mon_fac

############################# create a dictionary which keeps track of how many data entries you have per month #######################
    monthly_entries = dict()

# building dictionary for datetimes in site csv file

    for x in range(srt_yr, end_yr+1):
    	if (x == srt_yr):  #if first year, begin at starting month
    		for y in range(beg_mon,13):
    			month_dt = datetime.datetime(x,y,1,0,0)
    			month_key = month_dt.strftime(dformat)
    			monthly_entries[month_key] = 0
    	elif (x == end_yr):  #if final year, end at final month in dataset
    		for y in range(1,fin_mon+1):
    			month_dt = datetime.datetime(x,y,1,0,0)
    			month_key = month_dt.strftime(dformat)
    			monthly_entries[month_key] = 0
    	else:
    		for y in range(1,13): #else iterate over all 12 months
    			month_dt = datetime.datetime(x,y,1,0,0)
    			month_key = month_dt.strftime(dformat)
    			monthly_entries[month_key] = 0  			

    
################################## add in a second check where we check if the soil temperature data is valid for that particular day ####################################
    dframe_dt_stemp2 = dframe_dt_stemp.notna()

    stemp_new = []
    dat_new = []

###Now you have a dictionary which keeps track of how many data entries you have per month. 
###Then you can iterate over all of your data rows, and just increment the monthly_entries[month_key] counter by one for each entry.
###In order to generate the month_key from a datetime object, you'd just run:

    dt_idx = 0 #index of date and soil temperature value

    for dt in date_time:
    	dt2 = date_time[dt_idx]
    	stemp2 = dframe_dt_stemp.iloc[dt_idx,1]
    	stemp3 = dframe_dt_stemp2.iloc[dt_idx,1]
    	#print(type(dt))
    	key = dt.strftime(dformat)
    	if (stemp3 == True):  #if soil temp is not missing, add it to dat_new (else do not)
    		stemp_new.append(stemp2)
    		dat_new.append(dt2)
    		#print(dat_new)		
    	dt_idx += 1    


    for dt in dat_new:
    	#print(type(dt))
    	key = dt.strftime(dformat) 		
    	monthly_entries[key] += 1
    
    #print(tot_mon_entries)
    #print(len(tot_mon_entries))   
    #print(monthly_entries)
    #print(len(monthly_entries)) 
  

########################## calculate monthly average soil temperatures ##############################
    dframe_mon = dframe_new.resample('M', convention ='start').mean()
    dframe_mon['Date'] = dframe_mon.index.values.astype('datetime64[M]') #change datetime index 

########reindex Monthly Averages#######
    dframe_mon2 = dframe_mon.set_index(pd.DatetimeIndex(dframe_mon['Date'])) #reindex with new datetime index
    
    #print(dframe_mon2)
######## split dictionaries into values and keys, compare values #######
    mon_val = list(monthly_entries.values())
    mon_ky = list(monthly_entries.keys())

    tot_val = list(tot_mon_entries.values())
    tot_ky = list(tot_mon_entries.keys())

    dat_new2 = []
    dat_tot = []
    val_new = []
    val_tot = []

    for i in range(0,len(mon_val)):
    	str_ky = mon_ky[i]
    	str_tot = tot_ky[i]
    	new_dt = datetime.datetime.strptime(str_ky,'%Y_%m')
    	new_vl = mon_val[i]
    	tot_vl = tot_val[i]
    	dt_tot = datetime.datetime.strptime(str_tot,'%Y_%m')
    	dat_new2.append(new_dt)
    	dat_tot.append(dt_tot)
    	val_new.append(new_vl)
    	val_tot.append(tot_vl) 

######## calculate percent valid values #########
    val_new2 = np.array(val_new)
    val_tot2 = np.array(val_tot)
    val_pct = (val_new2/val_tot2)*100

    val_pct2 =[]

###### remove months with no data at all #######  
    dframe_mon2 = dframe_mon2[dframe_mon2['Soil_Temp'].notna()]

###### extract percent valid values for months with data #####
    for i in range(0,len(mon_val)):
    	val_pt = val_pct[i]
    	if(val_pt > 0):
    		val_pct2.append(val_pt)
    
###### append percent valid data column to monthly dataframe ######    
    dframe_mon2['Percent_Valid'] = val_pct2
    dframe_mon2['Date'] = dframe_mon2.index
    dframe_mon2 = dframe_mon2[['Date','Soil_Temp','Percent_Valid']]

##### remove months where percent valid < miss_threshold ########
    dframe_mon2 = dframe_mon2.loc[dframe_mon2['Percent_Valid'] >= miss_thr]


##### calculate anomalies of monthly means #######
    if (dframe_mon2['Soil_Temp'].empty == False):
    	dframe_mon2_val = dframe_mon2['Soil_Temp'].to_numpy()
    	dframe_mon2_anom = np.ma.anomalies(dframe_mon2_val).anom() 
    	anom_list = {'Date': dframe_mon2['Date'], 'Anomalies': dframe_mon2_anom, 'Percent_Valid': dframe_mon2['Percent_Valid']}
    	dframe_station_anom = pd.DataFrame(anom_list)
    	          
####### grab datetime from station location and convert it into Year-Mon format #######
    dmon_new = dframe_mon2['Date']
    dmon_n2 = []

    for i in range(0,len(dmon_new)):
    	dmon_n = datetime.datetime.strftime(dmon_new.iloc[i],'%Y-%m-%d')
    	dmon_n2.append(dmon_n)
    
    #print("list of valid dates: ",dmon_n2)


################################# grab reanalysis data #############################################
    base_dir = "/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/site_level/"
    CFSR_fi = "".join([str(base_dir),"CFSR/CFSR_site_",sitid,".nc"])
    CFSR2_fi = "".join([str(base_dir),"CFSR2/CFSR2_site_",sitid,".nc"])
    MERRA2_fi = "".join([str(base_dir),"MERRA2/MERRA2_site_",sitid,".nc"])
    ERA5_fi = "".join([str(base_dir),"ERA5/ERA5_site_",sitid,".nc"])
    ERAI_fi = "".join([str(base_dir),"ERA-Interim/ERA-Interim_site_",sitid,".nc"])
    JRA_fi = "".join([str(base_dir),"JRA55/JRA55_site_",sitid,".nc"])
    GLDAS_fi = "".join([str(base_dir),"GLDAS/GLDAS_site_",sitid,".nc"])

    GLDAS_fil = xr.open_dataset(GLDAS_fi)
    JRA_fil = xr.open_dataset(JRA_fi)
    ERAI_fil = xr.open_dataset(ERAI_fi)
    ERA5_fil = xr.open_dataset(ERA5_fi)
    MERRA2_fil = xr.open_dataset(MERRA2_fi)
    CFSR_fil = xr.open_dataset(CFSR_fi) #open NetCDF file with xarray
    CFSR2_fil = xr.open_dataset(CFSR2_fi) #open NetCDF file with xarray

########### extract soil temperatures and convert to celsius #######
    GLDAS_stemp = GLDAS_fil[GLDAS_layer] - 273.15
    JRA_stemp = JRA_fil[JRA_layer] - 273.15
    ERAI_stemp = ERAI_fil[ERAI_layer] - 273.15
    ERA5_stemp = ERA5_fil[ERA5_layer] - 273.15
    MERRA2_stemp = MERRA2_fil[MERRA2_layer] - 273.15 #convert from Kelvin to Celsius
    CFSR_stemp = CFSR_fil[CFSR_layer] - 273.15  #convert from Kelvin to Celsius
    CFSR2_stemp = CFSR2_fil[CFSR2_layer] - 273.15  #convert from Kelvin to Celsius

########## drop lon,lat coordinates #########

    GLDAS_stemp3 = GLDAS_stemp.isel(lon=0,lat=0,drop=True)
    JRA_stemp3 = JRA_stemp.isel(lon=0,lat=0,drop=True)
    ERAI_stemp3 = ERAI_stemp.isel(lon=0,lat=0,drop=True)
    ERA5_stemp3 = ERA5_stemp.isel(lon=0,lat=0,drop=True)
    MERRA2_stemp3 = MERRA2_stemp.isel(lon=0,lat=0,drop=True)
    CFSR_stemp3 = CFSR_stemp.isel(lon=0,lat=0,drop=True)
    CFSR2_stemp3 = CFSR2_stemp.isel(lon=0,lat=0,drop=True) 



############################### check to see if there are any valid dates in station #############################################    
    if (len(dmon_n2) > 0): #if there is valid data, move onto extracting reanalysis data (else move onto the next file)


######### check if reanalysis data is missing ########

    	GLDAS_valid = np.array(GLDAS_stemp3.count(keep_attrs=False))
    	JRA_valid = np.array(JRA_stemp3.count(keep_attrs=False))
    	ERAI_valid = np.array(ERAI_stemp3.count(keep_attrs=False))
    	ERA5_valid = np.array(ERA5_stemp3.count(keep_attrs=False))
    	MERRA2_valid = np.array(MERRA2_stemp3.count(keep_attrs=False))
    	CFSR_valid = np.array(CFSR_stemp3.count(keep_attrs=False))
    	CFSR2_valid = np.array(CFSR2_stemp3.count(keep_attrs=False))

    

###########grab corresponding reanalysis data############################# 
    
###### extract CFSR and CFSR2 values, and join them together in a contiguous array######
    	CFSR_all = []
    	for i in range(0,len(dmon_n2)):		
    		Cmon_n = datetime.datetime.strptime(dmon_n2[i],'%Y-%m-%d')
    		Cmon_n2 = dmon_n2[i]
    		if(Cmon_n > datetime.datetime(2010,12,31)):
    			continue
    		CFSR_n = CFSR_stemp3.sel(time = Cmon_n2, drop=True)
    		CFSR_n2 = CFSR_n.values  
    		CFSR_all.append(CFSR_n2)

    	if (datetime.datetime.strptime(dmon_n2[len(dmon_n2)-1],'%Y-%m-%d') > datetime.datetime(2010,12,31)): 
    		for i in range(0,len(dmon_n2)):	
    			C2mon_n = datetime.datetime.strptime(dmon_n2[i],'%Y-%m-%d')
    			C2mon_n2 = dmon_n2[i]
    			if(C2mon_n <= datetime.datetime(2010,12,31)):
    				continue	
    			CFSR2_n = CFSR2_stemp3.sel(time = C2mon_n2, drop=True)
    			CFSR2_n2 = CFSR2_n.values  
    			CFSR_all.append(CFSR2_n2)    

###################### extract values of other reanalysis products #######################
    	GLDAS_new = []
    	JRA_new = []
    	ERAI_new = []
    	ERA5_new = []
    	MERRA2_new = []
    	for i in range(0,len(dmon_new)):
    		dmon_n = dmon_new[i]
    		ddat = dmon_n
    		
    		if (ddat <= datetime.datetime(2020,7,1)):
    			GLDAS_n = GLDAS_stemp3.sel(time = ddat, drop=True)
    			GLDAS_n2 = GLDAS_n.values
    		elif (ddat > datetime.datetime(2020,7,1)):
    			GLDAS_n2 = np.nan				
    		GLDAS_new.append(GLDAS_n2)
        
    		if (ddat <= datetime.datetime(2019,12,1)):
    			JRA_n = JRA_stemp3.sel(time = ddat, drop=True)
    			JRA_n2 = JRA_n.values
    		elif (ddat > datetime.datetime(2019,12,1)):
    			JRA_n2 = np.nan				
    		JRA_new.append(JRA_n2)    
    
    		if (ddat <= datetime.datetime(2019,8,1)):
    			ERAI_n = ERAI_stemp3.sel(time = ddat, drop=True)
    			ERAI_n2 = ERAI_n.values
    		elif (ddat > datetime.datetime(2019,8,1)):
    			ERAI_n2 = np.nan				
    		ERAI_new.append(ERAI_n2)    
            
    		if (ddat <= datetime.datetime(2018,12,1)):
    			ERA5_n = ERA5_stemp3.sel(time = ddat, drop=True)
    			ERA5_n2 = ERA5_n.values
    		elif (ddat > datetime.datetime(2018,12,1)):
    			ERA5_n2 = np.nan				
    		ERA5_new.append(ERA5_n2)    
    
    		if (ddat <= datetime.datetime(2020,8,1)):
    			MERRA2_n = MERRA2_stemp3.sel(time = ddat, drop=True)
    			MERRA2_n2 = MERRA2_n.values
    		elif (ddat > datetime.datetime(2019,12,1)):
    			MERRA2_n2 = np.nan				
    		MERRA2_new.append(MERRA2_n2)    

    	GLDAS_new2 = np.array(GLDAS_new)
    	JRA_new2 = np.array(JRA_new)
    	ERAI_new2 = np.array(ERAI_new)
    	ERA5_new2 = np.array(ERA5_new)
    	MERRA2_new2 = np.array(MERRA2_new)
    	CFSR_all2 = np.array(CFSR_all)


####### Calculate corresponding reanalysis anomalies ########        
    	if(GLDAS_new2.size > 0):	
    		GLDAS_anom = np.ma.anomalies(GLDAS_new2).anom()
    	if(JRA_new2.size > 0):	
    		JRA_anom = np.ma.anomalies(JRA_new2).anom()
    	if(ERAI_new2.size > 0):    	
    		ERAI_anom = np.ma.anomalies(ERAI_new2).anom()
    	if(ERA5_new2.size > 0):
    		ERA5_anom = np.ma.anomalies(ERA5_new2).anom()
    	if(MERRA2_new2.size > 0):
    		MERRA2_anom = np.ma.anomalies(MERRA2_new2).anom()
    	if(CFSR_all2.size > 0):
    		CFSR_anom = np.ma.anomalies(CFSR_all2).anom()

########## create new Dataframe with date, station temp, reanalysis temp, %valid ##########
    	dframe_mon2['GLDAS'] = GLDAS_new2
    	dframe_mon2['JRA55'] = JRA_new2
    	dframe_mon2['ERA-Interim'] = ERAI_new2
    	dframe_mon2['ERA5'] = ERA5_new2
    	dframe_mon2['MERRA2'] = MERRA2_new2
    	dframe_mon2['CFSR'] = CFSR_all2
    	dframe_mon2['Site'] = sitid
    	dframe_mon2['Depth'] = sdepth
    	dframe_tot = dframe_mon2[['Date','Soil_Temp','CFSR','MERRA2','ERA5','ERA-Interim','JRA55','GLDAS','Percent_Valid','Site','Depth']]
    	#print(dframe_tot)

############ drop columns with NaN ###############
    	dframe_tot = dframe_tot[dframe_tot['GLDAS'].notna()]
    	dframe_tot = dframe_tot[dframe_tot['JRA55'].notna()]
    	dframe_tot = dframe_tot[dframe_tot['MERRA2'].notna()]
    	dframe_tot = dframe_tot[dframe_tot['ERA-Interim'].notna()]
    	dframe_tot = dframe_tot[dframe_tot['CFSR'].notna()]
    	dframe_tot = dframe_tot[dframe_tot['ERA5'].notna()]
    	#print(dframe_tot)


######### create new anomalies dataframe (if it exists for all files) ###########
    	if(dframe_mon2['Soil_Temp'].empty == False and GLDAS_new2.size > 0 and JRA_new2.size > 0 and ERAI_new2.size > 0 and ERA5_new2.size > 0 and MERRA2_new2.size > 0 and CFSR_all2.size > 0):
    		dframe_station_anom['GLDAS_anom'] = GLDAS_anom
    		dframe_station_anom['JRA55_anom'] = JRA_anom
    		dframe_station_anom['ERA-Interim_anom'] = ERAI_anom
    		dframe_station_anom['ERA5_anom'] = ERA5_anom
    		dframe_station_anom['CFSR_anom'] = CFSR_anom
    		dframe_station_anom['MERRA2_anom'] = MERRA2_anom
    		dframe_station_anom['Site'] = sitid
    		dframe_station_anom['Depth'] = sdepth		
    		dframe_tot_anom = dframe_station_anom[['Date','Anomalies','CFSR_anom','MERRA2_anom','ERA5_anom','ERA-Interim_anom','JRA55_anom','GLDAS_anom','Percent_Valid','Site','Depth']]		    			
    		#print(dframe_tot_anom)    		
############ drop columns with NaN ###############
    		dframe_tot_anom = dframe_tot_anom[dframe_tot_anom['GLDAS_anom'].notna()]
    		dframe_tot_anom = dframe_tot_anom[dframe_tot_anom['JRA55_anom'].notna()]
    		dframe_tot_anom = dframe_tot_anom[dframe_tot_anom['MERRA2_anom'].notna()]
    		dframe_tot_anom = dframe_tot_anom[dframe_tot_anom['ERA-Interim_anom'].notna()]
    		dframe_tot_anom = dframe_tot_anom[dframe_tot_anom['CFSR_anom'].notna()]
    		dframe_tot_anom = dframe_tot_anom[dframe_tot_anom['ERA5_anom'].notna()]
    		#print(dframe_tot_anom)

############ grab final values to store in master array ############
#    	station_final = dframe_tot['Soil_Temp']
#    	if (station_final.empty == False):
#    		station_master.append(station_final.values.flatten())
#    	
#    	site_final = dframe_tot['Site']
#    	if (site_final.empty == False):
#    		site_master.append(site_final.values.flatten())
#    
#    	CFSR_final = dframe_tot['CFSR']
#    	if (CFSR_final.empty == False):
#    		CFSR_master.append(CFSR_final.values.flatten())
#    
#    	ERAI_final = dframe_tot['ERA-Interim']
#    	if (ERAI_final.empty == False):
#    		ERAI_master.append(ERAI_final.values.flatten())
#    
#    	ERA5_final = dframe_tot['ERA5']
#    	if (ERA5_final.empty == False):
#    		ERA5_master.append(ERA5_final.values.flatten)
#    
#    	MERRA2_final = dframe_tot['MERRA2']
#    	if (MERRA2_final.empty == False):
#    		MERRA2_master.append(MERRA2_final.values.flatten())
#    
#    	JRA_final = dframe_tot['JRA55']
#    	if (JRA_final.empty == False):
#    		JRA_master.append(JRA_final.values.flatten())
#    
#    	GLDAS_final = dframe_tot['GLDAS']
#    	if (GLDAS_final.empty == False):	
#    		GLDAS_master.append(GLDAS_final.values.flatten())


    	station_final = dframe_tot['Soil_Temp']
    	if (station_final.empty == False):
    		station_master.append(station_final.tolist())
    	
    	site_final = dframe_tot['Site']
    	if (site_final.empty == False):
    		site_master.append(site_final.values.tolist())
    
    	CFSR_final = dframe_tot['CFSR']
    	if (CFSR_final.empty == False):
    		CFSR_master.append(CFSR_final.values.tolist())
    
    	ERAI_final = dframe_tot['ERA-Interim']
    	if (ERAI_final.empty == False):
    		ERAI_master.append(ERAI_final.values.tolist())
    
    	ERA5_final = dframe_tot['ERA5']
    	if (ERA5_final.empty == False):
    		ERA5_master.append(ERA5_final.values.tolist())
    
    	MERRA2_final = dframe_tot['MERRA2']
    	if (MERRA2_final.empty == False):
    		MERRA2_master.append(MERRA2_final.values.tolist())
    
    	JRA_final = dframe_tot['JRA55']
    	if (JRA_final.empty == False):
    		JRA_master.append(JRA_final.values.tolist())
    
    	GLDAS_final = dframe_tot['GLDAS']
    	if (GLDAS_final.empty == False):	
    		GLDAS_master.append(GLDAS_final.values.tolist())


########## grab final anomalies to store in master array ###########
#    	station_finala = dframe_tot_anom['Anomalies']
#    	if (station_finala.empty == False):
#    		station_master_a.append(station_finala.values.flatten())
#
#    	site_finala = dframe_tot_anom['Site']
#    	if (site_finala.empty == False):
#    		site_master_a.append(site_finala.values.flatten())
#    
#    	CFSR_finala = dframe_tot_anom['CFSR_anom']
#    	if (CFSR_finala.empty == False):
#    		CFSR_master_a.append(CFSR_finala.values.flatten())
#    
#    	ERAI_finala = dframe_tot_anom['ERA-Interim_anom']
#    	if (ERAI_finala.empty == False):
#    		ERAI_master_a.append(ERAI_finala.values.flatten())
#    
#    	ERA5_finala = dframe_tot_anom['ERA5_anom']
#    	if (ERA5_finala.empty == False):
#    		ERA5_master_a.append(ERA5_finala.values.flatten())
#    
#    	MERRA2_finala = dframe_tot_anom['MERRA2_anom']
#    	if (MERRA2_finala.empty == False):
#    		MERRA2_master_a.append(MERRA2_finala.values.flatten())
#    
#    	JRA_finala = dframe_tot_anom['JRA55_anom']
#    	if (JRA_finala.empty == False):
#    		JRA_master_a.append(JRA_finala.values.flatten())
#    
#    	GLDAS_finala = dframe_tot_anom['GLDAS_anom']
#    	if (GLDAS_finala.empty == False):	
#    		GLDAS_master_a.append(GLDAS_finala.values.flatten())


    	station_finala = dframe_tot_anom['Anomalies']
    	if (station_finala.empty == False):
    		station_master_a.append(station_finala.tolist())

    	site_finala = dframe_tot_anom['Site']
    	if (site_finala.empty == False):
    		site_master_a.append(site_finala.tolist())
    
    	CFSR_finala = dframe_tot_anom['CFSR_anom']
    	if (CFSR_finala.empty == False):
    		CFSR_master_a.append(CFSR_finala.tolist())
    
    	ERAI_finala = dframe_tot_anom['ERA-Interim_anom']
    	if (ERAI_finala.empty == False):
    		ERAI_master_a.append(ERAI_finala.tolist())
    
    	ERA5_finala = dframe_tot_anom['ERA5_anom']
    	if (ERA5_finala.empty == False):
    		ERA5_master_a.append(ERA5_finala.tolist())
    
    	MERRA2_finala = dframe_tot_anom['MERRA2_anom']
    	if (MERRA2_finala.empty == False):
    		MERRA2_master_a.append(MERRA2_finala.tolist())
    
    	JRA_finala = dframe_tot_anom['JRA55_anom']
    	if (JRA_finala.empty == False):
    		JRA_master_a.append(JRA_finala.values.tolist())
    
    	GLDAS_finala = dframe_tot_anom['GLDAS_anom']
    	if (GLDAS_finala.empty == False):	
    		GLDAS_master_a.append(GLDAS_finala.values.tolist())

def main():
#set files and directories
    from pathlib import Path
    directory = "/mnt/data/users/herringtont/soil_temp/In-Situ/All/depth_level/layers/no_outliers/"
    pthl = [directory,str(mthd),"/",str(lyr)+"/"]
    pthl2 = "".join(pthl)
    pathlist = Path(pthl2).glob('*.csv')
    for path in sorted(pathlist):
    	fil = str(path)
    	#print(fil)			
    	load_pandas(fil)
		
main()

print(station_master)

############### Flatten master lists to 1D ##########################

site_master_1D = []
for sublist in site_master:
    for item in sublist:
    	site_master_1D.append(item)
	
station_master_1D = []
for sublist in station_master:
    for item in sublist:
    	station_master_1D.append(item)

CFSR_master_1D = []
for sublist in CFSR_master:
    for item in sublist:
    	CFSR_master_1D.append(item)

ERAI_master_1D = []
for sublist in ERAI_master:
    for item in sublist:
    	ERAI_master_1D.append(item)

ERA5_master_1D = []
for sublist in ERA5_master:
    for item in sublist:
    	ERA5_master_1D.append(item)

JRA_master_1D = []
for sublist in JRA_master:
    for item in sublist:
    	JRA_master_1D.append(item)

MERRA2_master_1D = []
for sublist in MERRA2_master:
    for item in sublist:
    	MERRA2_master_1D.append(item)

GLDAS_master_1D = []
for sublist in GLDAS_master:
    for item in sublist:
    	GLDAS_master_1D.append(item)

site_master_a_1D = []
for sublist in site_master_a:
    for item in sublist:
    	site_master_a_1D.append(item)

station_master_a_1D = []
for sublist in station_master_a:
    for item in sublist:
    	station_master_a_1D.append(item)
	
CFSR_master_a_1D = []
for sublist in CFSR_master_a:
    for item in sublist:
    	CFSR_master_a_1D.append(item)
	
ERAI_master_a_1D = []
for sublist in ERAI_master_a:
    for item in sublist:
    	ERAI_master_a_1D.append(item)

ERA5_master_a_1D = []
for sublist in ERA5_master_a:
    for item in sublist:
    	ERAI_master_a_1D.append(item)
	
JRA_master_a_1D = []
for sublist in JRA_master_a:
    for item in sublist:
    	JRA_master_a_1D.append(item)
	
MERRA2_master_a_1D = []
for sublist in MERRA2_master_a:
    for item in sublist:
    	MERRA2_master_a_1D.append(item)
	
GLDAS_master_a_1D = []
for sublist in GLDAS_master_a:
    for item in sublist:
    	GLDAS_master_a_1D.append(item)


site_master_1D = np.array(site_master_1D)
station_master_1D = np.array(station_master_1D)
CFSR_master_1D = np.array(CFSR_master_1D)
ERAI_master_1D = np.array(ERAI_master_1D)
ERA5_master_1D = np.array(ERA5_master_1D)
JRA_master_1D = np.array(JRA_master_1D)
MERRA2_master_1D = np.array(MERRA2_master_1D)
GLDAS_master_1D = np.array(GLDAS_master_1D)

site_master_a_1D = np.array(site_master_a_1D)
station_master_a_1D = np.array(station_master_a_1D)
CFSR_master_a_1D = np.array(CFSR_master_a_1D)
ERAI_master_a_1D = np.array(ERAI_master_a_1D) 
ERA5_master_a_1D = np.array(ERA5_master_a_1D)
JRA_master_a_1D = np.array(JRA_master_a_1D) 
MERRA2_master_a_1D = np.array(JRA_master_a_1D)
GLDAS_master_a_1D = np.array(GLDAS_master_a_1D)


########################### Triple Collocation Triplets #############################################
UQ_sites = np.unique(site_master_1D)
sample_size = len(station_master_1D)
UQ_sitesa = np.unique(site_master_a_1D)
sample_sizea = len(station_master_a_1D)
print("unique sites: ",UQ_sites,", sample size: ", sample_size)
print("unique sites: ",UQ_sitesa,", sample size: ", sample_sizea)

TC_dir = "/mnt/data/users/herringtont/soil_temp/TC_Analysis/layer/"
odir ="".join([TC_dir,lyr,"/"])
print(odir)
