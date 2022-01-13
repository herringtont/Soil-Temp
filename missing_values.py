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

def str_to_datetime(column, date_fmt):

    date_list = []

    for dt_str in column:
        new_dt = datetime.datetime.strptime(dt_str, date_fmt)
        date_list.append(new_dt)
			

    return date_list


####################################### set thresholds ###########################################################
val_thresh = 100 #percent missing allowed (100, 75, 50, 25, 0)
layer_thick = 10  #thickness of soil layer in cm (10, 30)

miss_thr = 100 - val_thresh #percent valid required in order to be included in monthly average


########################## set in-situ files ####################################################
fil = "/mnt/data/users/herringtont/soil_temp/In-Situ/All/depth_level/0_4.9/site_109_depth_2.csv"

#### grab site and depth values from file name ####
sitid = fil.split("site_")[1].split("_depth")[0] #locate site id within filename
sdepth = fil.split("_depth_")[1].split(".csv")[0]
sdep = float(sdepth)

################################# grab reanalysis data #############################################
CFSR_fi = "/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/site_level/CFSR/new_taxis/new_reltime/CFSR_site_109.nc"
CFSR2_fi = "/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/site_level/CFSR2/new_taxis/new_reltime/CFSR2_site_109.nc"
MERRA2_fi = "/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/site_level/MERRA2/MERRA2_site_109.nc"
ERA5_fi = "/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/site_level/ERA5/ERA5_site_109.nc"
ERAI_fi = "/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/site_level/ERA-Interim/ERA-Interim_site_109.nc"
JRA_fi = "/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/site_level/JRA55/JRA55_site_109.nc"
GLDAS_fi = "/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/site_level/GLDAS/GLDAS_site_109.nc"

GLDAS_fil = xr.open_dataset(GLDAS_fi)
JRA_fil = xr.open_dataset(JRA_fi)
ERAI_fil = xr.open_dataset(ERAI_fi)
ERA5_fil = xr.open_dataset(ERA5_fi)
MERRA2_fil = xr.open_dataset(MERRA2_fi)
CFSR_fil = xr.open_dataset(CFSR_fi) #open NetCDF file with xarray
CFSR2_fil = xr.open_dataset(CFSR2_fi) #open NetCDF file with xarray

#extract soil temperatures and convert to celsius
GLDAS_stemp = GLDAS_fil[GLDAS_layer] - 273.15
JRA_stemp = JRA_fil[JRA_layer] - 273.15
ERAI_stemp = ERAI_fil[ERAI_layer] - 273.15
ERA5_stemp = ERA5_fil[ERA5_layer] - 273.15
MERRA2_stemp = MERRA2_fil[MERRA2_layer] - 273.15 #convert from Kelvin to Celsius
CFSR_stemp = CFSR_fil[CFSR_layer] - 273.15  #convert from Kelvin to Celsius
CFSR2_stemp = CFSR2_fil[CFSR2_layer] - 273.15  #convert from Kelvin to Celsius


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
    
    
#############################grab data from csv#########################
dframe = pd.read_csv(fil,parse_dates=True, infer_datetime_format=True)
#print(dframe)
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
    		#print(month_d)
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
    dt_idx += 1    


for dt in dat_new:
    #print(type(dt))
    key = dt.strftime(dformat) 		
    monthly_entries[key] += 1
       
     
exp_ent = len(date_time)
count = 0

#check to see if # of datapoints equals len(date_time)
for key in monthly_entries:
    count += monthly_entries[key]
    
if count != exp_ent:
    raise ValueError("Invalid number of expected entries")   


########################## calculate monthly average soil temperatures ##############################
dframe_mon = dframe_new.resample('M', convention ='start').mean()
dframe_mon['Date'] = dframe_mon.index.values.astype('datetime64[M]')

########reindex Monthly Averages#######
dframe_mon2 = dframe_mon.set_index(pd.DatetimeIndex(dframe_mon['Date']))

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


####### grab datetime from station location and convert it into Year-Mon format #######
dmon_new = dframe_mon2['Date']
dmon_n2 = []

for i in range(0,len(dmon_new)):
    dmon_n = datetime.datetime.strftime(dmon_new.iloc[i],'%Y-%m-%d')
    dmon_n2.append(dmon_n)

#print(GLDAS_stemp)

##################reindex reanalysis data over the date range of the station file (fill missing days with NaN)######################
tim_idx = pd.date_range(dmon_n2[0],dmon_n2[len(dmon_n2)-1],freq='M')
tim_idx2 = tim_idx - pd.offsets.MonthBegin(1, normalize=True)

CFSR_idx = pd.date_range(dmon_n2[0],'2010-12-16',freq='M') #CFSR has a different index because it ends in Dec 2010
CFSR2_idx = pd.date_range('2011-01-01',dmon_n2[len(dmon_n2)-1],freq='M') #CFSR2 has a different index because it begins in Jan 2011

GLDAS_stemp2 = GLDAS_stemp.reindex({"time":tim_idx2},method="bfill")
JRA_stemp2 = JRA_stemp.reindex({"time":tim_idx2},method="bfill")
ERAI_stemp2 = ERAI_stemp.reindex({"time":tim_idx2},method="bfill")
ERA5_stemp2 = ERA5_stemp.reindex({"time":tim_idx2},method="bfill")
MERRA2_stemp2 = MERRA2_stemp.reindex({"time":tim_idx2},method="bfill")
CFSR_stemp2 = CFSR_stemp.reindex({"time": CFSR_idx},method="bfill")
CFSR2_stemp2 = CFSR2_stemp.reindex({"time": CFSR2_idx},method="bfill")

GLDAS_stemp3 = GLDAS_stemp.isel(lon=0,lat=0,drop=True)
JRA_stemp3 = JRA_stemp.isel(lon=0,lat=0,drop=True)
ERAI_stemp3 = ERAI_stemp.isel(lon=0,lat=0,drop=True)
ERA5_stemp3 = ERA5_stemp.isel(lon=0,lat=0,drop=True)
MERRA2_stemp3 = MERRA2_stemp.isel(lon=0,lat=0,drop=True)
CFSR_stemp3 = CFSR_stemp.isel(lon=0,lat=0,drop=True)
CFSR2_stemp3 = CFSR2_stemp.isel(lon=0,lat=0,drop=True)


###########grab corresponding reanalysis data############################# 
#extract CFSR and CFSR2 values, and join them together in a contiguous array
CFSR_all = []
for i in range(0,len(dmon_n2)):
    Cmon_n = dmon_n2[i]
    Cdat = datetime.datetime.strptime(Cmon_n,'%Y-%m-%d')
    if (Cdat > datetime.datetime(2010,12,31)):
    	break
    CFSR_n = CFSR_stemp3.sel(time = Cmon_n, drop=True)
    CFSR_n2 = CFSR_n.values   
    CFSR_all.append(CFSR_n2)

if (datetime.datetime.strptime(dmon_n2[len(dmon_n2)-1],'%Y-%m-%d') > datetime.datetime(2010,12,31)): 
    for i in range(0,len(dmon_n2)):
    	C2mon_n = dmon_n2[i]
    	if (C2mon_n < datetime.datetime(2011,1,1)):
    		continue	
    	CFSR2_n = CFSR2_stemp3.sel(time = C2mon_n, drop=True)
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
    ddat = datetime.datetime.strftime(dmon_n,'%Y-%m-%d')
    GLDAS_n = GLDAS_stemp3.sel(time = ddat, drop=True)
    GLDAS_n2 = GLDAS_n.values
    GLDAS_new.append(GLDAS_n2)
        
    JRA_n = JRA_stemp3.sel(time = ddat, drop=True)
    JRA_n2 = JRA_n.values
    JRA_new.append(JRA_n2)    
    
    ERAI_n = ERAI_stemp3.sel(time = ddat,drop=True)
    ERAI_n2 = ERAI_n.values
    ERAI_new.append(ERAI_n2)
            
    ERA5_n = ERA5_stemp3.sel(time = ddat, drop=True)
    ERA5_n2 = ERA5_n.values
    ERA5_new.append(ERA5_n2)
    
    MERRA2_n = MERRA2_stemp3.sel(time = ddat, drop=True)
    MERRA2_n2 = MERRA2_n.values
    MERRA2_new.append(MERRA2_n2)

GLDAS_new2 = np.array(GLDAS_new)
JRA_new2 = np.array(JRA_new)
ERAI_new2 = np.array(ERAI_new)
ERA5_new2 = np.array(ERA5_new)
MERRA2_new2 = np.array(MERRA2_new)
CFSR_all2 = np.array(CFSR_all)


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

############ drop columns with NaN ###############
dframe_tot = dframe_tot[dframe_tot['GLDAS'].notna()]
dframe_tot = dframe_tot[dframe_tot['JRA55'].notna()]
dframe_tot = dframe_tot[dframe_tot['MERRA2'].notna()]
dframe_tot = dframe_tot[dframe_tot['ERA-Interim'].notna()]
dframe_tot = dframe_tot[dframe_tot['CFSR'].notna()]
dframe_tot = dframe_tot[dframe_tot['ERA5'].notna()]
print(dframe_tot)



##############################Run Triple Collocation Analysis##########################################

#dframe_station = dframe_tot['Soil_Temp']
#dframe_CFSR = dframe_tot['CFSR']
#dframe_ERA5 = dframe_tot['ERA5']
#
#TC_dict = {"Station":dframe_station,"CFSR":dframe_CFSR,"ERA5":dframe_ERA5}
#dframe_TC = pd.DataFrame(TC_dict)
#print(dframe_TC)
#
#
#
#x = dframe_station
#y = CFSR_all2
#z = ERA5_new2
#
#################################################
####### APPROACH 1 (SCALING)
#
#def mean_std(src, ref):
#    return ((src - np.nanmean(src)) /
#            np.nanstd(src)) * np.nanstd(ref) + np.nanmean(ref)
#
#def tcol_error(x, y, z):
#    e_x = np.sqrt(np.abs(np.nanmean((x - y) * (x - z))))
#    e_y = np.sqrt(np.abs(np.nanmean((y - x) * (y - z))))
#    e_z = np.sqrt(np.abs(np.nanmean((z - x) * (z - y))))
#
#    return e_x, e_y, e_z
#
#x = x
#y_scaled = mean_std(y, x) 
#z_scaled = mean_std(z, x)
#
#e_x, e_y, e_z = tcol_error(x, y_scaled, z_scaled)
#print("***Approach 1 - Scaling***")
#print("Errors:")
#print(e_x,e_y,e_z)
#
#################################################
####### APPROACH 2 (COVARIANCES)
#
#def triple_collocation_snr(x, y, z, ref_ind=0):
#    cov = np.cov(np.vstack((x, y, z)))
#    ind = (0, 1, 2, 0, 1, 2)
#    no_ref_ind = np.where(np.arange(3) != ref_ind)[0]
#    
#    snr = -10 * np.log10([abs(((cov[i, i] * cov[ind[i + 1], ind[i + 2]]) / (cov[i, ind[i + 1]] * cov[i, ind[i + 2]])) - 1)
#                         for i in np.arange(3)])
#    err_var = np.array([
#        abs(cov[i, i] -
#        (cov[i, ind[i + 1]] * cov[i, ind[i + 2]]) / cov[ind[i + 1], ind[i + 2]])
#        for i in np.arange(3)])
#
#    beta = np.array([cov[ref_ind, no_ref_ind[no_ref_ind != i][0]] /
#                     cov[i, no_ref_ind[no_ref_ind != i][0]] if i != ref_ind
#                     else 1 for i in np.arange(3)])
#
#    return snr, np.sqrt(err_var) * beta, beta
#
#snr, err, beta = triple_collocation_snr(x, y, z)
#
#print("***Approach 2 - Covariance***")
#print("Signal to Noise Ratios:")
#print(snr)
#print("Errors:")
#print(err)
#print("Inverse of Beta Y, Beta Z:")
#print(1/beta[1], 1/beta[2])
#
#y_beta_scaled = y * beta[1]
#z_beta_scaled = z * beta[2]
#
#y_ab_scaled = y_beta_scaled - np.mean(y_beta_scaled)
#z_ab_scaled = z_beta_scaled - np.mean(z_beta_scaled)
#
#print("R")
#R_xy = 1.0 / math.sqrt((1.0+(1.0/snr[0]))*(1.0+(1.0/snr[1])))
#R_yz = 1.0 / math.sqrt((1.0+(1.0/snr[1]))*(1.0+(1.0/snr[2])))
#R_xz = 1.0 / math.sqrt((1.0+(1.0/snr[0]))*(1.0+(1.0/snr[2])))
#print(R_xy, R_yz, R_xz)
#
#print("fRMSE")
#fRMSE_x = 1.0 / (1.0 + snr[0])
#fRMSE_y = 1.0 / (1.0 + snr[1])
#fRMSE_z = 1.0 / (1.0 + snr[2])
#print(fRMSE_x, fRMSE_y, fRMSE_z)
