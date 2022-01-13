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
import seaborn as sn
from calendar import isleap
from dateutil.relativedelta import *
from pathlib import Path
from matplotlib.ticker import (MultipleLocator, AutoLocator, AutoMinorLocator)
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from scipy.stats import pearsonr
from decimal import *


def is_empty(figure):
    """
    Return whether the figure contains no Artists (other than the default
    background patch).
    """
    contained_artists = figure.get_children()
    return len(contained_artists) <= 1

############################## grab sites and grid cells for each soil layer ##############################
geom_dir = "/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/"

olr = ['outliers','zscore','IQR']
lyr = ['0_9.9']
thr = ['0','25','50','75','100']
rmp_type = ['nn','bil']


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


CFSR_layer = "Soil_Temp_L1"
CFSR2_layer = "Soil_Temp_L1"
GLDAS_layer = "Soil_Temp_L1"
ERA5_layer = "Soil_Temp_L1"
ERAI_layer = "Soil_Temp_L1"
JRA_layer = "Soil_Temp"
MERRA2_layer = "Soil_Temp_L1"



####################### Raw Temperatures #################################
################# loop through in-situ files ###############
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
################################# create master arrays for in-situ, model data ######################################
    			date_master_all_DJF = []
    			CFSR_master_all_DJF = []
    			lat_master_all_DJF = []
    			lon_master_all_DJF = []
    			grid_master_all_DJF = []
    			station_master_all_DJF = []
    			ERAI_master_all_DJF = []
    			ERA5_master_all_DJF = []
    			JRA_master_all_DJF = []
    			MERRA2_master_all_DJF = []
    			GLDAS_master_all_DJF = []

    			date_master_all_MAM = []
    			CFSR_master_all_MAM = []
    			lat_master_all_MAM = []
    			lon_master_all_MAM = []
    			grid_master_all_MAM = []
    			station_master_all_MAM = []
    			ERAI_master_all_MAM = []
    			ERA5_master_all_MAM = []
    			JRA_master_all_MAM = []
    			MERRA2_master_all_MAM = []
    			GLDAS_master_all_MAM = []

    			date_master_all_JJA = []
    			CFSR_master_all_JJA = []
    			lat_master_all_JJA = []
    			lon_master_all_JJA = []
    			grid_master_all_JJA = []
    			station_master_all_JJA = []
    			ERAI_master_all_JJA = []
    			ERA5_master_all_JJA = []
    			JRA_master_all_JJA = []
    			MERRA2_master_all_JJA = []
    			GLDAS_master_all_JJA = []

    			date_master_all_SON = []
    			CFSR_master_all_SON = []
    			lat_master_all_SON = []
    			lon_master_all_SON = []
    			grid_master_all_SON = []
    			station_master_all_SON = []
    			ERAI_master_all_SON = []
    			ERA5_master_all_SON = []
    			JRA_master_all_SON = []
    			MERRA2_master_all_SON = []
    			GLDAS_master_all_SON = []

    			thrk = k
    			thrshld = "".join(["thr_"+str(thrk)])
    			indir = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/spatial_average/remap",rmph,"/no_outliers/",olri,"/",lyrj,"/thr_",thrk,"/"])
    			indira = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/spatial_average_anom/remap",rmph,"/no_outliers/",olri,"/",lyrj,"/thr_",thrk,"/"])
    			pathlist = Path(indir).glob('*.csv')
    			pathlista = Path(indira).glob('*anom.csv')
    			#print(indir)		

################################## loop through files within a threshold directory ##################################
    			for path in sorted(pathlista, key=lambda path: int(path.stem.split("_")[1].split("*.csv")[0])):
    				fil = str(path)
    				dframe = pd.read_csv(fil)
    				dat_mon = dframe['Date'].tolist()
    				date_mon = [datetime.datetime.strptime(x,'%Y-%m-%d') for x in dat_mon]
    				date_mon_CFSR = []
    				date_mon_CFSR2 = []				
    				for i in range(0,len(date_mon)):
    					date_mon_i = date_mon[i]
    					if (date_mon_i <= datetime.datetime(2010,12,31)):
    						date_mon_CFSR.append(date_mon_i)
    					elif (date_mon_i >= datetime.datetime(2011,1,1)):
    						date_mon_CFSR2.append(date_mon_i)
    				lat_cen = dframe['Central Lat'].iloc[0]
    				lon_cen = dframe['Central Lon'].iloc[0]
    				gcell = dframe['Grid Cell'].iloc[0]
    				stemp = dframe['Spatial Avg Anom'].tolist()
    				#print(date_mon)
    				#print(date_mon_CFSR) 
    				
################################## grab corresponding reanalysis data ##################################
    				base_dir  = "".join(["/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/common_grid/remap",rmph,"/grid_level/"])
    				CFSR_fi = "".join([base_dir,"CFSR/CFSR_grid_",str(gcell),".nc"])
    				CFSR2_fi = "".join([base_dir,"CFSR2/CFSR2_grid_",str(gcell),".nc"])
    				MERRA2_fi = "".join([base_dir,"MERRA2/MERRA2_grid_",str(gcell),".nc"])
    				ERA5_fi = "".join([base_dir,"ERA5/ERA5_grid_",str(gcell),".nc"])
    				ERAI_fi = "".join([base_dir,"ERA-Interim/ERA-Interim_grid_",str(gcell),".nc"])
    				JRA_fi = "".join([base_dir,"JRA55/JRA55_grid_",str(gcell),".nc"])
    				GLDAS_fi = "".join([base_dir,"GLDAS/GLDAS_grid_",str(gcell),".nc"])
    				#print(CFSR_fi)

    				GLDAS_fil = xr.open_dataset(GLDAS_fi)
    				JRA_fil = xr.open_dataset(JRA_fi)
    				ERAI_fil = xr.open_dataset(ERAI_fi)
    				ERA5_fil = xr.open_dataset(ERA5_fi)
    				MERRA2_fil = xr.open_dataset(MERRA2_fi)
    				CFSR_fil = xr.open_dataset(CFSR_fi) #open NetCDF file with xarray
    				CFSR2_fil = xr.open_dataset(CFSR2_fi) #open NetCDF file with xarray

########### extract soil temperatures and convert to celsius #######
    				GLDAS_stemp = GLDAS_fil[GLDAS_layer] -273.15
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
    				#print("GLDAS filename:",GLDAS_fi)
    				#print("GLDAS Temp Values:",GLDAS_stemp)
				
###################### extract values of other reanalysis products #######################
    				grid_new_all = []
    				CFSR_new_all = []
    				date_new_all = [] #if there is no CFSR or CFSR2 in triplet
    				station_new_all = [] #if there is no CFSR or CFSR2 in triplet    				
    				GLDAS_new_all = [] #if there is no CFSR or CFSR2 in triplet  
    				JRA_new_all = [] #if there is no CFSR or CFSR2 in triplet 
    				ERAI_new_all = [] #if there is no CFSR or CFSR2 in triplet 
    				ERA5_new_all = [] #if there is no CFSR or CFSR2 in triplet 
    				MERRA2_new_all = [] #if there is no CFSR or CFSR2 in triplet 				

    				for i in range(0,len(date_mon)):
    					date_mon_i = date_mon[i]
    					stemp_i = stemp[i]
    					#print(date_mon_i)			    					
    					grid_new_all.append(gcell)
					
    					if (date_mon_i <= datetime.datetime(2010,12,31)):
    						CFSR_stemp_i = CFSR_stemp3.sel(time=date_mon_i, drop=True)
    						CFSR_new_all.append(CFSR_stemp_i)
    						date_new_all.append(date_mon_i)
    						station_new_all.append(stemp_i)
    						JRA_stemp_i = JRA_stemp3.sel(time=date_mon_i, drop=True)
    						JRA_new_all.append(JRA_stemp_i)
    						GLDAS_stemp_i = GLDAS_stemp3.sel(time=date_mon_i, drop=True)
    						GLDAS_new_all.append(GLDAS_stemp_i)
    						ERA5_stemp_i = ERA5_stemp3.sel(time=date_mon_i, drop=True)
    						ERA5_new_all.append(ERA5_stemp_i)
    						ERAI_stemp_i = ERAI_stemp3.sel(time=date_mon_i, drop=True)
    						ERAI_new_all.append(ERAI_stemp_i)
    						MERRA2_stemp_i = MERRA2_stemp3.sel(time=date_mon_i, drop=True)
    						MERRA2_new_all.append(MERRA2_stemp_i)						
    						    						
    					elif (date_mon_i >= datetime.datetime(2011,1,1)):
    						date_new_all.append(date_mon_i)
    						station_new_all.append(stemp_i)
    						if (date_mon_i <= datetime.datetime(2019,9,30)):
    							CFSR2_stemp_i = CFSR2_stemp3.sel(time=date_mon_i, drop=True)
    							CFSR_new_all.append(CFSR2_stemp_i)
    						elif (date_mon_i >= datetime.datetime(2019,10,1)):
    							CFSR_new_all.append(np.nan)
    						if (date_mon_i <= datetime.datetime(2019,12,31)):
    							JRA_stemp_i = JRA_stemp3.sel(time=date_mon_i, drop=True)
    							JRA_new_all.append(JRA_stemp_i)
    						elif (date_mon_i >= datetime.datetime(2020,1,1)):
    							JRA_stemp_i = JRA_stemp3.sel(time=date_mon_i, drop=True)
    							JRA_new_all.append(np.nan)
    						if (date_mon_i <= datetime.datetime(2020,7,31)):
    							GLDAS_stemp_i = GLDAS_stemp3.sel(time=date_mon_i, drop=True)
    							GLDAS_new_all.append(GLDAS_stemp_i)
    						elif (date_mon_i >= datetime.datetime(2020,8,1)):
    							GLDAS_new_all.append(np.nan)
    						if (date_mon_i <= datetime.datetime(2018,12,31)):
    							ERA5_stemp_i = ERA5_stemp3.sel(time=date_mon_i, drop=True)
    							ERA5_new_all.append(ERA5_stemp_i)
    						elif (date_mon_i >= datetime.datetime(2019,1,1)):
    							ERA5_new_all.append(np.nan)
    						if (date_mon_i <= datetime.datetime(2019,8,31)):
    							ERAI_stemp_i = ERAI_stemp3.sel(time=date_mon_i, drop=True)
    							ERAI_new_all.append(ERAI_stemp_i)
    						elif (date_mon_i >= datetime.datetime(2019,9,1)):
    							ERAI_new_all.append(np.nan)
    						if (date_mon_i <= datetime.datetime(2020,7,31)):
    							MERRA2_stemp_i = MERRA2_stemp3.sel(time=date_mon_i, drop=True)
    							MERRA2_new_all.append(MERRA2_stemp_i)
    						elif (date_mon_i >= datetime.datetime(2020,8,1)):
    							MERRA2_new_all.append(np.nan) 

    				grid_new_all2 = np.array(grid_new_all).flatten()
    				date_new_all2 = np.array(date_new_all).flatten()
    				station_new_all2 = np.array(station_new_all).flatten()
    				GLDAS_new_all2 = np.array(GLDAS_new_all).flatten()
    				JRA_new_all2 = np.array(JRA_new_all).flatten()
    				ERAI_new_all2 = np.array(ERAI_new_all).flatten()
    				ERA5_new_all2 = np.array(ERA5_new_all).flatten()
    				MERRA2_new_all2 = np.array(MERRA2_new_all).flatten()
    				CFSR_new_all2 = np.array(CFSR_new_all).flatten()				


#################### create anomalies for reanalysis files #######################
    				rnysis_anom_master = []
    				rnysis_date_master = []
    				rnysis_name_master = []
    				rnysis_stemp_master = []
    				rnysis = [CFSR_new_all2,ERAI_new_all2,ERA5_new_all2,JRA_new_all2,MERRA2_new_all2,GLDAS_new_all2]
    				rnysis_name = ['CFSR','ERA-Interim','ERA-5','JRA-55','MERRA2','GLDAS']
    				dat_rowlist = date_new_all2
    				num_rows = len(dat_rowlist)
    				for i in range(0,6):
    					rnysisi = rnysis[i]
    					rnysis_namei = rnysis_name[i]
    					#print("Reanalysis Product:",rnysis_namei)
    					#print(rnysisi)
    					climatology = dict()
    					clim_averages = dict()
    					stemp_mstr = []
    					stemp_anom_master = []
    					date_mstr = []
    					name_mstr = []
    					for month in range(1,13):
    						month_key = f"{month:02}"
    						climatology[month_key] = list()

    					for j in range(0,num_rows):
					###add month data to list based on key
    						dat_row = dat_rowlist[j]
    						stemp_row = rnysisi[j]
    						month_key = dat_row.strftime("%m")
    						climatology[month_key].append(stemp_row)

    					climatology_keys = list(climatology.keys())
    					climatology_keys2 = np.array(climatology_keys).flatten()
    					#print(climatology)
					
    					for key in climatology:
					###take averages and write to averages dictionary
    						current_total = 0
    						len_current_list = 0
    						current_list = climatology[key]
    						for temp in current_list:
    							if (temp == np.nan):
    								current_total = current_total + 0
    								len_current_list = len_current_list + 0
    							else:
    								current_total = current_total + temp
    								len_current_list = len_current_list + 1
    						if (len_current_list == 0):
    							average = np.nan
    						else:
    							average = current_total/len_current_list
    						clim_averages[key] = average
    						#print(average)
							
    					clim_avg = list(clim_averages.values())
    					#print(clim_averages)
						
    					for k in range (0, num_rows):
    						stemp_rw = rnysisi[k]
    						dat_row = dat_rowlist[k]
    						dat_row_mon = dat_row.month
    						dat_row_mons = f"{dat_row_mon:02}"
    						#print(stemp_rw,dat_row_mon,clim_averages[dat_row_mons])
    						stemp_anom = stemp_rw - clim_averages[dat_row_mons]

    						rnysis_anom_master.append(stemp_anom)
    						rnysis_date_master.append(dat_row)					
    						rnysis_name_master.append(rnysis_namei)
    						rnysis_stemp_master.append(stemp_rw)


###################### create anom dataframe ########################

    				dframe_anom_master = pd.DataFrame(data=rnysis_date_master, columns=['Date'])
    				dframe_anom_master['Soil Temp'] = rnysis_stemp_master
    				dframe_anom_master['Soil Temp Anom'] = rnysis_anom_master
    				dframe_anom_master['Reanalysis Product'] = rnysis_name_master
    				#print(dframe_anom_master)					

    				dframe_anom_CFSR = dframe_anom_master[dframe_anom_master['Reanalysis Product'] == 'CFSR']
    				dframe_anom_CFSR = dframe_anom_CFSR['Soil Temp Anom'].values.tolist()
    				dframe_anom_GLDAS = dframe_anom_master[dframe_anom_master['Reanalysis Product'] == 'GLDAS']
    				dframe_anom_GLDAS = dframe_anom_GLDAS['Soil Temp Anom'].values.tolist()
    				dframe_anom_ERAI = dframe_anom_master[dframe_anom_master['Reanalysis Product'] == 'ERA-Interim']
    				dframe_anom_ERAI = dframe_anom_ERAI['Soil Temp Anom'].values.tolist()
    				dframe_anom_ERA5 = dframe_anom_master[dframe_anom_master['Reanalysis Product'] == 'ERA-5']
    				dframe_anom_ERA5 = dframe_anom_ERA5['Soil Temp Anom'].values.tolist()
    				dframe_anom_JRA = dframe_anom_master[dframe_anom_master['Reanalysis Product'] == 'JRA-55']
    				dframe_anom_JRA = dframe_anom_JRA['Soil Temp Anom'].values.tolist()
    				dframe_anom_MERRA2 = dframe_anom_master[dframe_anom_master['Reanalysis Product'] == 'MERRA2']
    				dframe_anom_MERRA2 = dframe_anom_MERRA2['Soil Temp Anom'].values.tolist()
##################### create new dataframe with date, station temp, reanalysis temp ###################

    				dframe_all = pd.DataFrame(data=date_new_all2, columns=['Date'])
    				dframe_all['Grid Cell'] = grid_new_all2
    				dframe_all['Lat Cen'] = lat_cen
    				dframe_all['Lon Cen'] = lon_cen
    				dframe_all['Station'] = station_new_all2
    				dframe_all['CFSR'] = dframe_anom_CFSR
    				dframe_all['GLDAS'] = dframe_anom_GLDAS
    				dframe_all['JRA55'] = dframe_anom_JRA
    				dframe_all['ERA5'] = dframe_anom_ERA5
    				dframe_all['ERA-Interim'] = dframe_anom_ERAI
    				dframe_all['MERRA2'] = dframe_anom_MERRA2
    				dframe_all['DateTime'] = pd.to_datetime(dframe_all['Date'])


############# drop rows with NaN ############
    				dframe_all = dframe_all[dframe_all['CFSR'].notna()]
    				dframe_all = dframe_all[dframe_all['GLDAS'].notna()]
    				dframe_all = dframe_all[dframe_all['JRA55'].notna()]
    				dframe_all = dframe_all[dframe_all['ERA5'].notna()]
    				dframe_all = dframe_all[dframe_all['ERA-Interim'].notna()]
    				dframe_all = dframe_all[dframe_all['MERRA2'].notna()]

    				dframe_all = dframe_all.set_index('DateTime')


    				#print(dframe_all)

############# select by season #############
#### Winter (DJF)

    				dframe_DJF = dframe_all[(dframe_all.index.month == 12) | (dframe_all.index.month == 1) | (dframe_all.index.month == 2)]
    				#print(dframe_DJF)

#### Spring (MAM)

    				dframe_MAM = dframe_all[(dframe_all.index.month == 3) | (dframe_all.index.month == 4) | (dframe_all.index.month == 5)]
    				#print(dframe_MAM)

#### Summer (JJA)

    				dframe_JJA = dframe_all[(dframe_all.index.month == 6) | (dframe_all.index.month == 7) | (dframe_all.index.month == 8)]
    				#print(dframe_JJA)

#### Autumn (SON)

    				dframe_SON = dframe_all[(dframe_all.index.month == 9) | (dframe_all.index.month == 10) | (dframe_all.index.month == 11)]
    				#print(dframe_SON)

################################# append values to master arrays ######################################

############## Winter (DJF) ################
    				date_final = dframe_DJF['Date']
    				if (date_final.empty == False):
    					date_master_all_DJF.append(date_final.tolist())

    				lat_final = dframe_DJF['Lat Cen']
    				if (lat_final.empty == False):
    					lat_master_all_DJF.append(lat_final.tolist())
					
    				lon_final = dframe_DJF['Lon Cen']
    				if (lon_final.empty == False):
    					lon_master_all_DJF.append(lon_final.tolist())

    				station_final = dframe_DJF['Station']
    				if (station_final.empty == False):
    					station_master_all_DJF.append(station_final.tolist())
    	
    				grid_final = dframe_DJF['Grid Cell']
    				if (grid_final.empty == False):
    					grid_master_all_DJF.append(grid_final.values.tolist())

    				CFSR_final = dframe_DJF['CFSR']
    				if (CFSR_final.empty == False):
    					CFSR_master_all_DJF.append(CFSR_final.values.tolist())
    
    				ERAI_final = dframe_DJF['ERA-Interim']
    				if (ERAI_final.empty == False):
    					ERAI_master_all_DJF.append(ERAI_final.values.tolist())
    
    				ERA5_final = dframe_DJF['ERA5']
    				if (ERA5_final.empty == False):
    					ERA5_master_all_DJF.append(ERA5_final.values.tolist())
    
    				MERRA2_final = dframe_DJF['MERRA2']
    				if (MERRA2_final.empty == False):
    					MERRA2_master_all_DJF.append(MERRA2_final.values.tolist())
    
    				JRA_final = dframe_DJF['JRA55']
    				if (JRA_final.empty == False):
    					JRA_master_all_DJF.append(JRA_final.values.tolist())
    
    				GLDAS_final = dframe_DJF['GLDAS']
    				if (GLDAS_final.empty == False):	
    					GLDAS_master_all_DJF.append(GLDAS_final.values.tolist())

############## Spring (MAM) ################

    				date_final = dframe_MAM['Date']
    				if (date_final.empty == False):
    					date_master_all_MAM.append(date_final.tolist())

    				lat_final = dframe_MAM['Lat Cen']
    				if (lat_final.empty == False):
    					lat_master_all_MAM.append(lat_final.tolist())
					
    				lon_final = dframe_MAM['Lon Cen']
    				if (lon_final.empty == False):
    					lon_master_all_MAM.append(lon_final.tolist())

    				station_final = dframe_MAM['Station']
    				if (station_final.empty == False):
    					station_master_all_MAM.append(station_final.tolist())
    	
    				grid_final = dframe_MAM['Grid Cell']
    				if (grid_final.empty == False):
    					grid_master_all_MAM.append(grid_final.values.tolist())

    				CFSR_final = dframe_MAM['CFSR']
    				if (CFSR_final.empty == False):
    					CFSR_master_all_MAM.append(CFSR_final.values.tolist())
    
    				ERAI_final = dframe_MAM['ERA-Interim']
    				if (ERAI_final.empty == False):
    					ERAI_master_all_MAM.append(ERAI_final.values.tolist())
    
    				ERA5_final = dframe_MAM['ERA5']
    				if (ERA5_final.empty == False):
    					ERA5_master_all_MAM.append(ERA5_final.values.tolist())
    
    				MERRA2_final = dframe_MAM['MERRA2']
    				if (MERRA2_final.empty == False):
    					MERRA2_master_all_MAM.append(MERRA2_final.values.tolist())
    
    				JRA_final = dframe_MAM['JRA55']
    				if (JRA_final.empty == False):
    					JRA_master_all_MAM.append(JRA_final.values.tolist())
    
    				GLDAS_final = dframe_MAM['GLDAS']
    				if (GLDAS_final.empty == False):	
    					GLDAS_master_all_MAM.append(GLDAS_final.values.tolist())


################ Summer (JJA) ################

    				date_final = dframe_JJA['Date']
    				if (date_final.empty == False):
    					date_master_all_JJA.append(date_final.tolist())

    				lat_final = dframe_JJA['Lat Cen']
    				if (lat_final.empty == False):
    					lat_master_all_JJA.append(lat_final.tolist())
					
    				lon_final = dframe_JJA['Lon Cen']
    				if (lon_final.empty == False):
    					lon_master_all_JJA.append(lon_final.tolist())

    				station_final = dframe_JJA['Station']
    				if (station_final.empty == False):
    					station_master_all_JJA.append(station_final.tolist())
    	
    				grid_final = dframe_JJA['Grid Cell']
    				if (grid_final.empty == False):
    					grid_master_all_JJA.append(grid_final.values.tolist())

    				CFSR_final = dframe_JJA['CFSR']
    				if (CFSR_final.empty == False):
    					CFSR_master_all_JJA.append(CFSR_final.values.tolist())
    
    				ERAI_final = dframe_JJA['ERA-Interim']
    				if (ERAI_final.empty == False):
    					ERAI_master_all_JJA.append(ERAI_final.values.tolist())
    
    				ERA5_final = dframe_JJA['ERA5']
    				if (ERA5_final.empty == False):
    					ERA5_master_all_JJA.append(ERA5_final.values.tolist())
    
    				MERRA2_final = dframe_JJA['MERRA2']
    				if (MERRA2_final.empty == False):
    					MERRA2_master_all_JJA.append(MERRA2_final.values.tolist())
    
    				JRA_final = dframe_JJA['JRA55']
    				if (JRA_final.empty == False):
    					JRA_master_all_JJA.append(JRA_final.values.tolist())
    
    				GLDAS_final = dframe_JJA['GLDAS']
    				if (GLDAS_final.empty == False):	
    					GLDAS_master_all_JJA.append(GLDAS_final.values.tolist())

################ Autumn (SON) ################

    				date_final = dframe_SON['Date']
    				if (date_final.empty == False):
    					date_master_all_SON.append(date_final.tolist())

    				lat_final = dframe_SON['Lat Cen']
    				if (lat_final.empty == False):
    					lat_master_all_SON.append(lat_final.tolist())
					
    				lon_final = dframe_SON['Lon Cen']
    				if (lon_final.empty == False):
    					lon_master_all_SON.append(lon_final.tolist())

    				station_final = dframe_SON['Station']
    				if (station_final.empty == False):
    					station_master_all_SON.append(station_final.tolist())
    	
    				grid_final = dframe_SON['Grid Cell']
    				if (grid_final.empty == False):
    					grid_master_all_SON.append(grid_final.values.tolist())

    				CFSR_final = dframe_SON['CFSR']
    				if (CFSR_final.empty == False):
    					CFSR_master_all_SON.append(CFSR_final.values.tolist())
    
    				ERAI_final = dframe_SON['ERA-Interim']
    				if (ERAI_final.empty == False):
    					ERAI_master_all_SON.append(ERAI_final.values.tolist())
    
    				ERA5_final = dframe_SON['ERA5']
    				if (ERA5_final.empty == False):
    					ERA5_master_all_SON.append(ERA5_final.values.tolist())
    
    				MERRA2_final = dframe_SON['MERRA2']
    				if (MERRA2_final.empty == False):
    					MERRA2_master_all_SON.append(MERRA2_final.values.tolist())
    
    				JRA_final = dframe_SON['JRA55']
    				if (JRA_final.empty == False):
    					JRA_master_all_SON.append(JRA_final.values.tolist())
    
    				GLDAS_final = dframe_SON['GLDAS']
    				if (GLDAS_final.empty == False):	
    					GLDAS_master_all_SON.append(GLDAS_final.values.tolist())

######################### Flatten Master Lists to 1D ############################


################## Winter (DJF) ####################

    			date_master_all_1D_DJF = []
    			for sublist in date_master_all_DJF:
    				for item in sublist:
    					date_master_all_1D_DJF.append(item)

    			lat_master_all_1D_DJF = []
    			for sublist in lat_master_all_DJF:
    				for item in sublist:
    					lat_master_all_1D_DJF.append(item)

    			lon_master_all_1D_DJF = []
    			for sublist in lon_master_all_DJF:
    				for item in sublist:
    					lon_master_all_1D_DJF.append(item)

    			grid_master_all_1D_DJF = []
    			for sublist in grid_master_all_DJF:
    				for item in sublist:
    					grid_master_all_1D_DJF.append(item)
				
    			station_master_all_1D_DJF = []
    			for sublist in station_master_all_DJF:
    				for item in sublist:
    					station_master_all_1D_DJF.append(item)
				
    			CFSR_master_all_1D_DJF = []
    			for sublist in CFSR_master_all_DJF:
    				for item in sublist:
    					CFSR_master_all_1D_DJF.append(item)
									
    			ERAI_master_all_1D_DJF = []
    			for sublist in ERAI_master_all_DJF:
    				for item in sublist:
    					ERAI_master_all_1D_DJF.append(item)
				
    			ERA5_master_all_1D_DJF = []
    			for sublist in ERA5_master_all_DJF:
    				for item in sublist:
    					ERA5_master_all_1D_DJF.append(item)
				
    			JRA_master_all_1D_DJF = []
    			for sublist in JRA_master_all_DJF:
    				for item in sublist:
    					JRA_master_all_1D_DJF.append(item)

    			MERRA2_master_all_1D_DJF = []
    			for sublist in MERRA2_master_all_DJF:
    				for item in sublist:
    					MERRA2_master_all_1D_DJF.append(item)

    			GLDAS_master_all_1D_DJF = []
    			for sublist in GLDAS_master_all_DJF:
    				for item in sublist:
    					GLDAS_master_all_1D_DJF.append(item)


################## Spring (MAM) ####################

    			date_master_all_1D_MAM = []
    			for sublist in date_master_all_MAM:
    				for item in sublist:
    					date_master_all_1D_MAM.append(item)

    			lat_master_all_1D_MAM = []
    			for sublist in lat_master_all_MAM:
    				for item in sublist:
    					lat_master_all_1D_MAM.append(item)

    			lon_master_all_1D_MAM = []
    			for sublist in lon_master_all_MAM:
    				for item in sublist:
    					lon_master_all_1D_MAM.append(item)

    			grid_master_all_1D_MAM = []
    			for sublist in grid_master_all_MAM:
    				for item in sublist:
    					grid_master_all_1D_MAM.append(item)
				
    			station_master_all_1D_MAM = []
    			for sublist in station_master_all_MAM:
    				for item in sublist:
    					station_master_all_1D_MAM.append(item)
				
    			CFSR_master_all_1D_MAM = []
    			for sublist in CFSR_master_all_MAM:
    				for item in sublist:
    					CFSR_master_all_1D_MAM.append(item)
									
    			ERAI_master_all_1D_MAM = []
    			for sublist in ERAI_master_all_MAM:
    				for item in sublist:
    					ERAI_master_all_1D_MAM.append(item)
				
    			ERA5_master_all_1D_MAM = []
    			for sublist in ERA5_master_all_MAM:
    				for item in sublist:
    					ERA5_master_all_1D_MAM.append(item)
				
    			JRA_master_all_1D_MAM = []
    			for sublist in JRA_master_all_MAM:
    				for item in sublist:
    					JRA_master_all_1D_MAM.append(item)

    			MERRA2_master_all_1D_MAM = []
    			for sublist in MERRA2_master_all_MAM:
    				for item in sublist:
    					MERRA2_master_all_1D_MAM.append(item)

    			GLDAS_master_all_1D_MAM = []
    			for sublist in GLDAS_master_all_MAM:
    				for item in sublist:
    					GLDAS_master_all_1D_MAM.append(item)

################## Summer (JJA) ####################

    			date_master_all_1D_JJA = []
    			for sublist in date_master_all_JJA:
    				for item in sublist:
    					date_master_all_1D_JJA.append(item)

    			lat_master_all_1D_JJA = []
    			for sublist in lat_master_all_JJA:
    				for item in sublist:
    					lat_master_all_1D_JJA.append(item)

    			lon_master_all_1D_JJA = []
    			for sublist in lon_master_all_JJA:
    				for item in sublist:
    					lon_master_all_1D_JJA.append(item)

    			grid_master_all_1D_JJA = []
    			for sublist in grid_master_all_JJA:
    				for item in sublist:
    					grid_master_all_1D_JJA.append(item)
				
    			station_master_all_1D_JJA = []
    			for sublist in station_master_all_JJA:
    				for item in sublist:
    					station_master_all_1D_JJA.append(item)
				
    			CFSR_master_all_1D_JJA = []
    			for sublist in CFSR_master_all_JJA:
    				for item in sublist:
    					CFSR_master_all_1D_JJA.append(item)
									
    			ERAI_master_all_1D_JJA = []
    			for sublist in ERAI_master_all_JJA:
    				for item in sublist:
    					ERAI_master_all_1D_JJA.append(item)
				
    			ERA5_master_all_1D_JJA = []
    			for sublist in ERA5_master_all_JJA:
    				for item in sublist:
    					ERA5_master_all_1D_JJA.append(item)
				
    			JRA_master_all_1D_JJA = []
    			for sublist in JRA_master_all_JJA:
    				for item in sublist:
    					JRA_master_all_1D_JJA.append(item)

    			MERRA2_master_all_1D_JJA = []
    			for sublist in MERRA2_master_all_JJA:
    				for item in sublist:
    					MERRA2_master_all_1D_JJA.append(item)

    			GLDAS_master_all_1D_JJA = []
    			for sublist in GLDAS_master_all_JJA:
    				for item in sublist:
    					GLDAS_master_all_1D_JJA.append(item)

################## Autumn (SON) ####################

    			date_master_all_1D_SON = []
    			for sublist in date_master_all_SON:
    				for item in sublist:
    					date_master_all_1D_SON.append(item)

    			lat_master_all_1D_SON = []
    			for sublist in lat_master_all_SON:
    				for item in sublist:
    					lat_master_all_1D_SON.append(item)

    			lon_master_all_1D_SON = []
    			for sublist in lon_master_all_SON:
    				for item in sublist:
    					lon_master_all_1D_SON.append(item)

    			grid_master_all_1D_SON = []
    			for sublist in grid_master_all_SON:
    				for item in sublist:
    					grid_master_all_1D_SON.append(item)
				
    			station_master_all_1D_SON = []
    			for sublist in station_master_all_SON:
    				for item in sublist:
    					station_master_all_1D_SON.append(item)
				
    			CFSR_master_all_1D_SON = []
    			for sublist in CFSR_master_all_SON:
    				for item in sublist:
    					CFSR_master_all_1D_SON.append(item)
									
    			ERAI_master_all_1D_SON = []
    			for sublist in ERAI_master_all_SON:
    				for item in sublist:
    					ERAI_master_all_1D_SON.append(item)
				
    			ERA5_master_all_1D_SON = []
    			for sublist in ERA5_master_all_SON:
    				for item in sublist:
    					ERA5_master_all_1D_SON.append(item)
				
    			JRA_master_all_1D_SON = []
    			for sublist in JRA_master_all_SON:
    				for item in sublist:
    					JRA_master_all_1D_SON.append(item)

    			MERRA2_master_all_1D_SON = []
    			for sublist in MERRA2_master_all_SON:
    				for item in sublist:
    					MERRA2_master_all_1D_SON.append(item)

    			GLDAS_master_all_1D_SON = []
    			for sublist in GLDAS_master_all_SON:
    				for item in sublist:
    					GLDAS_master_all_1D_SON.append(item)


######### Create Finalized Seasonal Dataframes ###########
########## Winter (DJF) ##############
    			dframe_master_DJF = pd.DataFrame(data=station_master_all_1D_DJF, columns=['Station'])
    			dframe_master_DJF['CFSR'] = CFSR_master_all_1D_DJF
    			dframe_master_DJF['ERA-Interim'] = ERAI_master_all_1D_DJF
    			dframe_master_DJF['ERA5'] = ERA5_master_all_1D_DJF
    			dframe_master_DJF['JRA-55'] = JRA_master_all_1D_DJF
    			dframe_master_DJF['MERRA2'] = MERRA2_master_all_1D_DJF
    			dframe_master_DJF['GLDAS'] = GLDAS_master_all_1D_DJF

########## Spring (MAM) ##############
    			dframe_master_MAM = pd.DataFrame(data=station_master_all_1D_MAM, columns=['Station'])
    			dframe_master_MAM['CFSR'] = CFSR_master_all_1D_MAM
    			dframe_master_MAM['ERA-Interim'] = ERAI_master_all_1D_MAM
    			dframe_master_MAM['ERA5'] = ERA5_master_all_1D_MAM
    			dframe_master_MAM['JRA-55'] = JRA_master_all_1D_MAM
    			dframe_master_MAM['MERRA2'] = MERRA2_master_all_1D_MAM
    			dframe_master_MAM['GLDAS'] = GLDAS_master_all_1D_MAM

########## Summer (JJA) ##############
    			dframe_master_JJA = pd.DataFrame(data=station_master_all_1D_JJA, columns=['Station'])
    			dframe_master_JJA['CFSR'] = CFSR_master_all_1D_JJA
    			dframe_master_JJA['ERA-Interim'] = ERAI_master_all_1D_JJA
    			dframe_master_JJA['ERA5'] = ERA5_master_all_1D_JJA
    			dframe_master_JJA['JRA-55'] = JRA_master_all_1D_JJA
    			dframe_master_JJA['MERRA2'] = MERRA2_master_all_1D_JJA
    			dframe_master_JJA['GLDAS'] = GLDAS_master_all_1D_JJA

########## Autumn (SON) ##############
    			dframe_master_SON = pd.DataFrame(data=station_master_all_1D_SON, columns=['Station'])
    			dframe_master_SON['CFSR'] = CFSR_master_all_1D_SON
    			dframe_master_SON['ERA-Interim'] = ERAI_master_all_1D_SON
    			dframe_master_SON['ERA5'] = ERA5_master_all_1D_SON
    			dframe_master_SON['JRA-55'] = JRA_master_all_1D_SON
    			dframe_master_SON['MERRA2'] = MERRA2_master_all_1D_SON
    			dframe_master_SON['GLDAS'] = GLDAS_master_all_1D_SON



################## Create Boxplots ####################
    			fig, axes = plt.subplots(2,2, figsize=(20,20), sharey=True)
    			fig.suptitle('Seasonal Boxplots of Soil Temperature Anomalies')
    			bDJF = sns.boxplot(ax=axes[0,0], data=dframe_master_DJF).set_title('DJF Boxplot')
    			bMAM = sns.boxplot(ax=axes[0,1], data=dframe_master_MAM).set_title('MAM Boxplot')
    			bJJA = sns.boxplot(ax=axes[1,0], data=dframe_master_JJA).set_title('JJA Boxplot')
    			bSON = sns.boxplot(ax=axes[1,1], data=dframe_master_SON).set_title('SON Boxplot')
    			fig.add_subplot(111, frameon=False) #create large subplot that will include labels
    			plt.tick_params(labelcolor='none', bottom=False,left=False) #hide ticks
    			plt.ylabel('Soil Temp Anomaly($^\circ$ C)',fontweight='bold')
    			bplot_fil = "".join(["/mnt/data/users/herringtont/soil_temp/plots/boxplots/"+str(remap_type)+"_"+str(olri)+"_"+lyrj+"_thr"+str(thrk)+"_boxplot_anom.png"])
    			print(bplot_fil)
    			plt.savefig(bplot_fil)
    			plt.close()
