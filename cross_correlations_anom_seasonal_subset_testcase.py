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
import pathlib
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
    			indir = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/spatial_average/subset_multiple/remap",rmph,"/",olri,"/",lyrj,"/thr_",thrk,"/"])
    			indira = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/spatial_average_anom/subset_multiple/remap",rmph,"/",olri,"/",lyrj,"/thr_",thrk,"/"])
    			pathlist = Path(indir).glob('*.csv')
    			pathlista = Path(indira).glob('*anom.csv')
    			#print(indir)		

################################## loop through files within a threshold directory ##################################
    			for path in sorted(pathlista, key=lambda path: int(path.stem.split("_")[1].split("*.csv")[0])):
    				fil = str(path)
    				dframe = pd.read_csv(fil)
    				#print(dframe)
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
    				col_nam = dframe.columns
    				sit_num = [s for s in col_nam if s.isdigit()] ######## check which elements of list are digits (as these are the site numbers)
    				site_1 = str(sit_num[1])
    				stemp = dframe[site_1].tolist()
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
    				dframe_all = dframe_all[dframe_all['Station'].notna()]
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


    			dframe_master_DJF = dframe_master_DJF.dropna()
    			dframe_master_MAM = dframe_master_MAM.dropna()
    			dframe_master_JJA = dframe_master_JJA.dropna()
    			dframe_master_SON = dframe_master_SON.dropna()


################### create arrays for correlations and scatterplots ####################

    			data_array_DJF = [[station_master_all_1D_DJF,station_master_all_1D_DJF],[station_master_all_1D_DJF,CFSR_master_all_1D_DJF],[station_master_all_1D_DJF,ERAI_master_all_1D_DJF],[station_master_all_1D_DJF,ERA5_master_all_1D_DJF],[station_master_all_1D_DJF,JRA_master_all_1D_DJF],[station_master_all_1D_DJF,MERRA2_master_all_1D_DJF],[station_master_all_1D_DJF,GLDAS_master_all_1D_DJF],[CFSR_master_all_1D_DJF,station_master_all_1D_DJF],[CFSR_master_all_1D_DJF,CFSR_master_all_1D_DJF],[CFSR_master_all_1D_DJF,ERAI_master_all_1D_DJF],[CFSR_master_all_1D_DJF,ERA5_master_all_1D_DJF],[CFSR_master_all_1D_DJF,JRA_master_all_1D_DJF],[CFSR_master_all_1D_DJF,MERRA2_master_all_1D_DJF],[CFSR_master_all_1D_DJF,GLDAS_master_all_1D_DJF],[ERAI_master_all_1D_DJF,station_master_all_1D_DJF],[ERAI_master_all_1D_DJF,CFSR_master_all_1D_DJF],[ERAI_master_all_1D_DJF,ERAI_master_all_1D_DJF],[ERAI_master_all_1D_DJF,ERA5_master_all_1D_DJF],[ERAI_master_all_1D_DJF,JRA_master_all_1D_DJF],[ERAI_master_all_1D_DJF,MERRA2_master_all_1D_DJF],[ERAI_master_all_1D_DJF,GLDAS_master_all_1D_DJF],[ERA5_master_all_1D_DJF,station_master_all_1D_DJF],[ERA5_master_all_1D_DJF,CFSR_master_all_1D_DJF],[ERA5_master_all_1D_DJF,ERAI_master_all_1D_DJF],[ERA5_master_all_1D_DJF,ERA5_master_all_1D_DJF],[ERA5_master_all_1D_DJF,JRA_master_all_1D_DJF],[ERA5_master_all_1D_DJF,MERRA2_master_all_1D_DJF],[ERA5_master_all_1D_DJF,GLDAS_master_all_1D_DJF],[JRA_master_all_1D_DJF,station_master_all_1D_DJF],[JRA_master_all_1D_DJF,CFSR_master_all_1D_DJF],[JRA_master_all_1D_DJF,ERAI_master_all_1D_DJF],[JRA_master_all_1D_DJF,ERA5_master_all_1D_DJF],[JRA_master_all_1D_DJF,JRA_master_all_1D_DJF],[JRA_master_all_1D_DJF,MERRA2_master_all_1D_DJF],[JRA_master_all_1D_DJF,GLDAS_master_all_1D_DJF],[MERRA2_master_all_1D_DJF,station_master_all_1D_DJF],[MERRA2_master_all_1D_DJF,CFSR_master_all_1D_DJF],[MERRA2_master_all_1D_DJF,ERAI_master_all_1D_DJF],[MERRA2_master_all_1D_DJF,ERA5_master_all_1D_DJF],[MERRA2_master_all_1D_DJF,JRA_master_all_1D_DJF],[MERRA2_master_all_1D_DJF,MERRA2_master_all_1D_DJF],[MERRA2_master_all_1D_DJF,GLDAS_master_all_1D_DJF],[GLDAS_master_all_1D_DJF,station_master_all_1D_DJF],[GLDAS_master_all_1D_DJF,CFSR_master_all_1D_DJF],[GLDAS_master_all_1D_DJF,ERAI_master_all_1D_DJF],[GLDAS_master_all_1D_DJF,ERA5_master_all_1D_DJF],[GLDAS_master_all_1D_DJF,JRA_master_all_1D_DJF],[GLDAS_master_all_1D_DJF,MERRA2_master_all_1D_DJF],[GLDAS_master_all_1D_DJF,GLDAS_master_all_1D_DJF]]
    			data_array_MAM = [[station_master_all_1D_MAM,station_master_all_1D_MAM],[station_master_all_1D_MAM,CFSR_master_all_1D_MAM],[station_master_all_1D_MAM,ERAI_master_all_1D_MAM],[station_master_all_1D_MAM,ERA5_master_all_1D_MAM],[station_master_all_1D_MAM,JRA_master_all_1D_MAM],[station_master_all_1D_MAM,MERRA2_master_all_1D_MAM],[station_master_all_1D_MAM,GLDAS_master_all_1D_MAM],[CFSR_master_all_1D_MAM,station_master_all_1D_MAM],[CFSR_master_all_1D_MAM,CFSR_master_all_1D_MAM],[CFSR_master_all_1D_MAM,ERAI_master_all_1D_MAM],[CFSR_master_all_1D_MAM,ERA5_master_all_1D_MAM],[CFSR_master_all_1D_MAM,JRA_master_all_1D_MAM],[CFSR_master_all_1D_MAM,MERRA2_master_all_1D_MAM],[CFSR_master_all_1D_MAM,GLDAS_master_all_1D_MAM],[ERAI_master_all_1D_MAM,station_master_all_1D_MAM],[ERAI_master_all_1D_MAM,CFSR_master_all_1D_MAM],[ERAI_master_all_1D_MAM,ERAI_master_all_1D_MAM],[ERAI_master_all_1D_MAM,ERA5_master_all_1D_MAM],[ERAI_master_all_1D_MAM,JRA_master_all_1D_MAM],[ERAI_master_all_1D_MAM,MERRA2_master_all_1D_MAM],[ERAI_master_all_1D_MAM,GLDAS_master_all_1D_MAM],[ERA5_master_all_1D_MAM,station_master_all_1D_MAM],[ERA5_master_all_1D_MAM,CFSR_master_all_1D_MAM],[ERA5_master_all_1D_MAM,ERAI_master_all_1D_MAM],[ERA5_master_all_1D_MAM,ERA5_master_all_1D_MAM],[ERA5_master_all_1D_MAM,JRA_master_all_1D_MAM],[ERA5_master_all_1D_MAM,MERRA2_master_all_1D_MAM],[ERA5_master_all_1D_MAM,GLDAS_master_all_1D_MAM],[JRA_master_all_1D_MAM,station_master_all_1D_MAM],[JRA_master_all_1D_MAM,CFSR_master_all_1D_MAM],[JRA_master_all_1D_MAM,ERAI_master_all_1D_MAM],[JRA_master_all_1D_MAM,ERA5_master_all_1D_MAM],[JRA_master_all_1D_MAM,JRA_master_all_1D_MAM],[JRA_master_all_1D_MAM,MERRA2_master_all_1D_MAM],[JRA_master_all_1D_MAM,GLDAS_master_all_1D_MAM],[MERRA2_master_all_1D_MAM,station_master_all_1D_MAM],[MERRA2_master_all_1D_MAM,CFSR_master_all_1D_MAM],[MERRA2_master_all_1D_MAM,ERAI_master_all_1D_MAM],[MERRA2_master_all_1D_MAM,ERA5_master_all_1D_MAM],[MERRA2_master_all_1D_MAM,JRA_master_all_1D_MAM],[MERRA2_master_all_1D_MAM,MERRA2_master_all_1D_MAM],[MERRA2_master_all_1D_MAM,GLDAS_master_all_1D_MAM],[GLDAS_master_all_1D_MAM,station_master_all_1D_MAM],[GLDAS_master_all_1D_MAM,CFSR_master_all_1D_MAM],[GLDAS_master_all_1D_MAM,ERAI_master_all_1D_MAM],[GLDAS_master_all_1D_MAM,ERA5_master_all_1D_MAM],[GLDAS_master_all_1D_MAM,JRA_master_all_1D_MAM],[GLDAS_master_all_1D_MAM,MERRA2_master_all_1D_MAM],[GLDAS_master_all_1D_MAM,GLDAS_master_all_1D_MAM]]
    			data_array_JJA = [[station_master_all_1D_JJA,station_master_all_1D_JJA],[station_master_all_1D_JJA,CFSR_master_all_1D_JJA],[station_master_all_1D_JJA,ERAI_master_all_1D_JJA],[station_master_all_1D_JJA,ERA5_master_all_1D_JJA],[station_master_all_1D_JJA,JRA_master_all_1D_JJA],[station_master_all_1D_JJA,MERRA2_master_all_1D_JJA],[station_master_all_1D_JJA,GLDAS_master_all_1D_JJA],[CFSR_master_all_1D_JJA,station_master_all_1D_JJA],[CFSR_master_all_1D_JJA,CFSR_master_all_1D_JJA],[CFSR_master_all_1D_JJA,ERAI_master_all_1D_JJA],[CFSR_master_all_1D_JJA,ERA5_master_all_1D_JJA],[CFSR_master_all_1D_JJA,JRA_master_all_1D_JJA],[CFSR_master_all_1D_JJA,MERRA2_master_all_1D_JJA],[CFSR_master_all_1D_JJA,GLDAS_master_all_1D_JJA],[ERAI_master_all_1D_JJA,station_master_all_1D_JJA],[ERAI_master_all_1D_JJA,CFSR_master_all_1D_JJA],[ERAI_master_all_1D_JJA,ERAI_master_all_1D_JJA],[ERAI_master_all_1D_JJA,ERA5_master_all_1D_JJA],[ERAI_master_all_1D_JJA,JRA_master_all_1D_JJA],[ERAI_master_all_1D_JJA,MERRA2_master_all_1D_JJA],[ERAI_master_all_1D_JJA,GLDAS_master_all_1D_JJA],[ERA5_master_all_1D_JJA,station_master_all_1D_JJA],[ERA5_master_all_1D_JJA,CFSR_master_all_1D_JJA],[ERA5_master_all_1D_JJA,ERAI_master_all_1D_JJA],[ERA5_master_all_1D_JJA,ERA5_master_all_1D_JJA],[ERA5_master_all_1D_JJA,JRA_master_all_1D_JJA],[ERA5_master_all_1D_JJA,MERRA2_master_all_1D_JJA],[ERA5_master_all_1D_JJA,GLDAS_master_all_1D_JJA],[JRA_master_all_1D_JJA,station_master_all_1D_JJA],[JRA_master_all_1D_JJA,CFSR_master_all_1D_JJA],[JRA_master_all_1D_JJA,ERAI_master_all_1D_JJA],[JRA_master_all_1D_JJA,ERA5_master_all_1D_JJA],[JRA_master_all_1D_JJA,JRA_master_all_1D_JJA],[JRA_master_all_1D_JJA,MERRA2_master_all_1D_JJA],[JRA_master_all_1D_JJA,GLDAS_master_all_1D_JJA],[MERRA2_master_all_1D_JJA,station_master_all_1D_JJA],[MERRA2_master_all_1D_JJA,CFSR_master_all_1D_JJA],[MERRA2_master_all_1D_JJA,ERAI_master_all_1D_JJA],[MERRA2_master_all_1D_JJA,ERA5_master_all_1D_JJA],[MERRA2_master_all_1D_JJA,JRA_master_all_1D_JJA],[MERRA2_master_all_1D_JJA,MERRA2_master_all_1D_JJA],[MERRA2_master_all_1D_JJA,GLDAS_master_all_1D_JJA],[GLDAS_master_all_1D_JJA,station_master_all_1D_JJA],[GLDAS_master_all_1D_JJA,CFSR_master_all_1D_JJA],[GLDAS_master_all_1D_JJA,ERAI_master_all_1D_JJA],[GLDAS_master_all_1D_JJA,ERA5_master_all_1D_JJA],[GLDAS_master_all_1D_JJA,JRA_master_all_1D_JJA],[GLDAS_master_all_1D_JJA,MERRA2_master_all_1D_JJA],[GLDAS_master_all_1D_JJA,GLDAS_master_all_1D_JJA]]
    			data_array_SON = [[station_master_all_1D_SON,station_master_all_1D_SON],[station_master_all_1D_SON,CFSR_master_all_1D_SON],[station_master_all_1D_SON,ERAI_master_all_1D_SON],[station_master_all_1D_SON,ERA5_master_all_1D_SON],[station_master_all_1D_SON,JRA_master_all_1D_SON],[station_master_all_1D_SON,MERRA2_master_all_1D_SON],[station_master_all_1D_SON,GLDAS_master_all_1D_SON],[CFSR_master_all_1D_SON,station_master_all_1D_SON],[CFSR_master_all_1D_SON,CFSR_master_all_1D_SON],[CFSR_master_all_1D_SON,ERAI_master_all_1D_SON],[CFSR_master_all_1D_SON,ERA5_master_all_1D_SON],[CFSR_master_all_1D_SON,JRA_master_all_1D_SON],[CFSR_master_all_1D_SON,MERRA2_master_all_1D_SON],[CFSR_master_all_1D_SON,GLDAS_master_all_1D_SON],[ERAI_master_all_1D_SON,station_master_all_1D_SON],[ERAI_master_all_1D_SON,CFSR_master_all_1D_SON],[ERAI_master_all_1D_SON,ERAI_master_all_1D_SON],[ERAI_master_all_1D_SON,ERA5_master_all_1D_SON],[ERAI_master_all_1D_SON,JRA_master_all_1D_SON],[ERAI_master_all_1D_SON,MERRA2_master_all_1D_SON],[ERAI_master_all_1D_SON,GLDAS_master_all_1D_SON],[ERA5_master_all_1D_SON,station_master_all_1D_SON],[ERA5_master_all_1D_SON,CFSR_master_all_1D_SON],[ERA5_master_all_1D_SON,ERAI_master_all_1D_SON],[ERA5_master_all_1D_SON,ERA5_master_all_1D_SON],[ERA5_master_all_1D_SON,JRA_master_all_1D_SON],[ERA5_master_all_1D_SON,MERRA2_master_all_1D_SON],[ERA5_master_all_1D_SON,GLDAS_master_all_1D_SON],[JRA_master_all_1D_SON,station_master_all_1D_SON],[JRA_master_all_1D_SON,CFSR_master_all_1D_SON],[JRA_master_all_1D_SON,ERAI_master_all_1D_SON],[JRA_master_all_1D_SON,ERA5_master_all_1D_SON],[JRA_master_all_1D_SON,JRA_master_all_1D_SON],[JRA_master_all_1D_SON,MERRA2_master_all_1D_SON],[JRA_master_all_1D_SON,GLDAS_master_all_1D_SON],[MERRA2_master_all_1D_SON,station_master_all_1D_SON],[MERRA2_master_all_1D_SON,CFSR_master_all_1D_SON],[MERRA2_master_all_1D_SON,ERAI_master_all_1D_SON],[MERRA2_master_all_1D_SON,ERA5_master_all_1D_SON],[MERRA2_master_all_1D_SON,JRA_master_all_1D_SON],[MERRA2_master_all_1D_SON,MERRA2_master_all_1D_SON],[MERRA2_master_all_1D_SON,GLDAS_master_all_1D_SON],[GLDAS_master_all_1D_SON,station_master_all_1D_SON],[GLDAS_master_all_1D_SON,CFSR_master_all_1D_SON],[GLDAS_master_all_1D_SON,ERAI_master_all_1D_SON],[GLDAS_master_all_1D_SON,ERA5_master_all_1D_SON],[GLDAS_master_all_1D_SON,JRA_master_all_1D_SON],[GLDAS_master_all_1D_SON,MERRA2_master_all_1D_SON],[GLDAS_master_all_1D_SON,GLDAS_master_all_1D_SON]]			
    			data_name_array = [["Station","Station"],["Station","CFSR"],["Station","ERA-Interim"],["Station","ERA-5"],["Station","JRA-55"],["Station","MERRA2"],["Station","GLDAS"],["CFSR","Station"],["CFSR","CFSR"],["CFSR","ERA-Interim"],["CFSR","ERA-5"],["CFSR","JRA-55"],["CFSR","MERRA2"],["CFSR","GLDAS"],["ERA-Interim","Station"],["ERA-Interim","CFSR"],["ERA-Interim","ERA-Interim"],["ERA-Interim","ERA-5"],["ERA-Interim","JRA-55"],["ERA-Interim","MERRA2"],["ERA-Interim","GLDAS"],["ERA-5","Station"],["ERA-5","CFSR"],["ERA-5","ERA-Interim"],["ERA-5","ERA-5"],["ERA-5","JRA-55"],["ERA-5","MERRA2"],["ERA-5","GLDAS"],["JRA-55","Station"],["JRA-55","CFSR"],["JRA-55","ERA-Interim"],["JRA-55","ERA-5"],["JRA-55","JRA-55"],["JRA-55","MERRA2"],["JRA-55","GLDAS"],["MERRA2","Station"],["MERRA2","CFSR"],["MERRA2","ERA-Interim"],["MERRA2","ERA-5"],["MERRA2","JRA-55"],["MERRA2","MERRA2"],["MERRA2","GLDAS"],["GLDAS","Station"],["GLDAS","CFSR"],["GLDAS","ERA-Interim"],["GLDAS","ERA-5"],["GLDAS","JRA-55"],["GLDAS","MERRA2"],["GLDAS","GLDAS"]]


#################### calculate cross-correlations ##########################
#    			correlation_array_S_DJF = []
#    			correlation_array_S_MAM = []
#    			correlation_array_S_JJA = []
#    			correlation_array_S_SON = []
#
#    			correlation_array_P_DJF = []
#    			correlation_array_P_MAM = []
#    			correlation_array_P_JJA = []
#    			correlation_array_P_SON = []
#
#    			#print(dframe_master_DJF)
#			
#    			correlation_name_array = []
#    			#fig = plt.figure()
#    			#fig,axs = plt.subplots(nrows = 7, ncols = 7, sharex = 'col', sharey = 'row', figsize=(28,28))
#    			for n in range(0,49):
#    				DataCombo = data_name_array[n]
#    				DataA_DJF = data_array_DJF[n][0]
#    				DataB_DJF = data_array_DJF[n][1]
#    				DataA_MAM = data_array_MAM[n][0]
#    				DataB_MAM = data_array_MAM[n][1]
#    				DataA_JJA = data_array_JJA[n][0]
#    				DataB_JJA = data_array_JJA[n][1]
#    				DataA_SON = data_array_SON[n][0]
#    				DataB_SON = data_array_SON[n][1]  
#    				NameA = data_name_array[n][0]
#    				NameB = data_name_array[n][1]
#    				#print(DataCombo)
#    				pltnum = n+1
#    			##### calculate Pearson's R #######
#    				coef_P_DJF, p_P_DJF = pearsonr(DataA_DJF,DataB_DJF)
#    				coef_P_MAM, p_P_MAM = pearsonr(DataA_MAM,DataB_MAM)
#    				coef_P_JJA, p_P_JJA = pearsonr(DataA_JJA,DataB_JJA)
#    				coef_P_SON, p_P_SON = pearsonr(DataA_SON,DataB_SON)    				
#    				#coef_Pdec = round(coef_P,2)
#    				#print(coef_Pdec)
#    				correlation_array_P_DJF.append(coef_P_DJF)
#    				correlation_array_P_MAM.append(coef_P_MAM)
#    				correlation_array_P_JJA.append(coef_P_JJA)
#    				correlation_array_P_SON.append(coef_P_SON)
#    			##### calculate Spearman's R ######
#    				coef_S_DJF, p_S_DJF = spearmanr(DataA_DJF,DataB_DJF)
#    				coef_S_MAM, p_S_MAM = spearmanr(DataA_MAM,DataB_MAM)
#    				coef_S_JJA, p_S_JJA = spearmanr(DataA_JJA,DataB_JJA)
#    				coef_S_SON, p_S_SON = spearmanr(DataA_SON,DataB_SON)
#    				#coef_Sdec = round(coef_S,2)
#    				#print(coef_Sdec)
#    				correlation_array_S_DJF.append(coef_S_DJF)
#    				correlation_array_S_MAM.append(coef_S_MAM)
#    				correlation_array_S_JJA.append(coef_S_JJA)
#    				correlation_array_S_SON.append(coef_S_SON)
#    			
#
#    			##### create DataFrames ######
#    			corr_P_DJF = {'Station': [correlation_array_P_DJF[0],correlation_array_P_DJF[1],correlation_array_P_DJF[2],correlation_array_P_DJF[3],correlation_array_P_DJF[4],correlation_array_P_DJF[5],correlation_array_P_DJF[6]],'CFSR': [correlation_array_P_DJF[7],correlation_array_P_DJF[8],correlation_array_P_DJF[9],correlation_array_P_DJF[10],correlation_array_P_DJF[11],correlation_array_P_DJF[12],correlation_array_P_DJF[13]],'ERA-Interim': [correlation_array_P_DJF[14],correlation_array_P_DJF[15],correlation_array_P_DJF[16],correlation_array_P_DJF[17],correlation_array_P_DJF[18],correlation_array_P_DJF[19],correlation_array_P_DJF[20]],'ERA-5': [correlation_array_P_DJF[21],correlation_array_P_DJF[22],correlation_array_P_DJF[23],correlation_array_P_DJF[24],correlation_array_P_DJF[25],correlation_array_P_DJF[26],correlation_array_P_DJF[27]],'JRA-55': [correlation_array_P_DJF[28],correlation_array_P_DJF[29],correlation_array_P_DJF[30],correlation_array_P_DJF[31],correlation_array_P_DJF[32],correlation_array_P_DJF[33],correlation_array_P_DJF[34]],'MERRA2': [correlation_array_P_DJF[35],correlation_array_P_DJF[36],correlation_array_P_DJF[37],correlation_array_P_DJF[38],correlation_array_P_DJF[39],correlation_array_P_DJF[40],correlation_array_P_DJF[41]],'GLDAS-Noah': [correlation_array_P_DJF[42],correlation_array_P_DJF[43],correlation_array_P_DJF[44],correlation_array_P_DJF[45],correlation_array_P_DJF[46],correlation_array_P_DJF[47],correlation_array_P_DJF[48]]}
#    			corr_P_MAM = {'Station': [correlation_array_P_MAM[0],correlation_array_P_MAM[1],correlation_array_P_MAM[2],correlation_array_P_MAM[3],correlation_array_P_MAM[4],correlation_array_P_MAM[5],correlation_array_P_MAM[6]],'CFSR': [correlation_array_P_MAM[7],correlation_array_P_MAM[8],correlation_array_P_MAM[9],correlation_array_P_MAM[10],correlation_array_P_MAM[11],correlation_array_P_MAM[12],correlation_array_P_MAM[13]],'ERA-Interim': [correlation_array_P_MAM[14],correlation_array_P_MAM[15],correlation_array_P_MAM[16],correlation_array_P_MAM[17],correlation_array_P_MAM[18],correlation_array_P_MAM[19],correlation_array_P_MAM[20]],'ERA-5': [correlation_array_P_MAM[21],correlation_array_P_MAM[22],correlation_array_P_MAM[23],correlation_array_P_MAM[24],correlation_array_P_MAM[25],correlation_array_P_MAM[26],correlation_array_P_MAM[27]],'JRA-55': [correlation_array_P_MAM[28],correlation_array_P_MAM[29],correlation_array_P_MAM[30],correlation_array_P_MAM[31],correlation_array_P_MAM[32],correlation_array_P_MAM[33],correlation_array_P_MAM[34]],'MERRA2': [correlation_array_P_MAM[35],correlation_array_P_MAM[36],correlation_array_P_MAM[37],correlation_array_P_MAM[38],correlation_array_P_MAM[39],correlation_array_P_MAM[40],correlation_array_P_MAM[41]],'GLDAS-Noah': [correlation_array_P_MAM[42],correlation_array_P_MAM[43],correlation_array_P_MAM[44],correlation_array_P_MAM[45],correlation_array_P_MAM[46],correlation_array_P_MAM[47],correlation_array_P_MAM[48]]}
#    			corr_P_JJA = {'Station': [correlation_array_P_JJA[0],correlation_array_P_JJA[1],correlation_array_P_JJA[2],correlation_array_P_JJA[3],correlation_array_P_JJA[4],correlation_array_P_JJA[5],correlation_array_P_JJA[6]],'CFSR': [correlation_array_P_JJA[7],correlation_array_P_JJA[8],correlation_array_P_JJA[9],correlation_array_P_JJA[10],correlation_array_P_JJA[11],correlation_array_P_JJA[12],correlation_array_P_JJA[13]],'ERA-Interim': [correlation_array_P_JJA[14],correlation_array_P_JJA[15],correlation_array_P_JJA[16],correlation_array_P_JJA[17],correlation_array_P_JJA[18],correlation_array_P_JJA[19],correlation_array_P_JJA[20]],'ERA-5': [correlation_array_P_JJA[21],correlation_array_P_JJA[22],correlation_array_P_JJA[23],correlation_array_P_JJA[24],correlation_array_P_JJA[25],correlation_array_P_JJA[26],correlation_array_P_JJA[27]],'JRA-55': [correlation_array_P_JJA[28],correlation_array_P_JJA[29],correlation_array_P_JJA[30],correlation_array_P_JJA[31],correlation_array_P_JJA[32],correlation_array_P_JJA[33],correlation_array_P_JJA[34]],'MERRA2': [correlation_array_P_JJA[35],correlation_array_P_JJA[36],correlation_array_P_JJA[37],correlation_array_P_JJA[38],correlation_array_P_JJA[39],correlation_array_P_JJA[40],correlation_array_P_JJA[41]],'GLDAS-Noah': [correlation_array_P_JJA[42],correlation_array_P_JJA[43],correlation_array_P_JJA[44],correlation_array_P_JJA[45],correlation_array_P_JJA[46],correlation_array_P_JJA[47],correlation_array_P_JJA[48]]}
#    			corr_P_SON = {'Station': [correlation_array_P_SON[0],correlation_array_P_SON[1],correlation_array_P_SON[2],correlation_array_P_SON[3],correlation_array_P_SON[4],correlation_array_P_SON[5],correlation_array_P_SON[6]],'CFSR': [correlation_array_P_SON[7],correlation_array_P_SON[8],correlation_array_P_SON[9],correlation_array_P_SON[10],correlation_array_P_SON[11],correlation_array_P_SON[12],correlation_array_P_SON[13]],'ERA-Interim': [correlation_array_P_SON[14],correlation_array_P_SON[15],correlation_array_P_SON[16],correlation_array_P_SON[17],correlation_array_P_SON[18],correlation_array_P_SON[19],correlation_array_P_SON[20]],'ERA-5': [correlation_array_P_SON[21],correlation_array_P_SON[22],correlation_array_P_SON[23],correlation_array_P_SON[24],correlation_array_P_SON[25],correlation_array_P_SON[26],correlation_array_P_SON[27]],'JRA-55': [correlation_array_P_SON[28],correlation_array_P_SON[29],correlation_array_P_SON[30],correlation_array_P_SON[31],correlation_array_P_SON[32],correlation_array_P_SON[33],correlation_array_P_SON[34]],'MERRA2': [correlation_array_P_SON[35],correlation_array_P_SON[36],correlation_array_P_SON[37],correlation_array_P_SON[38],correlation_array_P_SON[39],correlation_array_P_SON[40],correlation_array_P_SON[41]],'GLDAS-Noah': [correlation_array_P_SON[42],correlation_array_P_SON[43],correlation_array_P_SON[44],correlation_array_P_SON[45],correlation_array_P_SON[46],correlation_array_P_SON[47],correlation_array_P_SON[48]]}
#    			corr_S_DJF = {'Station': [correlation_array_S_DJF[0],correlation_array_S_DJF[1],correlation_array_S_DJF[2],correlation_array_S_DJF[3],correlation_array_S_DJF[4],correlation_array_S_DJF[5],correlation_array_S_DJF[6]],'CFSR': [correlation_array_S_DJF[7],correlation_array_S_DJF[8],correlation_array_S_DJF[9],correlation_array_S_DJF[10],correlation_array_S_DJF[11],correlation_array_S_DJF[12],correlation_array_S_DJF[13]],'ERA-Interim': [correlation_array_S_DJF[14],correlation_array_S_DJF[15],correlation_array_S_DJF[16],correlation_array_S_DJF[17],correlation_array_S_DJF[18],correlation_array_S_DJF[19],correlation_array_S_DJF[20]],'ERA-5': [correlation_array_S_DJF[21],correlation_array_S_DJF[22],correlation_array_S_DJF[23],correlation_array_S_DJF[24],correlation_array_S_DJF[25],correlation_array_S_DJF[26],correlation_array_S_DJF[27]],'JRA-55': [correlation_array_S_DJF[28],correlation_array_S_DJF[29],correlation_array_S_DJF[30],correlation_array_S_DJF[31],correlation_array_S_DJF[32],correlation_array_S_DJF[33],correlation_array_S_DJF[34]],'MERRA2': [correlation_array_S_DJF[35],correlation_array_S_DJF[36],correlation_array_S_DJF[37],correlation_array_S_DJF[38],correlation_array_S_DJF[39],correlation_array_S_DJF[40],correlation_array_S_DJF[41]],'GLDAS-Noah': [correlation_array_S_DJF[42],correlation_array_S_DJF[43],correlation_array_S_DJF[44],correlation_array_S_DJF[45],correlation_array_S_DJF[46],correlation_array_S_DJF[47],correlation_array_S_DJF[48]]}
#    			corr_S_MAM = {'Station': [correlation_array_S_MAM[0],correlation_array_S_MAM[1],correlation_array_S_MAM[2],correlation_array_S_MAM[3],correlation_array_S_MAM[4],correlation_array_S_MAM[5],correlation_array_S_MAM[6]],'CFSR': [correlation_array_S_MAM[7],correlation_array_S_MAM[8],correlation_array_S_MAM[9],correlation_array_S_MAM[10],correlation_array_S_MAM[11],correlation_array_S_MAM[12],correlation_array_S_MAM[13]],'ERA-Interim': [correlation_array_S_MAM[14],correlation_array_S_MAM[15],correlation_array_S_MAM[16],correlation_array_S_MAM[17],correlation_array_S_MAM[18],correlation_array_S_MAM[19],correlation_array_S_MAM[20]],'ERA-5': [correlation_array_S_MAM[21],correlation_array_S_MAM[22],correlation_array_S_MAM[23],correlation_array_S_MAM[24],correlation_array_S_MAM[25],correlation_array_S_MAM[26],correlation_array_S_MAM[27]],'JRA-55': [correlation_array_S_MAM[28],correlation_array_S_MAM[29],correlation_array_S_MAM[30],correlation_array_S_MAM[31],correlation_array_S_MAM[32],correlation_array_S_MAM[33],correlation_array_S_MAM[34]],'MERRA2': [correlation_array_S_MAM[35],correlation_array_S_MAM[36],correlation_array_S_MAM[37],correlation_array_S_MAM[38],correlation_array_S_MAM[39],correlation_array_S_MAM[40],correlation_array_S_MAM[41]],'GLDAS-Noah': [correlation_array_S_MAM[42],correlation_array_S_MAM[43],correlation_array_S_MAM[44],correlation_array_S_MAM[45],correlation_array_S_MAM[46],correlation_array_S_MAM[47],correlation_array_S_MAM[48]]}
#    			corr_S_JJA = {'Station': [correlation_array_S_JJA[0],correlation_array_S_JJA[1],correlation_array_S_JJA[2],correlation_array_S_JJA[3],correlation_array_S_JJA[4],correlation_array_S_JJA[5],correlation_array_S_JJA[6]],'CFSR': [correlation_array_S_JJA[7],correlation_array_S_JJA[8],correlation_array_S_JJA[9],correlation_array_S_JJA[10],correlation_array_S_JJA[11],correlation_array_S_JJA[12],correlation_array_S_JJA[13]],'ERA-Interim': [correlation_array_S_JJA[14],correlation_array_S_JJA[15],correlation_array_S_JJA[16],correlation_array_S_JJA[17],correlation_array_S_JJA[18],correlation_array_S_JJA[19],correlation_array_S_JJA[20]],'ERA-5': [correlation_array_S_JJA[21],correlation_array_S_JJA[22],correlation_array_S_JJA[23],correlation_array_S_JJA[24],correlation_array_S_JJA[25],correlation_array_S_JJA[26],correlation_array_S_JJA[27]],'JRA-55': [correlation_array_S_JJA[28],correlation_array_S_JJA[29],correlation_array_S_JJA[30],correlation_array_S_JJA[31],correlation_array_S_JJA[32],correlation_array_S_JJA[33],correlation_array_S_JJA[34]],'MERRA2': [correlation_array_S_JJA[35],correlation_array_S_JJA[36],correlation_array_S_JJA[37],correlation_array_S_JJA[38],correlation_array_S_JJA[39],correlation_array_S_JJA[40],correlation_array_S_JJA[41]],'GLDAS-Noah': [correlation_array_S_JJA[42],correlation_array_S_JJA[43],correlation_array_S_JJA[44],correlation_array_S_JJA[45],correlation_array_S_JJA[46],correlation_array_S_JJA[47],correlation_array_S_JJA[48]]}
#    			corr_S_SON = {'Station': [correlation_array_S_SON[0],correlation_array_S_SON[1],correlation_array_S_SON[2],correlation_array_S_SON[3],correlation_array_S_SON[4],correlation_array_S_SON[5],correlation_array_S_SON[6]],'CFSR': [correlation_array_S_SON[7],correlation_array_S_SON[8],correlation_array_S_SON[9],correlation_array_S_SON[10],correlation_array_S_SON[11],correlation_array_S_SON[12],correlation_array_S_SON[13]],'ERA-Interim': [correlation_array_S_SON[14],correlation_array_S_SON[15],correlation_array_S_SON[16],correlation_array_S_SON[17],correlation_array_S_SON[18],correlation_array_S_SON[19],correlation_array_S_SON[20]],'ERA-5': [correlation_array_S_SON[21],correlation_array_S_SON[22],correlation_array_S_SON[23],correlation_array_S_SON[24],correlation_array_S_SON[25],correlation_array_S_SON[26],correlation_array_S_SON[27]],'JRA-55': [correlation_array_S_SON[28],correlation_array_S_SON[29],correlation_array_S_SON[30],correlation_array_S_SON[31],correlation_array_S_SON[32],correlation_array_S_SON[33],correlation_array_S_SON[34]],'MERRA2': [correlation_array_S_SON[35],correlation_array_S_SON[36],correlation_array_S_SON[37],correlation_array_S_SON[38],correlation_array_S_SON[39],correlation_array_S_SON[40],correlation_array_S_SON[41]],'GLDAS-Noah': [correlation_array_S_SON[42],correlation_array_S_SON[43],correlation_array_S_SON[44],correlation_array_S_SON[45],correlation_array_S_SON[46],correlation_array_S_SON[47],correlation_array_S_SON[48]]}
#    			dframe_Pearson_DJF = pd.DataFrame(corr_P_DJF,index = ['Station','CFSR','ERA-Interim','ERA-5','JRA-55','MERRA2','GLDAS-Noah'])
#    			dframe_Pearson_MAM = pd.DataFrame(corr_P_MAM,index = ['Station','CFSR','ERA-Interim','ERA-5','JRA-55','MERRA2','GLDAS-Noah'])
#    			dframe_Pearson_JJA = pd.DataFrame(corr_P_JJA,index = ['Station','CFSR','ERA-Interim','ERA-5','JRA-55','MERRA2','GLDAS-Noah'])
#    			dframe_Pearson_SON  = pd.DataFrame(corr_P_SON,index = ['Station','CFSR','ERA-Interim','ERA-5','JRA-55','MERRA2','GLDAS-Noah'])
#    			dframe_Spearman_DJF = pd.DataFrame(corr_S_DJF,index = ['Station','CFSR','ERA-Interim','ERA-5','JRA-55','MERRA2','GLDAS-Noah'])
#    			dframe_Spearman_MAM = pd.DataFrame(corr_S_MAM,index = ['Station','CFSR','ERA-Interim','ERA-5','JRA-55','MERRA2','GLDAS-Noah'])
#    			dframe_Spearman_JJA = pd.DataFrame(corr_S_JJA,index = ['Station','CFSR','ERA-Interim','ERA-5','JRA-55','MERRA2','GLDAS-Noah'])
#    			dframe_Spearman_SON  = pd.DataFrame(corr_S_SON,index = ['Station','CFSR','ERA-Interim','ERA-5','JRA-55','MERRA2','GLDAS-Noah'])						
#    			#print(dframe_Pearson_DJF)
#    			#print(dframe_Spearman)
#
#    			###### create Spearman correlation matrix #######
#    			fig, axes = plt.subplots(2,2, figsize=(20,20), sharey=True)
#    			fig.suptitle('Spearmans R Correlation Matrix',fontweight='bold',fontsize='large')
#    			plt.subplot(221)
#    			sDJF = sn.heatmap(dframe_Spearman_DJF, annot=True, square=True,  vmin=0, vmax=1).set_title('DJF Correlations')
#    			plt.subplot(222)			
#    			sMAM = sn.heatmap(dframe_Spearman_MAM, annot=True, square=True,  vmin=0, vmax=1).set_title('MAM Correlations')
#    			plt.subplot(223)
#    			sJJA = sn.heatmap(dframe_Spearman_JJA, annot=True, square=True,  vmin=0, vmax=1).set_title('JJA Correlations')
#    			plt.subplot(224)
#    			sSON = sn.heatmap(dframe_Spearman_SON, annot=True, square=True,  vmin=0, vmax=1).set_title('SON Correlations')
#
#
######################## create figures #########################
#    			corr_filS = "".join(["/mnt/data/users/herringtont/soil_temp/plots/corr_matrix_anom/seasonal/subset_multiple/test_case2/Spearman/"+str(remap_type)+"_"+str(olri)+"_"+lyrj+"_thr"+str(thrk)+"_corr_matrix_Spearman_anom_seasonal_subset_multiple_testcase2.png"])
#    			path = pathlib.Path(corr_filS)
#    			path.parent.mkdir(parents=True, exist_ok=True)
#    			print(corr_filS)
#    			#plt.suptitle('Pearsons R Correlation Matrix SON')
#    			#plt.title("Remap Type: "+str(remap_type)+", outlier_type: "+str(olri)+", layer: "+str(lyrj)+", threshold: thr"+str(thrk))
#    			#plt.autoscale(enable=True,axis='both',tight=None)
#    			#plt.tight_layout()
#    			plt.savefig(corr_filS)
#    			plt.close()
#
############### create Pearson correlation matrix ##############
#    			fig, axes = plt.subplots(2,2, figsize=(20,20), sharey=True)
#    			fig.suptitle('Pearson R Correlation Matrix',fontweight='bold',fontsize='large')
#    			plt.subplot(221)
#    			pDJF = sn.heatmap(dframe_Pearson_DJF, annot=True, square=True, vmin=0, vmax=1).set_title('DJF Correlations')
#    			plt.subplot(222)
#    			pMAM = sn.heatmap(dframe_Pearson_MAM, annot=True, square=True,  vmin=0, vmax=1).set_title('MAM Correlations')
#    			plt.subplot(223)
#    			pJJA = sn.heatmap(dframe_Pearson_JJA, annot=True, square=True,  vmin=0, vmax=1).set_title('JJA Correlations')
#    			plt.subplot(224)
#    			pSON = sn.heatmap(dframe_Pearson_SON, annot=True, square=True,  vmin=0, vmax=1).set_title('SON Correlations')
#
#
######################## create figures #########################
#    			corr_filP = "".join(["/mnt/data/users/herringtont/soil_temp/plots/corr_matrix_anom/seasonal/subset_multiple/test_case2/Pearson/"+str(remap_type)+"_"+str(olri)+"_"+lyrj+"_thr"+str(thrk)+"_corr_matrix_Pearson_anom_seasonal_subset_multiple_testcase2.png"])
#    			path = pathlib.Path(corr_filP)
#    			path.parent.mkdir(parents=True, exist_ok=True)
#    			print(corr_filP)
#    			#plt.suptitle('Pearsons R Correlation Matrix SON')
#    			#plt.title("Remap Type: "+str(remap_type)+", outlier_type: "+str(olri)+", layer: "+str(lyrj)+", threshold: thr"+str(thrk))
#    			#plt.autoscale(enable=True,axis='both',tight=None)
#    			#plt.tight_layout()
#    			plt.savefig(corr_filP)
#    			plt.close()



####################### Create DJF Scatterplots #####################
    			fig = plt.figure()
    			fig,axs = plt.subplots(nrows = 7, ncols = 7, sharex = 'col', sharey = 'row', figsize=(28,28))

    			for p in range(0,49):
    				DataCombo = data_name_array[p]
    				DataA_DJF = data_array_DJF[p][0]
    				DataB_DJF = data_array_DJF[p][1]
 
    				NameA = data_name_array[p][0]
    				NameB = data_name_array[p][1]
				
    				#print(DataCombo)
    				pltnum = p+1
    				X_DJF = DataA_DJF
    				Y_DJF = DataB_DJF


    			##### calculate Pearson's R #######
    				coef_P_DJF, p_P_DJF = pearsonr(DataA_DJF,DataB_DJF)
    				coef_P_DJF = round(coef_P_DJF,2)  				
    			##### calculate Spearman's R ######
    				coef_S_DJF, p_S_DJF = spearmanr(DataA_DJF,DataB_DJF)
    				coef_S_DJF = round(coef_S_DJF,2)

    				ax = plt.subplot(7,7,pltnum)
    				ax.scatter(DataA_DJF,DataB_DJF,marker='o')
    				ax.xaxis.set_major_locator(MultipleLocator(5))
    				ax.xaxis.set_minor_locator(MultipleLocator(1))
    				ax.yaxis.set_major_locator(MultipleLocator(5))
    				ax.yaxis.set_minor_locator(MultipleLocator(1))				
    				ax.set_xlim(-20,20)
    				ax.set_ylim(-20,20)
    				handles = [mpl_patches.Rectangle((0,0), 1, 1, fc='white', ec='white', lw=0, alpha=0),mpl_patches.Rectangle((0,0), 1, 1, fc='white', ec='white', lw=0, alpha=0),mpl_patches.Rectangle((0,0), 1, 1, fc='white', ec='white', lw=0, alpha=0)]
    				axtext = []
    				axtext.append('Scatterplot of '+str(NameA)+' against '+str(NameB)+' DJF')
    				axtext.append("Pearson Correlation: "+str(coef_P_DJF))
    				axtext.append("Spearman Correlation: "+str(coef_S_DJF))				
    				ax.legend(handles, axtext, loc='best', fontsize = 'small', fancybox=False, framealpha=0, handlelength=0, handletextpad=0) 
    				lines = []
    				labels = []

    				for ax in fig.get_axes():
    					axLine, axLabel = ax.get_legend_handles_labels()
    					lines.extend(axLine)
    					labels.extend(axLabel)
    					

    			fig.add_subplot(111, frameon=False) #create large subplot which will include the plot labels for plots
    			plt.tick_params(labelcolor='none',bottom=False,left=False) #hide ticks
    			plt.xlabel('Soil Temp Anomaly($^\circ$ C)',fontweight='bold')
    			plt.ylabel('Soil Temp Anomaly($^\circ$ C)',fontweight='bold')
    			splot_fil = "".join(["/mnt/data/users/herringtont/soil_temp/plots/corr_matrix_anom/seasonal/Scatterplot/subset_multiple/test_case2/DJF/"+str(remap_type)+"_"+str(olri)+"_"+lyrj+"_thr"+str(thrk)+"_scatter_DJF_subset_multiple_testcase2.png"])
    			path = pathlib.Path(splot_fil)
    			path.parent.mkdir(parents=True, exist_ok=True)
    			print(splot_fil)
    			plt.tight_layout()
    			plt.savefig(splot_fil)
    			plt.close()


####################### Create MAM Scatterplots #####################
    			fig = plt.figure()
    			fig,axs = plt.subplots(nrows = 7, ncols = 7, sharex = 'col', sharey = 'row', figsize=(28,28))

    			for q in range(0,49):
    				DataCombo = data_name_array[q]

    				DataA_MAM = data_array_MAM[q][0]
    				DataB_MAM = data_array_MAM[q][1]
 
    				NameA = data_name_array[q][0]
    				NameB = data_name_array[q][1]
				
    				#print(DataCombo)
    				pltnum = q+1

    				X_MAM = DataA_MAM
    				Y_MAM = DataB_MAM


    			##### calculate Pearson's R #######
    				coef_P_MAM, p_P_MAM = pearsonr(DataA_MAM,DataB_MAM)
    				coef_P_MAM = round(coef_P_MAM,2)  				
    			##### calculate Spearman's R ######
    				coef_S_MAM, p_S_MAM = spearmanr(DataA_MAM,DataB_MAM)
    				coef_S_MAM = round(coef_S_MAM,2)
    				ax = plt.subplot(7,7,pltnum)
    				ax.scatter(DataA_MAM,DataB_MAM,marker='o')
    				ax.xaxis.set_major_locator(MultipleLocator(5))
    				ax.xaxis.set_minor_locator(MultipleLocator(1))
    				ax.yaxis.set_major_locator(MultipleLocator(5))
    				ax.yaxis.set_minor_locator(MultipleLocator(1))				
    				ax.set_xlim(-20,20)
    				ax.set_ylim(-20,20)
    				handles = [mpl_patches.Rectangle((0,0), 1, 1, fc='white', ec='white', lw=0, alpha=0),mpl_patches.Rectangle((0,0), 1, 1, fc='white', ec='white', lw=0, alpha=0),mpl_patches.Rectangle((0,0), 1, 1, fc='white', ec='white', lw=0, alpha=0)]
    				axtext = []
    				axtext.append('Scatterplot of '+str(NameA)+' against '+str(NameB)+' MAM')
    				axtext.append("Pearson Correlation: "+str(coef_P_MAM))
    				axtext.append("Spearman Correlation: "+str(coef_S_MAM))				
    				ax.legend(handles, axtext, loc='best', fontsize = 'small', fancybox=False, framealpha=0, handlelength=0, handletextpad=0) 
    				lines = []
    				labels = []

    				for ax in fig.get_axes():
    					axLine, axLabel = ax.get_legend_handles_labels()
    					lines.extend(axLine)
    					labels.extend(axLabel)
    					

    			fig.add_subplot(111, frameon=False) #create large subplot which will include the plot labels for plots
    			plt.tick_params(labelcolor='none',bottom=False,left=False) #hide ticks
    			plt.xlabel('Soil Temp Anomaly($^\circ$ C)',fontweight='bold')
    			plt.ylabel('Soil Temp Anomaly($^\circ$ C)',fontweight='bold')
    			splot_fil = "".join(["/mnt/data/users/herringtont/soil_temp/plots/corr_matrix_anom/seasonal/Scatterplot/subset_multiple/test_case2/MAM/"+str(remap_type)+"_"+str(olri)+"_"+lyrj+"_thr"+str(thrk)+"_scatter_MAM_subset_multiple_testcase2.png"])
    			path = pathlib.Path(splot_fil)
    			path.parent.mkdir(parents=True, exist_ok=True)
    			print(splot_fil)
    			plt.tight_layout()
    			plt.savefig(splot_fil)
    			plt.close()


####################### Create JJA Scatterplots #####################
    			fig = plt.figure()
    			fig,axs = plt.subplots(nrows = 7, ncols = 7, sharex = 'col', sharey = 'row', figsize=(28,28))

    			for r in range(0,49):
    				DataCombo = data_name_array[r]

    				DataA_JJA = data_array_JJA[r][0]
    				DataB_JJA = data_array_JJA[r][1]

    				NameA = data_name_array[r][0]
    				NameB = data_name_array[r][1]
				
    				#print(DataCombo)
    				pltnum = r+1

    				X_JJA = DataA_JJA
    				Y_JJA = DataB_JJA


    			##### calculate Pearson's R #######
    				coef_P_JJA, p_P_JJA = pearsonr(DataA_JJA,DataB_JJA)
    				coef_P_JJA = round(coef_P_JJA,2)  				
    			##### calculate Spearman's R ######
    				coef_S_JJA, p_S_JJA = spearmanr(DataA_JJA,DataB_JJA)
    				coef_S_JJA = round(coef_S_JJA,2)
    				ax = plt.subplot(7,7,pltnum)
    				ax.scatter(DataA_JJA,DataB_JJA,marker='o')
    				ax.xaxis.set_major_locator(MultipleLocator(5))
    				ax.xaxis.set_minor_locator(MultipleLocator(1))
    				ax.yaxis.set_major_locator(MultipleLocator(5))
    				ax.yaxis.set_minor_locator(MultipleLocator(1))				
    				ax.set_xlim(-20,20)
    				ax.set_ylim(-20,20)
    				handles = [mpl_patches.Rectangle((0,0), 1, 1, fc='white', ec='white', lw=0, alpha=0),mpl_patches.Rectangle((0,0), 1, 1, fc='white', ec='white', lw=0, alpha=0),mpl_patches.Rectangle((0,0), 1, 1, fc='white', ec='white', lw=0, alpha=0)]
    				axtext = []
    				axtext.append('Scatterplot of '+str(NameA)+' against '+str(NameB)+' JJA')
    				axtext.append("Pearson Correlation: "+str(coef_P_JJA))
    				axtext.append("Spearman Correlation: "+str(coef_S_JJA))				
    				ax.legend(handles, axtext, loc='best', fontsize = 'small', fancybox=False, framealpha=0, handlelength=0, handletextpad=0) 
    				lines = []
    				labels = []

    				for ax in fig.get_axes():
    					axLine, axLabel = ax.get_legend_handles_labels()
    					lines.extend(axLine)
    					labels.extend(axLabel)
    					

    			fig.add_subplot(111, frameon=False) #create large subplot which will include the plot labels for plots
    			plt.tick_params(labelcolor='none',bottom=False,left=False) #hide ticks
    			plt.xlabel('Soil Temp Anomaly($^\circ$ C)',fontweight='bold')
    			plt.ylabel('Soil Temp Anomaly($^\circ$ C)',fontweight='bold')
    			splot_fil = "".join(["/mnt/data/users/herringtont/soil_temp/plots/corr_matrix_anom/seasonal/Scatterplot/subset_multiple/test_case2/JJA/"+str(remap_type)+"_"+str(olri)+"_"+lyrj+"_thr"+str(thrk)+"_scatter_JJA_subset_multiple_testcase2.png"])
    			path = pathlib.Path(splot_fil)
    			path.parent.mkdir(parents=True, exist_ok=True)
    			print(splot_fil)
    			plt.tight_layout()
    			plt.savefig(splot_fil)
    			plt.close()


####################### Create SON Scatterplots #####################
    			fig = plt.figure()
    			fig,axs = plt.subplots(nrows = 7, ncols = 7, sharex = 'col', sharey = 'row', figsize=(28,28))

    			for s in range(0,49):
 
    				DataA_SON = data_array_SON[s][0]
    				DataB_SON = data_array_SON[s][1]  
    				NameA = data_name_array[s][0]
    				NameB = data_name_array[s][1]
				
    				#print(DataCombo)
    				pltnum = s+1

    				X_SON = DataA_SON
    				Y_SON = DataB_SON

    			##### calculate Pearson's R #######
    				coef_P_SON, p_P_SON = pearsonr(DataA_SON,DataB_SON)
    				coef_P_SON = round(coef_P_SON,2)  				
    			##### calculate Spearman's R ######
    				coef_S_SON, p_S_SON = spearmanr(DataA_SON,DataB_SON)
    				coef_S_SON = round(coef_S_SON,2)
    				ax = plt.subplot(7,7,pltnum)
    				ax.scatter(DataA_SON,DataB_SON,marker='o')
    				ax.xaxis.set_major_locator(MultipleLocator(5))
    				ax.xaxis.set_minor_locator(MultipleLocator(1))
    				ax.yaxis.set_major_locator(MultipleLocator(5))
    				ax.yaxis.set_minor_locator(MultipleLocator(1))				
    				ax.set_xlim(-20,20)
    				ax.set_ylim(-20,20)
    				handles = [mpl_patches.Rectangle((0,0), 1, 1, fc='white', ec='white', lw=0, alpha=0),mpl_patches.Rectangle((0,0), 1, 1, fc='white', ec='white', lw=0, alpha=0),mpl_patches.Rectangle((0,0), 1, 1, fc='white', ec='white', lw=0, alpha=0)]
    				axtext = []
    				axtext.append('Scatterplot of '+str(NameA)+' against '+str(NameB)+' SON')
    				axtext.append("Pearson Correlation: "+str(coef_P_SON))
    				axtext.append("Spearman Correlation: "+str(coef_S_SON))				
    				ax.legend(handles, axtext, loc='best', fontsize = 'small', fancybox=False, framealpha=0, handlelength=0, handletextpad=0) 
    				lines = []
    				labels = []

    				for ax in fig.get_axes():
    					axLine, axLabel = ax.get_legend_handles_labels()
    					lines.extend(axLine)
    					labels.extend(axLabel)
    					

    			fig.add_subplot(111, frameon=False) #create large subplot which will include the plot labels for plots
    			plt.tick_params(labelcolor='none',bottom=False,left=False) #hide ticks
    			plt.xlabel('Soil Temp Anomaly($^\circ$ C)',fontweight='bold')
    			plt.ylabel('Soil Temp Anomaly($^\circ$ C)',fontweight='bold')
    			splot_fil = "".join(["/mnt/data/users/herringtont/soil_temp/plots/corr_matrix_anom/seasonal/Scatterplot/subset_multiple/test_case2/SON/"+str(remap_type)+"_"+str(olri)+"_"+lyrj+"_thr"+str(thrk)+"_scatter_SON_subset_multiple_testcase2.png"])
    			path = pathlib.Path(splot_fil)
    			path.parent.mkdir(parents=True, exist_ok=True)
    			print(splot_fil)
    			plt.tight_layout()
    			plt.savefig(splot_fil)
    			plt.close()
