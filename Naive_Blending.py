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
import pathlib
from calendar import isleap
from dateutil.relativedelta import *
from pathlib import Path
import seaborn as sn
from calendar import isleap
from dateutil.relativedelta import *
from pathlib import Path
from matplotlib.ticker import (MultipleLocator, AutoLocator, AutoMinorLocator)
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from scipy.stats import pearsonr
from decimal import *

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
    			CFSR_master_all = []
    			CFSR_clim_master_all = []
    			CFSR_master_raw = []
    			grid_master_all = []
    			station_master_all = []
    			station_master_raw = []
    			ERAI_master_all = []
    			ERAI_clim_master_all = []
    			ERAI_master_raw = []
    			ERA5_master_all = []
    			ERA5_clim_master_all = []
    			ERA5_master_raw = []
    			JRA_master_all = []
    			JRA_clim_master_all = []
    			JRA_master_raw = []
    			MERRA2_master_all = []
    			MERRA2_clim_master_all = []
    			MERRA2_master_raw = []
    			GLDAS_master_all = []
    			GLDAS_clim_master_all = []
    			GLDAS_master_raw = []
    			date_master_all = []
    			lat_master_all = []
    			lon_master_all = []

    			thrk = k
    			thr_type = ''.join(['thr_',k])
    			thrshld = "".join(["thr_"+str(thrk)])
    			indir = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/spatial_average/remap",rmph,"/no_outliers/",olri,"/",lyrj,"/thr_",thrk,"/"])
    			indira = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/spatial_average_anom/remap",rmph,"/no_outliers/",olri,"/",lyrj,"/thr_",thrk,"/"])
    			pathlist = Path(indir).glob('*.csv')
    			pathlista = Path(indira).glob('*anom.csv')
    			#print(indir)			


########################## Create Anomaly Data ####################################################

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
    				stemp_raw = dframe['Spatial Avg Temp'].tolist()
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
    				CFSR_new_all = []
    				date_new_all = [] #if there is no CFSR or CFSR2 in triplet
    				station_new_all = [] #if there is no CFSR or CFSR2 in triplet
    				station_new_raw = []    				
    				GLDAS_new_all = [] #if there is no CFSR or CFSR2 in triplet  
    				JRA_new_all = [] #if there is no CFSR or CFSR2 in triplet 
    				ERAI_new_all = [] #if there is no CFSR or CFSR2 in triplet 
    				ERA5_new_all = [] #if there is no CFSR or CFSR2 in triplet 
    				MERRA2_new_all = [] #if there is no CFSR or CFSR2 in triplet 				

    				for i in range(0,len(date_mon)):
    					date_mon_i = date_mon[i]
    					stemp_i = stemp[i]
    					stemp_rawi = stemp_raw[i]
    					#print(date_mon_i)			    					

    					if (date_mon_i <= datetime.datetime(2010,12,31)):
    						CFSR_stemp_i = CFSR_stemp3.sel(time=date_mon_i, drop=True)
    						CFSR_new_all.append(CFSR_stemp_i)
    						date_new_all.append(date_mon_i)
    						station_new_all.append(stemp_i)
    						station_new_raw.append(stemp_rawi)
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
    						station_new_raw.append(stemp_rawi)
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

    				date_new_all2 = np.array(date_new_all).flatten()
    				station_new_all2 = np.array(station_new_all).flatten()
    				station_new_raw2 = np.array(station_new_raw).flatten()
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
    				rnysis_clim_avg_master =[]
				
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
    						climtgy = clim_averages[dat_row_mons]
    						rnysis_anom_master.append(stemp_anom)
    						rnysis_date_master.append(dat_row)					
    						rnysis_name_master.append(rnysis_namei)
    						rnysis_stemp_master.append(stemp_rw)
    						rnysis_clim_avg_master.append(climtgy)    						


###################### create anom dataframe ########################

    				dframe_anom_master = pd.DataFrame(data=rnysis_date_master, columns=['Date'])
    				dframe_anom_master['Soil Temp'] = rnysis_stemp_master
    				dframe_anom_master['Climatology'] = rnysis_clim_avg_master
    				dframe_anom_master['Soil Temp Anom'] = rnysis_anom_master
    				dframe_anom_master['Reanalysis Product'] = rnysis_name_master					

    				dframe_anom_CFSR = dframe_anom_master[dframe_anom_master['Reanalysis Product'] == 'CFSR']
    				dframe_clim_CFSR = dframe_anom_CFSR['Climatology'].values.tolist()
    				dframe_anom_CFSR = dframe_anom_CFSR['Soil Temp Anom'].values.tolist()

				
    				dframe_anom_GLDAS = dframe_anom_master[dframe_anom_master['Reanalysis Product'] == 'GLDAS']
    				dframe_clim_GLDAS = dframe_anom_GLDAS['Climatology'].values.tolist()
    				dframe_anom_GLDAS = dframe_anom_GLDAS['Soil Temp Anom'].values.tolist()


    				dframe_anom_ERAI = dframe_anom_master[dframe_anom_master['Reanalysis Product'] == 'ERA-Interim']
    				dframe_clim_ERAI = dframe_anom_ERAI['Climatology'].values.tolist()
    				dframe_anom_ERAI = dframe_anom_ERAI['Soil Temp Anom'].values.tolist()


    				dframe_anom_ERA5 = dframe_anom_master[dframe_anom_master['Reanalysis Product'] == 'ERA-5']
    				dframe_clim_ERA5 = dframe_anom_ERA5['Climatology'].values.tolist()
    				dframe_anom_ERA5 = dframe_anom_ERA5['Soil Temp Anom'].values.tolist()


    				dframe_anom_JRA = dframe_anom_master[dframe_anom_master['Reanalysis Product'] == 'JRA-55']
    				dframe_clim_JRA = dframe_anom_JRA['Climatology'].values.tolist()
    				dframe_anom_JRA = dframe_anom_JRA['Soil Temp Anom'].values.tolist()


    				dframe_anom_MERRA2 = dframe_anom_master[dframe_anom_master['Reanalysis Product'] == 'MERRA2']
    				dframe_clim_MERRA2 = dframe_anom_MERRA2['Climatology'].values.tolist()
    				dframe_anom_MERRA2 = dframe_anom_MERRA2['Soil Temp Anom'].values.tolist()

				
#################### create new dataframe with date, station temp, reanalysis temp ###################

    				dframe_all = pd.DataFrame(data=date_new_all2, columns=['Date'])
    				dframe_all['Grid Cell'] = gcell
    				dframe_all['Lat Cen'] = lat_cen
    				dframe_all['Lon Cen'] = lon_cen
    				dframe_all['Station'] = station_new_all2
    				dframe_all['Station Raw'] = station_new_raw2
    				dframe_all['CFSR'] = dframe_anom_CFSR
    				dframe_all['CFSR Clim'] = dframe_clim_CFSR
    				dframe_all['CFSR Raw'] = CFSR_new_all2
    				dframe_all['GLDAS'] = dframe_anom_GLDAS
    				dframe_all['GLDAS Clim'] = dframe_clim_GLDAS
    				dframe_all['GLDAS Raw'] = GLDAS_new_all2
    				dframe_all['JRA55'] = dframe_anom_JRA
    				dframe_all['JRA55 Clim'] = dframe_clim_JRA
    				dframe_all['JRA55 Raw'] = JRA_new_all2
    				dframe_all['ERA5'] = dframe_anom_ERA5
    				dframe_all['ERA5 Clim'] = dframe_clim_ERA5
    				dframe_all['ERA5 Raw'] = ERA5_new_all2
    				dframe_all['ERA-Interim'] = dframe_anom_ERAI
    				dframe_all['ERA-Interim Clim'] = dframe_clim_ERAI
    				dframe_all['ERA-Interim Raw'] = ERAI_new_all2
    				dframe_all['MERRA2'] = dframe_anom_MERRA2
    				dframe_all['MERRA2 Clim'] = dframe_clim_MERRA2
    				dframe_all['MERRA2 Raw'] = MERRA2_new_all2

############ drop rows with NaN ############
    				dframe_all = dframe_all[dframe_all['CFSR'].notna()]
    				dframe_all = dframe_all[dframe_all['GLDAS'].notna()]
    				dframe_all = dframe_all[dframe_all['JRA55'].notna()]
    				dframe_all = dframe_all[dframe_all['ERA5'].notna()]
    				dframe_all = dframe_all[dframe_all['ERA-Interim'].notna()]
    				dframe_all = dframe_all[dframe_all['MERRA2'].notna()]


    				#print(dframe_all)
				
################################# append values to master arrays ######################################

    				date_final = dframe_all['Date']
    				if(date_final.empty == False):
    					date_master_all.append(date_final.tolist())

    				station_final = dframe_all['Station']
    				if (station_final.empty == False):
    					station_master_all.append(station_final.tolist())

    				station_final_raw = dframe_all['Station Raw']
    				if (station_final_raw.empty == False):
    					station_master_raw.append(station_final_raw.tolist())
    	
    				grid_final = dframe_all['Grid Cell']
    				if (grid_final.empty == False):
    					grid_master_all.append(grid_final.values.tolist())

    				CFSR_final = dframe_all['CFSR']
    				if (CFSR_final.empty == False):
    					CFSR_master_all.append(CFSR_final.values.tolist())

    				CFSR_clim_final = dframe_all['CFSR Clim']
    				if (CFSR_clim_final.empty == False):
    					CFSR_clim_master_all.append(CFSR_clim_final.values.tolist())

    				CFSR_final_raw = dframe_all['CFSR Raw']
    				if (CFSR_final_raw.empty == False):
    					CFSR_master_raw.append(CFSR_final_raw.tolist())
    
    				ERAI_final = dframe_all['ERA-Interim']
    				if (ERAI_final.empty == False):
    					ERAI_master_all.append(ERAI_final.values.tolist())

    				ERAI_clim_final = dframe_all['ERA-Interim Clim']
    				if (ERAI_clim_final.empty == False):
    					ERAI_clim_master_all.append(ERAI_clim_final.values.tolist())

    				ERAI_final_raw = dframe_all['ERA-Interim Raw']
    				if (ERAI_final_raw.empty == False):
    					ERAI_master_raw.append(ERAI_final_raw.tolist())
    
    				ERA5_final = dframe_all['ERA5']
    				if (ERA5_final.empty == False):
    					ERA5_master_all.append(ERA5_final.values.tolist())

    				ERA5_clim_final = dframe_all['ERA5 Clim']
    				if (ERA5_clim_final.empty == False):
    					ERA5_clim_master_all.append(ERA5_clim_final.values.tolist())

    				ERA5_final_raw = dframe_all['ERA5 Raw']
    				if (ERA5_final_raw.empty == False):
    					ERA5_master_raw.append(ERA5_final_raw.tolist())
    
    				MERRA2_final = dframe_all['MERRA2']
    				if (MERRA2_final.empty == False):
    					MERRA2_master_all.append(MERRA2_final.values.tolist())

    				MERRA2_clim_final = dframe_all['MERRA2 Clim']
    				if (MERRA2_clim_final.empty == False):
    					MERRA2_clim_master_all.append(MERRA2_clim_final.values.tolist())

    				MERRA2_final_raw = dframe_all['MERRA2 Raw']
    				if (MERRA2_final_raw.empty == False):
    					MERRA2_master_raw.append(MERRA2_final_raw.tolist())
					    
    				JRA_final = dframe_all['JRA55']
    				if (JRA_final.empty == False):
    					JRA_master_all.append(JRA_final.values.tolist())
					
    				JRA_clim_final = dframe_all['JRA55 Clim']
    				if (JRA_clim_final.empty == False):
    					JRA_clim_master_all.append(JRA_clim_final.values.tolist())

    				JRA_final_raw = dframe_all['JRA55 Raw']
    				if (JRA_final_raw.empty == False):
    					JRA_master_raw.append(JRA_final_raw.tolist())
    
    				GLDAS_final = dframe_all['GLDAS']
    				if (GLDAS_final.empty == False):	
    					GLDAS_master_all.append(GLDAS_final.values.tolist())

    				GLDAS_clim_final = dframe_all['GLDAS Clim']
    				if (GLDAS_clim_final.empty == False):
    					GLDAS_clim_master_all.append(GLDAS_clim_final.values.tolist())

    				GLDAS_final_raw = dframe_all['GLDAS Raw']
    				if (GLDAS_final_raw.empty == False):
    					GLDAS_master_raw.append(GLDAS_final_raw.tolist())


######################### Flatten Master Lists to 1D ############################

    			date_master_all_1D = []
    			for sublist in date_master_all:
    				for item in sublist:
    					date_master_all_1D.append(item)

    			grid_master_all_1D = []
    			for sublist in grid_master_all:
    				for item in sublist:
    					grid_master_all_1D.append(item)
				
    			station_master_all_1D = []
    			for sublist in station_master_all:
    				for item in sublist:
    					station_master_all_1D.append(item)

    			station_master_raw_1D = []
    			for sublist in station_master_raw:
    				for item in sublist:
    					station_master_raw_1D.append(item)
				
    			CFSR_master_all_1D = []
    			for sublist in CFSR_master_all:
    				for item in sublist:
    					CFSR_master_all_1D.append(item)

    			CFSR_clim_master_all_1D = []
    			for sublist in CFSR_clim_master_all:
    				for item in sublist:
    					CFSR_clim_master_all_1D.append(item)

    			CFSR_master_raw_1D = []
    			for sublist in CFSR_master_raw:
    				for item in sublist:
    					CFSR_master_raw_1D.append(item)
									
    			ERAI_master_all_1D = []
    			for sublist in ERAI_master_all:
    				for item in sublist:
    					ERAI_master_all_1D.append(item)

    			ERAI_clim_master_all_1D = []
    			for sublist in ERAI_clim_master_all:
    				for item in sublist:
    					ERAI_clim_master_all_1D.append(item)

    			ERAI_master_raw_1D = []
    			for sublist in ERAI_master_raw:
    				for item in sublist:
    					ERAI_master_raw_1D.append(item)
				
    			ERA5_master_all_1D = []
    			for sublist in ERA5_master_all:
    				for item in sublist:
    					ERA5_master_all_1D.append(item)

    			ERA5_clim_master_all_1D = []
    			for sublist in ERA5_clim_master_all:
    				for item in sublist:
    					ERA5_clim_master_all_1D.append(item)

    			ERA5_master_raw_1D = []
    			for sublist in ERA5_master_raw:
    				for item in sublist:
    					ERA5_master_raw_1D.append(item)
				
    			JRA_master_all_1D = []
    			for sublist in JRA_master_all:
    				for item in sublist:
    					JRA_master_all_1D.append(item)

    			JRA_clim_master_all_1D = []
    			for sublist in JRA_clim_master_all:
    				for item in sublist:
    					JRA_clim_master_all_1D.append(item)

    			JRA_master_raw_1D = []
    			for sublist in JRA_master_raw:
    				for item in sublist:
    					JRA_master_raw_1D.append(item)

    			MERRA2_master_all_1D = []
    			for sublist in MERRA2_master_all:
    				for item in sublist:
    					MERRA2_master_all_1D.append(item)

    			MERRA2_clim_master_all_1D = []
    			for sublist in MERRA2_clim_master_all:
    				for item in sublist:
    					MERRA2_clim_master_all_1D.append(item)

    			MERRA2_master_raw_1D = []
    			for sublist in MERRA2_master_raw:
    				for item in sublist:
    					MERRA2_master_raw_1D.append(item)

    			GLDAS_master_all_1D = []
    			for sublist in GLDAS_master_all:
    				for item in sublist:
    					GLDAS_master_all_1D.append(item)

    			GLDAS_clim_master_all_1D = []
    			for sublist in GLDAS_clim_master_all:
    				for item in sublist:
    					GLDAS_clim_master_all_1D.append(item)

    			GLDAS_master_raw_1D = []
    			for sublist in GLDAS_master_raw:
    				for item in sublist:
    					GLDAS_master_raw_1D.append(item)

    			grid_celluq = np.unique(grid_master_all_1D)
    			#print('Number of Unique Grid Cells:', len(grid_celluq)) 
    			#print("Station Master")
    			#print(len(station_master_all_1D))
    			#print("ERA-Interim Master")
    			#print(len(ERAI_master_all_1D))   

    			grid_celluq = np.unique(grid_master_all_1D)
    			#print('Number of Unique Grid Cells:', len(grid_celluq)) 
    			#print("Station Master")
    			#print(len(station_master_all_1D))
    			#print("ERA-Interim Master")
    			#print(len(ERAI_master_all_1D))


##################### create finalized dataframe for blending purposes ###################

    			dframe_blended_raw = pd.DataFrame(data=CFSR_master_raw_1D, columns=['CFSR'])
    			dframe_blended_raw['ERA-Interim'] = ERAI_master_raw_1D
    			dframe_blended_raw['ERA5'] = ERA5_master_raw_1D
    			dframe_blended_raw['JRA-55'] = JRA_master_raw_1D
    			dframe_blended_raw['MERRA2'] = MERRA2_master_raw_1D
    			dframe_blended_raw['GLDAS'] = GLDAS_master_raw_1D
    			dframe_blended_raw['Naive Blending'] = dframe_blended_raw.mean(axis=1)
    			dframe_blended_raw.insert(0,'Date',date_master_all_1D)
    			dframe_blended_raw.insert(1,'Grid Cell',grid_master_all_1D)


    			dframe_blended_anom = pd.DataFrame(data=CFSR_master_all_1D, columns=['CFSR'])
    			dframe_blended_anom['ERA-Interim'] = ERAI_master_all_1D
    			dframe_blended_anom['ERA5'] = ERA5_master_all_1D
    			dframe_blended_anom['JRA-55'] = JRA_master_all_1D
    			dframe_blended_anom['MERRA2'] = MERRA2_master_all_1D
    			dframe_blended_anom['GLDAS'] = GLDAS_master_all_1D
    			dframe_blended_anom['Naive Blending'] = dframe_blended_anom.mean(axis=1)
    			dframe_blended_anom.insert(0,'Date',date_master_all_1D)
    			dframe_blended_anom.insert(1,'Grid Cell',grid_master_all_1D)

    			if (remap_type == 'remapnn'):
    				if (lyrj == '0_9.9'):
    					geom_fil = '/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/geometry_L1_nn.csv'
    				elif (lyrj == '10_29.9'):
    					geom_fil = '/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/geometry_L2_nn.csv'
    				elif (lyrj == '30_99.9'):
    					geom_fil = '/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/geometry_L3_nn.csv'
    				elif (lyrj == '100_299.9'):
    					geom_fil = '/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/geometry_L4_nn.csv'
    				elif (lyrj == '300_deeper'):
    					geom_fil = '/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/geometry_L5_nn.csv'					

    			elif (remap_type == 'remapbil'):
    				if (lyrj == '0_9.9'):
    					geom_fil = '/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/geometry_L1_bil.csv'
    				elif (lyrj == '10_29.9'):
    					geom_fil = '/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/geometry_L2_bil.csv'
    				elif (lyrj == '30_99.9'):
    					geom_fil = '/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/geometry_L3_bil.csv'
    				elif (lyrj == '100_299.9'):
    					geom_fil = '/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/geometry_L4_bil.csv'
    				elif (lyrj == '300_deeper'):
    					geom_fil = '/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/geometry_L5_bil.csv'

    			len_df_blending = len(dframe_blended_raw)
    			grid_cell_blending = dframe_blended_raw['Grid Cell'].values

    			dframe_geom = pd.read_csv(geom_fil) 

    			lat_cen_blending = []
    			lon_cen_blending = []
    			for x in range (0,len_df_blending):
    				grid_cell_x = grid_cell_blending[x]
    				dframe_geom_gcell = dframe_geom[dframe_geom['Grid Cell'] == grid_cell_x]
    				lat_cen_x = dframe_geom_gcell['Lat Cen'].iloc[0]
    				lon_cen_x = dframe_geom_gcell['Lat Cen'].iloc[0]  
    				lat_cen_blending.append(lat_cen_x)
    				lon_cen_blending.append(lon_cen_x)


    			dframe_blended_raw.insert(2,'Central Lat',lat_cen_blending)
    			dframe_blended_raw.insert(3,'Central Lon',lon_cen_blending)		
    			dframe_blended_raw.insert(4,'In-Situ',station_master_raw_1D)
    			dframe_blended_raw = dframe_blended_raw[['Date','Grid Cell','Central Lat','Central Lon','In-Situ','Naive Blending','CFSR','ERA-Interim','ERA5','JRA-55','MERRA2','GLDAS']]

    			dframe_blended_anom.insert(2,'Central Lat',lat_cen_blending)
    			dframe_blended_anom.insert(3,'Central Lon',lon_cen_blending)		
    			dframe_blended_anom.insert(4,'In-Situ',station_master_all_1D)
    			dframe_blended_anom = dframe_blended_anom[['Date','Grid Cell','Central Lat','Central Lon','In-Situ','Naive Blending','CFSR','ERA-Interim','ERA5','JRA-55','MERRA2','GLDAS']]			
    			#print('Remap Style:',remap_type,' Outlier:',olri,' Layer:',lyrj,' Threshold:',thr_type)
    			#print(dframe_blending)


    			blend_fil_raw = ''.join(['/mnt/data/users/herringtont/soil_temp/Blended_Product/collocated/Naive/raw/'+str(remap_type)+'_'+str(olri)+'_'+str(lyrj)+'_thr'+str(thrk)+'_naive_blending.csv'])
    			print(blend_fil_raw)			
    			path = pathlib.Path(blend_fil_raw)
    			path.parent.mkdir(parents=True, exist_ok=True)
    			dframe_blended_raw.to_csv(blend_fil_raw,index=False)

    			blend_fil_anom = ''.join(['/mnt/data/users/herringtont/soil_temp/Blended_Product/collocated/Naive/anom/'+str(remap_type)+'_'+str(olri)+'_'+str(lyrj)+'_thr'+str(thrk)+'_naive_blending_anom.csv'])
    			print(blend_fil_anom)			
    			path2 = pathlib.Path(blend_fil_anom)
    			path2.parent.mkdir(parents=True, exist_ok=True)
    			dframe_blended_anom.to_csv(blend_fil_anom,index=False)
