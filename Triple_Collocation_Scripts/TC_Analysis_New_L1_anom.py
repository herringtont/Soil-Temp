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
    			grid_master_all = []
    			station_master_all = []
    			ERAI_master_all = []
    			ERA5_master_all = []
    			JRA_master_all = []
    			MERRA2_master_all = []
    			GLDAS_master_all = []

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
				
#################### create new dataframe with date, station temp, reanalysis temp ###################

    				dframe_all = pd.DataFrame(data=date_new_all2, columns=['Date'])
    				dframe_all['Grid Cell'] = gcell
    				dframe_all['Lat Cen'] = lat_cen
    				dframe_all['Lon Cen'] = lon_cen
    				dframe_all['Station'] = station_new_all2
    				dframe_all['CFSR'] = dframe_anom_CFSR
    				dframe_all['GLDAS'] = dframe_anom_GLDAS
    				dframe_all['JRA55'] = dframe_anom_JRA
    				dframe_all['ERA5'] = dframe_anom_ERA5
    				dframe_all['ERA-Interim'] = dframe_anom_ERAI
    				dframe_all['MERRA2'] = dframe_anom_MERRA2


############ drop rows with NaN ############
    				dframe_all = dframe_all[dframe_all['CFSR'].notna()]
    				dframe_all = dframe_all[dframe_all['GLDAS'].notna()]
    				dframe_all = dframe_all[dframe_all['JRA55'].notna()]
    				dframe_all = dframe_all[dframe_all['ERA5'].notna()]
    				dframe_all = dframe_all[dframe_all['ERA-Interim'].notna()]
    				dframe_all = dframe_all[dframe_all['MERRA2'].notna()]


    				#print(dframe_all)
################################# append values to master arrays ######################################

    				station_final = dframe_all['Station']
    				if (station_final.empty == False):
    					station_master_all.append(station_final.tolist())
    	
    				grid_final = dframe_all['Grid Cell']
    				if (grid_final.empty == False):
    					grid_master_all.append(grid_final.values.tolist())

    				CFSR_final = dframe_all['CFSR']
    				if (CFSR_final.empty == False):
    					CFSR_master_all.append(CFSR_final.values.tolist())
    
    				ERAI_final = dframe_all['ERA-Interim']
    				if (ERAI_final.empty == False):
    					ERAI_master_all.append(ERAI_final.values.tolist())
    
    				ERA5_final = dframe_all['ERA5']
    				if (ERA5_final.empty == False):
    					ERA5_master_all.append(ERA5_final.values.tolist())
    
    				MERRA2_final = dframe_all['MERRA2']
    				if (MERRA2_final.empty == False):
    					MERRA2_master_all.append(MERRA2_final.values.tolist())
    
    				JRA_final = dframe_all['JRA55']
    				if (JRA_final.empty == False):
    					JRA_master_all.append(JRA_final.values.tolist())
    
    				GLDAS_final = dframe_all['GLDAS']
    				if (GLDAS_final.empty == False):	
    					GLDAS_master_all.append(GLDAS_final.values.tolist())



######################### Flatten Master Lists to 1D ############################

    			grid_master_all_1D = []
    			for sublist in grid_master_all:
    				for item in sublist:
    					grid_master_all_1D.append(item)
				
    			station_master_all_1D = []
    			for sublist in station_master_all:
    				for item in sublist:
    					station_master_all_1D.append(item)
				
    			CFSR_master_all_1D = []
    			for sublist in CFSR_master_all:
    				for item in sublist:
    					CFSR_master_all_1D.append(item)
									
    			ERAI_master_all_1D = []
    			for sublist in ERAI_master_all:
    				for item in sublist:
    					ERAI_master_all_1D.append(item)
				
    			ERA5_master_all_1D = []
    			for sublist in ERA5_master_all:
    				for item in sublist:
    					ERA5_master_all_1D.append(item)
				
    			JRA_master_all_1D = []
    			for sublist in JRA_master_all:
    				for item in sublist:
    					JRA_master_all_1D.append(item)

    			MERRA2_master_all_1D = []
    			for sublist in MERRA2_master_all:
    				for item in sublist:
    					MERRA2_master_all_1D.append(item)

    			GLDAS_master_all_1D = []
    			for sublist in GLDAS_master_all:
    				for item in sublist:
    					GLDAS_master_all_1D.append(item)
   

    			grid_celluq = np.unique(grid_master_all_1D)
    			#print('Number of Unique Grid Cells:', len(grid_celluq)) 
    			#print("Station Master")
    			#print(len(station_master_all_1D))
    			#print("ERA-Interim Master")
    			#print(len(ERAI_master_all_1D))



###################### Triple Collocation Triplets ###################################

    			TC_dir = "/mnt/data/users/herringtont/soil_temp/TC_Analysis/raw_data/"
    			odir ="".join([TC_dir,"remap",rmph,"/",olri,"/",lyrj,"/thr_",thrk,"/"])
    			ofil = "".join([odir,"TC_remap",rmph,"_",olri,"_",lyrj,"_thr_",thrk,".csv"])
    			print(ofil)

    			#In_Situ_Triplet = station_master_all_1D
    			CFSR_Triplet = CFSR_master_all_1D
    			GLDAS_Triplet = GLDAS_master_all_1D
    			ERAI_Triplet = ERAI_master_all_1D
    			ERA5_Triplet = ERA5_master_all_1D
    			JRA_Triplet = JRA_master_all_1D
    			MERRA2_Triplet = MERRA2_master_all_1D

    			Triplet1_master = []
    			Triplet2_master = []
    			Triplet3_master = []
    			x_bar_master = []
    			err_x_master = []
    			err_y_master = []
    			err_z_master = []
    			err_x_cov_master = []
    			err_y_cov_master = []
    			err_z_cov_master = []
    			scale_factor_x_master = []
    			scale_factor_y_master = []
    			scale_factor_z_master = []
    			scale_factor_x_cov_master = []
    			scale_factor_y_cov_master = []
    			scale_factor_z_cov_master = []
    			snr_x_master = []
    			snr_y_master = []
    			snr_z_master = []
    			R_xy_master = []
    			R_yz_master = []
    			R_xz_master = []
    			R_x_master = []
    			R_y_master = []
    			R_z_master = []
    			fMSE_x_master = []
    			fMSE_y_master = []												
    			fMSE_z_master = []

########################### Loop Through Triplets  ###########################

    			Triplet_1 = ['CFSR', 'ERA-Interim', 'ERA5', 'JRA-55', 'MERRA2', 'GLDAS']
    			Triplet_2 = ['CFSR', 'ERA-Interim', 'ERA5', 'JRA-55', 'MERRA2', 'GLDAS']
    			Triplet_3 = ['CFSR', 'ERA-Interim', 'ERA5', 'JRA-55', 'MERRA2', 'GLDAS']
    			
    			for a in Triplet_1:
    				triplet1_name = a
    				if (a == 'CFSR'):
    					triplet1_data = CFSR_master_all_1D
    				elif (a == 'ERA-Interim'):
    					triplet1_data = ERAI_master_all_1D
    				elif (a == 'ERA5'):
    					triplet1_data = ERA5_master_all_1D
    				elif (a == 'JRA-55'):
    					triplet1_data = JRA_master_all_1D
    				elif (a == 'MERRA2'):
    					triplet1_data = MERRA2_master_all_1D
    				elif (a == 'GLDAS'):
    					triplet1_data = GLDAS_master_all_1D


    				for b in Triplet_2:
    					triplet2_name = b
    					if (b == 'CFSR'):
    						triplet2_data = CFSR_master_all_1D
    					elif (b == 'ERA-Interim'):
    						triplet2_data = ERAI_master_all_1D
    					elif (b == 'ERA5'):
    						triplet2_data = ERA5_master_all_1D
    					elif (b == 'JRA-55'):
    						triplet2_data = JRA_master_all_1D
    					elif (b == 'MERRA2'):
    						triplet2_data = MERRA2_master_all_1D
    					elif (b == 'GLDAS'):
    						triplet2_data = GLDAS_master_all_1D

    					for c in Triplet_3:
    						triplet3_name = c
    						if (c == 'CFSR'):
    							triplet3_data = CFSR_master_all_1D
    						elif (c == 'ERA-Interim'):
    							triplet3_data = ERAI_master_all_1D
    						elif (c == 'ERA5'):
    							triplet3_data = ERA5_master_all_1D
    						elif (c == 'JRA-55'):
    							triplet3_data = JRA_master_all_1D
    						elif (c == 'MERRA2'):
    							triplet3_data = MERRA2_master_all_1D
    						elif (c == 'GLDAS'):
    							triplet3_data = GLDAS_master_all_1D

    						if(a == b or a == c or b == c):
    							continue
    						x = np.array(triplet1_data)
    						y = np.array(triplet2_data)
    						z = np.array(triplet3_data)


    						#print("Triplet 1:",triplet1_name," Triplet 2:",triplet2_name," Triplet 3:",triplet3_name)								
########################### APPROACH 1 (SCALING) ################################

    						x_df = x - np.mean(x)    ####This is the timeseries mean 				
    						y_df = y - np.mean(y)    ####This is the timeseries mean
    						z_df = z - np.mean(z)    ####This is the timeseries mean				
				
    						beta_ystar = np.mean(x_df*z_df)/np.mean(y_df*z_df) 
    						beta_zstar = np.mean(x_df*y_df)/np.mean(z_df*y_df) 

    						scaling_factor_Y = 1/beta_ystar ##rescaling factor for Y
    						scaling_factor_Z = 1/beta_zstar ##rescaling factor for Z

    						x_bar = np.mean(x)
    						y_bar = np.mean(y)
    						z_bar = np.mean(z)

    						y_diff = y-y_bar
    						z_diff = z-z_bar

    						y_rescaled = (beta_ystar*y_diff)+x_bar
    						z_rescaled = (beta_zstar*z_diff)+x_bar   				

    						err_varx_scaled = np.mean((x-y_rescaled)*(x-z_rescaled)) ## error variance of x using difference notation
    						err_vary_scaled = np.mean((y_rescaled-x)*(y_rescaled-z_rescaled)) ## error variance of y using difference notation
    						err_varz_scaled = np.mean((z_rescaled-x)*(z_rescaled-y_rescaled)) ## error variance of z using difference notation
				   		
								
    						#print("***Approach 1 - Scaling***")
    						#print("Error Variances:")
    						#print(err_varx_scaled,err_vary_scaled,err_varz_scaled)
    						#print("Scaling Factors:")
    						#print(scaling_factor_Y,scaling_factor_Z)


########################## APPROACH 2 (COVARIANCES) #############################
    						x_std = np.std(x)
    						y_std = np.std(y)
    						z_std = np.std(z)

    						signal_varx = (np.cov(x,y)[0][1]*np.cov(x,z)[0][1])/np.cov(y,z)[0][1] ###Signal to Noise Ratio of X (soil temperature sensitivity of the data set) 
    						signal_vary = (np.cov(y,x)[0][1]*np.cov(y,z)[0][1])/np.cov(x,z)[0][1] ###Signal to Noise Ratio of Y (soil temperature sensitivity of the data set) 
    						signal_varz = (np.cov(z,x)[0][1]*np.cov(z,y)[0][1])/np.cov(x,y)[0][1] ###Signal to Noise Ratio of Z (soil temperature sensitivity of the data set)

    						err_varx = np.var(x) - signal_varx ##Error variance of dataset X using covariance notation
    						err_vary = np.var(y) - signal_vary ##Error variance of dataset Y using covariance notation
    						err_varz = np.var(z) - signal_varz ##Error variance of dataset Z using covariance notation

    						snrx = signal_varx/err_varx    				
    						snry = signal_vary/err_vary
    						snrz = signal_varz/err_varz 
				
    						nsrx = err_varx/signal_varx ##Noise to Signal Ratio of dataset x
    						nsry = err_vary/signal_vary ##Noise to Signal Ratio of dataset y
    						nsrz = err_varz/signal_varz ##Noise to Signal Ratio of dataset z

    						Rxy = 1/math.sqrt((1+nsrx)*(1+nsry)) ##Pearson correlation between dataset X and dataset Y
    						Ryz = 1/math.sqrt((1+nsry)*(1+nsrz)) ##Pearson correlation between dataset Y and dataset Z
    						Rxz = 1/math.sqrt((1+nsrx)*(1+nsrz)) ##Pearson correlation between dataset X and dataset Z

    						beta_ystar_cov = np.cov(y,z)[0][1]/np.cov(x,z)[0][1]
    						beta_zstar_cov = np.cov(y,z)[0][1]/np.cov(x,y)[0][1]
    						scaling_factor_Y_cov = beta_ystar_cov
    						scaling_factor_Z_cov = beta_zstar_cov

    						#print("***Approach 2 - Covariance***")
    						#print("Signal to Noise Ratios:")
    						#print(snrx,snry,snrz)
    						#print("Error Variances:")
    						#print(err_varx,err_vary,err_varz)
    						#print("Scaling Factor of Y, Scaling Factor of Z:")
    						#print(scaling_factor_Y, scaling_factor_Z)

    						y_beta_scaled = y * beta_ystar_cov
    						z_beta_scaled = z * beta_zstar_cov

    						y_rescaled_cov = (beta_ystar_cov*(y - y_bar))+x_bar
    						z_rescaled_cov = (beta_zstar_cov*(z - z_bar))+x_bar


    						#print("Rxy, Ryz, and Rxz:")
    						#print(Rxy,Ryz,Rxz)

    						#print("Rx, Ry and Rz:")

    						Rx = math.sqrt(snrx/(1+snrx)) ##Correlation between Dataset X and true soil temp 
    						Ry = math.sqrt(snry/(1+snry)) ##Correlation between Dataset Y and true soil temp 
    						Rz = math.sqrt(snrz/(1+snrz)) ##Correlation between Dataset Y and true soil temp 
			
    						#print(Rx, Ry, Rz)

    						#print("fMSE:")
    						fMSE_x = 1/(1+snrx)
    						fMSE_y = 1/(1+snry)
    						fMSE_z = 1/(1+snrz)
    						#print(fMSE_x, fMSE_y, fMSE_z)
    				
    						Triplet1_master.append(triplet1_name)
    						Triplet2_master.append(triplet2_name)
    						Triplet3_master.append(triplet3_name)
    						x_bar_master.append(x_bar)
    						err_x_master.append(err_varx_scaled)
    						err_y_master.append(err_vary_scaled)
    						err_z_master.append(err_varz_scaled)
    						err_x_cov_master.append(err_varx)
    						err_y_cov_master.append(err_vary)
    						err_z_cov_master.append(err_varz)
    						scale_factor_x_master.append(1)
    						scale_factor_y_master.append(scaling_factor_Y)
    						scale_factor_z_master.append(scaling_factor_Z)
    						scale_factor_x_cov_master.append(1)
    						scale_factor_y_cov_master.append(scaling_factor_Y_cov)
    						scale_factor_z_cov_master.append(scaling_factor_Z_cov)
    						snr_x_master.append(snrx)
    						snr_y_master.append(snry)
    						snr_z_master.append(snrz)
    						R_xy_master.append(Rxy)
    						R_yz_master.append(Ryz)
    						R_xz_master.append(Rxz)
    						R_x_master.append(Rx)
    						R_y_master.append(Ry)
    						R_z_master.append(Rz)
    						fMSE_x_master.append(fMSE_x)
    						fMSE_y_master.append(fMSE_y)
    						fMSE_z_master.append(fMSE_z)																
				
    			Triplet1_master = np.array(Triplet1_master)
    			Triplet2_master = np.array(Triplet2_master)
    			Triplet3_master = np.array(Triplet3_master)
    			x_bar_master = np.array(x_bar_master)
    			err_x_master = np.array(err_x_master)
    			err_y_master = np.array(err_y_master)
    			err_z_master = np.array(err_z_master)
    			err_x_cov_master = np.array(err_x_cov_master)
    			err_y_cov_master = np.array(err_y_cov_master)
    			err_z_cov_master = np.array(err_z_cov_master)
    			scale_factor_x_master = np.array(scale_factor_x_master)
    			scale_factor_y_master = np.array(scale_factor_y_master)
    			scale_factor_z_master = np.array(scale_factor_z_master)
    			scale_factor_x_cov_master = np.array(scale_factor_x_cov_master)
    			scale_factor_y_cov_master = np.array(scale_factor_y_cov_master)
    			scale_factor_z_cov_master = np.array(scale_factor_z_cov_master)
    			snr_x_master = np.array(snr_x_master)
    			snr_y_master = np.array(snr_y_master)
    			snr_z_master = np.array(snr_z_master)
    			R_xy_master = np.array(R_xy_master)
    			R_yz_master = np.array(R_yz_master)
    			R_xz_master = np.array(R_xz_master)
    			R_x_master = np.array(R_x_master)
    			R_y_master = np.array(R_y_master)
    			R_z_master = np.array(R_z_master)
    			fMSE_x_master = np.array(fMSE_x_master)
    			fMSE_y_master = np.array(fMSE_y_master)												
    			fMSE_z_master = np.array(fMSE_z_master)


    			dframe_TC = pd.DataFrame(data=Triplet1_master, columns=['Triplet_1'])
    			dframe_TC['Triplet_2'] = Triplet2_master
    			dframe_TC['Triplet_3'] = Triplet3_master
    			dframe_TC['X-bar'] = x_bar_master
    			dframe_TC['E_x_Scaling'] = err_x_master
    			dframe_TC['E_y_Scaling'] = err_y_master
    			dframe_TC['E_z_Scaling'] = err_z_master
    			dframe_TC['E_x_Cov'] = err_x_cov_master
    			dframe_TC['E_y_Cov'] = err_y_cov_master
    			dframe_TC['E_z_Cov'] = err_z_cov_master
    			dframe_TC['Scale_Factor_x'] = scale_factor_x_master
    			dframe_TC['Scale_Factor_y'] = scale_factor_y_master
    			dframe_TC['Scale_Factor_z'] = scale_factor_z_master
    			dframe_TC['Scale_Factor_x_Cov'] = scale_factor_x_cov_master
    			dframe_TC['Scale_Factor_y_Cov'] = scale_factor_y_cov_master
    			dframe_TC['Scale_Factor_z_Cov'] = scale_factor_z_cov_master
    			dframe_TC['SNR_x'] = snr_x_master
    			dframe_TC['SNR_y'] = snr_y_master
    			dframe_TC['SNR_z'] = snr_z_master
    			dframe_TC['R_xy'] = R_xy_master
    			dframe_TC['R_yz'] = R_yz_master
    			dframe_TC['R_xz'] = R_xz_master
    			dframe_TC['R_x'] = R_x_master
    			dframe_TC['R_y'] = R_y_master
    			dframe_TC['R_z'] = R_z_master
    			dframe_TC['fMSE_x'] = fMSE_x_master
    			dframe_TC['fMSE_y'] = fMSE_y_master
    			dframe_TC['fMSE_z'] = fMSE_z_master
    			dframe_TC['Remap_type'] = remap_type
    			dframe_TC['Outlier_type'] = olri
    			dframe_TC['Layer'] = lyrj
    			dframe_TC['Threshold'] = thrshld

    			TC_odir = "/mnt/data/users/herringtont/soil_temp/TC_Analysis/anom/"
    			TC_ofil = ''.join([TC_odir+str(remap_type)+'_'+str(olri)+'_'+str(lyrj)+'_thr'+str(thrk)+'_TC_output.csv'])
    			print(TC_ofil)
    			#print(dframe_TC)
    			#dframe_TC.to_csv(TC_ofil,index=False)
			
						
######################## Loop Through Products to extract the product-specific values #######################
    			datasets = ["CFSR", "ERA-Interim", "ERA5", "JRA-55", "MERRA2", "GLDAS"]

    			#InSitu_E_scaling_mean = -1
    			#InSitu_E_cov_mean = -1
    			#InSitu_R_mean = -1
    			#InSitu_fRMSE_mean = -1

    			CFSR_E_scaling = -1
    			CFSR_scale_factor = -1
    			CFSR_E_cov = -1
    			CFSR_R = -1
    			CFSR_fMSE = -1
    			CFSR_E_scaling_mean = -1
    			CFSR_E_cov_mean = -1
    			CFSR_R_mean = -1
    			CFSR_fMSE_mean = -1


    			ERAI_E_scaling = -1
    			ERAI_scale_factor = -1
    			ERAI_E_cov = -1
    			ERAI_R = -1
    			ERAI_fMSE = -1			    			
    			ERAI_E_scaling_mean = -1
    			ERAI_E_cov_mean = -1
    			ERAI_R_mean = -1
    			ERAI_fMSE_mean = -1

    			ERA5_E_scaling = -1
    			ERA5_scale_factor = -1
    			ERA5_E_cov = -1
    			ERA5_R = -1
    			ERA5_fMSE = -1				
    			ERA5_E_scaling_mean = -1
    			ERA5_E_cov_mean = -1
    			ERA5_R_mean = -1
    			ERA5_fMSE_mean = -1


    			JRA_E_scaling = -1
    			JRA_scale_factor = -1
    			JRA_E_cov = -1
    			JRA_R = -1
    			JRA_fMSE = -1	
    			JRA_E_scaling_mean = -1
    			JRA_E_cov_mean = -1
    			JRA_R_mean = -1
    			JRA_fMSE_mean = -1			

    			MERRA2_E_scaling = -1
    			MERRA2_scale_factor = -1
    			MERRA2_E_cov = -1
    			MERRA2_R = -1
    			MERRA2_fMSE = -1
    			MERRA2_E_scaling_mean = -1
    			MERRA2_E_cov_mean = -1
    			MERRA2_R_mean = -1
    			MERRA2_fMSE_mean = -1

    			GLDAS_E_scaling = -1
    			GLDAS_scale_factor = -1
    			GLDAS_E_cov = -1
    			GLDAS_R = -1
    			GLDAS_fMSE = -1
    			GLDAS_E_scaling_mean = -1
    			GLDAS_E_cov_mean = -1
    			GLDAS_R_mean = -1
    			GLDAS_fMSE_mean = -1
			
    			for name in datasets:
    				#print(name)
    				E_scaling = []
    				E_cov = []
    				Pearson_R = []
    				Scale_Factor = []
    				fMSE = []			

    				triplet_cols = ['Triplet_1','Triplet_2','Triplet_3']
    				for i,col in enumerate(triplet_cols):
    					#print(col)
    					if i == 0:
    						E_scaling.append(dframe_TC.loc[(dframe_TC[col] == name)].E_x_Scaling.tolist())
    						E_cov.append(dframe_TC.loc[(dframe_TC[col] == name)].E_x_Cov.tolist())
    						Pearson_R.append(dframe_TC.loc[(dframe_TC[col] == name)].R_x.tolist())
    						fMSE.append(dframe_TC.loc[(dframe_TC[col] == name)].fMSE_x.tolist())
    						
    					elif i == 1:
    						E_scaling.append(dframe_TC.loc[(dframe_TC[col] == name)].E_y_Scaling.tolist())
    						E_cov.append(dframe_TC.loc[(dframe_TC[col] == name)].E_y_Cov.tolist())
    						Pearson_R.append(dframe_TC.loc[(dframe_TC[col] == name)].R_y.tolist())
    						fMSE.append(dframe_TC.loc[(dframe_TC[col] == name)].fMSE_y.tolist())
    						Scale_Factor.append(dframe_TC.loc[(dframe_TC[col] == name)].Scale_Factor_y_Cov.tolist())

    					elif i == 2:
    						E_scaling.append(dframe_TC.loc[(dframe_TC[col] == name)].E_z_Scaling.tolist())
    						E_cov.append(dframe_TC.loc[(dframe_TC[col] == name)].E_z_Cov.tolist())
    						Pearson_R.append(dframe_TC.loc[(dframe_TC[col] == name)].R_z.tolist())
    						fMSE.append(dframe_TC.loc[(dframe_TC[col] == name)].fMSE_z.tolist())
    						Scale_Factor.append(dframe_TC.loc[(dframe_TC[col] == name)].Scale_Factor_z_Cov.tolist())


    				E_scaling = [j for sub in E_scaling for j in sub]
    				E_cov = [j for sub in E_cov for j in sub]
    				Pearson_R = [j for sub in Pearson_R for j in sub]
    				fMSE = [j for sub in fMSE for j in sub]
    				Scale_Factor = [j for sub in Scale_Factor for j in sub]
										
    				#if name == "In-Situ":
    					#InSitu_E_scaling_mean = np.mean(E_scaling)
    					#InSitu_E_cov_mean = np.mean(E_cov)
    					#InSitu_R_mean = np.mean(Pearson_R)
    					#InSitu_fRMSE_mean = np.mean(fMSE)

    				if name == "CFSR":
    					CFSR_E_scaling = E_scaling
    					CFSR_E_cov = E_cov
    					CFSR_R = Pearson_R
    					CFSR_scale_factor = Scale_Factor
    					CFSR_fMSE = fMSE
    					CFSR_E_scaling_mean = np.mean(E_scaling)
    					CFSR_E_cov_mean = np.mean(E_cov)
    					CFSR_R_mean = np.mean(Pearson_R)
    					CFSR_fMSE_mean = np.mean(fMSE)
						
    				if name == "ERA-Interim":
    					ERAI_E_scaling = E_scaling
    					ERAI_E_cov = E_cov
    					ERAI_R = Pearson_R
    					ERAI_scale_factor = Scale_Factor
    					ERAI_fMSE = fMSE
    					ERAI_E_scaling_mean = np.mean(E_scaling)
    					ERAI_E_cov_mean = np.mean(E_cov)
    					ERAI_R_mean = np.mean(Pearson_R)
    					ERAI_fMSE_mean = np.mean(fMSE)

    				if name == "ERA5":
    					ERA5_E_scaling = E_scaling
    					ERA5_E_cov = E_cov
    					ERA5_R = Pearson_R
    					ERA5_scale_factor = Scale_Factor
    					ERA5_fMSE = fMSE
    					ERA5_E_scaling_mean = np.mean(E_scaling)
    					ERA5_E_cov_mean = np.mean(E_cov)
    					ERA5_R_mean = np.mean(Pearson_R)
    					ERA5_fMSE_mean = np.mean(fMSE)

    				if name == "JRA-55":
    					JRA_E_scaling = E_scaling
    					JRA_E_cov = E_cov
    					JRA_R = Pearson_R
    					JRA_scale_factor = Scale_Factor
    					JRA_fMSE = fMSE
    					JRA_E_scaling_mean = np.mean(E_scaling)
    					JRA_E_cov_mean = np.mean(E_cov)
    					JRA_R_mean = np.mean(Pearson_R)
    					JRA_fMSE_mean = np.mean(fMSE)

    				if name == "MERRA2":
    					MERRA2_E_scaling = E_scaling
    					MERRA2_E_cov = E_cov
    					MERRA2_R = Pearson_R
    					MERRA2_scale_factor = Scale_Factor
    					MERRA2_fMSE = fMSE
    					MERRA2_E_scaling_mean = np.mean(E_scaling)
    					MERRA2_E_cov_mean = np.mean(E_cov)
    					MERRA2_R_mean = np.mean(Pearson_R)
    					MERRA2_fMSE_mean = np.mean(fMSE)

    				if name == "GLDAS":
    					GLDAS_E_scaling = E_scaling
    					GLDAS_E_cov = E_cov
    					GLDAS_R = Pearson_R
    					GLDAS_scale_factor = Scale_Factor
    					GLDAS_fMSE = fMSE
    					GLDAS_E_scaling_mean = np.mean(E_scaling)
    					GLDAS_E_cov_mean = np.mean(E_cov)
    					GLDAS_R_mean = np.mean(Pearson_R)
    					GLDAS_fMSE_mean = np.mean(fMSE)


    			#print(CFSR_scale_factor)

    			dframe_scale_factor = pd.DataFrame(data=np.unique(CFSR_scale_factor), columns=['CFSR'])
    			dframe_scale_factor['ERA-Interim'] = np.unique(ERAI_scale_factor)
    			dframe_scale_factor['ERA5'] = np.unique(ERA5_scale_factor)
    			dframe_scale_factor['JRA-55'] = np.unique(JRA_scale_factor)
    			dframe_scale_factor['MERRA2'] = np.unique(MERRA2_scale_factor)
    			dframe_scale_factor['GLDAS'] = np.unique(GLDAS_scale_factor)
    			print(dframe_scale_factor)
    			sns.boxplot(data=dframe_scale_factor).set_title('Variations in TC Scale Factors')
    			dframe_scale_factor.boxplot(column=['CFSR','ERA-Interim','ERA5','JRA-55','MERRA2','GLDAS'])
    			plt.ylabel('Scale Factor')	
    			bplot_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/plots/scale_factor_variance/'+str(remap_type)+'_'+str(olri)+'_'+str(lyrj)+'_thr'+str(thrk)+'_scale_factor_boxplot.png'])
    			path = pathlib.Path(bplot_fil)
    			path.parent.mkdir(parents=True, exist_ok=True)
    			plt.savefig(bplot_fil)
    			plt.close()		
    			#print("******* In-Situ Averages ******")
    			#print("E_scaling:", InSitu_E_scaling_mean)
    			#print("E_cov:", InSitu_E_cov_mean)
    			#print("Pearson_R:", InSitu_R_mean)
    			#print("RMSE:", InSitu_fRMSE_mean)
#
#    			print("******* CFSR Averages ******")
#    			print("E_scaling:", CFSR_E_scaling_mean)
#    			print("E_cov:", CFSR_E_cov_mean)
#    			print("Pearson_R:", CFSR_R_mean)
#    			print("fMSE:", CFSR_fMSE_mean)
#
#    			print("******* ERA-Interim Averages ******")
#    			print("E_scaling:", ERAI_E_scaling_mean)
#    			print("E_cov:", ERAI_E_cov_mean)
#    			print("Pearson_R:", ERAI_R_mean)
#    			print("fMSE:", ERAI_fMSE_mean)
#
#    			print("******* ERA-5 Averages ******")
#    			print("E_scaling:", ERA5_E_scaling_mean)
#    			print("E_cov:", ERA5_E_cov_mean)
#    			print("Pearson_R:", ERA5_R_mean)
#    			print("fMSE:", ERA5_fMSE_mean)
#
#    			print("******* JRA-55 Averages ******")
#    			print("E_scaling:", JRA_E_scaling_mean)
#    			print("E_cov:", JRA_E_cov_mean)
#    			print("Pearson_R:", JRA_R_mean)
#    			print("fMSE:", JRA_fMSE_mean)
#
#    			print("******* MERRA2 Averages ******")
#    			print("E_scaling:", MERRA2_E_scaling_mean)
#    			print("E_cov:", MERRA2_E_cov_mean)
#    			print("Pearson_R:", MERRA2_R_mean)
#    			print("fMSE:", MERRA2_fMSE_mean)
#
#    			print("******* GLDAS Averages ******")
#    			print("E_scaling:", GLDAS_E_scaling_mean)
#    			print("E_cov:", GLDAS_E_cov_mean)
#    			print("Pearson_R:", GLDAS_R_mean)
#    			print("fMSE:", GLDAS_fMSE_mean)						        			
#    			#dframe_TC.to_csv(ofil, index=False)
