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
from calendar import isleap
from dateutil.relativedelta import *
from pathlib import Path
from matplotlib.ticker import (MultipleLocator, AutoLocator, AutoMinorLocator)


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
    			date_master_all = []
    			CFSR_master_all = []
    			lat_master_all = []
    			lon_master_all = []
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
    			for path in sorted(pathlist, key=lambda path: int(path.stem.split("_")[1].split("*.csv")[0])):
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
    				stemp = dframe['Spatial Avg'].tolist()
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
    						stemp_anom_master.append(stemp_anom)
    						date_mstr.append(dat_row)
    						stemp_mstr.append(stemp_rw)
    						name_mstr.append(rnysis_namei)

    					rnysis_anom_master.append(stemp_anom_master)
    					rnysis_date_master.append(date_mstr)					
    					rnysis_name_master.append(name_mstr)
    					rnysis_stemp_master.append(stemp_mstr)

    				rnysis_anom_master2 = [j for sub in rnysis_anom_master for j in sub]
    				rnysis_date_master2 = [j for sub in rnysis_date_master for j in sub]
    				rnysis_name_master2 = [j for sub in rnysis_name_master for j in sub]
    				rnysis_stemp_master2 = [j for sub in rnysis_stemp_master for j in sub]
##################### create anom dataframe ########################

    				dframe_anom_master = pd.DataFrame(data=rnysis_date_master2, columns=['Date'])
    				dframe_anom_master['Soil Temp'] = rnysis_stemp_master2
    				dframe_anom_master['Soil Temp Anom'] = rnysis_anom_master2
    				dframe_anom_master['Reanalysis Product'] = rnysis_name_master2
    				print(dframe_anom_master)					


##################### create new dataframe with date, station temp, reanalysis temp ###################

    				dframe_all = pd.DataFrame(data=date_new_all2, columns=['Date'])
    				dframe_all['Grid Cell'] = grid_new_all2
    				dframe_all['Lat Cen'] = lat_cen
    				dframe_all['Lon Cen'] = lon_cen
    				dframe_all['Station'] = station_new_all2
    				dframe_all['CFSR'] = CFSR_new_all2
    				dframe_all['GLDAS'] = GLDAS_new_all2
    				dframe_all['JRA55'] = JRA_new_all2
    				dframe_all['ERA5'] = ERA5_new_all2
    				dframe_all['ERA-Interim'] = ERAI_new_all2
    				dframe_all['MERRA2'] = MERRA2_new_all2
    				print(dframe_all)


############ drop rows with NaN ############
    				dframe_all = dframe_all[dframe_all['CFSR'].notna()]
    				dframe_all = dframe_all[dframe_all['GLDAS'].notna()]
    				dframe_all = dframe_all[dframe_all['JRA55'].notna()]
    				dframe_all = dframe_all[dframe_all['ERA5'].notna()]
    				dframe_all = dframe_all[dframe_all['ERA-Interim'].notna()]
    				dframe_all = dframe_all[dframe_all['MERRA2'].notna()]

################################# append values to master arrays ######################################
    				date_final = dframe_all['Date']
    				if (date_final.empty == False):
    					date_master_all.append(date_final.tolist())

    				lat_final = dframe_all['Lat Cen']
    				if (lat_final.empty == False):
    					lat_master_all.append(lat_final.tolist())
					
    				lon_final = dframe_all['Lon Cen']
    				if (lon_final.empty == False):
    					lon_master_all.append(lon_final.tolist())

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

    			date_master_all_1D = []
    			for sublist in date_master_all:
    				for item in sublist:
    					date_master_all_1D.append(item)

    			lat_master_all_1D = []
    			for sublist in lat_master_all:
    				for item in sublist:
    					lat_master_all_1D.append(item)

    			lon_master_all_1D = []
    			for sublist in lon_master_all:
    				for item in sublist:
    					lon_master_all_1D.append(item)

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
   

################### create master dataframe with all sites for a given remap/outlier/layer/threshold combo ###############

    			dframe_final = pd.DataFrame(data=date_master_all_1D, columns=['Date'])
    			dframe_final['Lat Cen'] = lat_master_all_1D
    			dframe_final['Lon Cen'] = lon_master_all_1D
    			dframe_final['Grid Cell'] = grid_master_all_1D
    			dframe_final['Station'] = station_master_all_1D
    			dframe_final['CFSR'] = CFSR_master_all_1D
    			dframe_final['ERA-Interim'] = ERAI_master_all_1D
    			dframe_final['ERA-5'] = ERA5_master_all_1D
    			dframe_final['JRA-55'] = JRA_master_all_1D
    			dframe_final['MERRA2'] = MERRA2_master_all_1D
    			dframe_final['GLDAS'] = GLDAS_master_all_1D						
    			grid_uq = np.unique(dframe_final['Grid Cell'].values)
    			print("Remap Type: ",rmph,"Outlier Type: ",olri," Threshold: ",thrk)
    			print(len(grid_uq))

################## multiplot ################################
# 12 subplots, 4x3 array in each figure

    			ymin = -45
    			ymax = 45
    			xmin = np.datetime64(datetime.date(1990,1,1),'Y')
    			xmax = np.datetime64(datetime.date(2020,1,1),'Y')

########## number of figures required ###############
#Remapnn
    			if (rmph == "nn"):
    #Outliers
    				if (olri == "outliers"):
    	#Threshold 0: 31 gridcells, 3 figs (last figure = 7 subplots)
    					if (thrk == "0"):
    						numfig = 3
    						lastfig = 7
    						lastrow = 3  			
    	#Threshold 25: 31 gridcells, 3 figs (last figure = 7 subplots)
    					elif (thrk == "25"):
    						numfig = 3
    						lastfig = 7
    						lastrow = 3
    	#Threshold 50: 31 gridcells, 3 figs (last figure = 7 subplots)
    					elif (thrk == "50"):
    						numfig = 3
    						lastfig = 7
    						lastrow = 3
    	#Threshold 75: 30 gridcells, 3 figs (last figure = 6 subplots)
    					elif (thrk == "75"):
    						numfig = 3
    						lastfig = 6
    						lastrow = 2
    	#Threshold 100: 24 gridcells, 2 figs (12 subplots each)
    					elif (thrk == "100"):
    						numfig = 2
    						lastfig = 12
    						lastrow = 4
    #Z-Score
    				elif (olri == "zscore"):
    	#Threshold 0: 31 gridcells, 3 figs (last figure = 7 subplots)
    					if (thrk == "0"):
    						numfig = 3
    						lastfig = 7
    						lastrow = 3						  			
    	#Threshold 25: 31 gridcells, 3 figs (last figure = 7 subplots)
    					elif (thrk == "25"):
    						numfig = 3
    						lastfig = 7
    						lastrow = 3
    	#Threshold 50: 31 gridcells, 3 figs (last figure = 7 subplots)
    					elif (thrk == "50"):
    						numfig = 3
    						lastfig = 7
    						lastrow = 3
    	#Threshold 75: 30 gridcells, 3 figs (last figure = 6 subplots)
    					elif (thrk == "75"):
    						numfig = 3
    						lastfig = 6
    						lastrow = 2
    	#Threshold 100: 24 gridcells, 2 figs (12 subplots each)
    					elif (thrk == "100"):
    						numfig = 2
    						lastfig = 12
    						lastrow = 4 	
    #IQR
    				elif (olri == "IQR"):
    	#Threshold 0: 31 gridcells, 3 figs (last figure = 7 subplots)
    					if (thrk == "0"):
    						numfig = 3
    						lastfig = 7
    						lastrow = 3  			
    	#Threshold 25: 31 gridcells, 3 figs (last figure = 7 subplots)
    					elif (thrk == "25"):
    						numfig = 3
    						lastfig = 7
    						lastrow = 3
    	#Threshold 50: 31 gridcells, 3 figs (last figure = 7 subplots)
    					elif (thrk == "50"):
    						numfig = 3
    						lastfig = 7
    						lastrow = 3
    	#Threshold 75: 30 gridcells, 3 figs (last figure = 6 subplots)
    					elif (thrk == "75"):
    						numfig = 3
    						lastfig = 6
    						lastrow = 2
    	#Threshold 100: 22 gridcells, 2 figs (10 subplots each)
    					elif (thrk == "100"):
    						numfig = 2
    						lastfig = 10
    						lastrow = 3
#Remapnn
    			elif (rmph == "bil"):
    #Outliers
    				if (olri == "outliers"):
    	#Threshold 0: 26 gridcells, 3 figs (last figure = 2 subplots)
    					if (thrk == "0"):
    						numfig = 3
    						lastfig = 2
    						lastrow = 1
    	#Threshold 25: 26 gridcells, 3 figs (last figure = 2 subplots)
    					elif (thrk == "25"):
    						numfig = 3
    						lastfig = 2
    						lastrow = 1
    	#Threshold 50: 26 gridcells, 3 figs (last figure = 2 subplots)
    					elif (thrk == "50"):
    						numfig = 3
    						lastfig = 2
    						lastrow = 1
    	#Threshold 75: 25 gridcells, 3 figs (last figure = 1 subplot)
    					elif (thrk == "75"):
    						numfig = 3
    						lastfig = 1
    						lastrow = 1
    	#Threshold 100: 21 gridcells, 2 figs (last figure = 9 subplots)
    					elif (thrk == "100"):
    						numfig = 2
    						lastfig = 9
    						lastrow = 3
    #Z-Score
    				elif (olri == "zscore"):
    	#Threshold 0: 26 gridcells, 3 figs (last figure = 2 subplots)
    					if (thrk == "0"):
    						numfig = 3
    						lastfig = 2
    						lastrow = 1
    	#Threshold 25: 26 gridcells, 3 figs (last figure = 2 subplots)
    					elif (thrk == "25"):
    						numfig = 3
    						lastfig = 2
    						lastrow = 1
    	#Threshold 50: 26 gridcells, 3 figs (last figure = 2 subplots)
    					elif (thrk == "50"):
    						numfig = 3
    						lastfig = 2
    						lastrow = 1
    	#Threshold 75: 25 gridcells, 3 figs (last figure = 1 subplot)
    					elif (thrk == "75"):
    						numfig = 3
    						lastfig = 1
    						lastrow = 1
    	#Threshold 100: 21 gridcells, 2 figs (last figure = 9 subplots)
    					elif (thrk == "100"):
    						numfig = 2
    						lastfig = 9
    						lastrow = 1 	
    #IQR
    				elif (olri == "IQR"):
    	#Threshold 0: 26 gridcells, 3 figs (last figure = 2 subplots)
    					if (thrk == "0"):
    						numfig = 3
    						lastfig = 2
    						lastrow = 1
    	#Threshold 25: 26 gridcells, 3 figs (last figure = 2 subplots)
    					elif (thrk == "25"):
    						numfig = 3
    						lastfig = 2
    						lastrow = 1
    	#Threshold 50: 26 gridcells, 3 figs (last figure = 2 subplots)
    					elif (thrk == "50"):
    						numfig = 3
    						lastfig = 2
    						lastrow = 1
    	#Threshold 75: 25 gridcells, 3 figs (last figure = 1 subplot)
    					elif (thrk == "75"):
    						numfig = 3
    						lastfig = 1
    						lastrow = 1
    	#Threshold 100: 19 gridcells, 2 figs (last figure = 7 subplots)
    					elif (thrk == "100"):
    						numfig = 2
    						lastfig = 7
    						lastrow = 3
    			print(grid_uq)
########################## create subplots ###################################
    			for i in range (0,numfig):
    				fig = plt.figure()
    				fig,axs = plt.subplots(nrows = 4, ncols = 3, sharex = 'col', sharey = 'row', figsize=(16,12)) # create a figure with 3x4 subplots
	
################### grab data for each site to plot ######################
    				if (i == (numfig-1)):
    					mxrg = lastfig+1
    				elif (i < (numfig-1)):
    					mxrg = 13
    				test = lastfig%3
    				print("MOD =",test)
    				if (lastfig%3 == 0): #if there is no remainder
    					numrow = int(lastfig/3)
    				elif (lastfig%3 != 0):
    					numrow = int((lastfig//3)+1)
    				print("last row =",numrow)
    				totfig = numrow*3 
    				min_grid = grid_uq[(i*12)]
    				max_grid = grid_uq[(i*12)+(mxrg-2)]
    				print(min_grid,max_grid)
    				line_labels = ["Station", "CFSR", "ERA-Interim", "ERA-5", "JRA-55", "MERRA2", "GLDAS-Noah"]   			
    				for j in range (1,mxrg): # 24 subplots per figure unless last figure    					    		
    					j0 = j-1
    					jgrd = (i*12) + j0
    					jgrid = grid_uq[jgrd]
    					dframe_grid = dframe_final[dframe_final['Grid Cell'] == jgrid]
    					date_grid = dframe_grid['Date'].values
    					lat_grid = np.round(dframe_grid['Lat Cen'].iloc[1],decimals=2)
    					lon_grid = np.round(dframe_grid['Lon Cen'].iloc[1],decimals=2)
    					stemp_stn = dframe_grid['Station']
    					stemp_CFSR = dframe_grid['CFSR']
    					stemp_ERAI = dframe_grid['ERA-Interim']
    					stemp_ERA5 = dframe_grid['ERA-5']
    					stemp_JRA = dframe_grid['JRA-55']
    					stemp_MERRA2 = dframe_grid['MERRA2']
    					stemp_GLDAS = dframe_grid['GLDAS']

    					if (i < (numfig-1)): #if figure has 12 subplots
    						ax = plt.subplot(4,3,j)
    					elif (i == (numfig-1)): #else if there are less than 12 subplots in figure
    						ax = plt.subplot(numrow,3,j)
    					ax.plot(date_grid,stemp_stn,label="Station",marker='o',markerfacecolor='dodgerblue',markersize=2,color='royalblue')
    					ax.plot(date_grid,stemp_CFSR,label="CFSR",marker='s',markerfacecolor='chartreuse',markersize=2,color='lawngreen')
    					ax.plot(date_grid,stemp_ERAI,label="ERA-Interim",marker='v',markerfacecolor='darkorchid',markersize=2,color='indigo')
    					ax.plot(date_grid,stemp_ERA5,label="ERA-5",marker='^',markerfacecolor='plum',markersize=2,color='mediumorchid')
    					ax.plot(date_grid,stemp_JRA,label="JRA-55",marker='*',markerfacecolor='orangered',markersize=2,color='red')
    					ax.plot(date_grid,stemp_MERRA2,label="MERRA2",marker='D',markerfacecolor='gold',markersize=2,color='goldenrod')
    					ax.plot(date_grid,stemp_GLDAS,label="GLDAS",marker='x',markerfacecolor='dimgrey',markersize=2,color='black')

    					ax.xaxis.set_major_locator(mdates.YearLocator(5)) #major tick every 5 years
    					ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y')) #only show the year
    					ax.xaxis.set_minor_locator(mdates.YearLocator(1)) #minor tick every year   					
    					ax.yaxis.set_major_locator(MultipleLocator(10)) #every 10 degrees will be a major tick
    					ax.yaxis.set_minor_locator(MultipleLocator(2)) #every 2 degrees will be a minor tick
    					ax.set_xlim(xmin,xmax)
    					ax.set_ylim(ymin,ymax)
    					handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)] 
    					axtext = []
    					axtext.append('Grid Cell: '+str(jgrid)+', Lat: '+str(lat_grid)+'$^\circ$N, Lon :'+str(lon_grid)+'$^\circ$')
    					ax.legend(handles, axtext, loc='best', fontsize = 'small', fancybox=False, framealpha=0, handlelength=0, handletextpad=0) 
    					lines = []
    					labels = []
    					for ax in fig.get_axes():
    						axLine, axLabel = ax.get_legend_handles_labels()
    						lines.extend(axLine)
    						labels.extend(axLabel)
						
    				if (i == (numfig-1)):
    					for k in range(mxrg,totfig+1):
    						plt.subplot(numrow,3,k).set_visible(False)
						
    				fig.add_subplot(111, frameon=False) #create large subplot which will include the plot labels for plots
    				plt.tick_params(labelcolor='none',bottom=False,left=False) #hide ticks
    				plt.xlabel('Date',fontweight='bold')
    				plt.ylabel('Soil Temp ($^\circ$ C)',fontweight='bold')
    				fig.legend(lines[0:6],labels[0:6],loc="right",title="Legend")
				
    				if ( i < (numfig -1)):    			
    					plt.tight_layout()
					   
    				L1fil = "".join(["/mnt/data/users/herringtont/soil_temp/plots/spatial_average/"+str(remap_type)+"/no_outliers/"+olri+"/"+lyrj+"/thr_"+thrk+"/"+str(remap_type)+"_"+str(olri)+"_"+lyrj+"_thr"+str(thrk)+"_grid"+str(min_grid)+"_grid"+str(max_grid)+".png"])
    				print(L1fil)
    				plt.savefig(L1fil)
    				plt.close()
