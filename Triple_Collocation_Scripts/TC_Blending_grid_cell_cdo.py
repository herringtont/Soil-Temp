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


############################## Reanalysis Products Coverage ################
#CFSR/CFSR2 01/1979 - 09/2020
#ERA-Interim 01/1979 - 08/2019
#ERA5 01/1979 - 12/2018
#JRA-55 01/1958 - 12/2019
#MERRA2 01/1980 - 08/2020
#GLDAS 01/1948 - 07/2020

#### Reanalysis Climatology = 1981-2010
#### Collocated Dates 01/1980 - 12/2018

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
#    			for path in sorted(pathlista, key=lambda path: int(path.stem.split("_")[1].split("*.csv")[0])):
#    				fil = str(path)
#    				dframe = pd.read_csv(fil)
#    				dat_mon = dframe['Date'].tolist()
#    				date_mon = [datetime.datetime.strptime(x,'%Y-%m-%d') for x in dat_mon]
#    				date_mon_CFSR = []
#    				date_mon_CFSR2 = []				
#    				for i in range(0,len(date_mon)):
#    					date_mon_i = date_mon[i]
#    					if (date_mon_i <= datetime.datetime(2010,12,31)):
#    						date_mon_CFSR.append(date_mon_i)
#    					elif (date_mon_i >= datetime.datetime(2011,1,1)):
#    						date_mon_CFSR2.append(date_mon_i)
    			coord_fil = '/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/common_grid/remapnn/common_date/lat_lon_grid.csv'
    			coord_dframe = pd.read_csv(coord_fil)    			
    			lat_cen1 = coord_dframe['Lat'].values
    			lon_cen1 = coord_dframe['Lon'].values
    			grcell = coord_dframe['Grid Cell'].values
    			print(coord_dframe)
    			print(lat_cen1)
    			len_dframe = len(coord_dframe)
    			for l in range(0,len_dframe):
    				gcell = grcell[l]
    				lat_cen = lat_cen1[l]
    				lon_cen = lon_cen1[l]
    				#print("Grid Cell:",gcell)
    				#print(len(dframe))
    				#stemp = dframe['Spatial Avg Anom'].tolist()
    				#stemp_raw = dframe['Spatial Avg Temp'].tolist()
    				#print(date_mon)
    				#print(date_mon_CFSR) 
    				
################################## grab corresponding reanalysis data ##################################
    				base_dir  = "".join(["/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/common_grid/remap",rmph,"/grid_level/"])
    				CFSR_fi = "".join([base_dir,"CFSR/CFSR_all_grid_"+str(gcell),".nc"])
    				MERRA2_fi = "".join([base_dir,"MERRA2/MERRA2_grid_"+str(gcell),".nc"])
    				ERA5_fi = "".join([base_dir,"ERA5/ERA5_grid_"+str(gcell),".nc"])
    				ERAI_fi = "".join([base_dir,"ERA-Interim/ERA-Interim_grid_"+str(gcell),".nc"])
    				JRA_fi = "".join([base_dir,"JRA55/JRA55_grid_"+str(gcell),".nc"])
    				GLDAS_fi = "".join([base_dir,"GLDAS/GLDAS_grid_"+str(gcell),".nc"])
    				#print(CFSR_fi)

    				GLDAS_fil = xr.open_dataset(GLDAS_fi)
    				JRA_fil = xr.open_dataset(JRA_fi)
    				ERAI_fil = xr.open_dataset(ERAI_fi)
    				ERA5_fil = xr.open_dataset(ERA5_fi)
    				MERRA2_fil = xr.open_dataset(MERRA2_fi)
    				CFSR_fil = xr.open_dataset(CFSR_fi) #open NetCDF file with xarray

########### extract soil temperatures and convert to celsius #######
    				GLDAS_stemp = GLDAS_fil[GLDAS_layer] -273.15
    				JRA_stemp = JRA_fil[JRA_layer] - 273.15
    				ERAI_stemp = ERAI_fil[ERAI_layer] - 273.15
    				ERA5_stemp = ERA5_fil[ERA5_layer] - 273.15
    				MERRA2_stemp = MERRA2_fil[MERRA2_layer] - 273.15 #convert from Kelvin to Celsius
    				CFSR_stemp = CFSR_fil[CFSR_layer] - 273.15  #convert from Kelvin to Celsius


########## drop lon,lat coordinates #########

    				GLDAS_stemp3 = GLDAS_stemp.isel(lat=0, lon=0 ,drop=True)
    				JRA_stemp3 = JRA_stemp.isel(lat=0, lon=0 ,drop=True)
    				ERAI_stemp3 = ERAI_stemp.isel(lat=0, lon=0 ,drop=True)
    				ERA5_stemp3 = ERA5_stemp.isel(lat=0, lon=0 ,drop=True)
    				MERRA2_stemp3 = MERRA2_stemp.isel(lat=0, lon=0 ,drop=True)
    				CFSR_stemp3 = CFSR_stemp.isel(lat=0, lon=0 ,drop=True)

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

    				dt_array = np.array([datetime.datetime(1980,1,1) + relativedelta(months=+i) for i in range(468)])

    				for i in range(0,len(dt_array)):
    					date_mon_i = dt_array[i]
    					#print(date_mon_i)			    					

    					if (date_mon_i <= datetime.datetime(2010,12,31)):
    						CFSR_stemp_i = CFSR_stemp3.sel(time=date_mon_i, drop=True).values
    						#print(CFSR_stemp_i)
    						CFSR_new_all.append(CFSR_stemp_i)
    						date_new_all.append(date_mon_i)
    						JRA_stemp_i = JRA_stemp3.sel(time=date_mon_i, drop=True).values
    						JRA_new_all.append(JRA_stemp_i)
    						GLDAS_stemp_i = GLDAS_stemp3.sel(time=date_mon_i, drop=True).values
    						GLDAS_new_all.append(GLDAS_stemp_i)
    						ERA5_stemp_i = ERA5_stemp3.sel(time=date_mon_i, drop=True).values
    						ERA5_new_all.append(ERA5_stemp_i)
    						ERAI_stemp_i = ERAI_stemp3.sel(time=date_mon_i, drop=True).values
    						ERAI_new_all.append(ERAI_stemp_i)
    						MERRA2_stemp_i = MERRA2_stemp3.sel(time=date_mon_i, drop=True).values
    						MERRA2_new_all.append(MERRA2_stemp_i)						
    						    						
    					elif (date_mon_i >= datetime.datetime(2011,1,1)):
    						date_new_all.append(date_mon_i)
    						if (date_mon_i <= datetime.datetime(2019,9,30)):
    							CFSR_stemp_i = CFSR_stemp3.sel(time=date_mon_i, drop=True).values
    							CFSR_new_all.append(CFSR_stemp_i)
    						elif (date_mon_i >= datetime.datetime(2019,10,1)):
    							CFSR_new_all.append(np.nan)
    						if (date_mon_i <= datetime.datetime(2019,12,31)):
    							JRA_stemp_i = JRA_stemp3.sel(time=date_mon_i, drop=True).values
    							JRA_new_all.append(JRA_stemp_i)
    						elif (date_mon_i >= datetime.datetime(2020,1,1)):
    							JRA_stemp_i = JRA_stemp3.sel(time=date_mon_i, drop=True).values
    							JRA_new_all.append(np.nan)
    						if (date_mon_i <= datetime.datetime(2020,7,31)):
    							GLDAS_stemp_i = GLDAS_stemp3.sel(time=date_mon_i, drop=True).values
    							GLDAS_new_all.append(GLDAS_stemp_i)
    						elif (date_mon_i >= datetime.datetime(2020,8,1)):
    							GLDAS_new_all.append(np.nan)
    						if (date_mon_i <= datetime.datetime(2018,12,31)):
    							ERA5_stemp_i = ERA5_stemp3.sel(time=date_mon_i, drop=True).values
    							ERA5_new_all.append(ERA5_stemp_i)
    						elif (date_mon_i >= datetime.datetime(2019,1,1)):
    							ERA5_new_all.append(np.nan)
    						if (date_mon_i <= datetime.datetime(2019,8,31)):
    							ERAI_stemp_i = ERAI_stemp3.sel(time=date_mon_i, drop=True).values
    							ERAI_new_all.append(ERAI_stemp_i)
    						elif (date_mon_i >= datetime.datetime(2019,9,1)):
    							ERAI_new_all.append(np.nan)
    						if (date_mon_i <= datetime.datetime(2020,7,31)):
    							MERRA2_stemp_i = MERRA2_stemp3.sel(time=date_mon_i, drop=True).values
    							MERRA2_new_all.append(MERRA2_stemp_i)
    						elif (date_mon_i >= datetime.datetime(2020,8,1)):
    							MERRA2_new_all.append(np.nan)   

    				date_new_all2 = np.array(date_new_all).flatten()
    				GLDAS_new_all2 = np.array(GLDAS_new_all).flatten()
    				JRA_new_all2 = np.array(JRA_new_all).flatten()
    				ERAI_new_all2 = np.array(ERAI_new_all).flatten()
    				ERA5_new_all2 = np.array(ERA5_new_all).flatten()
    				MERRA2_new_all2 = np.array(MERRA2_new_all).flatten()
    				CFSR_new_all2 = np.array(CFSR_new_all).flatten()

#################### create climatology arrays #######################

    				CFSR_new_clim = []
    				date_new_clim = [] #if there is no CFSR or CFSR2 in triplet				
    				GLDAS_new_clim = [] #if there is no CFSR or CFSR2 in triplet  
    				JRA_new_clim = [] #if there is no CFSR or CFSR2 in triplet 
    				ERAI_new_clim = [] #if there is no CFSR or CFSR2 in triplet 
    				ERA5_new_clim = [] #if there is no CFSR or CFSR2 in triplet 
    				MERRA2_new_clim = [] #if there is no CFSR or CFSR2 in triplet 				

    				dt_clim_array = np.array([datetime.datetime(1981,1,1) + relativedelta(months=+i) for i in range(360)])

    				for i in range(0,len(dt_clim_array)):
    					date_mon_i = dt_clim_array[i]
    					#print(date_mon_i)			    					

    					CFSR_stemp_i = CFSR_stemp3.sel(time=date_mon_i, drop=True).values
    					CFSR_new_clim.append(CFSR_stemp_i)
    					date_new_clim.append(date_mon_i)
    					JRA_stemp_i = JRA_stemp3.sel(time=date_mon_i, drop=True).values
    					JRA_new_clim.append(JRA_stemp_i)
    					GLDAS_stemp_i = GLDAS_stemp3.sel(time=date_mon_i, drop=True).values
    					GLDAS_new_clim.append(GLDAS_stemp_i)
    					ERA5_stemp_i = ERA5_stemp3.sel(time=date_mon_i, drop=True).values
    					ERA5_new_clim.append(ERA5_stemp_i)
    					ERAI_stemp_i = ERAI_stemp3.sel(time=date_mon_i, drop=True).values
    					ERAI_new_clim.append(ERAI_stemp_i)
    					MERRA2_stemp_i = MERRA2_stemp3.sel(time=date_mon_i, drop=True).values
    					MERRA2_new_clim.append(MERRA2_stemp_i)						
    						    						 

    				date_new_clim2 = np.array(date_new_clim).flatten()
    				GLDAS_new_clim2 = np.array(GLDAS_new_clim).flatten()
    				JRA_new_clim2 = np.array(JRA_new_clim).flatten()
    				ERAI_new_clim2 = np.array(ERAI_new_clim).flatten()
    				ERA5_new_clim2 = np.array(ERA5_new_clim).flatten()
    				MERRA2_new_clim2 = np.array(MERRA2_new_clim).flatten()
    				CFSR_new_clim2 = np.array(CFSR_new_clim).flatten()

#################### create anomalies for reanalysis files #######################
    				rnysis_anom_master = []
    				rnysis_date_master = []
    				rnysis_name_master = []
    				rnysis_stemp_master = []
    				rnysis_clim_avg_master =[]
				
    				rnysis = [CFSR_new_clim2,ERAI_new_clim2,ERA5_new_clim2,JRA_new_clim2,MERRA2_new_clim2,GLDAS_new_clim2]
    				rnysis_name = ['CFSR','ERA-Interim','ERA-5','JRA-55','MERRA2','GLDAS']
    				dat_rowlist = date_new_clim2
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

    					if (rnysis_namei == 'CFSR'):
    						rnysisi = CFSR_new_all2
    					if (rnysis_namei == 'ERA-Interim'):
    						rnysisi = ERAI_new_all2
    					if (rnysis_namei == 'ERA-5'):
    						rnysisi = ERA5_new_all2
    					if (rnysis_namei == 'JRA-55'):
    						rnysisi = JRA_new_all2
    					if (rnysis_namei == 'MERRA2'):
    						rnysisi = MERRA2_new_all2
    					if (rnysis_namei == 'GLDAS'):
    						rnysisi = GLDAS_new_all2
 						
    					for k in range (0, len(date_new_all2)):
    						stemp_rw = rnysisi[k]
    						dat_row = date_new_all2[k]
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
    				dframe_anom_master['Reanalysis Product'] = rnysis_name_master					
    				dframe_anom_master['Climatology'] = rnysis_clim_avg_master
    				dframe_anom_master['Soil Temp Anom'] = rnysis_anom_master

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

				
################################# append values to master arrays ######################################

    				date_final = dframe_all['Date']
    				if(date_final.empty == False):
    					date_master_all.append(date_final.tolist())
    	
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



####################### Loop Through Grid Cells #########################

    			master_df = pd.DataFrame(data=date_master_all_1D,columns=['Date'])
    			master_df['Grid Cell'] = grid_master_all_1D
    			master_df['CFSR'] = CFSR_master_all_1D
    			master_df['ERA-Interim'] = ERAI_master_all_1D
    			master_df['ERA5'] = ERA5_master_all_1D
    			master_df['JRA-55'] = JRA_master_all_1D
    			master_df['MERRA2'] = MERRA2_master_all_1D
    			master_df['GLDAS'] = GLDAS_master_all_1D

    			master_df_clim = pd.DataFrame(data=date_master_all_1D,columns=['Date'])
    			master_df_clim['Grid Cell'] = grid_master_all_1D
    			master_df_clim['CFSR'] = CFSR_clim_master_all_1D
    			master_df_clim['ERA-Interim'] = ERAI_clim_master_all_1D
    			master_df_clim['ERA5'] = ERA5_clim_master_all_1D
    			master_df_clim['JRA-55'] = JRA_clim_master_all_1D
    			master_df_clim['MERRA2'] = MERRA2_clim_master_all_1D
    			master_df_clim['GLDAS'] = GLDAS_clim_master_all_1D

    			master_df_raw = pd.DataFrame(data=date_master_all_1D,columns=['Date'])
    			master_df_raw['Grid Cell'] = grid_master_all_1D
    			master_df_raw['CFSR'] = CFSR_master_raw_1D
    			master_df_raw['ERA-Interim'] = ERAI_master_raw_1D
    			master_df_raw['ERA5'] = ERA5_master_raw_1D
    			master_df_raw['JRA-55'] = JRA_master_raw_1D
    			master_df_raw['MERRA2'] = MERRA2_master_raw_1D
    			master_df_raw['GLDAS'] = GLDAS_master_raw_1D






###############################################################################################################
################################################## CALCULATE TC METRICS #######################################
###############################################################################################################
#
#
#
#    			Triplet1_master = []
#    			Triplet2_master = []
#    			Triplet3_master = []
#    			x_bar_master = []
#    			err_x_master = []
#    			err_y_master = []
#    			err_z_master = []
#    			err_x_cov_master = []
#    			err_y_cov_master = []
#    			err_z_cov_master = []
#    			scale_factor_x_master = []
#    			scale_factor_y_master = []
#    			scale_factor_z_master = []
#    			scale_factor_x_cov_master = []
#    			scale_factor_y_cov_master = []
#    			scale_factor_z_cov_master = []
#    			snr_x_master = []
#    			snr_y_master = []
#    			snr_z_master = []
#    			R_xy_master = []
#    			R_yz_master = []
#    			R_xz_master = []
#    			R_x_master = []
#    			R_y_master = []
#    			R_z_master = []
#    			fMSE_x_master = []
#    			fMSE_y_master = []												
#    			fMSE_z_master = []
#
#
#
#
#
#    			date_array = []
#    			CFSR_array_raw = []
#    			ERAI_array_raw = []
#    			ERA5_array_raw = []
#    			JRA_array_raw = []
#    			MERRA2_array_raw = []
#    			GLDAS_array_raw = []
#    			TC_array_raw = []
#    			JRA_wght_array_raw = []
#    			MERRA2_wght_array_raw = []
#    			GLDAS_wght_array_raw = []
#    			naive_array_raw = []
#
#    			CFSR_array_anom = []
#    			ERAI_array_anom = []
#    			ERA5_array_anom = []
#    			JRA_array_anom = []
#    			MERRA2_array_anom = []
#    			GLDAS_array_anom = []
#    			TC_array_anom = []
#    			JRA_wght_array_anom = []
#    			MERRA2_wght_array_anom = []
#    			GLDAS_wght_array_anom = []
#    			naive_array_anom = []
#    			
#    			grid_cell_array = []
#    			sample_size_array = []    			
#    			grid_cell_array_TC = []
#
#    			triplet1_name = 'JRA55'
#    			triplet2_name = 'MERRA2'
#    			triplet3_name = 'GLDAS'
#    			for g in grid_celluq:
#    				grid_df = master_df[master_df['Grid Cell'] == g]
#    				date_df = grid_df['Date']
#    				gcells = grid_df['Grid Cell'].values.tolist()
#    				grid_df2 = grid_df[['JRA-55','MERRA2','GLDAS']]
#    				if (len(grid_df2) == 0):
#    					continue
#    				grid_df_raw = master_df_raw[master_df_raw['Grid Cell'] == g]
#    				grid_df_raw2 = grid_df_raw[['JRA-55','MERRA2','GLDAS']]
#    				if (len(grid_df_raw2) == 0):
#    					continue
#    				grid_df_clim = master_df_clim[master_df_clim['Grid Cell'] == g]
#    				grid_df_clim2 = grid_df_clim[['JRA-55','MERRA2','GLDAS']]
#    				grid_cell_array.append(gcells)
#    				grid_cell_array_TC.append(g)
#    				#date_array.append(date_df)
#
#
#
############################ Anomalies ############################
#    				x = grid_df2['JRA-55'].values
#    				y = grid_df2['MERRA2'].values
#    				z = grid_df2['GLDAS'].values
#    											
################ APPROACH 1 (SCALING) ############
#
#    				x_df = x - np.mean(x)    ####This is the timeseries mean 				
#    				y_df = y - np.mean(y)    ####This is the timeseries mean
#    				z_df = z - np.mean(z)    ####This is the timeseries mean				
#				
#    				beta_ystar = np.mean(x_df*z_df)/np.mean(y_df*z_df) 
#    				beta_zstar = np.mean(x_df*y_df)/np.mean(z_df*y_df) 
#
#    				scaling_factor_Y = 1/beta_ystar ##rescaling factor for Y
#    				scaling_factor_Z = 1/beta_zstar ##rescaling factor for Z
#
#    				x_bar = np.mean(x)
#    				y_bar = np.mean(y)
#    				z_bar = np.mean(z)
#
#    				y_diff = y-y_bar
#    				z_diff = z-z_bar
#
#    				y_rescaled = (beta_ystar*y_diff)+x_bar
#    				z_rescaled = (beta_zstar*z_diff)+x_bar   				
#
#    				err_varx_scaled = np.mean((x-y_rescaled)*(x-z_rescaled)) ## error variance of x using difference notation
#    				err_vary_scaled = np.mean((y_rescaled-x)*(y_rescaled-z_rescaled)) ## error variance of y using difference notation
#    				err_varz_scaled = np.mean((z_rescaled-x)*(z_rescaled-y_rescaled)) ## error variance of z using difference notation
#				   		
#								
#    						#print("***Approach 1 - Scaling***")
#    						#print("Error Variances:")
#    						#print(err_varx_scaled,err_vary_scaled,err_varz_scaled)
#    						#print("Scaling Factors:")
#    						#print(scaling_factor_Y,scaling_factor_Z)
#
#
################ APPROACH 2 (COVARIANCES) ##############
#    				x_std = np.std(x)
#    				y_std = np.std(y)
#    				z_std = np.std(z)
#
#    				signal_varx = (np.cov(x,y)[0][1]*np.cov(x,z)[0][1])/np.cov(y,z)[0][1] ###Signal to Noise Ratio of X (soil temperature sensitivity of the data set) 
#    				signal_vary = (np.cov(y,x)[0][1]*np.cov(y,z)[0][1])/np.cov(x,z)[0][1] ###Signal to Noise Ratio of Y (soil temperature sensitivity of the data set) 
#    				signal_varz = (np.cov(z,x)[0][1]*np.cov(z,y)[0][1])/np.cov(x,y)[0][1] ###Signal to Noise Ratio of Z (soil temperature sensitivity of the data set)
#
#    				err_varx = np.var(x) - signal_varx ##Error variance of dataset X using covariance notation
#    				err_vary = np.var(y) - signal_vary ##Error variance of dataset Y using covariance notation
#    				err_varz = np.var(z) - signal_varz ##Error variance of dataset Z using covariance notation
#
#    				snrx = signal_varx/err_varx    				
#    				snry = signal_vary/err_vary
#    				snrz = signal_varz/err_varz 
#				
#    				nsrx = err_varx/signal_varx ##Noise to Signal Ratio of dataset x
#    				nsry = err_vary/signal_vary ##Noise to Signal Ratio of dataset y
#    				nsrz = err_varz/signal_varz ##Noise to Signal Ratio of dataset z
#
#    				Rxy = 1/math.sqrt((1+nsrx)*(1+nsry)) ##Pearson correlation between dataset X and dataset Y
#    				Ryz = 1/math.sqrt((1+nsry)*(1+nsrz)) ##Pearson correlation between dataset Y and dataset Z
#    				Rxz = 1/math.sqrt((1+nsrx)*(1+nsrz)) ##Pearson correlation between dataset X and dataset Z
#
#    				beta_ystar_cov = np.cov(y,z)[0][1]/np.cov(x,z)[0][1]
#    				beta_zstar_cov = np.cov(y,z)[0][1]/np.cov(x,y)[0][1]
#    				scaling_factor_Y_cov = beta_ystar_cov
#    				scaling_factor_Z_cov = beta_zstar_cov
#
#    						#print("***Approach 2 - Covariance***")
#    						#print("Signal to Noise Ratios:")
#    						#print(snrx,snry,snrz)
#    						#print("Error Variances:")
#    						#print(err_varx,err_vary,err_varz)
#    						#print("Scaling Factor of Y, Scaling Factor of Z:")
#    						#print(scaling_factor_Y, scaling_factor_Z)
#
#    				y_beta_scaled = y * beta_ystar_cov
#    				z_beta_scaled = z * beta_zstar_cov
#
#    				y_rescaled_cov = (beta_ystar_cov*(y - y_bar))+x_bar
#    				z_rescaled_cov = (beta_zstar_cov*(z - z_bar))+x_bar
#
#
#    						#print("Rxy, Ryz, and Rxz:")
#    						#print(Rxy,Ryz,Rxz)
#
#    						#print("Rx, Ry and Rz:")
#
#    				Rx = math.sqrt(snrx/(1+snrx)) ##Correlation between Dataset X and true soil temp 
#    				Ry = math.sqrt(snry/(1+snry)) ##Correlation between Dataset Y and true soil temp 
#    				Rz = math.sqrt(snrz/(1+snrz)) ##Correlation between Dataset Y and true soil temp 
#			
#    						#print(Rx, Ry, Rz)
#
#    						#print("fMSE:")
#    				fMSE_x = 1/(1+snrx)
#    				fMSE_y = 1/(1+snry)
#    				fMSE_z = 1/(1+snrz)
#    						#print(fMSE_x, fMSE_y, fMSE_z)
#    				
#    				Triplet1_master.append(triplet1_name)
#    				Triplet2_master.append(triplet2_name)
#    				Triplet3_master.append(triplet3_name)
#    				x_bar_master.append(x_bar)
#    				err_x_master.append(err_varx_scaled)
#    				err_y_master.append(err_vary_scaled)
#    				err_z_master.append(err_varz_scaled)
#    				err_x_cov_master.append(err_varx)
#    				err_y_cov_master.append(err_vary)
#    				err_z_cov_master.append(err_varz)
#    				scale_factor_x_master.append(1)
#    				scale_factor_y_master.append(scaling_factor_Y)
#    				scale_factor_z_master.append(scaling_factor_Z)
#    				scale_factor_x_cov_master.append(1)
#    				scale_factor_y_cov_master.append(scaling_factor_Y_cov)
#    				scale_factor_z_cov_master.append(scaling_factor_Z_cov)
#    				snr_x_master.append(snrx)
#    				snr_y_master.append(snry)
#    				snr_z_master.append(snrz)
#    				R_xy_master.append(Rxy)
#    				R_yz_master.append(Ryz)
#    				R_xz_master.append(Rxz)
#    				R_x_master.append(Rx)
#    				R_y_master.append(Ry)
#    				R_z_master.append(Rz)
#    				fMSE_x_master.append(fMSE_x)
#    				fMSE_y_master.append(fMSE_y)
#    				fMSE_z_master.append(fMSE_z)																
#
#    			#grid_cell_array_TC = [j for sub in grid_cell_array_TC for j in sub]
#    			#Triplet1_master = [j for sub in Triplet1_master for j in sub]
#    			#Triplet2_master = [j for sub in Triplet2_master for j in sub]
#    			#Triplet1_master = [j for sub in Triplet3_master for j in sub]
#    			#x_bar_master = [j for sub in x_bar_master for j in sub]
##   			err_x_master = [j for sub in err_x_master for j in sub]
##    			err_y_master = [j for sub in err_y_master for j in sub]
##    			err_z_master = [j for sub in err_z_master for j in sub]
##    			err_x_cov_master = [j for sub in err_x_cov_master for j in sub]
##    			err_y_cov_master = [j for sub in err_y_cov_master for j in sub]
##    			err_z_cov_master = [j for sub in err_z_cov_master for j in sub]
##    			scale_factor_x_master = [j for sub in scale_factor_x_master for j in sub]
##    			scale_factor_y_master = [j for sub in scale_factor_y_master for j in sub]
##    			scale_factor_z_master = [j for sub in scale_factor_z_master for j in sub]
##    			scale_factor_x_cov_master = [j for sub in scale_factor_x_cov_master for j in sub]
##    			scale_factor_y_cov_master = [j for sub in scale_factor_y_cov_master for j in sub]
##    			scale_factor_z_cov_master = [j for sub in scale_factor_z_cov_master for j in sub]
##    			snr_x_master = [j for sub in snr_x_master for j in sub]
##    			snr_y_master = [j for sub in snr_y_master for j in sub]
##    			snr_z_master = [j for sub in snr_z_master for j in sub]
##    			R_xy_master = [j for sub in R_xy_master for j in sub]
##    			R_yz_master = [j for sub in R_yz_master for j in sub]
##    			R_xz_master = [j for sub in R_xz_master for j in sub]
##    			R_x_master = [j for sub in R_x_master for j in sub]
##    			R_y_master = [j for sub in R_y_master for j in sub]
##    			R_z_master = [j for sub in R_z_master for j in sub]
##    			fMSE_x_master = [j for sub in fMSE_x_master for j in sub]
##    			fMSE_y_master = [j for sub in fMSE_y_master for j in sub]
##    			fMSE_z_master = [j for sub in fMSE_z_master for j in sub]						
#
#    			#print(Triplet1_master)
#    			dframe_TC = pd.DataFrame(data=grid_cell_array_TC, columns=['Grid Cell'])
#    			dframe_TC['Triplet_1'] = Triplet1_master
#    			dframe_TC['Triplet_2'] = Triplet2_master
#    			dframe_TC['Triplet_3'] = Triplet3_master
#    			dframe_TC['X-bar'] = x_bar_master
#    			dframe_TC['E_x_Scaling'] = err_x_master
#    			dframe_TC['E_y_Scaling'] = err_y_master
#    			dframe_TC['E_z_Scaling'] = err_z_master
#    			dframe_TC['E_x_Cov'] = err_x_cov_master
#    			dframe_TC['E_y_Cov'] = err_y_cov_master
#    			dframe_TC['E_z_Cov'] = err_z_cov_master
#    			dframe_TC['Scale_Factor_x'] = scale_factor_x_master
#    			dframe_TC['Scale_Factor_y'] = scale_factor_y_master
#    			dframe_TC['Scale_Factor_z'] = scale_factor_z_master
#    			dframe_TC['Scale_Factor_x_Cov'] = scale_factor_x_cov_master
#    			dframe_TC['Scale_Factor_y_Cov'] = scale_factor_y_cov_master
#    			dframe_TC['Scale_Factor_z_Cov'] = scale_factor_z_cov_master
#    			dframe_TC['SNR_x'] = snr_x_master
#    			dframe_TC['SNR_y'] = snr_y_master
#    			dframe_TC['SNR_z'] = snr_z_master
#    			dframe_TC['R_xy'] = R_xy_master
#    			dframe_TC['R_yz'] = R_yz_master
#    			dframe_TC['R_xz'] = R_xz_master
#    			dframe_TC['R_x'] = R_x_master
#    			dframe_TC['R_y'] = R_y_master
#    			dframe_TC['R_z'] = R_z_master
#    			dframe_TC['fMSE_x'] = fMSE_x_master
#    			dframe_TC['fMSE_y'] = fMSE_y_master
#    			dframe_TC['fMSE_z'] = fMSE_z_master
#    			dframe_TC['Remap_type'] = remap_type
#    			dframe_TC['Outlier_type'] = olri
#    			dframe_TC['Layer'] = lyrj
#    			dframe_TC['Threshold'] = thrshld
#
#    			TC_odira = "/mnt/data/users/herringtont/soil_temp/TC_Analysis/grid_cell_level/anom/"
#    			TC_ofila = ''.join([TC_odira+str(remap_type)+'_'+str(olri)+'_'+str(lyrj)+'_thr'+str(thrk)+'_TC_output_gcell_level_anom_cdo.csv'])
#    			print(TC_ofila)
#    			#print(dframe_TC)
#    			dframe_TC.to_csv(TC_ofila,index=False)
#
#
#
#
#
#
#
#    			Triplet1_master = []
#    			Triplet2_master = []
#    			Triplet3_master = []
#    			x_bar_master = []
#    			err_x_master = []
#    			err_y_master = []
#    			err_z_master = []
#    			err_x_cov_master = []
#    			err_y_cov_master = []
#    			err_z_cov_master = []
#    			scale_factor_x_master = []
#    			scale_factor_y_master = []
#    			scale_factor_z_master = []
#    			scale_factor_x_cov_master = []
#    			scale_factor_y_cov_master = []
#    			scale_factor_z_cov_master = []
#    			snr_x_master = []
#    			snr_y_master = []
#    			snr_z_master = []
#    			R_xy_master = []
#    			R_yz_master = []
#    			R_xz_master = []
#    			R_x_master = []
#    			R_y_master = []
#    			R_z_master = []
#    			fMSE_x_master = []
#    			fMSE_y_master = []												
#    			fMSE_z_master = []
#
#    			grid_cell_array = []
#    			sample_size_array = []
#
#
#    			grid_cell_array_TC = []
#    			for f in grid_celluq:
#    				grid_df = master_df[master_df['Grid Cell'] == f]
#    				grid_df2 = grid_df[['JRA-55','MERRA2','GLDAS']]
#    				grid_df_raw = master_df_raw[master_df_raw['Grid Cell'] == f]
#    				grid_df_raw2 = grid_df_raw[['JRA-55','MERRA2','GLDAS']]
#    				grid_df_clim = master_df_clim[master_df_clim['Grid Cell'] == f]
#    				grid_df_clim2 = grid_df_clim[['JRA-55','MERRA2','GLDAS']]
#    				sample_size = len(grid_df)
#    				grid_cell_array_TC.append(f)    				
#    				#sample_size_array.append(sample_size)
#
#
#    				triplet1_name = 'JRA-55'
#    				triplet2_name = 'MERRA2'
#    				triplet3_name = 'GLDAS'
#
#
############################ Raw Temps ############################
#    				x = grid_df_raw2['JRA-55'].values
#    				y = grid_df_raw2['MERRA2'].values
#    				z = grid_df_raw2['GLDAS'].values
#    											
################ APPROACH 1 (SCALING) ############
#
#    				x_df = x - np.mean(x)    ####This is the timeseries mean 				
#    				y_df = y - np.mean(y)    ####This is the timeseries mean
#    				z_df = z - np.mean(z)    ####This is the timeseries mean				
#				
#    				beta_ystar = np.mean(x_df*z_df)/np.mean(y_df*z_df) 
#    				beta_zstar = np.mean(x_df*y_df)/np.mean(z_df*y_df) 
#
#    				scaling_factor_Y = 1/beta_ystar ##rescaling factor for Y
#    				scaling_factor_Z = 1/beta_zstar ##rescaling factor for Z
#
#    				x_bar = np.mean(x)
#    				y_bar = np.mean(y)
#    				z_bar = np.mean(z)
#
#    				y_diff = y-y_bar
#    				z_diff = z-z_bar
#
#    				y_rescaled = (beta_ystar*y_diff)+x_bar
#    				z_rescaled = (beta_zstar*z_diff)+x_bar   				
#
#    				err_varx_scaled = np.mean((x-y_rescaled)*(x-z_rescaled)) ## error variance of x using difference notation
#    				err_vary_scaled = np.mean((y_rescaled-x)*(y_rescaled-z_rescaled)) ## error variance of y using difference notation
#    				err_varz_scaled = np.mean((z_rescaled-x)*(z_rescaled-y_rescaled)) ## error variance of z using difference notation
#				   		
#								
#
################ APPROACH 2 (COVARIANCES) ##############
#    				x_std = np.std(x)
#    				y_std = np.std(y)
#    				z_std = np.std(z)
#
#    				signal_varx = (np.cov(x,y)[0][1]*np.cov(x,z)[0][1])/np.cov(y,z)[0][1] ###Signal to Noise Ratio of X (soil temperature sensitivity of the data set) 
#    				signal_vary = (np.cov(y,x)[0][1]*np.cov(y,z)[0][1])/np.cov(x,z)[0][1] ###Signal to Noise Ratio of Y (soil temperature sensitivity of the data set) 
#    				signal_varz = (np.cov(z,x)[0][1]*np.cov(z,y)[0][1])/np.cov(x,y)[0][1] ###Signal to Noise Ratio of Z (soil temperature sensitivity of the data set)
#
#    				err_varx = np.var(x) - signal_varx ##Error variance of dataset X using covariance notation
#    				err_vary = np.var(y) - signal_vary ##Error variance of dataset Y using covariance notation
#    				err_varz = np.var(z) - signal_varz ##Error variance of dataset Z using covariance notation
#
#    				snrx = signal_varx/err_varx    				
#    				snry = signal_vary/err_vary
#    				snrz = signal_varz/err_varz 
#				
#    				nsrx = err_varx/signal_varx ##Noise to Signal Ratio of dataset x
#    				nsry = err_vary/signal_vary ##Noise to Signal Ratio of dataset y
#    				nsrz = err_varz/signal_varz ##Noise to Signal Ratio of dataset z
#
#    				Rxy = 1/math.sqrt((1+nsrx)*(1+nsry)) ##Pearson correlation between dataset X and dataset Y
#    				Ryz = 1/math.sqrt((1+nsry)*(1+nsrz)) ##Pearson correlation between dataset Y and dataset Z
#    				Rxz = 1/math.sqrt((1+nsrx)*(1+nsrz)) ##Pearson correlation between dataset X and dataset Z
#
#    				beta_ystar_cov = np.cov(y,z)[0][1]/np.cov(x,z)[0][1]
#    				beta_zstar_cov = np.cov(y,z)[0][1]/np.cov(x,y)[0][1]
#    				scaling_factor_Y_cov = beta_ystar_cov
#    				scaling_factor_Z_cov = beta_zstar_cov
#
#    				y_beta_scaled = y * beta_ystar_cov
#    				z_beta_scaled = z * beta_zstar_cov
#
#    				y_rescaled_cov = (beta_ystar_cov*(y - y_bar))+x_bar
#    				z_rescaled_cov = (beta_zstar_cov*(z - z_bar))+x_bar
#
#
#
#    				Rx = math.sqrt(snrx/(1+snrx)) ##Correlation between Dataset X and true soil temp 
#    				Ry = math.sqrt(snry/(1+snry)) ##Correlation between Dataset Y and true soil temp 
#    				Rz = math.sqrt(snrz/(1+snrz)) ##Correlation between Dataset Y and true soil temp 
#			
#
#    				fMSE_x = 1/(1+snrx)
#    				fMSE_y = 1/(1+snry)
#    				fMSE_z = 1/(1+snrz)
#
#
#    				
#    				Triplet1_master.append(triplet1_name)
#    				Triplet2_master.append(triplet2_name)
#    				Triplet3_master.append(triplet3_name)
#    				x_bar_master.append(x_bar)
#    				err_x_master.append(err_varx_scaled)
#    				err_y_master.append(err_vary_scaled)
#    				err_z_master.append(err_varz_scaled)
#    				err_x_cov_master.append(err_varx)
#    				err_y_cov_master.append(err_vary)
#    				err_z_cov_master.append(err_varz)
#    				scale_factor_x_master.append(1)
#    				scale_factor_y_master.append(scaling_factor_Y)
#    				scale_factor_z_master.append(scaling_factor_Z)
#    				scale_factor_x_cov_master.append(1)
#    				scale_factor_y_cov_master.append(scaling_factor_Y_cov)
#    				scale_factor_z_cov_master.append(scaling_factor_Z_cov)
#    				snr_x_master.append(snrx)
#    				snr_y_master.append(snry)
#    				snr_z_master.append(snrz)
#    				R_xy_master.append(Rxy)
#    				R_yz_master.append(Ryz)
#    				R_xz_master.append(Rxz)
#    				R_x_master.append(Rx)
#    				R_y_master.append(Ry)
#    				R_z_master.append(Rz)
#    				fMSE_x_master.append(fMSE_x)
#    				fMSE_y_master.append(fMSE_y)
#    				fMSE_z_master.append(fMSE_z)																
#
#    			#grid_cell_array_TC = [j for sub in grid_cell_array_TC for j in sub]
#    			#Triplet1_master = [j for sub in Triplet1_master for j in sub]
#    			#Triplet2_master = [j for sub in Triplet2_master for j in sub]
#    			#Triplet1_master = [j for sub in Triplet3_master for j in sub]
#    			#x_bar_master = [j for sub in x_bar_master for j in sub]
##    			err_x_master = [j for sub in err_x_master for j in sub]
##    			err_y_master = [j for sub in err_y_master for j in sub]
##    			err_z_master = [j for sub in err_z_master for j in sub]
##    			err_x_cov_master = [j for sub in err_x_cov_master for j in sub]
##    			err_y_cov_master = [j for sub in err_y_cov_master for j in sub]
##    			err_z_cov_master = [j for sub in err_z_cov_master for j in sub]
##    			scale_factor_x_master = [j for sub in scale_factor_x_master for j in sub]
##    			scale_factor_y_master = [j for sub in scale_factor_y_master for j in sub]
##    			scale_factor_z_master = [j for sub in scale_factor_z_master for j in sub]
##    			scale_factor_x_cov_master = [j for sub in scale_factor_x_cov_master for j in sub]
##    			scale_factor_y_cov_master = [j for sub in scale_factor_y_cov_master for j in sub]
##    			scale_factor_z_cov_master = [j for sub in scale_factor_z_cov_master for j in sub]
##    			snr_x_master = [j for sub in snr_x_master for j in sub]
##    			snr_y_master = [j for sub in snr_y_master for j in sub]
##    			snr_z_master = [j for sub in snr_z_master for j in sub]
##    			R_xy_master = [j for sub in R_xy_master for j in sub]
##    			R_yz_master = [j for sub in R_yz_master for j in sub]
##    			R_xz_master = [j for sub in R_xz_master for j in sub]
##    			R_x_master = [j for sub in R_x_master for j in sub]
##    			R_y_master = [j for sub in R_y_master for j in sub]
##    			R_z_master = [j for sub in R_z_master for j in sub]
##    			fMSE_x_master = [j for sub in fMSE_x_master for j in sub]
##    			fMSE_y_master = [j for sub in fMSE_y_master for j in sub]
##    			fMSE_z_master = [j for sub in fMSE_z_master for j in sub]
#
#
#    			dframe_TC = pd.DataFrame(data=grid_cell_array_TC, columns=['Grid Cell'])
#    			dframe_TC['Triplet_1'] = Triplet1_master
#    			dframe_TC['Triplet_2'] = Triplet2_master
#    			dframe_TC['Triplet_3'] = Triplet3_master
#    			dframe_TC['X-bar'] = x_bar_master
#    			dframe_TC['E_x_Scaling'] = err_x_master
#    			dframe_TC['E_y_Scaling'] = err_y_master
#    			dframe_TC['E_z_Scaling'] = err_z_master
#    			dframe_TC['E_x_Cov'] = err_x_cov_master
#    			dframe_TC['E_y_Cov'] = err_y_cov_master
#    			dframe_TC['E_z_Cov'] = err_z_cov_master
#    			dframe_TC['Scale_Factor_x'] = scale_factor_x_master
#    			dframe_TC['Scale_Factor_y'] = scale_factor_y_master
#    			dframe_TC['Scale_Factor_z'] = scale_factor_z_master
#    			dframe_TC['Scale_Factor_x_Cov'] = scale_factor_x_cov_master
#    			dframe_TC['Scale_Factor_y_Cov'] = scale_factor_y_cov_master
#    			dframe_TC['Scale_Factor_z_Cov'] = scale_factor_z_cov_master
#    			dframe_TC['SNR_x'] = snr_x_master
#    			dframe_TC['SNR_y'] = snr_y_master
#    			dframe_TC['SNR_z'] = snr_z_master
#    			dframe_TC['R_xy'] = R_xy_master
#    			dframe_TC['R_yz'] = R_yz_master
#    			dframe_TC['R_xz'] = R_xz_master
#    			dframe_TC['R_x'] = R_x_master
#    			dframe_TC['R_y'] = R_y_master
#    			dframe_TC['R_z'] = R_z_master
#    			dframe_TC['fMSE_x'] = fMSE_x_master
#    			dframe_TC['fMSE_y'] = fMSE_y_master
#    			dframe_TC['fMSE_z'] = fMSE_z_master
#    			dframe_TC['Remap_type'] = remap_type
#    			dframe_TC['Outlier_type'] = olri
#    			dframe_TC['Layer'] = lyrj
#    			dframe_TC['Threshold'] = thrshld
#
#    			TC_odirr = "/mnt/data/users/herringtont/soil_temp/TC_Analysis/grid_cell_level/raw/"
#    			TC_ofilr = ''.join([TC_odirr+str(remap_type)+'_'+str(olri)+'_'+str(lyrj)+'_thr'+str(thrk)+'_TC_output_gcell_level_raw_cdo.csv'])
#    			print(TC_ofilr)
#    			#print(dframe_TC)
#    			dframe_TC.to_csv(TC_ofilr,index=False)



###############################################################################################################
################################################## CREATE BLENDED PRODUCT #####################################
###############################################################################################################

    			date_array = []
    			CFSR_array_raw = []
    			ERAI_array_raw = []
    			ERA5_array_raw = []
    			JRA_array_raw = []
    			MERRA2_array_raw = []
    			GLDAS_array_raw = []
    			TC_array_raw = []
    			JRA_wght_array_raw = []
    			MERRA2_wght_array_raw = []
    			GLDAS_wght_array_raw = []
    			naive_array_raw = []

    			CFSR_array_anom = []
    			ERAI_array_anom = []
    			ERA5_array_anom = []
    			JRA_array_anom = []
    			MERRA2_array_anom = []
    			GLDAS_array_anom = []
    			TC_array_anom = []
    			JRA_wght_array_anom = []
    			MERRA2_wght_array_anom = []
    			GLDAS_wght_array_anom = []
    			naive_array_anom = []
    			
    			grid_cell_array = []
    			sample_size_array = []    			



    			for g in grid_celluq:
    				grid_df = master_df[master_df['Grid Cell'] == g]
    				date_df = grid_df['Date']
    				gcells = grid_df['Grid Cell'].values.tolist()
    				grid_df2 = grid_df[['JRA-55','MERRA2','GLDAS']]
    				if (len(grid_df2) == 0):
    					continue
    				grid_df_raw = master_df_raw[master_df_raw['Grid Cell'] == g]
    				grid_df_raw2 = grid_df_raw[['JRA-55','MERRA2','GLDAS']]
    				if (len(grid_df_raw2) == 0):
    					continue
    				grid_df_clim = master_df_clim[master_df_clim['Grid Cell'] == g]
    				grid_df_clim2 = grid_df_clim[['JRA-55','MERRA2','GLDAS']]
    				if (len(grid_df_clim2) == 0):
    					continue
    				grid_cell_array.append(gcells)
    				date_array.append(date_df)




    				TC_dir_raw = "/mnt/data/users/herringtont/soil_temp/TC_Analysis/grid_cell_level/raw/"
    				TC_fil_raw = ''.join([TC_dir_raw+str(remap_type)+'_'+str(olri)+'_'+str(lyrj)+'_thr'+str(thrk)+'_TC_output_gcell_level_raw_cdo.csv'])
    				TC_dframe_raw = pd.read_csv(TC_fil_raw)
    				TC_dframe_raw_gcell = TC_dframe_raw[TC_dframe_raw['Grid Cell'] == g]

    				JRA_dframe_raw_gcell = grid_df_raw2['JRA-55'].values
    				MERRA2_dframe_raw_gcell = grid_df_raw2['MERRA2'].values
    				GLDAS_dframe_raw_gcell = grid_df_raw2['GLDAS'].values

    				JRA_SF_raw = TC_dframe_raw_gcell['Scale_Factor_x_Cov'].values
    				MERRA2_SF_raw = TC_dframe_raw_gcell['Scale_Factor_y_Cov'].values
    				GLDAS_SF_raw = TC_dframe_raw_gcell['Scale_Factor_z_Cov'].values

    				JRA_SNR_raw = TC_dframe_raw_gcell['SNR_x'].values
    				MERRA2_SNR_raw = TC_dframe_raw_gcell['SNR_y'].values
    				GLDAS_SNR_raw = TC_dframe_raw_gcell['SNR_z'].values

    				xbar_raw = TC_dframe_raw_gcell['X-bar']

    				TC_dir_anom = "/mnt/data/users/herringtont/soil_temp/TC_Analysis/grid_cell_level/anom/"
    				TC_fil_anom = ''.join([TC_dir_anom+str(remap_type)+'_'+str(olri)+'_'+str(lyrj)+'_thr'+str(thrk)+'_TC_output_gcell_level_anom_cdo.csv'])
    				TC_dframe_anom = pd.read_csv(TC_fil_anom)
    				TC_dframe_anom_gcell = TC_dframe_anom[TC_dframe_anom['Grid Cell'] == g]			
						
    				JRA_dframe_anom_gcell = grid_df2['JRA-55'].values
    				MERRA2_dframe_anom_gcell = grid_df2['MERRA2'].values
    				GLDAS_dframe_anom_gcell = grid_df2['GLDAS'].values    				

    				JRA_SF_anom = TC_dframe_anom_gcell['Scale_Factor_x_Cov'].values
    				MERRA2_SF_anom = TC_dframe_anom_gcell['Scale_Factor_y_Cov'].values
    				GLDAS_SF_anom = TC_dframe_anom_gcell['Scale_Factor_z_Cov'].values

    				JRA_SNR_anom = TC_dframe_anom_gcell['SNR_x'].values
    				MERRA2_SNR_anom = TC_dframe_anom_gcell['SNR_y'].values
    				GLDAS_SNR_anom = TC_dframe_anom_gcell['SNR_z'].values

    				xbar_anom = TC_dframe_anom_gcell['X-bar']





######################## Scale individual datasets ############

    				JRA_scaled_anom = JRA_dframe_anom_gcell  ## JRA55 doesn't need to be rescaled here because it is the reference 			
    				MERRA2_scaled_anom = [a*MERRA2_SF_anom for a in MERRA2_dframe_anom_gcell]
    				MERRA2_scaled_anom = [b+xbar_anom for b in MERRA2_scaled_anom]
    				GLDAS_scaled_anom = [a*GLDAS_SF_anom for a in GLDAS_dframe_anom_gcell]
    				GLDAS_scaled_anom = [b+xbar_anom for b in GLDAS_scaled_anom]

    				JRA_scaled_raw = JRA_dframe_raw_gcell  ## JRA55 doesn't need to be rescaled here because it is the reference
    				MERRA2_scaled_raw = [a*MERRA2_SF_raw for a in MERRA2_dframe_raw_gcell]
    				MERRA2_scaled_raw = [b+xbar_raw for b in MERRA2_scaled_raw]
    				GLDAS_scaled_raw = [a*GLDAS_SF_raw for a in GLDAS_dframe_raw_gcell]
    				GLDAS_scaled_raw = [b+xbar_raw for b in GLDAS_scaled_raw]

    				#print(MERRA2_scaled_raw)				  			
###################### Calculate weighted averages and create blended product ###############

#### calculate product weights based on SNR values ####

#### Anomalies ####
    				wght_denom_a = JRA_SNR_anom + MERRA2_SNR_anom + GLDAS_SNR_anom
    				print(wght_denom_a)				
    				JRA_wght_a = JRA_SNR_anom/wght_denom_a
    				print(JRA_wght_a)
    				MERRA2_wght_a = MERRA2_SNR_anom/wght_denom_a
    				print(MERRA2_wght_a)
    				GLDAS_wght_a = GLDAS_SNR_anom/wght_denom_a
    				print(GLDAS_wght_a)
    				print(JRA_wght_a+MERRA2_wght_a+GLDAS_wght_a)

    				d = 0
    				while d < len(JRA_scaled_anom):
    					JRA_wght_array_anom.append(JRA_wght_a)
    					MERRA2_wght_array_anom.append(MERRA2_wght_a)
    					GLDAS_wght_array_anom.append(GLDAS_wght_a)  
    					d += 1

    				#print('Grid Cell:',e)
    				#print('Scaling Factors (Anomalies):')
    				#print('MERRA2:',MERRA2_SF_anom)
    				#print('GLDAS:',GLDAS_SF_anom)

    				#print('Weightings (Anomalies):')
    				#print('JRA:',JRA_wght_a)  
    				#print('MERRA2:',MERRA2_wght_a)
    				#print('GLDAS:',GLDAS_wght_a)			

#### Raw Temps ####
    				wght_denom_r = JRA_SNR_raw + MERRA2_SNR_raw + GLDAS_SNR_raw
    				JRA_wght_r = JRA_SNR_raw/wght_denom_r
    				MERRA2_wght_r = MERRA2_SNR_raw/wght_denom_r
    				GLDAS_wght_r = GLDAS_SNR_raw/wght_denom_r

    				d = 0
    				while d < len(JRA_scaled_raw):
    					JRA_wght_array_raw.append(JRA_wght_r)
    					MERRA2_wght_array_raw.append(MERRA2_wght_r)
    					GLDAS_wght_array_raw.append(GLDAS_wght_r)  
    					d += 1

    				#print('Scaling Factors (Raw):')
    				#print('MERRA2:',MERRA2_SF_anom)
    				#print('GLDAS:',GLDAS_SF_anom)

    				#print('Weightings (Raw):')
    				#print('JRA:',JRA_wght_a)  
    				#print('MERRA2:',MERRA2_wght_a)
    				#print('GLDAS:',GLDAS_wght_a)


#### create TC blended product ####

    				blended_anom = []
    				for m in range(0,len(JRA_scaled_anom)):
    					JRAi = JRA_wght_a*JRA_scaled_anom[m]
    					MERRA2i = MERRA2_wght_a*MERRA2_scaled_anom[m]
    					GLDASi = GLDAS_wght_a*GLDAS_scaled_anom[m]
    					blended_anom_i = JRAi+MERRA2i+GLDASi
    					blended_anom_i = blended_anom_i.tolist()
    					blended_anom.append(blended_anom_i)
    					#print(blended_anom_i)
    				blended_anom = [i for sub in blended_anom for i in sub]
    				TC_array_anom.append(blended_anom)
    				#print(blended_anom)

    				blended_raw = []
    				for m in range(0,len(JRA_scaled_raw)):
    					JRAi = JRA_wght_a*JRA_scaled_raw[m]
    					MERRA2i = MERRA2_wght_a*MERRA2_scaled_raw[m]
    					GLDASi = GLDAS_wght_a*GLDAS_scaled_raw[m]
    					blended_raw_i = JRAi+MERRA2i+GLDASi
    					blended_raw_i = blended_raw_i.tolist()
    					blended_raw.append(blended_raw_i)
    					#print(blended_raw_i)
    				blended_raw = [i for sub in blended_raw for i in sub]
    				TC_array_raw.append(blended_raw)


#### create naive blended product ####

    				naive_anom = (JRA_dframe_anom_gcell + MERRA2_dframe_anom_gcell + GLDAS_dframe_anom_gcell)/3
    				naive_anom = naive_anom.tolist()
    				naive_array_anom.append(naive_anom)
    				#print(len(naive_anom))
    				naive_raw = (JRA_dframe_raw_gcell + MERRA2_dframe_raw_gcell + GLDAS_dframe_raw_gcell)/3
    				naive_raw = naive_raw.tolist()
    				naive_array_raw.append(naive_raw)

#### grab grid cell level product values ####

    				CFSR_raw = grid_df_raw['CFSR']
    				CFSR_raw = CFSR_raw.tolist()
    				CFSR_array_raw.append(CFSR_raw)

    				CFSR_anom = grid_df['CFSR']
    				CFSR_anom = CFSR_anom.tolist()
    				CFSR_array_anom.append(CFSR_anom)

    				ERAI_raw = grid_df_raw['ERA-Interim']
    				ERAI_raw = ERAI_raw.tolist()
    				ERAI_array_raw.append(ERAI_raw)

    				ERAI_anom = grid_df['ERA-Interim']
    				ERAI_anom = ERAI_anom.tolist()
    				ERAI_array_anom.append(ERAI_anom)

    				ERA5_raw = grid_df_raw['ERA5']
    				ERA5_raw = ERA5_raw.tolist()
    				ERA5_array_raw.append(ERA5_raw)

    				ERA5_anom = grid_df['ERA5']
    				ERA5_anom = ERA5_anom.tolist()
    				ERA5_array_anom.append(ERA5_anom)

    				JRA_raw = grid_df_raw['JRA-55']
    				JRA_raw = JRA_raw.tolist()
    				JRA_array_raw.append(JRA_raw)

    				JRA_anom = grid_df['JRA-55']
    				JRA_anom = JRA_anom.tolist()
    				JRA_array_anom.append(JRA_anom)

    				MERRA2_raw = grid_df_raw['MERRA2']
    				MERRA2_raw = MERRA2_raw.tolist()
    				MERRA2_array_raw.append(MERRA2_raw)

    				MERRA2_anom = grid_df['MERRA2']
    				MERRA2_anom = MERRA2_anom.tolist()
    				MERRA2_array_anom.append(MERRA2_anom)

    				GLDAS_raw = grid_df_raw['GLDAS']
    				GLDAS_raw = GLDAS_raw.tolist()
    				GLDAS_array_raw.append(GLDAS_raw)

    				GLDAS_anom = grid_df['GLDAS']
    				GLDAS_anom = GLDAS_anom.tolist()
    				GLDAS_array_anom.append(GLDAS_anom)

    			JRA_wght_array_anom = [ i for sub in JRA_wght_array_anom for i in sub]
    			JRA_wght_array_raw = [ i for sub in JRA_wght_array_raw for i in sub]
    			MERRA2_wght_array_anom = [ i for sub in MERRA2_wght_array_anom for i in sub]
    			MERRA2_wght_array_raw = [ i for sub in MERRA2_wght_array_raw for i in sub]
    			GLDAS_wght_array_anom = [ i for sub in GLDAS_wght_array_anom for i in sub]
    			GLDAS_wght_array_raw = [ i for sub in GLDAS_wght_array_raw for i in sub]
    			grid_cell_array = [i for sub in grid_cell_array for i in sub]
    			CFSR_array_anom = [i for sub in CFSR_array_anom for i in sub]
    			CFSR_array_raw = [i for sub in CFSR_array_raw for i in sub]
    			ERAI_array_anom = [i for sub in ERAI_array_anom for i in sub]
    			ERAI_array_raw = [i for sub in ERAI_array_raw for i in sub]			
    			ERA5_array_anom = [i for sub in ERA5_array_anom for i in sub]
    			ERA5_array_raw = [i for sub in ERA5_array_raw for i in sub]
    			JRA_array_anom = [i for sub in JRA_array_anom for i in sub]
    			JRA_array_raw = [i for sub in JRA_array_raw for i in sub]
    			MERRA2_array_anom = [i for sub in MERRA2_array_anom for i in sub]
    			MERRA2_array_raw = [i for sub in MERRA2_array_raw for i in sub]
    			GLDAS_array_anom = [i for sub in GLDAS_array_anom for i in sub]
    			GLDAS_array_raw = [i for sub in GLDAS_array_raw for i in sub]			
    			TC_array_anom = [i for sub in TC_array_anom for i in sub]
    			TC_array_raw = [i for sub in TC_array_raw for i in sub]
    			naive_array_anom = [i for sub in naive_array_anom for i in sub]
    			naive_array_raw = [i for sub in naive_array_raw for i in sub]


			
#### create finalized blended dataframes ####

    			dframe_blended_raw = pd.DataFrame(data=date_master_all_1D,columns=['Date'])
    			dframe_blended_raw['Grid Cell'] = grid_cell_array
    			
    			dframe_blended_anom = pd.DataFrame(data=date_master_all_1D,columns=['Date'])
    			dframe_blended_anom['Grid Cell'] = grid_cell_array

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


    			len_df_blended = len(dframe_blended_raw)
    			grid_cell_blended = dframe_blended_raw['Grid Cell'].values
    			dframe_geom = pd.read_csv(geom_fil)
			
    			lat_cen_blended = []
    			lon_cen_blended = [] 

    			for x in range (0,len_df_blended):
    				grid_cell_x = grid_cell_blended[x]
    				dframe_geom_gcell = dframe_geom[dframe_geom['Grid Cell'] == grid_cell_x]
    				lat_cen_x = dframe_geom_gcell['Lat Cen'].iloc[0]
    				lon_cen_x = dframe_geom_gcell['Lon Cen'].iloc[0]  
    				lat_cen_blended.append(lat_cen_x)
    				lon_cen_blended.append(lon_cen_x)

    			dframe_blended_raw['Central Lat'] = lat_cen_blended
    			dframe_blended_raw['Central Lon'] = lon_cen_blended
    			dframe_blended_raw['JRA weight'] = JRA_wght_array_raw
    			dframe_blended_raw['MERRA2 weight'] = MERRA2_wght_array_raw
    			dframe_blended_raw['GLDAS weight'] = GLDAS_wght_array_raw
    			dframe_blended_raw['TC Blended'] = TC_array_raw
    			dframe_blended_raw['Naive Blended'] = naive_array_raw
    			dframe_blended_raw['CFSR'] = CFSR_array_raw
    			dframe_blended_raw['ERA-Interim'] = ERAI_array_raw    			
    			dframe_blended_raw['ERA5'] = ERA5_array_raw
    			dframe_blended_raw['JRA-55'] = JRA_array_raw
    			dframe_blended_raw['MERRA2'] = MERRA2_array_raw
    			dframe_blended_raw['GLDAS'] = GLDAS_array_raw

    			dframe_blended_anom['Central Lat'] = lat_cen_blended
    			dframe_blended_anom['Central Lon'] = lon_cen_blended
    			dframe_blended_raw['JRA weight'] = JRA_wght_array_anom
    			dframe_blended_raw['MERRA2 weight'] = MERRA2_wght_array_anom
    			dframe_blended_raw['GLDAS weight'] = GLDAS_wght_array_anom
    			dframe_blended_anom['TC Blended'] = TC_array_anom
    			dframe_blended_anom['Naive Blended'] = naive_array_anom
    			dframe_blended_anom['CFSR'] = CFSR_array_anom
    			dframe_blended_anom['ERA-Interim'] = ERAI_array_anom    			
    			dframe_blended_anom['ERA5'] = ERA5_array_anom
    			dframe_blended_anom['JRA-55'] = JRA_array_anom
    			dframe_blended_anom['MERRA2'] = MERRA2_array_anom
    			dframe_blended_anom['GLDAS'] = GLDAS_array_anom

    			print(dframe_blended_anom['TC Blended'])
    			print(dframe_blended_raw['TC Blended'])

    			blend_fil_anom = ''.join(['/mnt/data/users/herringtont/soil_temp/Blended_Product/collocated/TC_blended/anom/'+str(remap_type)+'_'+str(olri)+'_'+str(lyrj)+'_thr'+str(thrk)+'_TC_blended_anom_cdo.csv'])
    			blend_fil_raw = ''.join(['/mnt/data/users/herringtont/soil_temp/Blended_Product/collocated/TC_blended/raw/'+str(remap_type)+'_'+str(olri)+'_'+str(lyrj)+'_thr'+str(thrk)+'_TC_blended_cdo.csv'])
    			print(blend_fil_anom)
    			print(blend_fil_raw)			
    			path = pathlib.Path(blend_fil_anom)
    			path.parent.mkdir(parents=True, exist_ok=True)
    			path2 = pathlib.Path(blend_fil_raw)
    			path2.parent.mkdir(parents=True, exist_ok=True)
    			#dframe_blended_raw.to_csv(blend_fil_raw,index=False)
    			#dframe_blended_anom.to_csv(blend_fil_anom,index=False)















