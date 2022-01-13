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
    				#print("Grid Cell:",gcell)
    				#print(len(dframe))
    				stemp = dframe['Spatial Avg Anom'].tolist()
    				stemp_raw = dframe['Spatial Avg Temp'].tolist()
    				#print(date_mon)
    				#print(date_mon_CFSR) 
    				
    				for l in gcell:
    					gcell_l = l    				
################################## grab corresponding reanalysis data ##################################
    					base_dir  = "".join(["/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/common_grid/remap",rmph,"/grid_level/"])
    					CFSR_fi = "".join([base_dir,"CFSR/CFSR_all_grid_"+str(gcell_l)+".nc"])
    					MERRA2_fi = "".join([base_dir,"MERRA2/MERRA2_grid_"+str(gcell_l)+".nc"])
    					ERA5_fi = "".join([base_dir,"ERA5/ERA5_grid_"+str(gcell_l)+".nc"])
    					ERAI_fi = "".join([base_dir,"ERA-Interim/ERA-Interim_grid_"+str(gcell_l)+".nc"])
    					JRA_fi = "".join([base_dir,"JRA55/JRA55_grid_"+str(gcell_l)+".nc"])
    					GLDAS_fi = "".join([base_dir,"GLDAS/GLDAS_grid_"+str(gcell_l)+".nc"])
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

    					GLDAS_stemp3 = GLDAS_stemp.isel(lat=0, lon=0, drop=True)
    					JRA_stemp3 = JRA_stemp.isel(lat=0, lon=0, drop=True)
    					ERAI_stemp3 = ERAI_stemp.isel(lat=0, lon=0, drop=True)
    					ERA5_stemp3 = ERA5_stemp.isel(lat=0, lon=0, drop=True)
    					MERRA2_stemp3 = MERRA2_stemp.isel(lat=0, lon=0, drop=True)
    					CFSR_stemp3 = CFSR_stemp.isel(lat=0, lon=0,drop=True)


#################### create anomalies for reanalysis files #######################
    					cdo = Cdo()
    					GLDAS_clim_f = "".join([base_dir,"GLDAS/GLDAS_grid_"+str(gcell_l)+"_clim.nc"])
    					GLDAS_clim_fi = cdo.ymonavg(input="-selyear,1981/2010 %s"% GLDAS_fi, output = GLDAS_clim_f)
    					JRA_clim_f = "".join([base_dir,"JRA55/JRA55_grid_"+str(gcell_l)+"_clim.nc"])
    					JRA_clim_fi = cdo.ymonavg(input="-selyear,1981/2010 %s"% JRA_fi, output = JRA_clim_f)
    					ERAI_clim_f = "".join([base_dir,"ERA-Interim/ERA-Interim_grid_"+str(gcell_l)+"_clim.nc"])
    					ERAI_clim_fi = cdo.ymonavg(input="-selyear,1981/2010 %s"% ERAI_fi, output = ERAI_clim_f)
    					ERA5_clim_f = "".join([base_dir,"ERA5/ERA5_grid_"+str(gcell_l)+"_clim.nc"])
    					ERA5_clim_fi = cdo.ymonavg(input="-selyear,1981/2010 %s"% ERA5_fi, output = ERA5_clim_f)
    					MERRA2_clim_f = "".join([base_dir,"MERRA2/MERRA2_grid_"+str(gcell_l)+"_clim.nc"])
    					MERRA2_clim_fi = cdo.ymonavg(input="-selyear,1981/2010 %s"% MERRA2_fi, output = MERRA2_clim_f)
    					CFSR_clim_f = "".join([base_dir,"CFSR/CFSR_all_grid_"+str(gcell_l)+"_clim.nc"])
    					CFSR_clim_fi = cdo.ymonavg(input="-selyear,1981/2010 %s"% CFSR_fi, output = CFSR_clim_f)

    					GLDAS_anom_f = "".join([base_dir,"GLDAS/GLDAS_grid_"+str(gcell_l)+"_anom.nc"])
    					GLDAS_anom_fi = cdo.ymonsub(input= GLDAS_fi GLDAS_clim_fi, output = GLDAS_anom_f)

##################### create anomalies for reanalysis files #######################
#    					rnysis_anom_master = []
#    					rnysis_date_master = []
#    					rnysis_name_master = []
#    					rnysis_stemp_master = []
#    					rnysis_clim_avg_master =[]
#				
#    					rnysis = [CFSR_new_clim2,ERAI_new_clim2,ERA5_new_clim2,JRA_new_clim2,MERRA2_new_clim2,GLDAS_new_clim2]
#    					rnysis_name = ['CFSR','ERA-Interim','ERA-5','JRA-55','MERRA2','GLDAS']
#    					dat_rowlist = date_new_clim2
#    					num_rows = len(dat_rowlist)
#    					
#					for i in range(0,6):
#    						rnysisi = rnysis[i]
#    						rnysis_namei = rnysis_name[i]
#    						#print("Reanalysis Product:",rnysis_namei)
#    						#print(rnysisi)
#    						climatology = dict()
#    						clim_averages = dict()
#    						stemp_mstr = []
#    						stemp_anom_master = []
#    						date_mstr = []
#    						name_mstr = []
#    						for month in range(1,13):
#    							month_key = f"{month:02}"
#    							climatology[month_key] = list()
#
#    						for j in range(0,num_rows):
#						###add month data to list based on key
#    							dat_row = dat_rowlist[j]
#    							stemp_row = rnysisi[j]
#    							month_key = dat_row.strftime("%m")
#    							climatology[month_key].append(stemp_row)
#
#    						climatology_keys = list(climatology.keys())
#    						climatology_keys2 = np.array(climatology_keys).flatten()
#    						#print(climatology)
#					
#    						for key in climatology:
#						###take averages and write to averages dictionary
#    							current_total = 0
#    							len_current_list = 0
#    							current_list = climatology[key]
#    							for temp in current_list:
#    								if (temp == np.nan):
#    									current_total = current_total + 0
#    									len_current_list = len_current_list + 0
#    								else:
#    									current_total = current_total + temp
#    									len_current_list = len_current_list + 1
#    							if (len_current_list == 0):
#    								average = np.nan
#    							else:
#    								average = current_total/len_current_list
#    							clim_averages[key] = average
#    							#print(average)
#							
#    						clim_avg = list(clim_averages.values())
#    						#print(clim_averages)
#
#    						if (rnysis_namei == 'CFSR'):
#    							rnysisi = CFSR_new_all2
#    						if (rnysis_namei == 'ERA-Interim'):
#    							rnysisi = ERAI_new_all2
#    						if (rnysis_namei == 'ERA-5'):
#    						rnysisi = ERA5_new_all2
#    					if (rnysis_namei == 'JRA-55'):
#    						rnysisi = JRA_new_all2
#    					if (rnysis_namei == 'MERRA2'):
#    						rnysisi = MERRA2_new_all2
#    					if (rnysis_namei == 'GLDAS'):
#    						rnysisi = GLDAS_new_all2
# 						
#    					for k in range (0, len(date_new_all2)):
#    						stemp_rw = rnysisi[k]
#    						dat_row = date_new_all2[k]
#    						dat_row_mon = dat_row.month
#    						dat_row_mons = f"{dat_row_mon:02}"
#    						#print(stemp_rw,dat_row_mon,clim_averages[dat_row_mons])
#    						stemp_anom = stemp_rw - clim_averages[dat_row_mons]
#    						climtgy = clim_averages[dat_row_mons]
#    						rnysis_anom_master.append(stemp_anom)
#    						rnysis_date_master.append(dat_row)					
#    						rnysis_name_master.append(rnysis_namei)
#    						rnysis_stemp_master.append(stemp_rw)
#    						rnysis_clim_avg_master.append(climtgy)    						
#
#
####################### create anom dataframe ########################
#
#    				dframe_anom_master = pd.DataFrame(data=rnysis_date_master, columns=['Date'])
#    				dframe_anom_master['Reanalysis Product'] = rnysis_name_master					
#    				dframe_anom_master['Climatology'] = rnysis_clim_avg_master
#    				dframe_anom_master['Soil Temp Anom'] = rnysis_anom_master
#
#    				dframe_anom_CFSR = dframe_anom_master[dframe_anom_master['Reanalysis Product'] == 'CFSR']
#    				dframe_clim_CFSR = dframe_anom_CFSR['Climatology'].values.tolist()
#    				dframe_anom_CFSR = dframe_anom_CFSR['Soil Temp Anom'].values.tolist()
#
#				
#    				dframe_anom_GLDAS = dframe_anom_master[dframe_anom_master['Reanalysis Product'] == 'GLDAS']
#    				dframe_clim_GLDAS = dframe_anom_GLDAS['Climatology'].values.tolist()
#    				dframe_anom_GLDAS = dframe_anom_GLDAS['Soil Temp Anom'].values.tolist()
#
#
#    				dframe_anom_ERAI = dframe_anom_master[dframe_anom_master['Reanalysis Product'] == 'ERA-Interim']
#    				dframe_clim_ERAI = dframe_anom_ERAI['Climatology'].values.tolist()
#    				dframe_anom_ERAI = dframe_anom_ERAI['Soil Temp Anom'].values.tolist()
#
#
#    				dframe_anom_ERA5 = dframe_anom_master[dframe_anom_master['Reanalysis Product'] == 'ERA-5']
#    				dframe_clim_ERA5 = dframe_anom_ERA5['Climatology'].values.tolist()
#    				dframe_anom_ERA5 = dframe_anom_ERA5['Soil Temp Anom'].values.tolist()
#
#
#    				dframe_anom_JRA = dframe_anom_master[dframe_anom_master['Reanalysis Product'] == 'JRA-55']
#    				dframe_clim_JRA = dframe_anom_JRA['Climatology'].values.tolist()
#    				dframe_anom_JRA = dframe_anom_JRA['Soil Temp Anom'].values.tolist()
#
#
#    				dframe_anom_MERRA2 = dframe_anom_master[dframe_anom_master['Reanalysis Product'] == 'MERRA2']
#    				dframe_clim_MERRA2 = dframe_anom_MERRA2['Climatology'].values.tolist()
#    				dframe_anom_MERRA2 = dframe_anom_MERRA2['Soil Temp Anom'].values.tolist()
#
#
#    								
##################### create new dataframe with date, station temp, reanalysis temp ###################
#
#    				dframe_all = pd.DataFrame(data=date_new_all2, columns=['Date'])
#    				dframe_all['Grid Cell'] = gcell
#    				dframe_all['Lat Cen'] = lat_cen
#    				dframe_all['Lon Cen'] = lon_cen
#    				dframe_all['CFSR'] = dframe_anom_CFSR
#    				dframe_all['CFSR Clim'] = dframe_clim_CFSR
#    				dframe_all['CFSR Raw'] = CFSR_new_all2
#    				dframe_all['GLDAS'] = dframe_anom_GLDAS
#    				dframe_all['GLDAS Clim'] = dframe_clim_GLDAS
#    				dframe_all['GLDAS Raw'] = GLDAS_new_all2
#    				dframe_all['JRA55'] = dframe_anom_JRA
#    				dframe_all['JRA55 Clim'] = dframe_clim_JRA
#    				dframe_all['JRA55 Raw'] = JRA_new_all2
#    				dframe_all['ERA5'] = dframe_anom_ERA5
#    				dframe_all['ERA5 Clim'] = dframe_clim_ERA5
#    				dframe_all['ERA5 Raw'] = ERA5_new_all2
#    				dframe_all['ERA-Interim'] = dframe_anom_ERAI
#    				dframe_all['ERA-Interim Clim'] = dframe_clim_ERAI
#    				dframe_all['ERA-Interim Raw'] = ERAI_new_all2
#    				dframe_all['MERRA2'] = dframe_anom_MERRA2
#    				dframe_all['MERRA2 Clim'] = dframe_clim_MERRA2
#    				dframe_all['MERRA2 Raw'] = MERRA2_new_all2
#
############# drop rows with NaN ############
#    				dframe_all = dframe_all[dframe_all['CFSR'].notna()]
#    				dframe_all = dframe_all[dframe_all['GLDAS'].notna()]
#    				dframe_all = dframe_all[dframe_all['JRA55'].notna()]
#    				dframe_all = dframe_all[dframe_all['ERA5'].notna()]
#    				dframe_all = dframe_all[dframe_all['ERA-Interim'].notna()]
#    				dframe_all = dframe_all[dframe_all['MERRA2'].notna()]
#
#				
################################## append values to master arrays ######################################
#
#    				date_final = dframe_all['Date']
#    				if(date_final.empty == False):
#    					date_master_all.append(date_final.tolist())
#    	
#    				grid_final = dframe_all['Grid Cell']
#    				if (grid_final.empty == False):
#    					grid_master_all.append(grid_final.values.tolist())
#
#    				CFSR_final = dframe_all['CFSR']
#    				if (CFSR_final.empty == False):
#    					CFSR_master_all.append(CFSR_final.values.tolist())
#
#    				CFSR_clim_final = dframe_all['CFSR Clim']
#    				if (CFSR_clim_final.empty == False):
#    					CFSR_clim_master_all.append(CFSR_clim_final.values.tolist())
#
#    				CFSR_final_raw = dframe_all['CFSR Raw']
#    				if (CFSR_final_raw.empty == False):
#    					CFSR_master_raw.append(CFSR_final_raw.tolist())
#    
#    				ERAI_final = dframe_all['ERA-Interim']
#    				if (ERAI_final.empty == False):
#    					ERAI_master_all.append(ERAI_final.values.tolist())
#
#    				ERAI_clim_final = dframe_all['ERA-Interim Clim']
#    				if (ERAI_clim_final.empty == False):
#    					ERAI_clim_master_all.append(ERAI_clim_final.values.tolist())
#
#    				ERAI_final_raw = dframe_all['ERA-Interim Raw']
#    				if (ERAI_final_raw.empty == False):
#    					ERAI_master_raw.append(ERAI_final_raw.tolist())
#    
#    				ERA5_final = dframe_all['ERA5']
#    				if (ERA5_final.empty == False):
#    					ERA5_master_all.append(ERA5_final.values.tolist())
#
#    				ERA5_clim_final = dframe_all['ERA5 Clim']
#    				if (ERA5_clim_final.empty == False):
#    					ERA5_clim_master_all.append(ERA5_clim_final.values.tolist())
#
#    				ERA5_final_raw = dframe_all['ERA5 Raw']
#    				if (ERA5_final_raw.empty == False):
#    					ERA5_master_raw.append(ERA5_final_raw.tolist())
#    
#    				MERRA2_final = dframe_all['MERRA2']
#    				if (MERRA2_final.empty == False):
#    					MERRA2_master_all.append(MERRA2_final.values.tolist())
#
#    				MERRA2_clim_final = dframe_all['MERRA2 Clim']
#    				if (MERRA2_clim_final.empty == False):
#    					MERRA2_clim_master_all.append(MERRA2_clim_final.values.tolist())
#
#    				MERRA2_final_raw = dframe_all['MERRA2 Raw']
#    				if (MERRA2_final_raw.empty == False):
#    					MERRA2_master_raw.append(MERRA2_final_raw.tolist())
#					    
#    				JRA_final = dframe_all['JRA55']
#    				if (JRA_final.empty == False):
#    					JRA_master_all.append(JRA_final.values.tolist())
#					
#    				JRA_clim_final = dframe_all['JRA55 Clim']
#    				if (JRA_clim_final.empty == False):
#    					JRA_clim_master_all.append(JRA_clim_final.values.tolist())
#
#    				JRA_final_raw = dframe_all['JRA55 Raw']
#    				if (JRA_final_raw.empty == False):
#    					JRA_master_raw.append(JRA_final_raw.tolist())
#    
#    				GLDAS_final = dframe_all['GLDAS']
#    				if (GLDAS_final.empty == False):	
#    					GLDAS_master_all.append(GLDAS_final.values.tolist())
#
#    				GLDAS_clim_final = dframe_all['GLDAS Clim']
#    				if (GLDAS_clim_final.empty == False):
#    					GLDAS_clim_master_all.append(GLDAS_clim_final.values.tolist())
#
#    				GLDAS_final_raw = dframe_all['GLDAS Raw']
#    				if (GLDAS_final_raw.empty == False):
#    					GLDAS_master_raw.append(GLDAS_final_raw.tolist())
#
#
########################## Flatten Master Lists to 1D ############################
#
#    			date_master_all_1D = []
#    			for sublist in date_master_all:
#    				for item in sublist:
#    					date_master_all_1D.append(item)
#
#    			grid_master_all_1D = []
#    			for sublist in grid_master_all:
#    				for item in sublist:
#    					grid_master_all_1D.append(item)				
#				
#    			CFSR_master_all_1D = []
#    			for sublist in CFSR_master_all:
#    				for item in sublist:
#    					CFSR_master_all_1D.append(item)
#
#    			CFSR_clim_master_all_1D = []
#    			for sublist in CFSR_clim_master_all:
#    				for item in sublist:
#    					CFSR_clim_master_all_1D.append(item)
#
#    			CFSR_master_raw_1D = []
#    			for sublist in CFSR_master_raw:
#    				for item in sublist:
#    					CFSR_master_raw_1D.append(item)
#									
#    			ERAI_master_all_1D = []
#    			for sublist in ERAI_master_all:
#    				for item in sublist:
#    					ERAI_master_all_1D.append(item)
#
#    			ERAI_clim_master_all_1D = []
#    			for sublist in ERAI_clim_master_all:
#    				for item in sublist:
#    					ERAI_clim_master_all_1D.append(item)
#
#    			ERAI_master_raw_1D = []
#    			for sublist in ERAI_master_raw:
#    				for item in sublist:
#    					ERAI_master_raw_1D.append(item)
#				
#    			ERA5_master_all_1D = []
#    			for sublist in ERA5_master_all:
#    				for item in sublist:
#    					ERA5_master_all_1D.append(item)
#
#    			ERA5_clim_master_all_1D = []
#    			for sublist in ERA5_clim_master_all:
#    				for item in sublist:
#    					ERA5_clim_master_all_1D.append(item)
#
#    			ERA5_master_raw_1D = []
#    			for sublist in ERA5_master_raw:
#    				for item in sublist:
#    					ERA5_master_raw_1D.append(item)
#				
#    			JRA_master_all_1D = []
#    			for sublist in JRA_master_all:
#    				for item in sublist:
#    					JRA_master_all_1D.append(item)
#
#    			JRA_clim_master_all_1D = []
#    			for sublist in JRA_clim_master_all:
#    				for item in sublist:
#    					JRA_clim_master_all_1D.append(item)
#
#    			JRA_master_raw_1D = []
#    			for sublist in JRA_master_raw:
#    				for item in sublist:
#    					JRA_master_raw_1D.append(item)
#
#    			MERRA2_master_all_1D = []
#    			for sublist in MERRA2_master_all:
#    				for item in sublist:
#    					MERRA2_master_all_1D.append(item)
#
#    			MERRA2_clim_master_all_1D = []
#    			for sublist in MERRA2_clim_master_all:
#    				for item in sublist:
#    					MERRA2_clim_master_all_1D.append(item)
#
#    			MERRA2_master_raw_1D = []
#    			for sublist in MERRA2_master_raw:
#    				for item in sublist:
#    					MERRA2_master_raw_1D.append(item)
#
#    			GLDAS_master_all_1D = []
#    			for sublist in GLDAS_master_all:
#    				for item in sublist:
#    					GLDAS_master_all_1D.append(item)
#
#    			GLDAS_clim_master_all_1D = []
#    			for sublist in GLDAS_clim_master_all:
#    				for item in sublist:
#    					GLDAS_clim_master_all_1D.append(item)
#
#    			GLDAS_master_raw_1D = []
#    			for sublist in GLDAS_master_raw:
#    				for item in sublist:
#    					GLDAS_master_raw_1D.append(item)
#
#    			grid_celluq = np.unique(grid_master_all_1D)
#    			#print('Number of Unique Grid Cells:', len(grid_celluq)) 
#    			#print("Station Master")
#    			#print(len(station_master_all_1D))
#    			#print("ERA-Interim Master")
#    			#print(len(ERAI_master_all_1D))
#
#
#
######################## Loop Through Grid Cells #########################
#
#    			master_df = pd.DataFrame(data=date_master_all_1D,columns=['Date'])
#    			master_df['Grid Cell'] = grid_master_all_1D
#    			master_df['CFSR'] = CFSR_master_all_1D
#    			master_df['ERA-Interim'] = ERAI_master_all_1D
#    			master_df['ERA5'] = ERA5_master_all_1D
#    			master_df['JRA-55'] = JRA_master_all_1D
#    			master_df['MERRA2'] = MERRA2_master_all_1D
#    			master_df['GLDAS'] = GLDAS_master_all_1D
#
#    			master_df_clim = pd.DataFrame(data=date_master_all_1D,columns=['Date'])
#    			master_df_clim['Grid Cell'] = grid_master_all_1D
#    			master_df_clim['CFSR'] = CFSR_clim_master_all_1D
#    			master_df_clim['ERA-Interim'] = ERAI_clim_master_all_1D
#    			master_df_clim['ERA5'] = ERA5_clim_master_all_1D
#    			master_df_clim['JRA-55'] = JRA_clim_master_all_1D
#    			master_df_clim['MERRA2'] = MERRA2_clim_master_all_1D
#    			master_df_clim['GLDAS'] = GLDAS_clim_master_all_1D
#
#    			master_df_raw = pd.DataFrame(data=date_master_all_1D,columns=['Date'])
#    			master_df_raw['Grid Cell'] = grid_master_all_1D
#    			master_df_raw['CFSR'] = CFSR_master_raw_1D
#    			master_df_raw['ERA-Interim'] = ERAI_master_raw_1D
#    			master_df_raw['ERA5'] = ERA5_master_raw_1D
#    			master_df_raw['JRA-55'] = JRA_master_raw_1D
#    			master_df_raw['MERRA2'] = MERRA2_master_raw_1D
#    			master_df_raw['GLDAS'] = GLDAS_master_raw_1D
