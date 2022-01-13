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
import scipy.cluster.hierarchy as shc
from calendar import isleap
from dateutil.relativedelta import *
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram,linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
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
#################### create new dataframe with date, station temp, reanalysis temp ###################

    				dframe_all = pd.DataFrame(data=date_new_all2, columns=['Date'])
    				dframe_all['Grid Cell'] = gcell
    				dframe_all['Lat Cen'] = lat_cen
    				dframe_all['Lon Cen'] = lon_cen
    				dframe_all['Station'] = station_new_all2
    				dframe_all['CFSR'] = CFSR_new_all2
    				dframe_all['GLDAS'] = GLDAS_new_all2
    				dframe_all['JRA55'] = JRA_new_all2
    				dframe_all['ERA5'] = ERA5_new_all2
    				dframe_all['ERA-Interim'] = ERAI_new_all2
    				dframe_all['MERRA2'] = MERRA2_new_all2


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
   

#################### Create Final DataFrame for Cluster Analysis ###################

    			dframe_master = pd.DataFrame(data=date_master_all_1D, columns = ['Date'])
    			dframe_master['Grid Cell'] = grid_master_all_1D
    			dframe_master['Station'] = station_master_all_1D
    			dframe_master['CFSR'] = CFSR_master_all_1D
    			dframe_master['ERA-Interim'] = ERAI_master_all_1D
    			dframe_master['ERA-5'] = ERA5_master_all_1D
    			dframe_master['JRA-55'] = JRA_master_all_1D
    			dframe_master['MERRA2'] = MERRA2_master_all_1D
    			dframe_master['GLDAS-Noah'] = GLDAS_master_all_1D			
    			dframe_master['Remap_type'] = remap_type
    			dframe_master['Outlier_type'] = olri
    			dframe_master['Layer'] = lyrj
    			dframe_master['Threshold'] = thrshld

    			#print(dframe_master)


    			dframe_cluster = dframe_master[['Station','CFSR','ERA-Interim','ERA-5','JRA-55','MERRA2','GLDAS-Noah']]
    			Station_cluster = dframe_cluster['Station']
    			CFSR_cluster = dframe_cluster['CFSR']
    			ERAI_cluster = dframe_cluster['ERA-Interim']
    			ERA5_cluster = dframe_cluster['ERA-5']
    			JRA_cluster = dframe_cluster['JRA-55']
    			MERRA2_cluster = dframe_cluster['MERRA2']
    			GLDAS_cluster = dframe_cluster['GLDAS-Noah']
    			dframe_len = len(dframe_cluster)
    			data_matrix = []

    			for i in range (0,dframe_len):
    				row = [Station_cluster.iloc[i],CFSR_cluster.iloc[i],ERAI_cluster.iloc[i],ERA5_cluster.iloc[i],JRA_cluster.iloc[i],MERRA2_cluster.iloc[i],GLDAS_cluster.iloc[i]]
    				data_matrix.append(row)

    				
################### Perform Cluster Analysis #####################

    			names = ['In-Situ','CFSR','ERA-Interim','ERA-5','JRA-55','MERRA2','GLDAS-Noah']
    			plt.figure(figsize=(10,7))
    			plt.title("Model Dendrograms")
    			Z = shc.linkage(data_matrix, method='ward')
    			dend = shc.dendrogram(Z,leaf_rotation=90)
    			dend_file = "".join(["/mnt/data/users/herringtont/soil_temp/plots/dendrograms/dendrograms_"+str(remap_type)+"_"+str(olri)+"_"+str(lyrj)+"_thr"+str(thrk)+".png"])
    			print(dend_file)
    			plt.savefig(dend_file)
    			plt.close()


#### dendrograms suggest 2-3 clusters depending on remapping/outlier method/threshold

########### clusters=2 ##############
#
#    			ClusterA = AgglomerativeClustering(n_clusters=2, affinity='euclidean',method='ward')
#    			ClusterA.fit_predict(dframe_master)
#    			plt.figure(figsize=(10,7))
#    			plt.scatter(dframe_cluster['Station'],dframe_cluster['CFSR'], c=clusterA.labels_)
#    			plt.scatter(dframe_cluster['Station'],dframe_cluster['ERA-Interim'], c = clusterA.labels_)
#    			plt.scatter(dframe_cluster['Station'],dframe_cluster['ERA-5'], c = clusterA.labels_)
#    			plt.scatter(dframe_cluster['Station'],dframe_cluster['JRA-55'], c = clusterA.labels_)
#    			plt.scatter(dframe_cluster['Station'],dframe_cluster['MERRA2'], c = clusterA.labels_)
#    			plt.scatter(dframe_cluster['Station'],dframe_cluster['GLDAS-Noah'], c = clusterA.labels_)
#    			clusterA_file = "".join("/mnt/data/users/herringtont/soil_temp/plots/clusters/2cluster/2cluster_"+str(remap_type)+"_"+str(olri)+"_"+str(lyrj)+"_thr"+str(thrk)+".png"])
#    			print(clusterA_file)
#    			plt.savefig(clusterA_file)
#    			plt.close()
