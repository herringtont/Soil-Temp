import os
import csv
import datetime
import numpy as np
import pandas as pd
import matplotlib as mpl
import xarray as xr
import matplotlib.pyplot as plt
import scipy
import cdo
from cdo import Cdo
import re
import math
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cftime

##################### make master arrays ##################
CFSR_array_L1 = [[]]
CFSR2_array_L1 = [[]]

CFSR_array_L2 = [[]]
CFSR2_array_L2 = [[]]

CFSR_array_L3 = [[]]
CFSR2_array_L3 = [[]]

CFSR_array_L4 = [[]]
CFSR2_array_L4 = [[]] 

##################### set file locations #################
CFSR_fi = "/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/CFSR.nc"
CFSR2_fi = "/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/CFSR2.nc"
obs_loc = str("/mnt/data/users/herringtont/soil_temp/In-Situ/Master_Lat_Long_Obs.csv")

##################### read in lat/long coordinates from in situ station locations ####################
dframe = pd.read_csv(obs_loc)
lat_csv = dframe['Lat']
lon_csv = dframe['Long']
lon_csv2 = ((lon_csv+360)%360).round(decimals=2)


##################### read in data using XArray ####################
CFSR_fil = xr.open_dataset(CFSR_fi)
CFSR2_fil = xr.open_dataset(CFSR2_fi)

##################### read in soil temperatures and time ####################
CFSR_stemp_L1 = CFSR_fil.Soil_Temp_L1
CFSR2_stemp_L1 = CFSR2_fil.Soil_Temp_L1

CFSR_stemp_L2 = CFSR_fil.Soil_Temp_L2
CFSR2_stemp_L2 = CFSR2_fil.Soil_Temp_L2

CFSR_stemp_L3 = CFSR_fil.Soil_Temp_L3
CFSR2_stemp_L3 = CFSR2_fil.Soil_Temp_L3

CFSR_stemp_L4 = CFSR_fil.Soil_Temp_L4
CFSR2_stemp_L4 = CFSR2_fil.Soil_Temp_L4

CFSR_time = CFSR_fil.time
CFSR2_time = CFSR2_fil.time

#################### grab attribute data from file, set soil temp units to deg C
CFSR_time.attrs = CFSR_fil.time.attrs
CFSR2_time.attrs = CFSR2_fil.time.attrs

#grab attribute data from file, set units to deg C
CFSR_stemp_L1.attrs = CFSR_fil.Soil_Temp_L1.attrs
#CFSR_stemp_L1.attrs["units"] = "deg C"
CFSR2_stemp_L1.attrs = CFSR2_fil.Soil_Temp_L1.attrs
#CFSR2_stemp_L1.attrs["units"] = "deg C"

CFSR_stemp_L2.attrs = CFSR_fil.Soil_Temp_L2.attrs
#CFSR_stemp_L2.attrs["units"] = "deg C"
CFSR2_stemp_L2.attrs = CFSR2_fil.Soil_Temp_L2.attrs
#CFSR2_stemp_L2.attrs["units"] = "deg C"

CFSR_stemp_L3.attrs = CFSR_fil.Soil_Temp_L3.attrs
#CFSR_stemp_L3.attrs["units"] = "deg C"
CFSR2_stemp_L3.attrs = CFSR2_fil.Soil_Temp_L3.attrs
#CFSR2_stemp_L3.attrs["units"] = "deg C"

CFSR_stemp_L4.attrs = CFSR_fil.Soil_Temp_L4.attrs
#CFSR_stemp_L4.attrs["units"] = "deg C"
CFSR2_stemp_L4.attrs = CFSR2_fil.Soil_Temp_L4.attrs
#CFSR2_stemp_L4.attrs["units"] = "deg C"

#################### loop through station lat/lon pairs from csv file ########################
#for i in range (0,1):
for i in range (0,299):
    i2 = i+1
    str_i = str(i2)
    latf = lat_csv[i]
    lonf = lon_csv2[i]
    lonf2 = lon_csv[i]
    latfs = str(latf)
    lonfs = str(lonf2)
################### select nearest grid cell from each netCDF file #####################
    CFSR_stemploc_L1 = CFSR_stemp_L1.sel(lon=lonf,lat=latf,method='nearest') - 273.15
    CFSR2_stemploc_L1 = CFSR2_stemp_L1.sel(lon=lonf,lat=latf,method='nearest') - 273.15
    
    CFSR_stemploc_L2 = CFSR_stemp_L2.sel(lon=lonf,lat=latf,method='nearest') - 273.15
    CFSR2_stemploc_L2 = CFSR2_stemp_L2.sel(lon=lonf,lat=latf,method='nearest') - 273.15   
    
    CFSR_stemploc_L3 = CFSR_stemp_L3.sel(lon=lonf,lat=latf,method='nearest') - 273.15
    CFSR2_stemploc_L3 = CFSR2_stemp_L3.sel(lon=lonf,lat=latf,method='nearest') - 273.15
    
    CFSR_stemploc_L4 = CFSR_stemp_L4.sel(lon=lonf,lat=latf,method='nearest') - 273.15
    CFSR2_stemploc_L4 = CFSR2_stemp_L4.sel(lon=lonf,lat=latf,method='nearest') - 273.15
    print(CFSR_stemploc_L1)

#################### fill 2D arrays ######################
    #CFSR_array_L1[0].append(i2)
    CFSR_array_L1[0].append(CFSR_stemploc_L1)
    #CFSR_array_L2[0].append(i2)
    CFSR_array_L2[0].append(CFSR_stemploc_L2)    
    #CFSR_array_L3[0].append(i2)
    CFSR_array_L3[0].append(CFSR_stemploc_L3)
    #CFSR_array_L4[0].append(i2)
    CFSR_array_L4[0].append(CFSR_stemploc_L4)
    
    #CFSR2_array_L1[0].append(i2)
    CFSR2_array_L1[0].append(CFSR2_stemploc_L1)
    #CFSR2_array_L2[0].append(i2)
    CFSR2_array_L2[0].append(CFSR2_stemploc_L2)    
    #CFSR2_array_L3[0].append(i2)
    CFSR2_array_L3[0].append(CFSR2_stemploc_L3)
    #CFSR2_array_L4[0].append(i2)
    CFSR2_array_L4[0].append(CFSR2_stemploc_L4) 

CFSR_array_L1b = np.array(CFSR_array_L1).squeeze()
CFSR_array_L2b = np.array(CFSR_array_L2).squeeze()
CFSR_array_L3b = np.array(CFSR_array_L3).squeeze()
CFSR_array_L4b = np.array(CFSR_array_L4).squeeze()
    
CFSR2_array_L1b = np.array(CFSR2_array_L1).squeeze()
CFSR2_array_L2b = np.array(CFSR2_array_L2).squeeze()
CFSR2_array_L3b = np.array(CFSR2_array_L3).squeeze()
CFSR2_array_L4b = np.array(CFSR2_array_L4).squeeze()

#print(CFSR_array_L1b)

x1 = CFSR_time
x2 = CFSR2_time
xmin = datetime.date(1979,1,1)
xmax = datetime.date(2019,12,31)
#
ymin = -30
ymax = 25
#################### Single Plots
#for i in range (1,300):
#    fig = plt.figure()
#    site_id = i
#    y1 = CFSR_array_L1b[i-1,:]
#    y2 = CFSR2_array_L1b[i-1,:]
#    
#    plt.plot(x1,y1,label = "CFSR", marker='o', markerfacecolor='indigo', markersize=2, color='darkorchid')
#    plt.plot(x2,y2,label = "CFSR2",marker='^', markerfacecolor='red', markersize=2, color='tomato') 
#    plt.xlim(xmin,xmax)
#    plt.ylim(ymin,ymax)
#    plt.xlabel('Date/Time')
#    plt.ylabel('Soil Temp Anomalies ($^\circ$ C)')			
#    plt.title('Site: '+str(site_id)+', Latitude: '+str(latfs)+', Longitude: '+str(lonfs))
#    
#    leg = plt.legend()
#    plt.xticks(rotation=60)
#    pfils = "".join(["/mnt/data/users/herringtont/soil_temp/plots/CFSR_anom2/indiv/site_"+str(site_id)+"_L1.png"])
#    print(pfils)
#    plt.savefig(pfils)
#    plt.close()              
################## Multi-plotting #######################



################# Soil Layer 2 ###############
for i in range (1,26): #we are going to create 25 figures
    fig = plt.figure()

##### add a big subplot for x-label and y-label, hide frame ######
    fig.add_subplot(111, frameon=False)
    ### hide tick and tick label of big axis
    plt.tick_params(labelcolor='none',top=False,bottom=False,left=False,right=False)
    plt.xlabel("Date")
    plt.ylabel("Soil Temp ($^\circ$ C)") 

#    if (i < 19):
#    	fig, axs = plt.subplots(nrows = 4, ncols = 3, sharex='col',sharey='col')
#    elif(i == 19):
#    	fig, axs = plt.subplots(nrows = 3, ncols = 3, sharex='col',sharey='row')
#    if (i == 19):
#    	mxrg = 12
#    else:
#    	mxrg = 17 

    fig,axs = plt.subplots(nrows = 4, ncols = 3, sharex='col',sharey='row',figsize=(12,8))
    
    if (i == 25):
    	mxrg = 12
    else:
    	mxrg = 13
    	    
    for j in range (1,mxrg): #each figure will have 12 subplots (in a 4x3 layout)
    	i0 = i-1
    	#print(i0)
    	i1 = i0*12
    	#print(i1)
    	site_id = j + i1

    	y1 = CFSR_array_L4b[site_id-1,:]
    	y2 = CFSR2_array_L4b[site_id-1,:]
	
    	min_site = 1 + i1
    	if (i == 25):
    		max_site = 299
    	else:
    		max_site = 12 + i1

    	ax = plt.subplot(4,3,j)
    	ax.plot(x1,y1,label = "CFSR", marker='o', markerfacecolor='indigo', markersize=2, color='darkorchid')
    	ax.plot(x2,y2,label = "CFSR2",marker='^', markerfacecolor='red', markersize=2, color='tomato')
    	ax.yaxis.set_major_locator(plt.MaxNLocator(9))
    	ax.yaxis.get_minor_locator()	
    	ax.set_xlim(xmin,xmax)    	
    	ax.set_ylim(ymin,ymax)
    	for ax in fig.get_axes():
    		ax.label_outer()
    		plt.xticks(rotation=75)	
	#ax.set_xlabel('Date/Time')
    	#ax.set_ylabel('Soil Temperature ($^\circ$ C)')			
    	ax.text(0.20,0.15,'Site: '+str(site_id)+', Layer: 4', horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)    		
    	ax.legend()
    plt.tight_layout()   
    L1fil = "".join(["/mnt/data/users/herringtont/soil_temp/plots/CFSR_ts/site"+str(min_site)+"_site"+str(max_site),"_L4.png"])
    print(L1fil)
    plt.savefig(L1fil)
    plt.close()
#
#    
