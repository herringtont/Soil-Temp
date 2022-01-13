
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

#set file locations
CFSR_fi = "/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/CFSR.nc"
CFSR2_fi = "/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/CFSR2.nc"
ERAI_fi = "/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/ERA-Interim.nc"
ERA5_fi = "/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/ERA5.nc"
GLDAS_fi = "/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/GLDAS.nc"
JRA_fi = "/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/JRA55.nc"
MERRA2_fi = "/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/MERRA2.nc"
obs_loc = str("/mnt/data/users/herringtont/soil_temp/In-Situ/Master_Lat_Long_Obs.csv")
#otdir = str("/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/CFSR_site_level/")

#read in lat/long coordinates from in situ station locations
dframe = pd.read_csv(obs_loc)
lat_csv = dframe['Lat']
lon_csv = dframe['Long']
lon_csv2 = ((lon_csv+360)%360).round(decimals=2)

#open netCDF files with xarray
CFSR_fil = xr.open_dataset(CFSR_fi)
CFSR2_fil = xr.open_dataset(CFSR2_fi)
ERAI_fil = xr.open_dataset(ERAI_fi)
ERA5_fil = xr.open_dataset(ERA5_fi)
GLDAS_fil = xr.open_dataset(GLDAS_fi)
JRA_fil = xr.open_dataset(JRA_fi)
MERRA2_fil = xr.open_dataset(MERRA2_fi)

#extract soil temperatures
CFSR_stemp = CFSR_fil.Soil_Temp_L1 - 273.15
CFSR2_stemp = CFSR2_fil.Soil_Temp_L1 - 273.15
ERAI_stemp = ERAI_fil.Soil_Temp_L1 - 273.15

ERA5_st = ERA5_fil['Soil_Temp_L1']
ERA5_stemp = ERA5_st - 273.15

GLDAS_stemp = GLDAS_fil.Soil_Temp_L1 - 273.15
JRA_stemp = JRA_fil.Soil_Temp - 273.15
MERRA2_stemp = MERRA2_fil.Soil_Temp_L1 - 273.15

#extract time and attributes
CFSR_time = CFSR_fil.time
CFSR2_time = CFSR2_fil.time
ERAI_time = ERAI_fil.time
ERA5_time = ERA5_fil.time
GLDAS_time = GLDAS_fil.time
JRA_time = JRA_fil.time
MERRA2_time = MERRA2_fil.time
#print(MERRA2_time)

CFSR_time.attrs = CFSR_fil.time.attrs
CFSR2_time.attrs = CFSR2_fil.time.attrs
ERAI_time.attrs = ERAI_fil.time.attrs
ERA5_time.attrs = ERA5_fil.time.attrs
GLDAS_time.attrs = GLDAS_fil.time.attrs
JRA_time.attrs = JRA_fil.time.attrs
#MERRA2_time = MERRA2_fil.time.attrs

#print(MERRA2_time)
#grab attribute data from file, set units to deg C
CFSR_stemp.attrs = CFSR_fil.Soil_Temp_L1.attrs
CFSR_stemp.attrs["units"] = "deg C"

CFSR2_stemp.attrs = CFSR2_fil.Soil_Temp_L1.attrs
CFSR2_stemp.attrs["units"] = "deg C"

ERAI_stemp.attrs = ERAI_fil.Soil_Temp_L1.attrs
ERAI_stemp.attrs["units"] = "deg C"

ERA5_stemp.attrs = ERA5_fil.Soil_Temp_L1.attrs
ERA5_stemp.attrs["units"] = "deg C"

GLDAS_stemp.attrs = GLDAS_fil.Soil_Temp_L1.attrs
GLDAS_stemp.attrs["units"] = "deg C"

JRA_stemp.attrs = JRA_fil.Soil_Temp.attrs
JRA_stemp.attrs["units"] = "deg C"

MERRA2_stemp.attrs = MERRA2_fil.Soil_Temp_L1.attrs
MERRA2_stemp.attrs["units"] = "deg C"


##array structure: (site,soilT) = (7,299,N) CFSR = 384, CFSR2 = 117, ERAI = 448, ERA5 = 480, GLDAS = 838, JRA = 744, MERRA2 = 488 
CFSR_array = [[]]
CFSR2_array = [[]]
ERAI_array = [[]]
ERA5_array = [[]]
GLDAS_array = [[]]
JRA_array = [[]]
MERRA2_array = [[]]

#test = np.empty((299,384))
#print(test)

#loop through station lat/lon pairs from csv file
#for i in range (0,1):
for i in range (0,299):
    i2 = i+1
    str_i = str(i2)
    latf = lat_csv[i]
    lonf = lon_csv2[i]
    lonf2 = lon_csv[i]
    latfs = str(latf)
    lonfs = str(lonf2)
#select nearest grid cell from each netCDF file
    CFSR_stemploc = CFSR_stemp.sel(lon=lonf,lat=latf,method='nearest')
    CFSR2_stemploc = CFSR2_stemp.sel(lon=lonf,lat=latf,method='nearest')
    ERAI_stemploc = ERAI_stemp.sel(lon=lonf,lat=latf,method='nearest')
    ERA5_stemploc = ERA5_stemp.sel(longitude=lonf,latitude=latf,method='nearest')
    GLDAS_stemploc = GLDAS_stemp.sel(lon=lonf,lat=latf,method='nearest')
    JRA_stemploc = JRA_stemp.sel(lon=lonf,lat=latf,method='nearest')
    MERRA2_stemploc = MERRA2_stemp.sel(lon=lonf,lat=latf,method='nearest')

#calculate length of stemp_loc array
    CFSR_len = len(CFSR_stemploc)
    CFSR2_len = len(CFSR2_stemploc)
    ERAI_len = len(ERAI_stemploc)
    ERA5_len = len(ERA5_stemploc)
    GLDAS_len = len(GLDAS_stemploc)
    JRA_len = len(JRA_stemploc)
    MERRA2_len = len(MERRA2_stemploc)
    lenarray = [CFSR_len,CFSR2_len,ERAI_len,ERA5_len,GLDAS_len,JRA_len,MERRA2_len]
    #print(lenarray)  
    
    
      
#create site level files           
    #CFSR_ofil = "".join([otdir,"CFSR_site_",str_i32.5,".nc"])
    #CFSR2_ofil = "".join([otdir,"CFSR2_site_",str_i,".nc"])
    #CFSR_r_ofil = "".join([otdir,"CFSR_remap_site_",str_i,".nc"])
    #print(CFSR_ofil)
    #print(CFSR2_ofil)
    #print(CFSR_r_ofil)
    #cdo = Cdo()
    #cdo.remapnn('lon='+str(lonf)+'/lat='+str(latf), input=CFSR_fi, output=CFSR_ofil, options = '-f nc')
    #cdo.remapnn('lon='+str(lonf)+'/lat='+str(latf), input=CFSR2_fi, output=CFSR2_ofil, options = '-f nc')      
    #cdo.remapnn('lon='+str(lonf)+'/lat='+str(latf), input=CFSR_r_fi, output=CFSR_r_ofil, options = '-f nc')


#fill 2D arrays
    CFSR_array[0].append(i2)
    CFSR_array.append(CFSR_stemploc)
    
    CFSR2_array[0].append(i2)
    CFSR2_array.append(CFSR2_stemploc)
    
    ERAI_array[0].append(i2)
    ERAI_array.append(ERAI_stemploc)
    
    ERA5_array[0].append(i2)
    ERA5_array.append(ERA5_stemploc)
    
    GLDAS_array[0].append(i2)
    GLDAS_array.append(GLDAS_stemploc)
    
    JRA_array[0].append(i2)
    JRA_array.append(JRA_stemploc)
    
    MERRA2_array[0].append(i2)
    MERRA2_array.append(MERRA2_stemploc)

#print(MERRA2_time)   
#print(MERRA2_array[20])  


x1 = CFSR_time
x2 = CFSR2_time
x3 = ERAI_time
x4 = ERA5_time
x5 = GLDAS_time
x6 = JRA_time
x7 = MERRA2_time

xmin = datetime.date(1948,1,1)
xmax = datetime.date(2020,12,31)
ymin = -40
ymax = 30

#print(CFSR_time)
#print(CFSR_array[20])

####Single Plots#####
for i in range (1,300):
    fig = plt.figure()
    site_id = i
    y1 = CFSR_array[site_id]
    y2 = CFSR2_array[site_id]
    y3 = ERAI_array[site_id]
    y4 = ERA5_array[site_id]
    y5 = GLDAS_array[site_id]
    y6 = JRA_array[site_id]
    y7 = MERRA2_array[site_id]

    plt.plot(x1,y1,label = "CFSR", marker='o', markerfacecolor='indigo', markersize=2, color='darkorchid')
    plt.plot(x2,y2,label = "CFSR2",marker='^', markerfacecolor='red', markersize=2, color='tomato')
    plt.plot(x3,y3,label = "ERA-Interim",marker='d', markerfacecolor='greenyellow', markersize=2, color='yellowgreen')
    plt.plot(x4,y4,label = "ERA5",marker='s',markerfacecolor='black',markersize=2,color='dimgray')
    plt.plot(x5,y5,label = "GLDAS",marker='<',markerfacecolor='saddlebrown',markersize=2,color='sienna')
    plt.plot(x6,y6,label = "JRA55",marker='*',markerfacecolor='chartreuse',markersize=2,color='darkseagreen')
    plt.plot(x7,y7,label = "MERRA2",marker='v',markerfacecolor='dodgerblue',markersize=2,color='aqua')
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    plt.xlabel('Date/Time')
    plt.ylabel('Soil Temperature ($^\circ$ C)')			
    plt.title('Site: '+str(site_id)+', Latitude: '+str(latfs)+', Longitude: '+str(lonfs))
    
    leg = plt.legend(frameon=False, loc='lower center', ncol=3)
    pfils = "".join(["/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/plots/site_"+str(site_id)+".png"])
    print(pfils)
    plt.savefig(pfils)
    plt.close()

####Multiplots#####
#for i in range (1,20): #we are going to create 19 figures
#    fig = plt.figure(figsize =(20,16))
#    fig
#    if (i < 19):
#    	fig, axs = plt.subplots(nrows = 4, ncols = 4)
#    elif(i == 19):
#    	fig, axs = plt.subplots(nrows = 4, ncols = 3)
#    xmin = datetime.date(1948,1,1)
#    xmax = datetime.date(2020,12,31)
#    ymin = -40
#    ymax = 30
#    if (i == 19):
#    	mxrg = 12
#    else:
#    	mxrg = 17 
#    for j in range (1,mxrg): #each figure will have 16 subplots (in a 4x4 layout)
#    	i0 = i-1
#    	#print(i0)
#    	i1 = i0*16
#    	#print(i1)
#    	site_id = j + i1
#    	print(site_id)
#    	#print(site_id)
#    	y1 = CFSR_array[site_id]
#    	y2 = CFSR2_array[site_id]
#    	y3 = ERAI_array[site_id]
#    	y4 = ERA5_array[site_id]
#    	y5 = GLDAS_array[site_id]
#    	y6 = JRA_array[site_id]
#    	y7 = MERRA2_array[site_id]
#
#    	min_site = 1 + i1
#    	if (i == 19):	
#    		max_site = 299
#    	else:
#    		max_site = 16 + i1
#	
#    	ax = plt.subplot(4,4,j)
#    	ax.plot(x1,y1,label = "CFSR", marker='o', markerfacecolor='indigo', markersize=2, color='darkorchid')
#    	ax.plot(x2,y2,label = "CFSR2",marker='^', markerfacecolor='red', markersize=2, color='tomato')
#    	ax.plot(x3,y3,label = "ERA-Interim",marker='d', markerfacecolor='greenyellow', markersize=2, color='yellowgreen')
#    	ax.plot(x4,y4,label = "ERA5",marker='s',markerfacecolor='black',markersize=2,color='dimgray')
#    	ax.plot(x5,y5,label = "GLDAS",marker='<',markerfacecolor='saddlebrown',markersize=2,color='sienna')
#    	ax.plot(x6,y6,label = "JRA55",marker='*',markerfacecolor='chartreuse',markersize=2,color='darkseagreen')
#    	ax.plot(x7,y7,label = "MERRA2",marker='v',markerfacecolor='dodgerblue',markersize=2,color='aqua')
#    	
#    	ax.set_xlim(xmin,xmax)
#    	ax.set_ylim(ymin,ymax)
#    	#ax.set_xlabel('Date/Time')
#    	#ax.set_ylabel('Soil Temperature ($^\circ$ C)')			
#    	#ax.set_title('Site: '+str(site_id)+' Latitude: '+str(latfs)+' Longitude: '+str(lonfs))	
#    
#    plt.tight_layout()	    
#    pfil = "".join(["/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/multi_plots/site"+str(min_site)+"_site"+str(max_site),".png"])
#    print(pfil)
#    plt.savefig(pfil)
#    plt.close()

    
   
