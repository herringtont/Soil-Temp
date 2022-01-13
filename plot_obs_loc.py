import os
import csv
import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import numpy as np
import xarray as xr
import scipy
import pandas as pd
import re
import math
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature
import matplotlib.pyplot as plt
import matplotlib.path as mpath

from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim, cartopy_ylim, latlon_coords)

#set file locations
obs_loc = str("/mnt/data/users/herringtont/soil_temp/In-Situ/Master_Lat_Long_Obs.csv")
CFSR_fi = "/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/land_sea_mask/CFSR_ls.nc"
CFSR2_fi = "/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/land_sea_mask/CFSR2_ls.nc"
ERAI_fi = "/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/land_sea_mask/ERA-Interim_ls2.nc"
ERA5_fi = "/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/land_sea_mask/ERA5_ls2.nc"
JRA_fi = "/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/land_sea_mask/JRA55_ls2.nc"


#read in lat/long coordinates from in situ station locations
dframe = pd.read_csv(obs_loc)
lat_csv = np.array(dframe['Lat'])
lon_csv = np.array(dframe['Long'])
#lon_csv2 = ((lon_csv+360)%360).round(decimals=2)

#open netCDF files with xarray
CFSR_fil = xr.open_dataset(CFSR_fi)
CFSR2_fil = xr.open_dataset(CFSR2_fi)
ERAI_fil = xr.open_dataset(ERAI_fi)
ERA5_fil = xr.open_dataset(ERA5_fi)
JRA_fil = xr.open_dataset(JRA_fi)

#extract lat/lon of reanalysis products
#CFSR_lon = ((CFSR_fil.lon + 180)%360)-180
CFSR_lon = CFSR_fil.lon
CFSR_lat = CFSR_fil.lat

#CFSR2_lon = ((CFSR2_fil.lon + 180)%360)-180
CFSR2_lon = CFSR2_fil.lon
CFSR2_lat = CFSR2_fil.lat

#ERAI_lon = ((ERAI_fil.lon + 180)%360)-180
ERAI_lon = ERAI_fil.lon
ERAI_lat = ERAI_fil.lat

#ERA5_lon = ((ERA5_fil.longitude + 180)%360)-180
ERA5_lon = ERA5_fil.longitude
ERA5_lat = ERA5_fil.latitude
print(ERA5_lon)

#JRA_lon = ((JRA_fil.lon + 180)%360)-180
JRA_lon = JRA_fil.lon
JRA_lat = JRA_fil.lat

#extract land/sea mask
CFSR_ls = CFSR_fil.LSM
CFSR2_ls = CFSR2_fil.LSM
ERAI_ls = ERAI_fil.LSM
ERA5_ls = ERA5_fil.LSM
JRA_ls = JRA_fil.LSM

#extract attributes from netCDF
CFSR_ls.attrs = CFSR_fil.LSM.attrs
CFSR2_ls.attrs = CFSR2_fil.LSM.attrs
ERAI_ls.attrs = ERAI_fil.LSM.attrs
ERA5_ls.attrs = ERA5_fil.LSM.attrs
JRA_ls.attrs = JRA_fil.LSM.attrs

#remove time dimension
CFSR_ls2 = CFSR_ls.isel(time=0)
CFSR2_ls2 = CFSR2_ls.isel(time=0)
ERAI_ls2 = ERAI_ls.isel(time=0)
ERA5_ls2 = ERA5_ls.isel(time=0) 
JRA_ls2 = JRA_ls.isel(time=0)

#print(lat_csv)
#print(lon_csv)

#create multiplot with 2 columns, 3 rows
proj = ccrs.NorthPolarStereo()

fig = plt.figure(figsize=[15,10])

ax1 = plt.subplot(231, projection=proj)
ax2 = plt.subplot(232, projection=proj, sharex=ax1, sharey=ax1)
ax3 = plt.subplot(233, projection=proj, sharex=ax1, sharey=ax1)
ax4 = plt.subplot(234, projection=proj, sharex=ax1, sharey=ax1)
ax5 = plt.subplot(235, projection=proj, sharex=ax1, sharey=ax1)

#plot land/sea masks

ax1.contourf(CFSR_lon,CFSR_lat,CFSR_ls2,cmap='Reds',transform=ccrs.PlateCarree())
ax2.contourf(CFSR2_lon,CFSR2_lat,CFSR2_ls2,cmap='Reds',transform=ccrs.PlateCarree())
ax3.contourf(ERAI_lon,ERAI_lat,ERAI_ls2,cmap='Reds',transform=ccrs.PlateCarree())
#ax4.contourf(ERA5_lon2,ERA5_lat,ERA5_ls2,cmap='Reds',transform=ccrs.PlateCarree())
ax5.contourf(JRA_lon,JRA_lat,JRA_ls2,cmap='Reds',transform=ccrs.PlateCarree())

#overlay lat/lon coordinates of obs sites on map

ax1.scatter(lon_csv,lat_csv, color='dodgerblue', s=2, alpha=1, transform=ccrs.PlateCarree())
ax2.scatter(lon_csv,lat_csv, color='dodgerblue', s=2, alpha=1, transform=ccrs.PlateCarree())
ax3.scatter(lon_csv,lat_csv, color='dodgerblue', s=2, alpha=1, transform=ccrs.PlateCarree())
ax4.scatter(lon_csv,lat_csv, color='dodgerblue', s=2, alpha=1, transform=ccrs.PlateCarree())
ax5.scatter(lon_csv,lat_csv, color='dodgerblue', s=2, alpha=1, transform=ccrs.PlateCarree())

#ax1.plot(lat_csv,lon_csv,markersize=2,marker='o',linestyle='',color='dodgerblue',transform=ccrs.PlateCarree())
#Limit map extent between 90N and 50N
extent = [-180,180,50,90]
ax1.set_extent(extent, crs=ccrs.PlateCarree())


#add land/water/gridlines/coastlines to map

#ax1.add_feature(cartopy.feature.COASTLINE, edgecolor="black")
#ax2.add_feature(cartopy.feature.COASTLINE, edgecolor="black")
#ax3.add_feature(cartopy.feature.COASTLINE, edgecolor="black")
#ax4.add_feature(cartopy.feature.COASTLINE, edgecolor="black")
#ax5.add_feature(cartopy.feature.COASTLINE, edgecolor="black")

ax1.gridlines()
ax2.gridlines()
ax3.gridlines()
ax4.gridlines()
ax5.gridlines()


##create a circular boundary for the map
#theta = np.linspace(0, 2*np.pi, 100)
#center, radius = [0.5, 0.5], 0.5
#verts = np.vstack([np.sin(theta), np.cos(theta)]).T
#circle = mpath.Path(verts * radius + center)
#
#ax1.set_boundary(circle, transform=ax1.transAxes)
#ax2.set_boundary(circle, transform=ax2.transAxes)
#ax3.set_boundary(circle, transform=ax3.transAxes)
#ax4.set_boundary(circle, transform=ax4.transAxes)
#ax5.set_boundary(circle, transform=ax5.transAxes)

ax1.set_title('CFSR Land Mask')
ax2.set_title('CFSR2 Land Mask')
ax3.set_title('ERA-Interim Land Mask')
ax4.set_title('ERA5 Land Mask')
ax5.set_title('JRA55 Land Mask')

plt.savefig('Observation_Locations.png')
plt.show()
