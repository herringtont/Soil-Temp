import os
import pathlib
from pathlib import Path
import csv
import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import numpy as np
import scipy
import pandas as pd
import geopandas as gpd
import rioxarray
import xarray as xr
import seaborn as sns
import shapely
import math
import pathlib
import cftime
import re
import cdms2
import cartopy.crs as ccrs
import cdo
from cdo import Cdo
from decimal import *
from calendar import isleap
from shapely.geometry import Polygon, Point, GeometryCollection
from dateutil.relativedelta import *
from mpl_toolkits.basemap import Basemap



################################# Set Parameters #################################

dtype = ['raw_temp','anom']
remap_type = ['remapnn','remapbil']

for i in dtype: #loop through data type (absolute temp, anomalies)
    dtype_i = i
    if (dtype_i == 'raw_temp'):
    	tmp_type = 'Absolute Temps'
    	bldsfx = 'raw'
    	bldvar = 'TC_blended_stemp'
    	nvar = 'naive_blended_stemp'
    if (dtype_i == 'anom'):
    	tmp_type = 'Anomalies'
    	bldsfx = 'anom'
    	bldvar = 'TC_blended_anom'
    	nvar = 'naive_blended_anom'
   	
    for j in remap_type:
    	remap_type_j = j   	
    	basedir = ''.join(['/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/'+str(dtype_i)+'/'+str(remap_type_j)+'/'])
    	basedir_polar = ''.join([basedir,'NH_extratrop/'])
    	Path(basedir_polar).mkdir(parents=True, exist_ok=True)
    	blddir = ''.join(['/mnt/data/users/herringtont/soil_temp/global_triple_collocationE/'+str(dtype_i)+'/'+str(remap_type_j)+'/blended_products/'])
    	blddir_polar = ''.join([blddir,'NH_extratrop/'])
    	Path(blddir_polar).mkdir(parents=True, exist_ok=True)
    	print(basedir)



    	JRA55_errvar_fi = ''.join([basedir+str(remap_type_j)+'_JRA55_err_var_cov.nc'])
    	ERAI_errvar_fi = ''.join([basedir+str(remap_type_j)+'_ERA-Interim_err_var_cov.nc'])
    	ERA5_errvar_fi = ''.join([basedir+str(remap_type_j)+'_ERA5_err_var_cov.nc'])
    	blended_prod_fi = ''.join([blddir+str(remap_type_j)+'_diff_blended_clim_'+str(bldsfx)+'.nc'])


################################# Set lonlatbox coordinates #############################
    	min_lon = 0
    	max_lon = 360
    	min_lat = 40
    	max_lat = 90


################################## Calculate Climatologies of blended products #############################
    	cdo = Cdo()
    	TC_prod_fi = ''.join([blddir+str(remap_type_j)+'_TC_blended_'+str(bldsfx)+'.nc'])
    	TC_prod_clim_fi = ''.join([blddir+str(remap_type_j)+'_TC_blended_clim_'+str(bldsfx)+'.nc'])
    	cdo.timmean(input="-selyear,1981/2010 %s" % TC_prod_fi, output=str(TC_prod_clim_fi)) ## Create TC 1981-2010 annual climatology	
    	n_prod_fi = ''.join([blddir+str(remap_type_j)+'_naive_blended_'+str(bldsfx)+'.nc'])
    	n_prod_clim_fi = ''.join([blddir+str(remap_type_j)+'_naive_blended_clim_'+str(bldsfx)+'.nc'])
    	cdo.timmean(input="-selyear,1981/2010 %s" % n_prod_fi, output=str(n_prod_clim_fi)) ## Create Naive 1981-2010 annual climatology
    	cdo.sub(input=str(TC_prod_clim_fi)+' '+str(n_prod_clim_fi), output=str(blended_prod_fi))		
    
################################# Sellonlatbox to extract NH extratropics north of 40N ########################

    	JRA55_errvar_fi_polar = ''.join([basedir_polar+str(remap_type_j)+'_JRA55_err_var_cov_NH40.nc'])
    	cdo.sellonlatbox(str(min_lon)+','+str(max_lon)+','+str(min_lat)+','+str(max_lat), input=JRA55_errvar_fi, output=JRA55_errvar_fi_polar, options = '-f nc')
    	ERAI_errvar_fi_polar = ''.join([basedir_polar+str(remap_type_j)+'_ERA-Interim_err_var_cov_NH40.nc'])
    	cdo.sellonlatbox(str(min_lon)+','+str(max_lon)+','+str(min_lat)+','+str(max_lat), input=ERAI_errvar_fi, output=ERAI_errvar_fi_polar, options = '-f nc')	    	
    	ERA5_errvar_fi_polar = ''.join([basedir_polar+str(remap_type_j)+'_ERA5_err_var_cov_NH40.nc'])
    	cdo.sellonlatbox(str(min_lon)+','+str(max_lon)+','+str(min_lat)+','+str(max_lat), input=ERA5_errvar_fi, output=ERA5_errvar_fi_polar, options = '-f nc')
    	blended_prod_fi_polar = ''.join([blddir_polar+str(remap_type_j)+'_diff_blended_clim_'+str(bldsfx)+'_polar.nc'])
    	cdo.sellonlatbox(str(min_lon)+','+str(max_lon)+','+str(min_lat)+','+str(max_lat), input=blended_prod_fi, output=blended_prod_fi_polar, options = '-f nc')


################################## Grab Error Variance Data ##############################
#
#    	JRA55_errvar_fil = xr.open_dataset(JRA55_errvar_fi_polar)
#    	ERAI_errvar_fil = xr.open_dataset(ERAI_errvar_fi_polar)
#    	ERA5_errvar_fil = xr.open_dataset(ERA5_errvar_fi_polar)
#    	blended_prod_fil = xr.open_dataset(blended_prod_fi_polar)
#
#    	JRA55_errvar = JRA55_errvar_fil['err_var_cov']
#    	ERAI_errvar = ERAI_errvar_fil['err_var_cov']
#    	ERA5_errvar = ERA5_errvar_fil['err_var_cov']
#    	blended_prod = blended_prod_fil[bldvar]
#    	blended_prod_2D = blended_prod.isel(time=0,drop=True)
#    	#print(JRA55_errvar)
#    	#print(JRA55_errvar)
#
#    	JRA55_lat = JRA55_errvar_fil.coords['lat'].values
#    	ERAI_lat = ERAI_errvar_fil.coords['lat'].values
#    	ERA5_lat = ERA5_errvar_fil.coords['lat'].values
#    	blend_lat = blended_prod_fil.coords['lat'].values
#
##### since longitude is listed between 0-360, we must convert it to a -180 to 180 range
#    	#JRA55_lon = ((JRA55_errvar_fil.coords['lon'] + 180) % 360) - 180
#    	#ERAI_lon = ((ERAI_errvar_fil.coords['lon'] + 180) % 360) - 180
#    	#ERA5_lon = ((ERA5_errvar_fil.coords['lon'] + 180) % 360) - 180
#    	JRA55_lon = JRA55_errvar_fil.coords['lon'].values
#    	ERAI_lon = ERAI_errvar_fil.coords['lon'].values
#    	ERA5_lon = ERA5_errvar_fil.coords['lon'].values
#    	blend_lon = blended_prod_fil.coords['lon'].values
#
#    	#print(JRA55_lon.values)
##### convert to 1D array ####	
#    	JRA55_errvar_val = JRA55_errvar.values.tolist()
#    	ERAI_errvar_val = ERAI_errvar.values.tolist()
#    	ERA5_errvar_val = ERA5_errvar.values.tolist()
#
#    	JRA55_errvar_1D = np.array([j for sub in JRA55_errvar_val for j in sub])
#    	ERAI_errvar_1D = np.array([j for sub in ERAI_errvar_val for j in sub])
#    	ERA5_errvar_1D = np.array([j for sub in ERA5_errvar_val for j in sub])		
#
#
##### remove NaN and return only land grid cells ####
#    	JRA55_errvar_notna = JRA55_errvar_1D[~np.isnan(JRA55_errvar_1D)] 
#    	ERAI_errvar_notna = ERAI_errvar_1D[~np.isnan(ERAI_errvar_1D)] 
#    	ERA5_errvar_notna = ERA5_errvar_1D[~np.isnan(ERA5_errvar_1D)] 
#
##### determine grid cells with negative error variances ####
#    	JRA55_errvar_neg = JRA55_errvar_notna[np.where(JRA55_errvar_notna<0)]
#    	ERAI_errvar_neg = ERAI_errvar_notna[np.where(ERAI_errvar_notna<0)]
#    	ERA5_errvar_neg = ERA5_errvar_notna[np.where(ERA5_errvar_notna<0)]
#
##### determine grid cells with error variances < -1 ####
#    	JRA55_errvar_neg1 = JRA55_errvar_notna[np.where(JRA55_errvar_notna<-1)]
#    	ERAI_errvar_neg1 = ERAI_errvar_notna[np.where(ERAI_errvar_notna<-1)]
#    	ERA5_errvar_neg1 = ERA5_errvar_notna[np.where(ERA5_errvar_notna<-1)]
#
##### calculate % of land grid cells with negative error var ####
#    	JRA55_pct_neg = (len(JRA55_errvar_neg)/len(JRA55_errvar_notna))*100
#    	JRA55_pct_neg_rnd = round(JRA55_pct_neg,2)
#    	print('Percent negative JRA55:',JRA55_pct_neg)
#    	ERAI_pct_neg = (len(ERAI_errvar_neg)/len(ERAI_errvar_notna))*100
#    	ERAI_pct_neg_rnd = round(ERAI_pct_neg,2)
#    	print('Percent negative ERAI:',ERAI_pct_neg)
#    	ERA5_pct_neg = (len(ERA5_errvar_neg)/len(ERA5_errvar_notna))*100
#    	ERA5_pct_neg_rnd = round(ERA5_pct_neg,2)
#    	print('Percent negative ERA5:',ERA5_pct_neg)
#
#
##### calculate % of land grid cells with negative error var ####
#    	JRA55_pct_neg1 = (len(JRA55_errvar_neg1)/len(JRA55_errvar_notna))*100
#    	JRA55_pct_neg1_rnd = round(JRA55_pct_neg1,2)
#    	print('Percent JRA55 < -1:',JRA55_pct_neg1)
#    	ERAI_pct_neg1 = (len(ERAI_errvar_neg1)/len(ERAI_errvar_notna))*100
#    	ERAI_pct_neg1_rnd = round(ERAI_pct_neg1,2)
#    	print('Percent ERAI < -1:',ERAI_pct_neg1)
#    	ERA5_pct_neg1 = (len(ERA5_errvar_neg1)/len(ERA5_errvar_notna))*100
#    	ERA5_pct_neg1_rnd = round(ERA5_pct_neg1,2)
#    	print('Percent ERA5 < -1:',ERA5_pct_neg1)
#
#    	pltnam = ''.join(['/mnt/data/users/herringtont/soil_temp/plots/err_var_global/'+str(remap_type_j)+'_'+str(dtype_i)+'_errvar_JRA55_ERAI_ERA5_polar.png'])
#	
#
##### JRA55 fig ####
##### plot error variance data ####
#    	fig,axs = plt.subplots(nrows = 2,ncols = 2,figsize=(20,20))
#    	ax1 = plt.subplot(221,projection=ccrs.NorthPolarStereo())
#    	ax1.set_extent([-180,180,40,90], ccrs.PlateCarree())
#    	ax1.coastlines()
#    	ax1.gridlines()
#
##### set contour levels, then draw plot and colorbar #####
#    	cf1 = ax1.contourf(JRA55_lon,JRA55_lat,JRA55_errvar,transform=ccrs.PlateCarree(),levels=[-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30],cmap='bwr')
#    	#cb = plt.colorbar(ax=ax, orientation='horizontal')
#    	ax1.set_title('JRA55 Error Variances, '+str(remap_type_j)+', '+str(tmp_type)+', Pct Neg = '+str(JRA55_pct_neg_rnd)+'%')
#
#
##### ERAI fig ####
##### plot error variance data ####
#    	ax2 = plt.subplot(222,projection=ccrs.NorthPolarStereo())
#    	ax2.set_extent([-180,180,40,90], ccrs.PlateCarree())
#    	ax2.coastlines()
#    	ax2.gridlines()
#
##### set contour levels, then draw plot and colorbar #####
#    	cf2 = ax2.contourf(ERAI_lon,ERAI_lat,ERAI_errvar,transform=ccrs.PlateCarree(),levels=[-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30],cmap='bwr')
#    	ax2.set_title('ERA-Interim Error Variances, '+str(remap_type_j)+', '+str(tmp_type)+', Pct Neg = '+str(ERAI_pct_neg_rnd)+'%')
#
#
##### ERA5 fig ####
##### plot error variance data ####
#    	ax3 = plt.subplot(223,projection=ccrs.NorthPolarStereo())
#    	ax3.set_extent([-180,180,40,90], ccrs.PlateCarree())
#    	ax3.coastlines()
#    	ax3.gridlines()
#
##### set contour levels, then draw plot and colorbar #####
#    	cf3 = ax3.contourf(ERA5_lon,ERA5_lat,ERA5_errvar,transform=ccrs.PlateCarree(),levels=[-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30],cmap='bwr')
#    	ax3.set_title('ERA5 Error Variances, '+str(remap_type_j)+', '+str(tmp_type)+', Pct Neg = '+str(ERA5_pct_neg_rnd)+'%')
#
#
##### Blended Product ####
#    	ax4 = plt.subplot(224,projection=ccrs.NorthPolarStereo())
#    	ax4.set_extent([-180,180,40,90], ccrs.PlateCarree())
#    	ax4.coastlines()
#    	ax4.gridlines()
#
##### set contour levels, then draw plot and colorbar #####
#    	cf4 = ax4.contourf(ERA5_lon,ERA5_lat,blended_prod_2D,transform=ccrs.PlateCarree(),levels=[-10,-8,-6,-4,-2,0,2,4,6,8,10],cmap='bwr')
#    	ax4.set_title('1981-2010 Blended Soil '+str(tmp_type)+' Diff ($^\circ$ C), '+str(remap_type_j))
#
#    	fig.colorbar(cf1, ax=ax1, shrink=0.75)
#    	fig.colorbar(cf2, ax=ax2, shrink=0.75)
#    	fig.colorbar(cf3, ax=ax3, shrink=0.75)
#    	fig.colorbar(cf4, ax=ax4, shrink=0.75)
#    	plt.tight_layout()
#    	fig.savefig(pltnam)
#    	plt.close()
#
#
#
#
#
#
#
