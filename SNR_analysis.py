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
import geopandas as gpd
import xarray as xr
import seaborn as sns
import shapely
import math
import cftime
import re
import cdms2
import cartopy.crs as ccrs
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
    if (dtype_i == 'anom'):
    	tmp_type = 'Anomalies'
    	bldsfx = 'anom'
    	bldvar = 'TC_blended_anom'
    for j in remap_type:
    	remap_type_j = j   	
    	basedir = ''.join(['/mnt/data/users/herringtont/soil_temp/global_triple_collocation/'+str(dtype_i)+'/'+str(remap_type_j)+'/'])
    	blddir = ''.join(['/mnt/data/users/herringtont/soil_temp/global_triple_collocation/'+str(dtype_i)+'/'+str(remap_type_j)+'/blended_products/'])
    	print(basedir)

################################# Grab Error Variance Data ##############################

    	JRA_errvar_fi = ''.join([basedir+str(remap_type_j)+'_JRA55_SNR.nc'])
    	MERRA2_errvar_fi = ''.join([basedir+str(remap_type_j)+'_MERRA2_SNR.nc'])
    	GLDAS_errvar_fi = ''.join([basedir+str(remap_type_j)+'_GLDAS_SNR.nc'])
    	blended_prod_fi = ''.join([blddir+str(remap_type_j)+'_diff_blended_clim_'+str(bldsfx)+'.nc'])

    	JRA_errvar_fil = xr.open_dataset(JRA_errvar_fi)
    	MERRA2_errvar_fil = xr.open_dataset(MERRA2_errvar_fi)
    	GLDAS_errvar_fil = xr.open_dataset(GLDAS_errvar_fi)
    	blended_prod_fil = xr.open_dataset(blended_prod_fi)

    	JRA_errvar = JRA_errvar_fil['SNR_JRA55']
    	MERRA2_errvar = MERRA2_errvar_fil['SNR_MERRA2']
    	GLDAS_errvar = GLDAS_errvar_fil['SNR_GLDAS']
    	blended_prod = blended_prod_fil[bldvar]
    	blended_prod_2D = blended_prod.isel(time=0,drop=True)
    	print(blended_prod_2D)
    	#print(JRA_errvar)

    	JRA_lat = JRA_errvar_fil.coords['lat'].values
    	MERRA2_lat = MERRA2_errvar_fil.coords['lat'].values
    	GLDAS_lat = GLDAS_errvar_fil.coords['lat'].values
    	blend_lat = blended_prod_fil.coords['lat'].values

#### since longitude is listed between 0-360, we must convert it to a -180 to 180 range
    	#JRA_lon = ((JRA_errvar_fil.coords['lon'] + 180) % 360) - 180
    	#MERRA2_lon = ((MERRA2_errvar_fil.coords['lon'] + 180) % 360) - 180
    	#GLDAS_lon = ((GLDAS_errvar_fil.coords['lon'] + 180) % 360) - 180
    	JRA_lon = JRA_errvar_fil.coords['lon'].values
    	MERRA2_lon = MERRA2_errvar_fil.coords['lon'].values
    	GLDAS_lon = GLDAS_errvar_fil.coords['lon'].values
    	blend_lon = blended_prod_fil.coords['lon'].values

    	#print(JRA_lon.values)
#### convert to 1D array ####	
    	JRA_errvar_val = JRA_errvar.values.tolist()
    	MERRA2_errvar_val = MERRA2_errvar.values.tolist()
    	GLDAS_errvar_val = GLDAS_errvar.values.tolist()

    	JRA_errvar_1D = np.array([j for sub in JRA_errvar_val for j in sub])
    	MERRA2_errvar_1D = np.array([j for sub in MERRA2_errvar_val for j in sub])
    	GLDAS_errvar_1D = np.array([j for sub in GLDAS_errvar_val for j in sub])		


#### remove NaN and return only land grid cells ####
    	JRA_errvar_notna = JRA_errvar_1D[~np.isnan(JRA_errvar_1D)] 
    	MERRA2_errvar_notna = MERRA2_errvar_1D[~np.isnan(MERRA2_errvar_1D)] 
    	GLDAS_errvar_notna = GLDAS_errvar_1D[~np.isnan(GLDAS_errvar_1D)] 

#### determine grid cells with negative error variances ####
    	JRA_errvar_neg = JRA_errvar_notna[np.where(JRA_errvar_notna<0)]
    	MERRA2_errvar_neg = MERRA2_errvar_notna[np.where(MERRA2_errvar_notna<0)]
    	GLDAS_errvar_neg = GLDAS_errvar_notna[np.where(GLDAS_errvar_notna<0)]

#### calculate % of land grid cells with negative error var ####
    	JRA_pct_neg = (len(JRA_errvar_neg)/len(JRA_errvar_notna))*100
    	JRA_pct_neg_rnd = round(JRA_pct_neg,2)
    	print('Percent negative JRA55:',JRA_pct_neg)
    	MERRA2_pct_neg = (len(MERRA2_errvar_neg)/len(MERRA2_errvar_notna))*100
    	MERRA2_pct_neg_rnd = round(MERRA2_pct_neg,2)
    	print('Percent negative MERRA2:',MERRA2_pct_neg)
    	GLDAS_pct_neg = (len(GLDAS_errvar_neg)/len(GLDAS_errvar_notna))*100
    	GLDAS_pct_neg_rnd = round(GLDAS_pct_neg,2)
    	print('Percent negative GLDAS:',GLDAS_pct_neg)


    	pltnam = ''.join(['/mnt/data/users/herringtont/soil_temp/plots/err_var_global/'+str(remap_type_j)+'_'+str(dtype_i)+'_SNR.png'])
	

#### JRA fig ####
#### plot error variance data ####
    	fig,axs = plt.subplots(nrows = 2,ncols = 2,figsize=(20,20))
    	ax1 = plt.subplot(221,projection=ccrs.Robinson())
    	ax1.set_global()
    	ax1.coastlines()
    	ax1.gridlines()

#### set contour levels, then draw plot and colorbar #####
    	cf1 = ax1.contourf(JRA_lon,JRA_lat,JRA_errvar,transform=ccrs.PlateCarree(),levels=[-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90,100],cmap='bwr')
    	#cb = plt.colorbar(ax=ax, orientation='horizontal')
    	ax1.set_title('JRA55 SNR, '+str(remap_type_j)+', '+str(tmp_type)+', Pct Neg = '+str(JRA_pct_neg_rnd)+'%')


#### MERRA2 fig ####
#### plot error variance data ####
    	ax2 = plt.subplot(222,projection=ccrs.Robinson())
    	ax2.set_global()
    	ax2.coastlines()
    	ax2.gridlines()

#### set contour levels, then draw plot and colorbar #####
    	cf2 = ax2.contourf(MERRA2_lon,MERRA2_lat,MERRA2_errvar,transform=ccrs.PlateCarree(),levels=[-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90,100],cmap='bwr')
    	ax2.set_title('MERRA2 SNR, '+str(remap_type_j)+', '+str(tmp_type)+', Pct Neg = '+str(MERRA2_pct_neg_rnd)+'%')


#### GLDAS fig ####
#### plot error variance data ####
    	ax3 = plt.subplot(223,projection=ccrs.Robinson())
    	ax3.set_global()
    	ax3.coastlines()
    	ax3.gridlines()

#### set contour levels, then draw plot and colorbar #####
    	cf3 = ax3.contourf(GLDAS_lon,GLDAS_lat,GLDAS_errvar,transform=ccrs.PlateCarree(),levels=[-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90,100],cmap='bwr')
    	ax3.set_title('GLDAS SNR, '+str(remap_type_j)+', '+str(tmp_type)+', Pct Neg = '+str(GLDAS_pct_neg_rnd)+'%')


#### Blended Product ####
    	ax4 = plt.subplot(224,projection=ccrs.Robinson())
    	ax4.set_global()
    	ax4.coastlines()
    	ax4.gridlines()

#### set contour levels, then draw plot and colorbar #####
    	cf4 = ax4.contourf(GLDAS_lon,GLDAS_lat,blended_prod_2D,transform=ccrs.PlateCarree(),levels=[-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60],cmap='bwr')
    	ax4.set_title('1981-2010 Blended Soil '+str(tmp_type)+' Diff ($^\circ$ C), '+str(remap_type_j))

    	fig.colorbar(cf1, ax=ax1, shrink=0.75)
    	fig.colorbar(cf2, ax=ax2, shrink=0.75)
    	fig.colorbar(cf3, ax=ax3, shrink=0.75)
    	fig.colorbar(cf4, ax=ax4, shrink=0.75)
    	plt.tight_layout()
    	fig.savefig(pltnam)
    	plt.close()
