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

cdo = Cdo()

################################# Set Parameters #################################

dtype = ['raw_temp','anom']
remap_type = ['remapnn','remapbil']
TC_dir = ['global_triple_collocation','global_triple_collocationB','global_triple_collocationC','global_triple_collocationD','global_triple_collocationE']

min_lon = 0
max_lon = 360
min_lat = 40
max_lat = 90


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

    	for k in TC_dir:
    		TC_dir_k = k   	
    		basedir = ''.join(['/mnt/data/users/herringtont/soil_temp/'+str(TC_dir_k)+'/'+str(dtype_i)+'/'+str(remap_type_j)+'/'])
    		basedir_polar = ''.join([basedir,'NH_extratrop/'])
    		Path(basedir_polar).mkdir(parents=True, exist_ok=True)
    		pathlist = Path(basedir).glob('*.nc')
    		blddir = ''.join(['/mnt/data/users/herringtont/soil_temp/'+str(TC_dir_k)+'/'+str(dtype_i)+'/'+str(remap_type_j)+'/blended_products/'])
    		blddir_polar = ''.join([blddir,'NH_extratrop/'])
    		Path(blddir_polar).mkdir(parents=True, exist_ok=True)
    		pathlistb = Path(blddir).glob('*.nc')
    		print(basedir)
	
################################# Calculate Climatologies of blended products #############################
    		cdo = Cdo()
    		TC_prod_fi = ''.join([blddir+str(remap_type_j)+'_TC_blended_'+str(bldsfx)+'.nc'])
    		TC_prod_clim_fi = ''.join([blddir+str(remap_type_j)+'_TC_blended_clim_'+str(bldsfx)+'.nc'])
    		cdo.timmean(input="-selyear,1981/2010 %s" % TC_prod_fi, output=str(TC_prod_clim_fi)) ## Create TC 1981-2010 annual climatology	
    		n_prod_fi = ''.join([blddir+str(remap_type_j)+'_naive_blended_'+str(bldsfx)+'.nc'])
    		n_prod_clim_fi = ''.join([blddir+str(remap_type_j)+'_naive_blended_clim_'+str(bldsfx)+'.nc'])
    		cdo.timmean(input="-selyear,1981/2010 %s" % n_prod_fi, output=str(n_prod_clim_fi)) ## Create Naive 1981-2010 annual climatology
    		blended_prod_fi = ''.join([blddir+str(remap_type_j)+'_diff_blended_clim_'+str(bldsfx)+'.nc'])
    		cdo.sub(input=str(TC_prod_clim_fi)+' '+str(n_prod_clim_fi), output=str(blended_prod_fi))	

################################ Extract extratropical NH north of 40N #############################    	
    		for path in pathlist:
    			fil = str(path)
    			print(fil)
    			fil_no_ext = os.path.splitext(os.path.basename(fil))[0]
    			print(fil_no_ext)
    			fil_polar = ''.join([basedir_polar,fil_no_ext+'_NH40.nc'])
    			if(fil == fil_polar or fil == TC_prod_clim_fi or fil == n_prod_fi or fil == blended_prod_fi):
    				continue		    			
    			cdo.sellonlatbox(str(min_lon)+','+str(max_lon)+','+str(min_lat)+','+str(max_lat), input=fil, output=fil_polar, options = '-f nc')
    			print(fil_polar)
		
    		for pathb in pathlistb:
    			filb = str(pathb)
    			print(filb)
    			filb_no_ext = os.path.splitext(os.path.basename(fil))[0]
    			filb_polar = ''.join([blddir_polar,filb_no_ext+'_NH40.nc'])
    			if(filb == filb_polar or filb == TC_prod_clim_fi or filb == n_prod_fi or filb == blended_prod_fi):
    				continue		    			
    			cdo.sellonlatbox(str(min_lon)+','+str(max_lon)+','+str(min_lat)+','+str(max_lat), input=filb, output=filb_polar, options = '-f nc')
    			print(filb_polar)
