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
import fiona
import xarray as xr
import seaborn as sns
import shapely
import math
import cftime
import re
import cdms2
from decimal import *
from calendar import isleap
from shapely.geometry import Polygon, Point, GeometryCollection
from dateutil.relativedelta import *


#################### set directories ######################

naive_dir = "/mnt/data/users/herringtont/soil_temp/Blended_Product/collocated/Naive/raw_temp/"
naive_dir_anom = "/mnt/data/users/herringtont/soil_temp/Blended_Product/collocated/Naive/anom/"

states_fil = "/mnt/data/users/herringtont/soil_temp/by_region/shpfiles/AK/cb_2018_us_state_5m.shp"
continent_fil = "/mnt/data/users/herringtont/soil_temp/by_region/shpfiles/World_Continents/ne_10m_graticules_10.shp"


olr = ['outliers','zscore','IQR']
lyr = ['0_9.9']
thr = ['0','25','50','75','100']
rmp_type = ['nn','bil']


#################### store alaska polygon in geodataframe #################

states_shp = gpd.read_file(states_fil)
continents_shp = gpd.read_file(continent_fil)

continents_gdf = gpd.GeoDataFrame(continents_shp)

alaska_shp = states_shp.loc[states_shp['STUSPS'] == 'AK']
alaska_geom = alaska_shp['geometry']
alaska_gpd = gpd.GeoDataFrame(geometry=alaska_geom) 
alaska_gpd.crs = "EPSG:4326" 

#print(alaska_gpd)

#################### get grid cell information ####################
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
    			thrk = k
    			thr_type = ''.join(['thr_'+str(k)])


#################### grab lat/lon of grid cells
    			blended_fil = ''.join(["/mnt/data/users/herringtont/soil_temp/Blended_Product/collocated/Naive/anom/"+str(remap_type)+'_'+str(olri)+'_'+str(lyrj)+'_thr'+str(thrk)+'_naive_blending_anom.csv'])
    			blended_df = pd.read_csv(blended_fil)
    			lat_cen = blended_df['Central Lat']
    			lon_cen = blended_df['Central Lon']
    			gcell = blended_df['Grid Cell']
    			blend_df = pd.DataFrame({'Grid Cell':gcell,'lat':lat_cen,'lon':lon_cen})
    			blend_gpd = gpd.GeoDataFrame(blend_df, geometry=gpd.points_from_xy(lon_cen, lat_cen))
    			blend_gpd.crs = "EPSG:4326"
			    						
#################### conduct a spatial join to figure out which grid cells are in Alaska ###############
    			alaska_blend = gpd.sjoin(blend_gpd, alaska_gpd, how='inner', op ='intersects')
    			print(alaska_blend) 
