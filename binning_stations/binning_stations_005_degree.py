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
from decimal import *
from calendar import isleap
from shapely.geometry import Polygon, Point, GeometryCollection
from dateutil.relativedelta import *
from pathlib import Path
from shapely import wkt

################### define functions ################
def str_to_datetime(column, date_fmt):

    date_list = []

    for dt_str in column:
        new_dt = datetime.datetime.strptime(dt_str, date_fmt)
        date_list.append(new_dt)
			

    return date_list
    
def remove_trailing_zeros(x):
    return str(x).rstrip('0').rstrip('.')

def LonTo360(dlon):
    # Convert longitudes to 0-360 deg
    dlon = ((360 + (dlon % 360)) % 360)
    return dlon


############### define variables #####################

directory = "/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/"
olr = ['outliers','zscore','IQR']
layers = [(0,30),(30,300),(0,50),(50,100),(100,200.1)]

rmp_type = ['bil']
    
################## grab lat/lon grid from reanalysis ############

ERAI_fic = "/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/ERA5_Land/005_degree/ERA5_Land_1970_2021_remapbil.nc"

ERAI_filc = xr.open_dataset(ERAI_fic)


latRc = ERAI_filc.lat
lonRc = ERAI_filc.lon
latRc = np.array(latRc.values)
lonRc = np.array(lonRc.values)
print(latRc)
print(lonRc)


################## create grid cells ##########################
grid_cellsn = []
grid_cellsb = []
grid_cellsc = []

for f in range(0,len(lonRc)-1):
    for g in  range(0,len(latRc)-1):  ### loop through latitude values
    	x1 = lonRc[f] #leftmost x-coordinate of grid cell
    	x2 = lonRc[f+1] #rightmost x-coordinate of grid cell
    	y1 = latRc[g] #topmost y-coordinate of grid cell 
    	y2 = latRc[g+1] #bottommost y-coordinate of grid cell
    	grid = Polygon([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])
    	#print(grid)
    	grid_cellsc.append(grid)

grid_cellsc = np.array(grid_cellsc)
grid_numc = np.arange(1,len(grid_cellsc)+1,1)


#### create geopandas dataframe of reanalysis grids ###### 

dframe_gridc = pd.DataFrame({'Grid Cell':grid_numc})
dframe_grid_geoc = gpd.GeoDataFrame(dframe_gridc, geometry = grid_cellsc)

################## find centroid of grid cell #####################################
dframe_grid_geoc_cen = dframe_grid_geoc.centroid

dframe_grid_geoc_lat = dframe_grid_geoc_cen.y.values
dframe_grid_geoc_lon = dframe_grid_geoc_cen.x.values
 
dframe_grid_geoc['Lat Cen'] = dframe_grid_geoc_lat
dframe_grid_geoc['Lon Cen'] = dframe_grid_geoc_lon

cen_geoc_fil = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/005_degree/centroid/dframe_geob.csv"])
dframe_grid_geoc.to_csv(cen_geoc_fil,na_rep=np.nan,index=False)


###### create geopandas dataframe of reanalysis grids ######
#
#cen_geoc_fil = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/005_degree/centroid/dframe_geoc.csv"])
#
#dframe_grid_geo1 = pd.read_csv(cen_geoc_fil)
#
#grid_cells = dframe_grid_geo1['Grid Cell']
#lat_cen = dframe_grid_geo1['Lat Cen']
#lon_cen = dframe_grid_geo1['Lon Cen']
#geom = dframe_grid_geo1['geometry']
#geom_conv = geom.apply(wkt.loads)
#
#dframe_grid_geoc = gpd.GeoDataFrame(grid_cells, geometry = geom_conv)
#dframe_grid_geoc['Lat Cen'] = lat_cen
#dframe_grid_geoc['Lon Cen'] = lon_cen
#
#print(dframe_grid_geoc)


## find the matching lat/lon separately

#look into mesh grid function

#find index that is closest to the point (arcindex) for each of lat/lon and then return the value at that index and store it in mesh grid




################## grab latitude and longitude coordinates of station data ################


for i in olr:
    olr_i = i
    print(i)
    latC_master = []
    lonC_master = []
    site_master = []

    for j in layers:
    	print(j)
    	top_layer = j[0]
    	btm_layer = j[1]
    	btm_bdy = float(btm_layer)

    	if(top_layer == 0 and btm_layer == 30):
    		layer_name = "top_30cm"

    	elif(top_layer == 30 and btm_layer == 300):
    		btm_bdy2 = btm_bdy-0.1
    		layer_name = ''.join([str(int(top_layer))+'_'+str(btm_bdy2)])
    	pthl2 = [directory,str(i),"/",layer_name,"/"]	
    	pthl3 = "".join(pthl2)
    	#print(pthl2)
    	pathlist = Path(pthl3).glob('*.csv')
    	#print(pathlist)		
    	for path in sorted(pathlist, key=lambda path: int(path.stem.rsplit("_",1)[1])):

    		file_name = str(path)
    		#print(file_name)

    		#print("Loading file: ", file_name)
    		dframe = pd.read_csv(file_name)
    		#print(dframe)
    		latC = dframe.iloc[1,2]
    		lonC = dframe.iloc[1,3]
    		sitid = file_name.split("site_")[1].split(".csv")[0] #locate site id within filename
    		sitnam = "".join(["site_",sitid])
    		lonC2 = LonTo360(lonC) #csv file longitudes are offset by 180 relative to netcdf
    		olrid = file_name.split("no_outliers/")[1].split("/")[0]
    		lyrid = file_name.split("/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/"+str(olrid)+"/")[1].split("/site_")[0]


    		latC_master.append(latC)
    		lonC_master.append(lonC)
    		site_master.append(sitid)    		
    		
########## store master lists as numpy arrrays ##############

    	latC_mastern = np.array(latC_master)
    	lonC_mastern = np.array(lonC_master)
    	site_mastern = np.array(site_master)
    	#print(site_mastern)

################ create geopandas dataframes #################

##### obs locations ######
    	dframe_stn = pd.DataFrame({'site':site_mastern,'lat':latC_mastern,'lon':lonC_mastern})

    	dframe_stn_geo = gpd.GeoDataFrame(dframe_stn, geometry=gpd.points_from_xy(dframe_stn.lon, dframe_stn.lat))

    	#print(type(dframe_stn_geo))

    	#print("dframe_stn_geo: ",dframe_stn_geo)


    	stn_fil = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/005_degree/pre_join/geometry_"+str(olr_i)+"_"+str(layer_name)+"_005_degree_"+str(rmp_type)+"_prejoin.csv"])


    	dframe_stn_geo.to_csv(stn_fil,na_rep=np.nan,index=False)


################## do a spatial join to figure out which grid cell each station belongs to ################

    	stn_gridc = gpd.sjoin(dframe_stn_geo, dframe_grid_geoc, how="inner", op ='intersects')

################# find lat/lon of grid cell centroid ##########################

    	print(stn_gridc)

    	stn_gridc_gc = stn_gridc['Grid Cell']

    	master_latc = []
    	master_lonc = []

    	for f in stn_gridc_gc:
    		dframe_grid_geoc_latc = dframe_grid_geoc['Lat Cen'][dframe_grid_geoc['Grid Cell'] == f].values
    		dframe_grid_geoc_lonc = dframe_grid_geoc['Lon Cen'][dframe_grid_geoc['Grid Cell'] == f].values
    		master_latc.append(dframe_grid_geoc_latc)
    		master_lonc.append(dframe_grid_geoc_lonc)

    	latc_cen = master_latc
    	lonc_cen = master_lonc


################# add lat/lon of centroid to dataframe ################

    	stn_gridc['Lat Cen'] = latc_cen
    	stn_gridc['Lon Cen'] = lonc_cen

    	print(stn_gridc)

    	filc = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/005_degree/geometry_"+str(olr_i)+"_"+str(layer_name)+"_"+str(rmp_type)+"_005_degree.csv"])

    	print(filc)

    	stn_gridc.to_csv(filc,na_rep=np.nan,index=False)





