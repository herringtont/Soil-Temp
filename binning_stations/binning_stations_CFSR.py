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
    
################## grab lat/lon grid from reanalysis ############

ERAI_fic = "/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/obs_depths/common_grid_CFSR/remapcon/common_date/masked_files/CFSR_all.nc"

ERAI_filc = xr.open_dataset(ERAI_fic)


latRc = ERAI_filc.lat
lonRc = ERAI_filc.lon
latRc = np.array(latRc.values)
lonRc = np.array(lonRc.values)
#print(latR)
#print(lonR)


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

########## create master arrays for all outlier removal methods and layers

latC_master_olr_top30 = []
lonC_master_olr_top30 = []
site_master_olr_top30 = []

latC_master_olr_L7 = []
lonC_master_olr_L7 = []
site_master_olr_L7 = []

latC_master_Z_top30 = []
lonC_master_Z_top30 = []
site_master_Z_top30 = []

latC_master_Z_L7 = []
lonC_master_Z_L7 = []
site_master_Z_L7 = []

latC_master_I_top30 = []
lonC_master_I_top30 = []
site_master_I_top30 = []

latC_master_I_L7 = []
lonC_master_I_L7 = []
site_master_I_L7 = []
################## grab latitude and longitude coordinates of station data ################

def load_pandas(file_name):
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
    #print(olrid)
    #print(lyrid)
    #print(sitnam)
    #print(latC)
    #print(lonC2)
    if(olrid == "outliers"):
    	if(lyrid == "top_30cm"):
    		latC_master_olr_top30.append(latC)
    		lonC_master_olr_top30.append(lonC)
    		site_master_olr_top30.append(sitid)
    	elif(lyrid == "30_299.9"):
    		latC_master_olr_L7.append(latC)
    		lonC_master_olr_L7.append(lonC)
    		site_master_olr_L7.append(sitid)

    elif(olrid == "zscore"):
    	if(lyrid == "top_30cm"):
    		latC_master_Z_top30.append(latC)
    		lonC_master_Z_top30.append(lonC)
    		site_master_Z_top30.append(sitid)
    	elif(lyrid == "30_299.9"):
    		latC_master_Z_L7.append(latC)
    		lonC_master_Z_L7.append(lonC)
    		site_master_Z_L7.append(sitid)


    elif(olrid == "IQR"):
    	if(lyrid == "top_30cm"):
    		latC_master_I_top30.append(latC)
    		lonC_master_I_top30.append(lonC)
    		site_master_I_top30.append(sitid)
    	elif(lyrid == "30_299.9"):
    		latC_master_I_L7.append(latC)
    		lonC_master_I_L7.append(lonC)
    		site_master_I_L7.append(sitid)

def main():
#set files and directories
    from pathlib import Path
    directory = "/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/"
    olr = ['outliers','zscore','IQR']
    dep = ['top_30cm','30_299.9']

    for i in olr:
    	#print(i)
    	for j in dep:
    		pthl2 = [directory,str(i),"/",str(j),"/"]	
    		pthl3 = "".join(pthl2)
    		#print(pthl2)
    		pathlist = Path(pthl3).glob('*.csv')
    		#print(pathlist)		
    		for path in sorted(pathlist, key=lambda path: int(path.stem.rsplit("_",1)[1])):
    			fil = str(path)
    			load_pandas(fil)

main()


########## store master lists as numpy arrrays ##############

latC_master_olr_top30n = np.array(latC_master_olr_top30)
lonC_master_olr_top30n = np.array(lonC_master_olr_top30)
site_master_olr_top30n = np.array(site_master_olr_top30)

latC_master_olr_L7n = np.array(latC_master_olr_L7)
lonC_master_olr_L7n = np.array(lonC_master_olr_L7)
site_master_olr_L7n = np.array(site_master_olr_L7)

latC_master_Z_top30n = np.array(latC_master_Z_top30)
lonC_master_Z_top30n = np.array(lonC_master_Z_top30)
site_master_Z_top30n = np.array(site_master_Z_top30)

latC_master_Z_L7n = np.array(latC_master_Z_L7)
lonC_master_Z_L7n = np.array(lonC_master_Z_L7)
site_master_Z_L7n = np.array(site_master_Z_L7)

latC_master_I_top30n = np.array(latC_master_I_top30)
lonC_master_I_top30n = np.array(lonC_master_I_top30)
site_master_I_top30n = np.array(site_master_I_top30)

latC_master_I_L7n = np.array(latC_master_I_L7)
lonC_master_I_L7n = np.array(lonC_master_I_L7)
site_master_I_L7n = np.array(site_master_I_L7)

print(site_master_olr_L7n)
print(site_master_olr_top30n)

################ create geopandas dataframes #################

rmp_type = ['con']

#### reanalysis grids ###### 

dframe_gridc = pd.DataFrame({'Grid Cell':grid_numc})
dframe_grid_geoc = gpd.GeoDataFrame(dframe_gridc, geometry = grid_cellsc)

##### with outliers ######
dframe_stn_O_top30 = pd.DataFrame({'site':site_master_olr_top30n,'lat':latC_master_olr_top30n,'lon':lonC_master_olr_top30n})
dframe_stn_O_L7 = pd.DataFrame({'site':site_master_olr_L7n,'lat':latC_master_olr_L7n,'lon':lonC_master_olr_L7n})

dframe_stn_O_top30_geo = gpd.GeoDataFrame(dframe_stn_O_top30, geometry=gpd.points_from_xy(dframe_stn_O_top30.lon, dframe_stn_O_top30.lat))
dframe_stn_O_L7_geo = gpd.GeoDataFrame(dframe_stn_O_L7, geometry=gpd.points_from_xy(dframe_stn_O_L7.lon, dframe_stn_O_L7.lat))

print("dframe_stn_O_top30_geo: ",dframe_stn_O_top30_geo)
print("dframe_stn_O_L7_geo: ",dframe_stn_O_L7_geo)

stn_top30_fil = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/CFSR/pre_join/geometry_top30_CFSR_prejoin.csv"])
stn_L7_fil = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/CFSR/pre_join/geometry_L7_CFSR_prejoin.csv"])

dframe_stn_O_top30_geo.to_csv(stn_top30_fil,na_rep=np.nan,index=False)
dframe_stn_O_L7_geo.to_csv(stn_L7_fil,na_rep=np.nan,index=False)


################## find centroid of grid cell #####################################
dframe_grid_geoc_cen = dframe_grid_geoc.centroid

dframe_grid_geoc_lat = dframe_grid_geoc_cen.y.values
dframe_grid_geoc_lon = dframe_grid_geoc_cen.x.values
 
dframe_grid_geoc['Lat Cen'] = dframe_grid_geoc_lat
dframe_grid_geoc['Lon Cen'] = dframe_grid_geoc_lon

cen_geoc_fil = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/CFSR/centroid/dframe_geoc.csv"])

dframe_grid_geoc.to_csv(cen_geoc_fil,na_rep=np.nan,index=False)

print(dframe_grid_geoc)

################## do a spatial join to figure out which grid cell each station belongs to ################

##### with outliers ######

stn_grid_O_top30c = gpd.sjoin(dframe_stn_O_top30_geo, dframe_grid_geoc, how="inner", op ='intersects')
stn_grid_O_L7c = gpd.sjoin(dframe_stn_O_L7_geo, dframe_grid_geoc, how="inner", op ='intersects')

print(stn_grid_O_top30c)
print(stn_grid_O_L7c)

################# find lat/lon of grid cell centroid ##########################

stn_grid_O_top30c_gc = stn_grid_O_top30c['Grid Cell']
stn_grid_O_L7c_gc = stn_grid_O_L7c['Grid Cell']

master_lat_O_top30c = []
master_lon_O_top30c = []
for f in stn_grid_O_top30c_gc:
    dframe_grid_geoc_latc = dframe_grid_geoc['Lat Cen'][dframe_grid_geoc['Grid Cell'] == f].values
    dframe_grid_geoc_lonc = dframe_grid_geoc['Lon Cen'][dframe_grid_geoc['Grid Cell'] == f].values
    master_lat_O_top30c.append(dframe_grid_geoc_latc)
    master_lon_O_top30c.append(dframe_grid_geoc_lonc)

master_lat_O_top30c = np.array(master_lat_O_top30c).flatten()
master_lon_O_top30c = np.array(master_lon_O_top30c).flatten()

master_lat_O_L7c = []
master_lon_O_L7c = []
for g in stn_grid_O_L7c_gc:
    dframe_grid_geoc_latc = dframe_grid_geoc['Lat Cen'][dframe_grid_geoc['Grid Cell'] == g].values
    dframe_grid_geoc_lonc = dframe_grid_geoc['Lon Cen'][dframe_grid_geoc['Grid Cell'] == g].values
    master_lat_O_L7c.append(dframe_grid_geoc_latc)
    master_lon_O_L7c.append(dframe_grid_geoc_lonc)

master_lat_O_L7c = np.array(master_lat_O_L7c).flatten()
master_lon_O_L7c = np.array(master_lon_O_L7c).flatten()

latc_cen_top30 = master_lat_O_top30c
lonc_cen_top30 = master_lon_O_top30c
latc_cen_L7 = master_lat_O_L7c
lonc_cen_L7 = master_lon_O_L7c

################# add lat/lon of centroid to dataframe ################

stn_grid_O_top30c['Lat Cen'] = latc_cen_top30
stn_grid_O_top30c['Lon Cen'] = lonc_cen_top30
stn_grid_O_L7c['Lat Cen'] = latc_cen_L7
stn_grid_O_L7c['Lon Cen'] = lonc_cen_L7

O_top30_filc = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/CFSR/geometry_top30_con_CFSR.csv"])
O_L7_filc = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/CFSR/geometry_L7_con_CFSR.csv"])
#print(stn_grid_O_L1b)

stn_grid_O_top30c.to_csv(O_top30_filc,na_rep=np.nan,index=False)
stn_grid_O_L7c.to_csv(O_L7_filc,na_rep=np.nan,index=False)
