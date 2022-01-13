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

ERAI_fic = "/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/obs_depths/common_grid_CLSM/remapcon/common_date/GLDAS_CLSM.nc"
ERAI_fib = "/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/common_grid/remapbil/ERA-Interim.nc"
ERAI_fin = "/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/common_grid/remapnn/ERA-Interim.nc"

ERAI_filc = xr.open_dataset(ERAI_fic)
ERAI_filb = xr.open_dataset(ERAI_fib)
ERAI_filn = xr.open_dataset(ERAI_fin)

latRb = ERAI_filb.lat
lonRb = ERAI_filb.lon
latRb = np.array(latRb.values)
lonRb = np.array(lonRb.values)

latRn = ERAI_filb.lat
lonRn = ERAI_filb.lon
latRn = np.array(latRn.values)
lonRn = np.array(lonRn.values)

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

for b in range(0,len(lonRn)-1):
    for c in  range(0,len(latRn)-1):  ### loop through latitude values
    	x1 = lonRn[b] #leftmost x-coordinate of grid cell
    	x2 = lonRn[b+1] #rightmost x-coordinate of grid cell
    	y1 = latRn[c] #topmost y-coordinate of grid cell 
    	y2 = latRn[c+1] #bottommost y-coordinate of grid cell
    	grid = Polygon([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])
    	#print(grid)
    	grid_cellsn.append(grid)

grid_cellsn = np.array(grid_cellsn)
grid_numn = np.arange(1,len(grid_cellsn)+1,1)

for i in range(0,len(lonRb)-1): ### loop through longitude values
    for j in  range(0,len(latRb)-1):  ### loop through latitude values
    	x1 = lonRb[i] #leftmost x-coordinate of grid cell
    	x2 = lonRb[i+1] #rightmost x-coordinate of grid cell
    	y1 = latRb[j] #topmost y-coordinate of grid cell 
    	y2 = latRb[j+1] #bottommost y-coordinate of grid cell
    	grid = Polygon([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])
    	grid_cellsb.append(grid)

grid_cellsb = np.array(grid_cellsb)
grid_numb = np.arange(1,len(grid_cellsb)+1,1)

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
latC_master_olr_L1 = []
lonC_master_olr_L1 = []
site_master_olr_L1 = []

latC_master_olr_L2 = []
lonC_master_olr_L2 = []
site_master_olr_L2 = []

latC_master_olr_L3 = []
lonC_master_olr_L3 = []
site_master_olr_L3 = []

latC_master_olr_L4 = []
lonC_master_olr_L4 = []
site_master_olr_L4 = []

latC_master_olr_L5 = []
lonC_master_olr_L5 = []
site_master_olr_L5 = []

latC_master_olr_top30 = []
lonC_master_olr_top30 = []
site_master_olr_top30 = []

latC_master_olr_L7 = []
lonC_master_olr_L7 = []
site_master_olr_L7 = []

latC_master_Z_L1 = []
lonC_master_Z_L1 = []
site_master_Z_L1 = []

latC_master_Z_L2 = []
lonC_master_Z_L2 = []
site_master_Z_L2 = []

latC_master_Z_L3 = []
lonC_master_Z_L3 = []
site_master_Z_L3 = []

latC_master_Z_L4 = []
lonC_master_Z_L4 = []
site_master_Z_L4 = []

latC_master_Z_L5 = []
lonC_master_Z_L5 = []
site_master_Z_L5 = []

latC_master_Z_top30 = []
lonC_master_Z_top30 = []
site_master_Z_top30 = []

latC_master_Z_L7 = []
lonC_master_Z_L7 = []
site_master_Z_L7 = []

latC_master_I_L1 = []
lonC_master_I_L1 = []
site_master_I_L1 = []

latC_master_I_L2 = []
lonC_master_I_L2 = []
site_master_I_L2 = []

latC_master_I_L3 = []
lonC_master_I_L3 = []
site_master_I_L3 = []

latC_master_I_L4 = []
lonC_master_I_L4 = []
site_master_I_L4 = []

latC_master_I_L5 = []
lonC_master_I_L5 = []
site_master_I_L5 = []

latC_master_I_top30 = []
lonC_master_I_top30 = []
site_master_I_top30 = []

latC_master_I_L7 = []
lonC_master_I_L7 = []
site_master_I_L7 = []
################## grab latitude and longitude coordinates of station data ################

def load_pandas(file_name):
    print("Loading file: ", file_name)
    dframe = pd.read_csv(file_name)
    print(dframe)
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
    	if(lyrid == "0_9.9"):
    		latC_master_olr_L1.append(latC)
    		lonC_master_olr_L1.append(lonC2)
    		site_master_olr_L1.append(sitid)
    	elif(lyrid == "10_29.9"):
    		latC_master_olr_L2.append(latC)
    		lonC_master_olr_L2.append(lonC2)
    		site_master_olr_L2.append(sitid)
    	elif(lyrid == "30_99.9"):
    		latC_master_olr_L3.append(latC)
    		lonC_master_olr_L3.append(lonC2)
    		site_master_olr_L3.append(sitid)    			
    	elif(lyrid == "100_299.9"):
    		latC_master_olr_L4.append(latC)
    		lonC_master_olr_L4.append(lonC2)
    		site_master_olr_L4.append(sitid)
    	elif(lyrid == "300_deeper"):
    		latC_master_olr_L5.append(latC)
    		lonC_master_olr_L5.append(lonC2)
    		site_master_olr_L5.append(sitid)
    	elif(lyrid == "top_30cm"):
    		latC_master_olr_top30.append(latC)
    		lonC_master_olr_top30.append(lonC2)
    		site_master_olr_top30.append(sitid)
    	elif(lyrid == "30_299.9"):
    		latC_master_olr_L7.append(latC)
    		lonC_master_olr_L7.append(lonC2)
    		site_master_olr_L7.append(sitid)

    elif(olrid == "zscore"):
    	if(lyrid == "0_9.9"):
    		latC_master_Z_L1.append(latC)
    		lonC_master_Z_L1.append(lonC2)
    		site_master_Z_L1.append(sitid)
    	elif(lyrid == "10_29.9"):
    		latC_master_Z_L2.append(latC)
    		lonC_master_Z_L2.append(lonC2)
    		site_master_Z_L2.append(sitid)
    	elif(lyrid == "30_99.9"):
    		latC_master_Z_L3.append(latC)
    		lonC_master_Z_L3.append(lonC2)
    		site_master_Z_L3.append(sitid)    			
    	elif(lyrid == "100_299.9"):
    		latC_master_Z_L4.append(latC)
    		lonC_master_Z_L4.append(lonC2)
    		site_master_Z_L4.append(sitid)
    	elif(lyrid == "300_deeper"):
    		latC_master_Z_L5.append(latC)
    		lonC_master_Z_L5.append(lonC2)
    		site_master_Z_L5.append(sitid)
    	elif(lyrid == "top_30cm"):
    		latC_master_Z_top30.append(latC)
    		lonC_master_Z_top30.append(lonC2)
    		site_master_Z_top30.append(sitid)
    	elif(lyrid == "30_299.9"):
    		latC_master_Z_L7.append(latC)
    		lonC_master_Z_L7.append(lonC2)
    		site_master_Z_L7.append(sitid)


    elif(olrid == "IQR"):
    	if(lyrid == "0_9.9"):
    		latC_master_I_L1.append(latC)
    		lonC_master_I_L1.append(lonC2)
    		site_master_I_L1.append(sitid)
    	elif(lyrid == "10_29.9"):
    		latC_master_I_L2.append(latC)
    		lonC_master_I_L2.append(lonC2)
    		site_master_I_L2.append(sitid)
    	elif(lyrid == "30_99.9"):
    		latC_master_I_L3.append(latC)
    		lonC_master_I_L3.append(lonC2)
    		site_master_I_L3.append(sitid)    			
    	elif(lyrid == "100_299.9"):
    		latC_master_I_L4.append(latC)
    		lonC_master_I_L4.append(lonC2)
    		site_master_I_L4.append(sitid)
    	elif(lyrid == "300_deeper"):
    		latC_master_I_L5.append(latC)
    		lonC_master_I_L5.append(lonC2)
    		site_master_I_L5.append(sitid)
    	elif(lyrid == "top_30cm"):
    		latC_master_I_top30.append(latC)
    		lonC_master_I_top30.append(lonC2)
    		site_master_I_top30.append(sitid)
    	elif(lyrid == "30_299.9"):
    		latC_master_I_L4.append(latC)
    		lonC_master_I_L4.append(lonC2)
    		site_master_I_L4.append(sitid)

def main():
#set files and directories
    from pathlib import Path
    directory = "/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/"
    olr = ['outliers','zscore','IQR']
    dep = ['0_9.9','10_29.9','30_99.9','100_299.9','300_deeper','top_30cm','30_299.9']
    print(len(dep))
    #dep = ['0_4.9']

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

		
latC_master_olr_L1n = np.array(latC_master_olr_L1)
lonC_master_olr_L1n = np.array(lonC_master_olr_L1)
site_master_olr_L1n = np.array(site_master_olr_L1)

latC_master_olr_L2n = np.array(latC_master_olr_L2)
lonC_master_olr_L2n = np.array(lonC_master_olr_L2)
site_master_olr_L2n = np.array(site_master_olr_L2)

latC_master_olr_L3n = np.array(latC_master_olr_L3)
lonC_master_olr_L3n = np.array(lonC_master_olr_L3)
site_master_olr_L3n = np.array(site_master_olr_L3)

latC_master_olr_L4n = np.array(latC_master_olr_L4)
lonC_master_olr_L4n = np.array(lonC_master_olr_L4)
site_master_olr_L4n = np.array(site_master_olr_L4)

latC_master_olr_L5n = np.array(latC_master_olr_L5)
lonC_master_olr_L5n = np.array(lonC_master_olr_L5)
site_master_olr_L5n = np.array(site_master_olr_L5)

latC_master_olr_top30n = np.array(latC_master_olr_top30)
lonC_master_olr_top30n = np.array(lonC_master_olr_top30)
site_master_olr_top30n = np.array(site_master_olr_top30)

latC_master_olr_L7n = np.array(latC_master_olr_L7)
lonC_master_olr_L7n = np.array(lonC_master_olr_L7)
site_master_olr_L7n = np.array(site_master_olr_L7)


latC_master_Z_L1n = np.array(latC_master_Z_L1)
lonC_master_Z_L1n = np.array(lonC_master_Z_L1)
site_master_Z_L1n = np.array(site_master_Z_L1)

latC_master_Z_L2n = np.array(latC_master_Z_L2)
lonC_master_Z_L2n = np.array(lonC_master_Z_L2)
site_master_Z_L2n = np.array(site_master_Z_L2)

latC_master_Z_L3n = np.array(latC_master_Z_L3)
lonC_master_Z_L3n = np.array(lonC_master_Z_L3)
site_master_Z_L3n = np.array(site_master_Z_L3)

latC_master_Z_L4n = np.array(latC_master_Z_L4)
lonC_master_Z_L4n = np.array(lonC_master_Z_L4)
site_master_Z_L4n = np.array(site_master_Z_L4)

latC_master_Z_L5n = np.array(latC_master_Z_L5)
lonC_master_Z_L5n = np.array(lonC_master_Z_L5)
site_master_Z_L5n = np.array(site_master_Z_L5)

latC_master_Z_top30n = np.array(latC_master_Z_top30)
lonC_master_Z_top30n = np.array(lonC_master_Z_top30)
site_master_Z_top30n = np.array(site_master_Z_top30)

latC_master_Z_L7n = np.array(latC_master_Z_L7)
lonC_master_Z_L7n = np.array(lonC_master_Z_L7)
site_master_Z_L7n = np.array(site_master_Z_L7)


latC_master_I_L1n = np.array(latC_master_I_L1)
lonC_master_I_L1n = np.array(lonC_master_I_L1)
site_master_I_L1n = np.array(site_master_I_L1)

latC_master_I_L2n = np.array(latC_master_I_L2)
lonC_master_I_L2n = np.array(lonC_master_I_L2)
site_master_I_L2n = np.array(site_master_I_L2)

latC_master_I_L3n = np.array(latC_master_I_L3)
lonC_master_I_L3n = np.array(lonC_master_I_L3)
site_master_I_L3n = np.array(site_master_I_L3)

latC_master_I_L4n = np.array(latC_master_I_L4)
lonC_master_I_L4n = np.array(lonC_master_I_L4)
site_master_I_L4n = np.array(site_master_I_L4)

latC_master_I_L5n = np.array(latC_master_I_L5)
lonC_master_I_L5n = np.array(lonC_master_I_L5)
site_master_I_L5n = np.array(site_master_I_L5)

latC_master_I_top30n = np.array(latC_master_I_top30)
lonC_master_I_top30n = np.array(lonC_master_I_top30)
site_master_I_top30n = np.array(site_master_I_top30)

latC_master_I_L7n = np.array(latC_master_I_L7)
lonC_master_I_L7n = np.array(lonC_master_I_L7)
site_master_I_L7n = np.array(site_master_I_L7)

print(site_master_olr_L1)
print(site_master_olr_top30)

################ create geopandas dataframes #################

rmp_type = ['nn','bil','con']

#### reanalysis grids ###### 
dframe_gridb = pd.DataFrame({'Grid Cell':grid_numb})
dframe_grid_geob = gpd.GeoDataFrame(dframe_gridb, geometry = grid_cellsb)

dframe_gridn = pd.DataFrame({'Grid Cell':grid_numn})
dframe_grid_geon = gpd.GeoDataFrame(dframe_gridn, geometry = grid_cellsn)

dframe_gridc = pd.DataFrame({'Grid Cell':grid_numc})
dframe_grid_geoc = gpd.GeoDataFrame(dframe_gridc, geometry = grid_cellsc)

##### with outliers ######
dframe_stn_O_L1 = pd.DataFrame({'site':site_master_olr_L1n,'lat':latC_master_olr_L1n,'lon':lonC_master_olr_L1n})
dframe_stn_O_L2 = pd.DataFrame({'site':site_master_olr_L2n,'lat':latC_master_olr_L2n,'lon':lonC_master_olr_L2n})
dframe_stn_O_L3 = pd.DataFrame({'site':site_master_olr_L3n,'lat':latC_master_olr_L3n,'lon':lonC_master_olr_L3n})
dframe_stn_O_L4 = pd.DataFrame({'site':site_master_olr_L4n,'lat':latC_master_olr_L4n,'lon':lonC_master_olr_L4n})
dframe_stn_O_L5 = pd.DataFrame({'site':site_master_olr_L5n,'lat':latC_master_olr_L5n,'lon':lonC_master_olr_L5n})
dframe_stn_O_top30 = pd.DataFrame({'site':site_master_olr_top30n,'lat':latC_master_olr_top30n,'lon':lonC_master_olr_top30n})
dframe_stn_O_L7 = pd.DataFrame({'site':site_master_olr_L7n,'lat':latC_master_olr_L7n,'lon':lonC_master_olr_L7n})


dframe_stn_O_L1_geo = gpd.GeoDataFrame(dframe_stn_O_L1, geometry=gpd.points_from_xy(dframe_stn_O_L1.lon, dframe_stn_O_L1.lat)) 
dframe_stn_O_L2_geo = gpd.GeoDataFrame(dframe_stn_O_L2, geometry=gpd.points_from_xy(dframe_stn_O_L2.lon, dframe_stn_O_L2.lat))
dframe_stn_O_L3_geo = gpd.GeoDataFrame(dframe_stn_O_L3, geometry=gpd.points_from_xy(dframe_stn_O_L3.lon, dframe_stn_O_L3.lat))
dframe_stn_O_L4_geo = gpd.GeoDataFrame(dframe_stn_O_L4, geometry=gpd.points_from_xy(dframe_stn_O_L4.lon, dframe_stn_O_L4.lat))
dframe_stn_O_L5_geo = gpd.GeoDataFrame(dframe_stn_O_L5, geometry=gpd.points_from_xy(dframe_stn_O_L5.lon, dframe_stn_O_L5.lat))
dframe_stn_O_top30_geo = gpd.GeoDataFrame(dframe_stn_O_top30, geometry=gpd.points_from_xy(dframe_stn_O_top30.lon, dframe_stn_O_top30.lat))
dframe_stn_O_L7_geo = gpd.GeoDataFrame(dframe_stn_O_L7, geometry=gpd.points_from_xy(dframe_stn_O_L7.lon, dframe_stn_O_L7.lat))

print("dframe_stn_O_top30_geo: ",dframe_stn_O_top30_geo)
print("dframe_stn_O_L7_geo: ",dframe_stn_O_L7_geo)


################## find centroid of grid cell #####################################
dframe_grid_geob_cen = dframe_grid_geob.centroid
dframe_grid_geon_cen = dframe_grid_geon.centroid
dframe_grid_geoc_cen = dframe_grid_geoc.centroid

dframe_grid_geob['Lat Cen'] = dframe_grid_geob_cen.y.values
dframe_grid_geob['Lon Cen'] = dframe_grid_geob_cen.x.values

dframe_grid_geon['Lat Cen'] = dframe_grid_geon_cen.y.values
dframe_grid_geon['Lon Cen'] = dframe_grid_geon_cen.x.values

dframe_grid_geoc['Lat Cen'] = dframe_grid_geoc_cen.y.values
dframe_grid_geoc['Lon Cen'] = dframe_grid_geoc_cen.x.values

################## do a spatial join to figure out which grid cell each station belongs to ################

##### with outliers ######
stn_grid_O_L1b = gpd.sjoin(dframe_stn_O_L1_geo, dframe_grid_geob, how="inner", op ='intersects')
stn_grid_O_L2b = gpd.sjoin(dframe_stn_O_L2_geo, dframe_grid_geob, how="inner", op ='intersects')
stn_grid_O_L3b = gpd.sjoin(dframe_stn_O_L3_geo, dframe_grid_geob, how="inner", op ='intersects')
stn_grid_O_L4b = gpd.sjoin(dframe_stn_O_L4_geo, dframe_grid_geob, how="inner", op ='intersects')
stn_grid_O_L5b = gpd.sjoin(dframe_stn_O_L5_geo, dframe_grid_geob, how="inner", op ='intersects')
stn_grid_O_top30b = gpd.sjoin(dframe_stn_O_top30_geo, dframe_grid_geob, how="inner", op ='intersects')
stn_grid_O_L7b = gpd.sjoin(dframe_stn_O_L7_geo, dframe_grid_geob, how="inner", op ='intersects')

stn_grid_O_L1n = gpd.sjoin(dframe_stn_O_L1_geo, dframe_grid_geon, how="inner", op ='intersects')
stn_grid_O_L2n = gpd.sjoin(dframe_stn_O_L2_geo, dframe_grid_geon, how="inner", op ='intersects')
stn_grid_O_L3n = gpd.sjoin(dframe_stn_O_L3_geo, dframe_grid_geon, how="inner", op ='intersects')
stn_grid_O_L4n = gpd.sjoin(dframe_stn_O_L4_geo, dframe_grid_geon, how="inner", op ='intersects')
stn_grid_O_L5n = gpd.sjoin(dframe_stn_O_L5_geo, dframe_grid_geon, how="inner", op ='intersects')
stn_grid_O_top30n = gpd.sjoin(dframe_stn_O_top30_geo, dframe_grid_geon, how="inner", op ='intersects')
stn_grid_O_L7n = gpd.sjoin(dframe_stn_O_L7_geo, dframe_grid_geon, how="inner", op ='intersects')

stn_grid_O_L1c = gpd.sjoin(dframe_stn_O_L1_geo, dframe_grid_geoc, how="inner", op ='intersects')
stn_grid_O_L2c = gpd.sjoin(dframe_stn_O_L2_geo, dframe_grid_geoc, how="inner", op ='intersects')
stn_grid_O_L3c = gpd.sjoin(dframe_stn_O_L3_geo, dframe_grid_geoc, how="inner", op ='intersects')
stn_grid_O_L4c = gpd.sjoin(dframe_stn_O_L4_geo, dframe_grid_geoc, how="inner", op ='intersects')
stn_grid_O_L5c = gpd.sjoin(dframe_stn_O_L5_geo, dframe_grid_geoc, how="inner", op ='intersects')
stn_grid_O_top30c = gpd.sjoin(dframe_stn_O_top30_geo, dframe_grid_geoc, how="inner", op ='intersects')
stn_grid_O_L7c = gpd.sjoin(dframe_stn_O_L7_geo, dframe_grid_geoc, how="inner", op ='intersects')

################# find lat/lon of grid cell centroid ##########################


stn_grid_O_L1b_gc = stn_grid_O_L1b['Grid Cell']
stn_grid_O_L2b_gc = stn_grid_O_L2b['Grid Cell']
stn_grid_O_L3b_gc = stn_grid_O_L3b['Grid Cell']
stn_grid_O_L4b_gc = stn_grid_O_L4b['Grid Cell']
stn_grid_O_L5b_gc = stn_grid_O_L5b['Grid Cell']
stn_grid_O_top30b_gc = stn_grid_O_top30b['Grid Cell']
stn_grid_O_L7b_gc = stn_grid_O_L7b['Grid Cell']

master_lat_O_L1b = []
master_lon_O_L1b = []
for a in stn_grid_O_L1b_gc:
    dframe_grid_geob_latc = dframe_grid_geob['Lat Cen'][dframe_grid_geob['Grid Cell'] == a].values
    dframe_grid_geob_lonc = dframe_grid_geob['Lon Cen'][dframe_grid_geob['Grid Cell'] == a].values
    master_lat_O_L1b.append(dframe_grid_geob_latc)
    master_lon_O_L1b.append(dframe_grid_geob_lonc)

master_lat_O_L1b = np.array(master_lat_O_L1b).flatten()
master_lon_O_L1b = np.array(master_lon_O_L1b).flatten()

master_lat_O_L2b = []
master_lon_O_L2b = []
for b in stn_grid_O_L2b_gc:
    dframe_grid_geob_latc = dframe_grid_geob['Lat Cen'][dframe_grid_geob['Grid Cell'] == b].values
    dframe_grid_geob_lonc = dframe_grid_geob['Lon Cen'][dframe_grid_geob['Grid Cell'] == b].values
    master_lat_O_L2b.append(dframe_grid_geob_latc)
    master_lon_O_L2b.append(dframe_grid_geob_lonc)

master_lat_O_L2b = np.array(master_lat_O_L2b).flatten()
master_lon_O_L2b = np.array(master_lon_O_L2b).flatten()

master_lat_O_L3b = []
master_lon_O_L3b = []
for c in stn_grid_O_L3b_gc:
    dframe_grid_geob_latc = dframe_grid_geob['Lat Cen'][dframe_grid_geob['Grid Cell'] == c].values
    dframe_grid_geob_lonc = dframe_grid_geob['Lon Cen'][dframe_grid_geob['Grid Cell'] == c].values
    master_lat_O_L3b.append(dframe_grid_geob_latc)
    master_lon_O_L3b.append(dframe_grid_geob_lonc)

master_lat_O_L3b = np.array(master_lat_O_L3b).flatten()
master_lon_O_L3b = np.array(master_lon_O_L3b).flatten()

master_lat_O_L4b = []
master_lon_O_L4b = []
for d in stn_grid_O_L4b_gc:
    dframe_grid_geob_latc = dframe_grid_geob['Lat Cen'][dframe_grid_geob['Grid Cell'] == d].values
    dframe_grid_geob_lonc = dframe_grid_geob['Lon Cen'][dframe_grid_geob['Grid Cell'] == d].values
    master_lat_O_L4b.append(dframe_grid_geob_latc)
    master_lon_O_L4b.append(dframe_grid_geob_lonc)

master_lat_O_L4b = np.array(master_lat_O_L4b).flatten()
master_lon_O_L4b = np.array(master_lon_O_L4b).flatten()


master_lat_O_L5b = []
master_lon_O_L5b = []
for e in stn_grid_O_L5b_gc:
    dframe_grid_geob_latc = dframe_grid_geob['Lat Cen'][dframe_grid_geob['Grid Cell'] == e].values
    dframe_grid_geob_lonc = dframe_grid_geob['Lon Cen'][dframe_grid_geob['Grid Cell'] == e].values
    master_lat_O_L5b.append(dframe_grid_geob_latc)
    master_lon_O_L5b.append(dframe_grid_geob_lonc)

master_lat_O_L5b = np.array(master_lat_O_L5b).flatten()
master_lon_O_L5b = np.array(master_lon_O_L5b).flatten()


master_lat_O_top30b = []
master_lon_O_top30b = []
for f in stn_grid_O_top30b_gc:
    dframe_grid_geob_latc = dframe_grid_geob['Lat Cen'][dframe_grid_geob['Grid Cell'] == f].values
    dframe_grid_geob_lonc = dframe_grid_geob['Lon Cen'][dframe_grid_geob['Grid Cell'] == f].values
    master_lat_O_top30b.append(dframe_grid_geob_latc)
    master_lon_O_top30b.append(dframe_grid_geob_lonc)

master_lat_O_top30b = np.array(master_lat_O_top30b).flatten()
master_lon_O_top30b = np.array(master_lon_O_top30b).flatten()


master_lat_O_L7b = []
master_lon_O_L7b = []
for g in stn_grid_O_L7b_gc:
    dframe_grid_geob_latc = dframe_grid_geob['Lat Cen'][dframe_grid_geob['Grid Cell'] == g].values
    dframe_grid_geob_lonc = dframe_grid_geob['Lon Cen'][dframe_grid_geob['Grid Cell'] == g].values
    master_lat_O_L7b.append(dframe_grid_geob_latc)
    master_lon_O_L7b.append(dframe_grid_geob_lonc)

master_lat_O_L7b = np.array(master_lat_O_L7b).flatten()
master_lon_O_L7b = np.array(master_lon_O_L7b).flatten()

stn_grid_O_L1n_gc = stn_grid_O_L1n['Grid Cell']
stn_grid_O_L2n_gc = stn_grid_O_L2n['Grid Cell']
stn_grid_O_L3n_gc = stn_grid_O_L3n['Grid Cell']
stn_grid_O_L4n_gc = stn_grid_O_L4n['Grid Cell']
stn_grid_O_L5n_gc = stn_grid_O_L5n['Grid Cell']
stn_grid_O_top30n_gc = stn_grid_O_top30n['Grid Cell']
stn_grid_O_L7n_gc = stn_grid_O_L7n['Grid Cell']


master_lat_O_L1n = []
master_lon_O_L1n = []
for a in stn_grid_O_L1n_gc:
    dframe_grid_geon_latc = dframe_grid_geon['Lat Cen'][dframe_grid_geon['Grid Cell'] == a].values
    dframe_grid_geon_lonc = dframe_grid_geon['Lon Cen'][dframe_grid_geon['Grid Cell'] == a].values
    master_lat_O_L1n.append(dframe_grid_geon_latc)
    master_lon_O_L1n.append(dframe_grid_geon_lonc)

master_lat_O_L1n = np.array(master_lat_O_L1n).flatten()
master_lon_O_L1n = np.array(master_lon_O_L1n).flatten()

master_lat_O_L2n = []
master_lon_O_L2n = []
for b in stn_grid_O_L2n_gc:
    dframe_grid_geon_latc = dframe_grid_geon['Lat Cen'][dframe_grid_geon['Grid Cell'] == b].values
    dframe_grid_geon_lonc = dframe_grid_geon['Lon Cen'][dframe_grid_geon['Grid Cell'] == b].values
    master_lat_O_L2n.append(dframe_grid_geon_latc)
    master_lon_O_L2n.append(dframe_grid_geon_lonc)

master_lat_O_L2n = np.array(master_lat_O_L2n).flatten()
master_lon_O_L2n = np.array(master_lon_O_L2n).flatten()

master_lat_O_L3n = []
master_lon_O_L3n = []
for c in stn_grid_O_L3n_gc:
    dframe_grid_geon_latc = dframe_grid_geon['Lat Cen'][dframe_grid_geon['Grid Cell'] == c].values
    dframe_grid_geon_lonc = dframe_grid_geon['Lon Cen'][dframe_grid_geon['Grid Cell'] == c].values
    master_lat_O_L3n.append(dframe_grid_geon_latc)
    master_lon_O_L3n.append(dframe_grid_geon_lonc)

master_lat_O_L3n = np.array(master_lat_O_L3n).flatten()
master_lon_O_L3n = np.array(master_lon_O_L3n).flatten()

master_lat_O_L4n = []
master_lon_O_L4n = []
for d in stn_grid_O_L4n_gc:
    dframe_grid_geon_latc = dframe_grid_geon['Lat Cen'][dframe_grid_geon['Grid Cell'] == d].values
    dframe_grid_geon_lonc = dframe_grid_geon['Lon Cen'][dframe_grid_geon['Grid Cell'] == d].values
    master_lat_O_L4n.append(dframe_grid_geon_latc)
    master_lon_O_L4n.append(dframe_grid_geon_lonc)

master_lat_O_L4n = np.array(master_lat_O_L4n).flatten()
master_lon_O_L4n = np.array(master_lon_O_L4n).flatten()


master_lat_O_L5n = []
master_lon_O_L5n = []
for e in stn_grid_O_L5n_gc:
    dframe_grid_geon_latc = dframe_grid_geon['Lat Cen'][dframe_grid_geon['Grid Cell'] == e].values
    dframe_grid_geon_lonc = dframe_grid_geon['Lon Cen'][dframe_grid_geon['Grid Cell'] == e].values
    master_lat_O_L5n.append(dframe_grid_geon_latc)
    master_lon_O_L5n.append(dframe_grid_geon_lonc)

master_lat_O_L5n = np.array(master_lat_O_L5n).flatten()
master_lon_O_L5n = np.array(master_lon_O_L5n).flatten()


master_lat_O_top30n = []
master_lon_O_top30n = []
for f in stn_grid_O_top30n_gc:
    dframe_grid_geon_latc = dframe_grid_geon['Lat Cen'][dframe_grid_geon['Grid Cell'] == f].values
    dframe_grid_geon_lonc = dframe_grid_geon['Lon Cen'][dframe_grid_geon['Grid Cell'] == f].values
    master_lat_O_top30n.append(dframe_grid_geon_latc)
    master_lon_O_top30n.append(dframe_grid_geon_lonc)

master_lat_O_top30n = np.array(master_lat_O_top30n).flatten()
master_lon_O_top30n = np.array(master_lon_O_top30n).flatten()

master_lat_O_L7n = []
master_lon_O_L7n = []
for g in stn_grid_O_L7n_gc:
    dframe_grid_geon_latc = dframe_grid_geon['Lat Cen'][dframe_grid_geon['Grid Cell'] == g].values
    dframe_grid_geon_lonc = dframe_grid_geon['Lon Cen'][dframe_grid_geon['Grid Cell'] == g].values
    master_lat_O_L7n.append(dframe_grid_geon_latc)
    master_lon_O_L7n.append(dframe_grid_geon_lonc)

master_lat_O_L7n = np.array(master_lat_O_L7n).flatten()
master_lon_O_L7n = np.array(master_lon_O_L7n).flatten()



stn_grid_O_L1c_gc = stn_grid_O_L1c['Grid Cell']
stn_grid_O_L2c_gc = stn_grid_O_L2c['Grid Cell']
stn_grid_O_L3c_gc = stn_grid_O_L3c['Grid Cell']
stn_grid_O_L4c_gc = stn_grid_O_L4c['Grid Cell']
stn_grid_O_L5c_gc = stn_grid_O_L5c['Grid Cell']
stn_grid_O_top30c_gc = stn_grid_O_top30c['Grid Cell']
stn_grid_O_L7c_gc = stn_grid_O_L7c['Grid Cell']

master_lat_O_L1c = []
master_lon_O_L1c = []
for a in stn_grid_O_L1c_gc:
    dframe_grid_geoc_latc = dframe_grid_geoc['Lat Cen'][dframe_grid_geoc['Grid Cell'] == a].values
    dframe_grid_geoc_lonc = dframe_grid_geoc['Lon Cen'][dframe_grid_geoc['Grid Cell'] == a].values
    master_lat_O_L1c.append(dframe_grid_geoc_latc)
    master_lon_O_L1c.append(dframe_grid_geoc_lonc)

master_lat_O_L1c = np.array(master_lat_O_L1c).flatten()
master_lon_O_L1c = np.array(master_lon_O_L1c).flatten()

master_lat_O_L2c = []
master_lon_O_L2c = []
for b in stn_grid_O_L2c_gc:
    dframe_grid_geoc_latc = dframe_grid_geoc['Lat Cen'][dframe_grid_geoc['Grid Cell'] == b].values
    dframe_grid_geoc_lonc = dframe_grid_geoc['Lon Cen'][dframe_grid_geoc['Grid Cell'] == b].values
    master_lat_O_L2c.append(dframe_grid_geoc_latc)
    master_lon_O_L2c.append(dframe_grid_geoc_lonc)

master_lat_O_L2c = np.array(master_lat_O_L2c).flatten()
master_lon_O_L2c = np.array(master_lon_O_L2c).flatten()

master_lat_O_L3c = []
master_lon_O_L3c = []
for c in stn_grid_O_L3c_gc:
    dframe_grid_geoc_latc = dframe_grid_geoc['Lat Cen'][dframe_grid_geoc['Grid Cell'] == c].values
    dframe_grid_geoc_lonc = dframe_grid_geoc['Lon Cen'][dframe_grid_geoc['Grid Cell'] == c].values
    master_lat_O_L3c.append(dframe_grid_geoc_latc)
    master_lon_O_L3c.append(dframe_grid_geoc_lonc)

master_lat_O_L3c = np.array(master_lat_O_L3c).flatten()
master_lon_O_L3c = np.array(master_lon_O_L3c).flatten()

master_lat_O_L4c = []
master_lon_O_L4c = []
for d in stn_grid_O_L4c_gc:
    dframe_grid_geoc_latc = dframe_grid_geoc['Lat Cen'][dframe_grid_geoc['Grid Cell'] == d].values
    dframe_grid_geoc_lonc = dframe_grid_geoc['Lon Cen'][dframe_grid_geoc['Grid Cell'] == d].values
    master_lat_O_L4c.append(dframe_grid_geoc_latc)
    master_lon_O_L4c.append(dframe_grid_geoc_lonc)

master_lat_O_L4c = np.array(master_lat_O_L4c).flatten()
master_lon_O_L4c = np.array(master_lon_O_L4c).flatten()


master_lat_O_L5c = []
master_lon_O_L5c = []
for e in stn_grid_O_L5c_gc:
    dframe_grid_geoc_latc = dframe_grid_geoc['Lat Cen'][dframe_grid_geoc['Grid Cell'] == e].values
    dframe_grid_geoc_lonc = dframe_grid_geoc['Lon Cen'][dframe_grid_geoc['Grid Cell'] == e].values
    master_lat_O_L5c.append(dframe_grid_geoc_latc)
    master_lon_O_L5c.append(dframe_grid_geoc_lonc)

master_lat_O_L5c = np.array(master_lat_O_L5c).flatten()
master_lon_O_L5c = np.array(master_lon_O_L5c).flatten()


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


latb_cen_L1 = master_lat_O_L1b
lonb_cen_L1 = master_lon_O_L1b
latb_cen_L2 = master_lat_O_L2b
lonb_cen_L2 = master_lon_O_L2b
latb_cen_L3 = master_lat_O_L3b
lonb_cen_L3 = master_lon_O_L3b
latb_cen_L4 = master_lat_O_L4b
lonb_cen_L4 = master_lon_O_L4b
latb_cen_L5 = master_lat_O_L5b
lonb_cen_L5 = master_lon_O_L5b
latb_cen_top30 = master_lat_O_top30b
lonb_cen_top30 = master_lon_O_top30b
latb_cen_L7 = master_lat_O_L7b
lonb_cen_L7 = master_lon_O_L7b

latn_cen_L1 = master_lat_O_L1n
lonn_cen_L1 = master_lon_O_L1n
latn_cen_L2 = master_lat_O_L2n
lonn_cen_L2 = master_lon_O_L2n
latn_cen_L3 = master_lat_O_L3n
lonn_cen_L3 = master_lon_O_L3n
latn_cen_L4 = master_lat_O_L4n
lonn_cen_L4 = master_lon_O_L4n
latn_cen_L5 = master_lat_O_L5n
lonn_cen_L5 = master_lon_O_L5n
latn_cen_top30 = master_lat_O_top30n
lonn_cen_top30 = master_lon_O_top30n
latn_cen_L7 = master_lat_O_L7n
lonn_cen_L7 = master_lon_O_L7n

latc_cen_L1 = master_lat_O_L1c
lonc_cen_L1 = master_lon_O_L1c
latc_cen_L2 = master_lat_O_L2c
lonc_cen_L2 = master_lon_O_L2c
latc_cen_L3 = master_lat_O_L3c
lonc_cen_L3 = master_lon_O_L3c
latc_cen_L4 = master_lat_O_L4c
lonc_cen_L4 = master_lon_O_L4c
latc_cen_L5 = master_lat_O_L5c
lonc_cen_L5 = master_lon_O_L5c
latc_cen_top30 = master_lat_O_top30c
lonc_cen_top30 = master_lon_O_top30c
latc_cen_L7 = master_lat_O_L7c
lonc_cen_L7 = master_lon_O_L7c
################# add lat/lon of centroid to dataframe ################
stn_grid_O_L1b['Lat Cen'] = latb_cen_L1
stn_grid_O_L1b['Lon Cen'] = lonb_cen_L1
stn_grid_O_L2b['Lat Cen'] = latb_cen_L2
stn_grid_O_L2b['Lon Cen'] = lonb_cen_L2
stn_grid_O_L3b['Lat Cen'] = latb_cen_L3
stn_grid_O_L3b['Lon Cen'] = lonb_cen_L3
stn_grid_O_L4b['Lat Cen'] = latb_cen_L4
stn_grid_O_L4b['Lon Cen'] = lonb_cen_L4
stn_grid_O_L5b['Lat Cen'] = latb_cen_L5
stn_grid_O_L5b['Lon Cen'] = lonb_cen_L5
stn_grid_O_top30b['Lat Cen'] = latb_cen_top30
stn_grid_O_top30b['Lon Cen'] = lonb_cen_top30
stn_grid_O_L7b['Lat Cen'] = latb_cen_L7
stn_grid_O_L7b['Lon Cen'] = lonb_cen_L7

stn_grid_O_L1n['Lat Cen'] = latn_cen_L1
stn_grid_O_L1n['Lon Cen'] = lonn_cen_L1
stn_grid_O_L2n['Lat Cen'] = latn_cen_L2
stn_grid_O_L2n['Lon Cen'] = lonn_cen_L2
stn_grid_O_L3n['Lat Cen'] = latn_cen_L3
stn_grid_O_L3n['Lon Cen'] = lonn_cen_L3
stn_grid_O_L4n['Lat Cen'] = latn_cen_L4
stn_grid_O_L4n['Lon Cen'] = lonn_cen_L4
stn_grid_O_L5n['Lat Cen'] = latn_cen_L5
stn_grid_O_L5n['Lon Cen'] = lonn_cen_L5
stn_grid_O_top30n['Lat Cen'] = latn_cen_top30
stn_grid_O_top30n['Lon Cen'] = lonn_cen_top30
stn_grid_O_L7n['Lat Cen'] = latn_cen_L7
stn_grid_O_L7n['Lon Cen'] = lonn_cen_L7

stn_grid_O_L1c['Lat Cen'] = latc_cen_L1
stn_grid_O_L1c['Lon Cen'] = lonc_cen_L1
stn_grid_O_L2c['Lat Cen'] = latc_cen_L2
stn_grid_O_L2c['Lon Cen'] = lonc_cen_L2
stn_grid_O_L3c['Lat Cen'] = latc_cen_L3
stn_grid_O_L3c['Lon Cen'] = lonc_cen_L3
stn_grid_O_L4c['Lat Cen'] = latc_cen_L4
stn_grid_O_L4c['Lon Cen'] = lonc_cen_L4
stn_grid_O_L5c['Lat Cen'] = latc_cen_L5
stn_grid_O_L5c['Lon Cen'] = lonc_cen_L5
stn_grid_O_top30c['Lat Cen'] = latc_cen_top30
stn_grid_O_top30c['Lon Cen'] = lonc_cen_top30
stn_grid_O_L7c['Lat Cen'] = latc_cen_L7
stn_grid_O_L7c['Lon Cen'] = lonc_cen_L7

O_L1_filb = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/geometry_L1_bil.csv"])
O_L2_filb = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/geometry_L2_bil.csv"])
O_L3_filb = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/geometry_L3_bil.csv"])
O_L4_filb = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/geometry_L4_bil.csv"])
O_L5_filb = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/geometry_L5_bil.csv"])
O_top30_filb = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/geometry_top30_bil.csv"])
O_L7_filb = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/geometry_L7_bil.csv"])

O_L1_filn = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/geometry_L1_nn.csv"])
O_L2_filn = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/geometry_L2_nn.csv"])
O_L3_filn = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/geometry_L3_nn.csv"])
O_L4_filn = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/geometry_L4_nn.csv"])
O_L5_filn = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/geometry_L5_nn.csv"])
O_top30_filn = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/geometry_top30_nn.csv"])
O_L7_filn = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/geometry_L7_nn.csv"])

O_L1_filc = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/geometry_L1_con.csv"])
O_L2_filc = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/geometry_L2_con.csv"])
O_L3_filc = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/geometry_L3_con.csv"])
O_L4_filc = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/geometry_L4_con.csv"])
O_L5_filc = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/geometry_L5_con.csv"])
O_top30_filc = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/geometry_top30_con.csv"])
O_L7_filc = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/geometry_L7_con.csv"])
#print(stn_grid_O_L1b)

stn_grid_O_L1b.to_csv(O_L1_filb,na_rep=np.nan,index=False)
stn_grid_O_L2b.to_csv(O_L2_filb,na_rep=np.nan,index=False)
stn_grid_O_L3b.to_csv(O_L3_filb,na_rep=np.nan,index=False)
stn_grid_O_L4b.to_csv(O_L4_filb,na_rep=np.nan,index=False)
stn_grid_O_L5b.to_csv(O_L5_filb,na_rep=np.nan,index=False)
stn_grid_O_top30b.to_csv(O_top30_filb,na_rep=np.nan,index=False)
stn_grid_O_L7b.to_csv(O_L7_filb,na_rep=np.nan,index=False)

stn_grid_O_L1n.to_csv(O_L1_filn,na_rep=np.nan,index=False)
stn_grid_O_L2n.to_csv(O_L2_filn,na_rep=np.nan,index=False)
stn_grid_O_L3n.to_csv(O_L3_filn,na_rep=np.nan,index=False)
stn_grid_O_L4n.to_csv(O_L4_filn,na_rep=np.nan,index=False)
stn_grid_O_L5n.to_csv(O_L5_filn,na_rep=np.nan,index=False)
stn_grid_O_top30n.to_csv(O_top30_filn,na_rep=np.nan,index=False)
stn_grid_O_L7n.to_csv(O_L7_filn,na_rep=np.nan,index=False)

stn_grid_O_L1c.to_csv(O_L1_filc,na_rep=np.nan,index=False)
stn_grid_O_L2c.to_csv(O_L2_filc,na_rep=np.nan,index=False)
stn_grid_O_L3c.to_csv(O_L3_filc,na_rep=np.nan,index=False)
stn_grid_O_L4c.to_csv(O_L4_filc,na_rep=np.nan,index=False)
stn_grid_O_L5c.to_csv(O_L5_filc,na_rep=np.nan,index=False)
stn_grid_O_top30c.to_csv(O_top30_filc,na_rep=np.nan,index=False)
stn_grid_O_L7c.to_csv(O_L7_filc,na_rep=np.nan,index=False)
