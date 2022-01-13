import cdo
from cdo import Cdo
import os
import subprocess
from subprocess import call
import cdo
import csv
import datetime as dt  # Python standard library datetime  module
import xarray as xr
import numpy as np
import pandas as pd
import scipy
import netCDF4
from netCDF4 import Dataset, num2date  # http://code.google.com/p/netcdf4-python/
import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid


# Read in central lat/lon coordinates of each grid cell
obs_loc = str("/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/Grid_Cell_Centroid.csv")
dframe = pd.read_csv(obs_loc)

lat_cen = dframe['Lat Cen'].values
lon_cen = dframe['Lon Cen'].values
grid_cell = dframe['Grid Cell'].values



# loop through reanalysis products
# set directory of nc files
from pathlib import Path

directory = "/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/common_grid/remapnn/common_date/"

dir_as_str = str(directory)
pathlist = Path(dir_as_str).glob('*.nc') #choose all nc files

# loop through nc files
for path in pathlist:
    path_in_str = str(path)
    nc_f = path_in_str 
    #print(nc_f)

# read Reanalysis product from filename

    rnys = nc_f.split("/rename/common_grid/remapnn/common_date/")[1].split(".nc")[0] 
    print(rnys) 
    if(rnys == 'CFSR_all'):
    	rnys2 = 'CFSR'
    else:
    	rnys2 = rnys
#set directory of out files 

    odir = ["/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/common_grid/remapnn/grid_level/", rnys2,"/"]
    odir2 = "".join(odir)
    wfil = os.path.basename(nc_f) #capture the filename only
    ofil = wfil.replace('.nc','') #remove .nc at end of filename, replace with space
    ofil2 = ofil.rstrip() #remove space at end
    ofil3 = str(ofil2)
    #print(ofil3)
#loop through each lat/lon pair from csv file
    #loop through grid cells and extract central lat/lon
    for i in range(0,len(grid_cell)):
    	gcell_i = grid_cell[i]
    	lat_i = lat_cen[i]
    	lon_i = lon_cen[i]
    	str_gcell = str(gcell_i)
    	ofil4 = [odir2,ofil3,"_grid_",str_gcell,".nc"]
    	otfil = "".join(ofil4)
    	#print(otfil)

#grab the soil temperatures for the grid cell closest to each lat/lon pair
    	print ("the product is: ", rnys)
    	print ("the site level is: ", str_gcell)
    	print (nc_f)
    	print (otfil)
    	cdo = Cdo()
    	cdo.remapnn('lon='+str(lon_i)+'/lat='+str(lat_i), input=nc_f, output= otfil, options = '-f nc')

