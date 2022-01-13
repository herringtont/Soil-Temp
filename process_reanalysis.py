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


# Read in lat/lon coordinates of in-situ locations from .csv file
obs_loc = str("/mnt/data/users/herringtont/soil_temp/In-Situ/Master_Lat_Long_Obs.csv")
dframe = pd.read_csv(obs_loc)
lat_csv = dframe['Lat']
lon_csv = dframe['Long']
lon_csv2 = ((lon_csv+360)%360).round(decimals=2) #csv file longitudes are offset by 180 relative to netcdf
#print(lon_csv2)


# loop through reanalysis products
# set directory of nc files
from pathlib import Path

#are we using the remapped files?
remap = "No" #"No" "Yes"

directory = "/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/"

dir_as_str = str(directory)
pathlist = Path(dir_as_str).glob('CFS*.nc') #choose all nc files

# loop through nc files
for path in pathlist:
    path_in_str = str(path)
    nc_f = path_in_str 
    #print(nc_f)

# read Reanalysis product from filename

    rnys = nc_f.split("/rename/")[1].split(".nc")[0] 
    #print(rnys) 
#set directory of out files 

    odir = ["/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/site_level/", rnys,"/"]
    odir2 = "".join(odir)
    wfil = os.path.basename(nc_f) #capture the filename only
    ofil = wfil.replace('.nc','') #remove .nc at end of filename, replace with space
    ofil2 = ofil.rstrip() #remove space at end
    ofil3 = str(ofil2)
    #print(ofil3)
#loop through each lat/lon pair from csv file
    for i in range (0,299):
    	i2 = i+1
    	str_i = str(i2)
    	ofil4 = [odir2,ofil3,"_site_",str_i,".nc"]
    	pfil = [odir2,ofil3,"_site_",str_i,".png"]
    	otfil = "".join(ofil4)
    	ptfil = "".join(pfil)
    	#print(otfil)
    	latf = lat_csv[i]
    	lonf = lon_csv2[i]

#grab the soil temperatures for the grid cell closest to each lat/lon pair
    	print ("the product is: ", rnys)
    	print ("the site level is: ", str_i)
    	print (nc_f)
    	print (otfil)
    	print (ptfil)
    	cdo = Cdo()
    	cdo.remapnn('lon='+str(lonf)+'/lat='+str(latf), input=nc_f, output= otfil, options = '-f nc')
    	#cdo.remapnn('lon=297.58/lat=56.9', input=wkfil, output= otfil, options = '-f nc')
#    	ds = xr.open_dataset(nc_f)
#    	dsloc = ds.sel(lon=lonf,lat=latf,method='nearest')
#    	dsloc[ 



##store lat_idc and lon_idc in csv file:
#  #field names
#fields = ['lat_index', 'lon_index']
#  #data rows of csv file
#rows = [lat_idc, lon_idc]
#
#  #write to csv file
#with open('lat_lon_idx', 'w') as f:
#    write = csv.writer(f)
#    write.writerow(fields)
#    write.writerow(rows)

##grab the timeseries of the reanalysis soil temperature for the gridbox that is closest to the insitu lat/long    
#
#stemp = nc_fid.variables['stl1'][:, lat_idx, lon_idx] 
#
##convert the timestamp into the same format as the insitu data
#    		units = nc_fid.variables['time'].units
#    		times = nc_fid.variables['time']
#    		dates = netCDF4.num2date(times[:], units=units, calendar='gregorian')
#    		date_fmt = "%Y-%m-%d %H:%M:%S"
#    #pydate = dates.indexes['time'].to_datetimeindex()
#    #dates['time'] = datetimeindex
#    #print(dates)
#        
    
