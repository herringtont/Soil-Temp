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
from mpl_toolkits.basemap import Basemap


Rnysis_f = "/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/common_grid/remapnn/ERA-Interim.nc"
sit_f = "/mnt/data/users/herringtont/soil_temp/In-Situ/Master_Lat_Long_Obs.csv"

############## Grab Reanalysis Grid Data (remapped to ERA-Interim) ###############
Rnysis_fil = xr.open_dataset(Rnysis_f)
latR = Rnysis_fil.lat
latR = np.array(latR.values)
lonR = Rnysis_fil.lon
lonR = np.array(lonR.values)



############# Grab Station Lat/Lon Data
dframeO = pd.read_csv(sit_f)
dframeG = dframeO[dframeO['Dataset'] == 'GTN-P']
dframeK = dframeO[dframeO['Dataset'] == 'Kropp']
latG = dframeG['Lat'].values
lonG = dframeG['Long'].values
#lonG = ((lonG+360)%360).round(decimals=2).values #station longitudes are offset by 180 relative to reanalysis

latK = dframeK['Lat'].values
lonK = dframeK['Long'].values
#lonK = ((lonK+360)%360).round(decimals=2).values #station longitudes are offset by 180 relative to reanalysis

sitO = dframeO['Site_ID'].values
dtstO = dframeO['Dataset'].values

#print(latO,lonO)


############# create Basemap ###############
m = Basemap(projection='npstere',boundinglat=50,lon_0=180,resolution='l')
m.bluemarble()
m.drawcoastlines()
m.drawmapboundary(fill_color='white')
x,y = m(lonG,latG)
x2,y2 = m(lonK,latK)
GTN = plt.scatter(x,y,10,marker='o', color='Yellow') ## plot GTN-P sites as yellow circles
Kr = plt.scatter(x2,y2, 10,marker='^', color = 'Red') ## plot Kropp sites as red triangles
plt.legend((GTN,Kr),('GTN-P','Kropp'), loc = 'best')
plt.tight_layout()
plt.show()
