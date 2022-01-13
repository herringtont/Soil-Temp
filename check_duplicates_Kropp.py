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
import re
import math
import seaborn as sns


def checkIfDuplicates_1(listOfElems):
    ''' Check if given list contains any duplicates '''    
    ''' Check if given list contains any duplicates '''
    if len(listOfElems) == len(set(listOfElems)):
        return False
    else:
        return True

filename = str("/mnt/data/users/herringtont/soil_temp/In-Situ/Kropp/data/soil_temp_date_coord.csv")

dframe = pd.read_csv(filename)
##read date/time


#read in data
dframe = pd.read_csv(filename)
dframe.replace(-999, np.nan, inplace =True)
levels = dframe.columns.values.tolist()
 ###store all unique site id values
sid = np.unique(dframe['site_id'])
sitid = sid[~np.isnan(sid)]

#col1 = dframe['Date']
#print(col1)
# Sample date: 2011-06-21
date_fmt = "%Y-%m-%d"
   
#datetime_column = str_to_datetime(col1, date_fmt)

#date = np.array(date)

###group by site id
#for i in range(2):
duplicates = []
for i in sitid:
	dframe_siteid = dframe[dframe['site_id'] == i]
	sdepth = np.unique(dframe_siteid['st_depth'])
	sdep = sdepth[~np.isnan(sdepth)]
	i2 = i + 68
	sint = str(i2)
	nam = "site_"
	snam = nam + sint
	#date = dframe_siteid['Date']
	#date2 = np.array(date)
	#print(dframe_siteid)
	
	for j in sdep:
		wdep = int(j)
		strdep = str(wdep)
		dep = "depth_"
		sdepth = dep + strdep
		dframe_sdep = dframe_siteid[dframe_siteid['st_depth'] == j]
		date = dframe_sdep['Date']
		date2 = np.array(date)
					
    #print(type(date))
		result = checkIfDuplicates_1(date2)
	    
		if result:
			duplicates.append("site_"+str(i)+"_"+sdepth+" has duplicates")
fil = "/mnt/data/users/herringtont/soil_temp/In-Situ/Kropp_dupes.csv"
df_dupes = pd.DataFrame(duplicates,columns=['Dupes'])
print(df_dupes)
#df_dupes.to_csv(fil,index=False)
		
    	

