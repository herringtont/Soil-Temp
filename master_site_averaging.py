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
import xarray as xr
import seaborn as sns
import pytesmo
import math
import cftime
import re
from decimal import *
from calendar import isleap
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


################## groupings to average over #########################
site_pairs = [[2,3,4,5,6],[7,8],[9,10,11],[13,14,15,16,17],[19,112],[20,21],[22,25],[23,24],[32,33,34,109,255],[45,46],\
[47,48,49],[50,51],[53,67],[58,61],[71,72],[74,75,76,77],[78,79],[81,82],[91,92,93],[94,95,96],list(range(97,107)), \
[102,107,108],[124,125],[127,128],[132,136],[133,134,135],list(range(140,156)),[147,148],[157,158],list(range(170,248)),[249,250,251,252,253],\
[260,262],[291,292],[294,295,296],[297,298,299]]


################## create master storage arrays ######################
dat_m = []
dataset_m = []
lat_m = []
lon_m = []
depth_m = []
stemp_m = []


for i in range (len(site_pairs)): ####grab the 1st dimension of site_pairs (the groupings)
    group = site_pairs[i]
    print(group)
    for j in range(len(group)):  ####loop through each site within a grouping
    	site = group[j]

#################if the sites are from GTN-P dataset ###################
    	if (site < 69):
    		wdir = "/mnt/data/users/herringtont/soil_temp/In-Situ/GTN-P/site_level/"
    		wfil = "".join([wdir,"site_",str(site),".csv"])
    		print(wfil)  	   	
    		dframe = pd.read_csv(wfil)
    		dframe.replace(-999,np.nan,inplace=True)
    		levels = dframe.columns.values.tolist()
    		date = dframe['Date/Depth']
    		lat = dframe['Lat']
					
    		total_col = len(dframe.axes[1]) #count number of columns
    		col_val = np.array(dframe.columns) #store column names
    		col1 = dframe[levels[0]]
    		date_fmt = "%Y-%m-%d %H:%M:%S"
    		datetime_column = str_to_datetime(col1, date_fmt)
    		dframe_stemp = dframe.drop(['Date/Depth','Lat','Long'],axis=1)
		dframe_stemp_10_29.9 =
    		print(dframe_stemp)
###### loop through soil depths ######
    		for i in range (3,total_col):
    			stemp = dframe.iloc[:,[i]] # select ith column
			
    			print(col_val[i])
    			a = "." in col_val[i]

    			if ( a == False):
    				if ( col_val[i] == "0" ):
    					sdepth = int(col_val[i])
    					sdepth2 = Decimal(sdepth * 100)
    					sdepth_cm = (sdepth * 100)
    				else:
    					sdepth = int(col_val[i])
    					sdepth2 = Decimal(sdepth * 100)
    					#print("sdepth type is :", type(sdepth))
    					sdepth_cm = (sdepth * 100)
    			else:
    				sdepth = Decimal(col_val[i])
    				sdepth2 = sdepth * 100
    				#print("sdepth type is :", type(sdepth))
    				sdepth_cm = remove_trailing_zeros(sdepth * 100)
				
    			sdep_s = str(sdepth_cm)
