# -*- coding: utf-8 -*-

import os
import csv
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import numpy as np
import scipy
import pandas as pd
from decimal import *
getcontext().prec = 4
#set files and directories
wdir = str("/mnt/data/users/herringtont/soil_temp/In-Situ/GTN-P/site_level/")
odir = str("/mnt/data/users/herringtont/soil_temp/In-Situ/All/depth_level/")
pdir = str("/mnt/data/users/herringtont/soil_temp/In-Situ/plots/")


import re

def remove_trailing_zeros(x):
    return str(x).rstrip('0').rstrip('.')
    
def str_to_datetime(column, date_fmt):

    date_list = []

    for dt_str in column:
        new_dt = datetime.datetime.strptime(dt_str, date_fmt)
        date_list.append(new_dt)
	# specify offset for UTC
		

    return date_list


def load_pandas(file_name):
    pattern = re.compile(r'\d+')
    substring = pattern.findall(file_name)
    substring2 = "".join(substring)
    
    print("Loading file:", file_name)

    dframe = pd.read_csv(file_name)
    dframe.replace(-999, np.nan, inplace =True)
    levels = dframe.columns.values.tolist()
    date = dframe['Date/Depth']
    #print(date)
    total_col = len(dframe.axes[1]) #count number of columns
    col_val = np.array(dframe.columns) #store column names
    print(total_col)
    
    
    #print("Levels:", levels)
    #print(levels[0])
    #print("Column types:", dframe.dtypes)

    #print(dframe)

    col1 = dframe[levels[0]]
    # Sample date: 2011-06-21 08:00:00
    date_fmt = "%Y-%m-%d %H:%M:%S"
    
    datetime_column = str_to_datetime(col1, date_fmt)
    # The pandas builtin seems to have issues
    #datetime_column = pd.to_datetime(dframe[levels[0]], date_fmt)
    #print("Length of datetime column:", len(datetime_column))    
   
    #dframe['Datetime'] = pd.to_datetime(dframe['Date/Depth'], format=date_fmt)
    #dframe2 = dframe['Date/Depth'].dt.strftime('%Y-%m-%d %H:%M:%S')
    #dframe2 = dframe.set_index(pd.DatetimeIndex(dframe['Date/Depth']))
    
    #Loop through soil depths
    for i in range (3,total_col):
    	stemp = dframe.iloc[:,[0,1,2,i]] # select ith column
    	print(col_val[i])
    	a = "." in col_val[i]
    	#print(a)
    	if ( a == False):
    		if ( col_val[i] == "0" ):
    			sdepth = int(col_val[i])
    			sdepth2 = Decimal(sdepth * 100)
    			#print("sdepth type is :", type(sdepth))
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
    	#print(sdepth2)
    	#print(sdepth_cm)
    	if ( sdepth2 < 0 ):
    		bins = "above_ground"
    	elif ( sdepth2 == 0 ):
    		bins = "0_4.9"
    	elif ( 0 <= sdepth2 < 5 ):
    		bins = "0_4.9"
    	elif ( 5 <= sdepth2 < 10 ):
    		bins = "5_9.9"
    	elif ( 10 <= sdepth2 < 15 ):
    		bins = "10_14.9"
    	elif ( 15 <= sdepth2 < 20 ):
    		bins = "15_19.9"
    	elif ( 20 <= sdepth2 < 30 ):
    		bins = "20_29.9"
    	elif ( 30 <= sdepth2 < 50 ):
    		bins = "30_49.9"
    	elif ( 50 <= sdepth2 < 70 ): 
    		bins = "50_69.9"
    	elif ( 70 <= sdepth2 < 100 ):
    		bins = "70_99.9"
    	elif ( 100 <= sdepth2 < 150 ):
    		bins = "100_149.9"
    	elif ( 150 <= sdepth2 < 200 ):
    		bins = "150_199.9"
    	elif ( 200 <= sdepth2 < 300 ):
    		bins = "200_299.9"
    	elif ( sdepth2 >= 300 ):
    		bins = "300_deeper"		
    	print(bins)
    	print(sdepth2)	
    	print(sdepth_cm)
    	sdep_s = str(sdepth_cm)
    	#stemp.columns = ['soil_temp']
    	#date2 = date.apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    	#stemp2 = stemp.rename(index={'Date/Depth': 'Date/Time'})
    	#print(stemp2)	
    	odir2 = [odir,bins,"/"]
    	odir3 = "".join(odir2)
    	#print(odir3)
    	ofil = [odir3,"site_",substring2,"_depth_",sdep_s,".csv"]
    	#print(ofil)
    	ofil2 = "".join(ofil)
    	print(ofil2)	
    	stemp.insert(0,'Dataset','GTN-P')
    	stemp.insert(4,'depth_cm',sdepth_cm)
    	#print(stemp)

    	#print(stemp)
    	stemp.to_csv(ofil2,na_rep="NaN", header=['Dataset','Date','Lat','Lon','Depth_cm','Soil_Temp'], index=False)
#	

		
def main():

	from pathlib import Path
	directory = "/mnt/data/users/herringtont/soil_temp/In-Situ/GTN-P/site_level/"
	directory_as_str = str(directory)
	pathlist = Path(directory_as_str).glob('*.csv')
	for path in pathlist:
	#because path is object not string
		path_in_str = str(path)
		pattern = re.compile(r'\d+')
		substring = pattern.findall(path_in_str)
		#print(substring)
		#print(path_in_str)
		boreholes = path_in_str
		load_pandas(boreholes)
		

main()
