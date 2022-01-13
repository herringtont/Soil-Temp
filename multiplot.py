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
import re
import math

def str_to_datetime(column, date_fmt):

    date_list = []

    for dt_str in column:
        new_dt = datetime.datetime.strptime(dt_str, date_fmt)
        date_list.append(new_dt)
			

    return date_list


def load_pandas(file_name):
    print("Loading file: ", file_name)
    dframe = pd.read_csv(file_name)
##read depth from csv file
    dep = dframe.iloc[1,4]
    
##check if depth is integer or float
    a = "." in dep #look for decimal in dep
    if ( a == False ):
    	sdep = int(substring2)
    	sdep2 = sdep
    elif ( a == True ):
    	sdep = float(substring2)
    	sdep2 = math.trunc(sdep)
    #print(type(sdep))    				    

#separate files into bins by depth
    if ( sdep2 < 0 ):
    	bins = "above_ground"
    elif ( sdep2 == "0" ):
    	bins = "0_4.9"
    elif ( sdep2 in range(0, 5) ):
    	bins = "0_4.9"
    elif ( sdep2 in range(5,10) ):
    	bins = "5_9.9"
    elif ( sdep2 in range(10,15) ):
    	bins = "10_14.9"
    elif ( sdep2 in range(15,20) ):
    	bins = "15_19.9"
    elif ( sdep2 in range(20,30) ):
    	bins = "20_29.9"
    elif ( sdep2 in range(30,50) ):
    	bins = "30_49.9"
    elif ( sdep2 in range(50,70) ): 
    	bins = "50_69.9"
    elif ( sdep2 in range(70,100) ):
    	bins = "70_99.9"
    elif ( sdep2 in range(100,150) ):
    	bins = "100_149.9"
    elif ( sdep2 in range(150,200) ):
    	bins = "150_199.9"
    elif ( sdep2 in range(200,300) ):
    	bins = "200_299.9"
    elif ( sdep2 >= 300 ):
    	bins = "300_deeper"
    #print("Bin: ",bins)		    
    dframe = pd.read_csv(file_name)
    dframe.replace(-999, np.nan, inplace =True)
    levels = dframe.columns.values.tolist()
    date = dframe.iloc[:,0]
    #print(date)
    total_col = len(dframe.axes[1]) #count number of columns
    col_val = np.array(dframe.columns) #store column names    
    col1 = dframe[levels[0]]
    # Sample date: 2011-06-21 08:00:00
    if ( sitid < 69 ):
    	date_fmt = "%Y-%m-%d %H:%M:%S"
    elif ( sitid >= 69 ):
    	date_fmt = "%Y-%m-%d"
    datetime_column = str_to_datetime(col1, date_fmt)

#read values from csv
    dataset = dframe.iloc[1,0] #take value in 2nd row of column 0
    dat_tim = dframe.iloc[:,1] #grab values from second column
    stemp = dframe.iloc[:,5]
    lat = dframe.iloc[1,2]
    lon = dframe.iloc[1,3]
#date/time format will vary depending on dataset
    if ( dataset == "GTN-P" ):
    	date_fmt = "%Y-%m-%d %H:%M:%S"
    elif ( dataset == "Kropp" ):
    	date_fmt = "%Y-%m-%d" 
    lat2 = round(lat,2)
    lon2 = round(lon,2)
    slat = str(lat2)
    slon = str(lon2)

    years = mdates.YearLocator() #every year
    months = mdates.MonthLocator() #every month
    years_fmt = mdates.DateFormatter('%Y')
	
##set axes
    fig, ax = plt.subplots()
    x_val = [datetime.datetime.strptime(d, date_fmt).date() for d in dat_tim]
    y_val = stemp
    formatter = mdates.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(formatter)
    ax.plot(x_val, y_val)	

##format the ticks
    ax.get_xlim()
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_minor_locator(months)	
	
##set title
    ttl_list = ["Site: ",substring2,", ","Soil depth: ", substring2, " cm, lat: ", slat, ", long: ", slon]
    ttl_name = "".join(ttl_list)
    ax.set(xlabel = 'Date', ylabel='Soil Temperature ($^\circ$ C)', title=ttl_name) ### $^\circ$ allows us to insert a degree symbol

##make room for axes
    ax.grid()  
    fig.autofmt_xdate()
    plt.xticks(rotation=75)
    #mpl.rcParams['xtick.labelsize'] = 10
	
##create figure
    pfil = "".join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/plots/",bins,"/","site_",substring,"_depth_",substring2,".png"])
    print(pfil)
    fig.savefig(pfil)
    #plt.show()
    plt.close()	

def main():
#set files and directories
    from pathlib import Path
    directory = "/mnt/data/users/herringtont/soil_temp/In-Situ/All/depth_level/"
    #dep = ['above_ground','0_4.9','5_9.9','10_14.9','15_19.9','20_29.9','30_49.9','50_69.9','70_99.9','100_149.9','150_199.9','200_299.99','300_deeper']
    dep = ['0_4.9']

    for i in dep:
    	#print(i)
    	pthl = [directory,str(i),"/"]
    	pthl2 = "".join(pthl)
    	#print(pthl2)
    	pathlist = Path(pthl2).glob('*.csv')
    	for path in pathlist:
    		fil = str(path)
    		load_pandas(fil)

main()
    
