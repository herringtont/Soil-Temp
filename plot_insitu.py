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
import natsort
def str_to_datetime(column, date_fmt):

    date_list = []

    for dt_str in column:
        new_dt = datetime.datetime.strptime(dt_str, date_fmt)
        date_list.append(new_dt)
			

    return date_list


def load_pandas(file_name):
    print("Loading file: ", file_name)
    dframe = pd.read_csv(file_name)
    
    
#read values from csv
    dataset = dframe.iloc[1,1] #take value in 2nd row of column 1
    dat_tim = dframe.iloc[:,0] #grab values from first column
    stemp = dframe['Layer_Avg']
    lat = dframe.iloc[1,2]
    lon = dframe.iloc[1,3]
#date/time format will vary depending on dataset
    sitid = file_name.split("site_")[1].split(".csv")[0] #locate site id within filename
    olrid = file_name.split("no_outliers/")[1].split("/")[0]
    lyrid = file_name.split("/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/"+str(olrid)+"/")[1].split("/site_")[0]
    print(sitid)
    print(olrid)
    print(lyrid)
    #print(sitid)
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
	
#set axes
    fig, ax = plt.subplots()
    x_val = [datetime.datetime.strptime(d, date_fmt).date() for d in dat_tim]
    y_val = stemp
    formatter = mdates.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(formatter)
    ax.plot(x_val, y_val)	

#format the ticks
    ax.get_xlim()
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_minor_locator(months)	
	
#set title
    ttl_list = ["Site: ",sitid,", ","Layer: ", lyrid, " cm, lat: ", slat, ", long: ", slon]
    ttl_name = "".join(ttl_list)
    ax.set(xlabel = 'Date', ylabel='Soil Temperature ($^\circ$ C)', title=ttl_name) ### $^\circ$ allows us to insert a degree symbol

#make room for axes
    ax.grid()  
    fig.autofmt_xdate()
    plt.xticks(rotation=75)
    #mpl.rcParams['xtick.labelsize'] = 10
	
#create figure
    pfil = "".join(["/mnt/data/users/herringtont/soil_temp/plots/InSitu/layer_average/",str(olrid),"/",str(lyrid),"/","site_",str(sitid),".png"])
    print(pfil)
    fig.savefig(pfil)
    plt.close()	

def main():
#set files and directories
    from pathlib import Path
    directory = "/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/"
    olr = ['outliers','zscore','IQR']
    dep = ['0_9.9','10_29.9','30_99.9','100_299.9','300_deeper']
    #dep = ['0_4.9']

    for i in olr:
    	#print(i)
    	for j in dep:
    		pthl2 = [directory,str(i),"/",str(j),"/"]	
    		pthl3 = "".join(pthl2)
    		#print(pthl2)
    		pathlist = Path(pthl3).glob('*.csv')
    		print(pathlist)		
    		for path in sorted(pathlist, key=lambda path: int(path.stem.rsplit("_",1)[1])):
    			fil = str(path)
    			load_pandas(fil)

main()
    
