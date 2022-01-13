# -*- coding: utf-8 -*-
"""
Quick script for getting started
-Daniel
"""


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


def str_to_datetime(column, date_fmt):

    date_list = []

    for dt_str in column:
        new_dt = datetime.datetime.strptime(dt_str, date_fmt)
        date_list.append(new_dt)
	# specify offset for UTC
		

    return date_list


def load_pandas(file_name):

    print("Loading file:", file_name)

    dframe = pd.read_csv(file_name)
    dframe.replace(-999, np.nan, inplace =True)
    levels = dframe.columns.values.tolist()

    print("Levels:", levels)
    print(levels[0])
    print("Column types:", dframe.dtypes)

    print(dframe)

    col1 = dframe[levels[0]]
    # Sample date: 2011-06-21 08:00:00
    date_fmt = "%Y-%m-%d %H:%M:%S"
    
    datetime_column = str_to_datetime(col1, date_fmt)
    # The pandas builtin seems to have issues
    #datetime_column = pd.to_datetime(dframe[levels[0]], date_fmt)
    print("Length of datetime column:", len(datetime_column))    
   
    dframe['Datetime'] = pd.to_datetime(dframe['Date/Depth'], format=date_fmt)
    dframe = dframe.set_index(pd.DatetimeIndex(dframe['Date/Depth']))
    
    #calculate monthly mean from daily or hourly data
    dframe_mon = dframe.resample('M', convention='start').mean()
    #dframe_mon.reset_index(inplace=True)
    
    ###obtain lat/long information from csv file
    #mon_lat = dframe_mon.iloc[0, dframe_mon.columns.get_loc('Lat')]
    #mon_long = dframe_mon.iloc[0, dframe_mon.columns.get_loc('Long')
    #slat = str(mon_lat)
    #slong = str(mon_long)
    
#    ###format dates for figure
#    years = mdates.YearLocator() #every year
#    months = mdates.MonthLocator() #every month
#    years_fmt = mdates.DateFormatter('%Y')
#    
#    ###set axes
#    fig, ax = plt.subplots()
#    ax.plot(mon_tim, mon_stemp)
#		
#    ###format the ticks
#    ax.get_xlim()
#    ax.xaxis.set_major_locator(years)
#    ax.xaxis.set_major_formatter(years_fmt)
#    ax.xaxis.set_minor_locator(months)
    
    #count number of missing cells by month
    dframe_miss = dframe.isnull().groupby(pd.Grouper(freq='M')).sum() #Group data by month/year
    
    #dframe_miss.reset_index(inplace=True)
    dt_fmt2 = "%Y-%m-%d $H-%M-%S"
    #print(dframe_mon)
    #dframe_mon['Datetime'] = pd.to_datetime(dframe['Date/Depth'], format=dt_fmt2)
    
    #grab filename and remove .csv extension from string
    wk_file = file_name.replace('.csv','')
    wk_file2 = wk_file.rstrip()
    
    #create new filename with _mon.csv at end
    my_list1= [wk_file2,"_mon.csv"]
    mon_fil = "".join(my_list1)
    dframe_mon.to_csv(mon_fil,na_rep="NaN")
    #print(dframe_miss)
    #dframe_miss['Datetime'] = pd.to_datetime(dframe['Date/Depth'], format=dt_fmt2)
    
    #create new filename with _miss.csv at end
    my_list2 = [wk_file2,"_miss.csv"]
    miss_fil = "".join(my_list2)
    dframe_miss.to_csv(miss_fil,na_rep="NaN")
    
def main():

	from pathlib import Path
	directory = "/praid/users/herringtont/soil_temp/In-Situ/GTN-P/"
	directory_as_str = str(directory)
	pathlist = Path(directory_as_str).glob('*.csv')
	for path in pathlist:
	# because path is object not string
		path_in_str = str(path)
		#print(path_in_str)
		boreholes = path_in_str
		load_pandas(boreholes)
		


main()
