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
			

    return date_list


def load_pandas(file_name):

    dframe = pd.read_csv(file_name)
    dframe.replace(-999, np.nan, inplace =True)
    levels = dframe.columns.values.tolist()
    ###store all unique site id values
    sid = np.unique(dframe['site_id'])
    sitid = sid[~np.isnan(sid)]
    print("Levels:", levels)
    print("Column types:", dframe.dtypes)
    print(dframe)
    
    col1 = dframe['Date']
    print(col1)
    # Sample date: 2011-06-21
    date_fmt = "%Y-%m-%d"
    
    datetime_column = str_to_datetime(col1, date_fmt)
    # The pandas builtin seems to have issues
    #datetime_column = pd.to_datetime(datcol, date_fmt)
    print("Length of datetime column:", len(datetime_column))
    
    #dframe['Date'] = pd.to_datetime(dframe['Date'], format=date_fmt)
    dframe = dframe.set_index(pd.DatetimeIndex(dframe['Date']))
    
    ###group by site id
    #for i in range(2):
    for i in sitid:
    	dframe_siteid = dframe[dframe['site_id'] == i]
    	sdepth = np.unique(dframe_siteid['st_depth'])
    	sdep = sdepth[~np.isnan(sdepth)]
    	sint = str(i)
    	nam = "site_"
    	snam = nam + sint
    	#print(snam)
	###Need to fix NaN values in sdepth and sid
    	#print(sdep)
    	for j in sdep:
    		wdep = int(j)
    		strdep = str(wdep)
    		dep = "depth_"
    		sdepth = dep + strdep
    		dframe_sdep =dframe_siteid[dframe_siteid['st_depth'] == j]
    		#print(dframe_sdep.dtypes)
                   
    		###calculate monthly mean from daily or hourly data
    		dframe_mon = dframe_sdep.resample('M', convention='start').mean()
    		dframe_mon.reset_index(inplace=True)
		
    		#print(dframe_mon)
		####plot monthly timeseries
    		mon_tim = dframe_mon['Date']
    		mon_stemp = dframe_mon['soil_t']
    		mon_sdepth = dframe_mon['st_depth']
    		mon_lat = dframe_mon.iloc[0, dframe_mon.columns.get_loc('lat')]
    		mon_latd = round(mon_lat, 2)
    		mon_long = dframe_mon.iloc[0, dframe_mon.columns.get_loc('long')]
    		mon_longd = round(mon_long, 2)
    		slat = str(mon_latd)
    		slong = str(mon_longd)
    		#print(slat)
    		#print(slong)
    		years = mdates.YearLocator() #every year
    		months = mdates.MonthLocator() #every month
    		years_fmt = mdates.DateFormatter('%Y')
    		
		###set axes
    		fig, ax = plt.subplots()
    		ax.plot(mon_tim, mon_stemp)
		
		###format the ticks
    		ax.get_xlim()
    		ax.xaxis.set_major_locator(years)
    		ax.xaxis.set_major_formatter(years_fmt)
    		ax.xaxis.set_minor_locator(months)
		
		
		#set title
    		ttl_list = ["Site: ", sint, ", ", "Soil depth: ", strdep, " cm", ", lat: ", slat, ", long: ", slong]
    		ttl_name = "".join(ttl_list)
    		ax.set(xlabel = 'Date', ylabel='Soil Temperature ($^\circ$ C)', title=ttl_name) ### $^\circ$ allows us to insert a degree symbol
    		
		
		###make room for axes
    		ax.grid() 
    		fig.autofmt_xdate()
		
    		#print(ttl_name)
		
    		###count number of missing cells by month
    		dframe_miss = dframe_sdep.isnull().groupby(pd.Grouper(freq='M')).sum() #Group data by month/year
    		#dt_fmt2 = "%Y-%m-%d"
    		#print("Site: ",snam)
    		#print("Soil Depth: ",strdep)
    		#print(dframe_mon)
    		#dframe_mon['Datetime'] = pd.to_datetime(dframe['Date/Depth'], format=dt_fmt2)
    
    		####write monthly averages to csv file
    		###grab filename and remove .csv extension from string
    		wk_file = file_name.replace('.csv','')
    		wk_file2 = wk_file.rstrip()
    		
		####create new filename with _mon.csv at end
    		my_list1= [wk_file2,"_", snam, "_", sdepth,"_mon.csv"]
    		mon_fil = "".join(my_list1)
    		  ####create figure filename
    		my_listfig= [wk_file2,"_", snam, "_", sdepth,"_mon.png"]
    		#print(my_list1)
    		mon_fig = "".join(my_listfig)
    		
		###create figure
    		fig.savefig(mon_fig)
    		matplotlib.pyplot.close()
    		#plt.show()
    		#print(mon_fil)
    		dframe_mon.to_csv(mon_fil,na_rep="NaN")
    		#print(dframe_miss)
    		#dframe_miss['Datetime'] = pd.to_datetime(dframe['Date/Depth'], format=dt_fmt2)
    		
		####create new filename with _miss.csv at end
    		my_list2 = [wk_file2,"_", snam, "_", sdepth,"_miss.csv"]
    		miss_fil = "".join(my_list2)
    		dframe_miss.to_csv(miss_fil,na_rep="NaN")
    
def main():

	path = "/praid/users/herringtont/soil_temp/In-Situ/Kropp/data/soil_temp_date_coord.csv"
	path_in_str = str(path)
	print("File:",path_in_str)
	boreholes = path_in_str
	load_pandas(boreholes)
		


main()
