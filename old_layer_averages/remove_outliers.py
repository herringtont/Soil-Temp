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
from calendar import isleap
from dateutil.relativedelta import *


def load_pandas(file_name):
    print("Loading file: ", file_name)
    dframe = pd.read_csv(file_name)


#read depth from csv file
    dep = dframe.iloc[1,4]
    dep2 = str(dep)
##check if depth is integer or float
    a = "." in dep2 #look for decimal in dep
    if ( a == False ):
    	sdep = int(dep2)
    elif ( a == True ):
    	sdep = float(dep2)

#separate files into bins by depth
    if (sdep == "0"):
    	bins = "0_9.9"
    elif (0 <= sdep < 10):
    	bins = "0_9.9"
    elif (10 <= sdep < 30):
    	bins = "10_29.9"
    elif (30 <= sdep < 100):
    	bins = "30_99.9"
    elif (100 <= sdep < 300):
    	bins = "100_299.9"
    elif (sdep >= 300):
    	bins = "300_deeper"

    #print("Bin :",bins)

    dframe.replace(-999, np.nan, inplace =True)
    levels = dframe.columns.values.tolist()
    date = dframe.iloc[:,0]
    #print(date)
    total_col = len(dframe.axes[1]) #count number of columns
    col_val = np.array(dframe.columns) #store column names    
    col1 = dframe[levels[0]]
    # Sample date: 2011-06-21 08:00:00
    sitid = file_name.split("site_")[1].split("_depth")[0] #locate site id within filename
    sdepth = file_name.split("_depth_")[1].split(".csv")[0]
# read values from csv files
    dataset = dframe.iloc[1,0] #take value in 2nd row of column 0
    sdep =  dframe.iloc[1,4]
    dat_tim = dframe.iloc[:,1] #grab values from second column
    stemp = dframe.iloc[:,5]
    lat = dframe.iloc[1,2]
    lon = dframe.iloc[1,3]
    stemp2 = np.array(stemp)
    
    if ( dataset == "GTN-P" ):
    	date_fmt = "%Y-%m-%d %H:%M:%S"
    elif ( dataset == "Kropp" ):
    	date_fmt = "%Y-%m-%d" 
	
#######Test for outliers

####1 - Standard Deviation Method

    threshold = 3.5 #set outlier threshold to this many standard deviations away from mean

    mean_value = stemp.mean(axis = 0, skipna = True)
    stdev = stemp.std()

    stemp_new_m1 = []
    dat_new_m1 = []
    for i in range(0,len(stemp)):
    	stmp = stemp.iloc[i]
    	dat2 = dat_tim[i]

    	z = (stmp - mean_value)/stdev
    	z_abs = abs(z)
    
    	if (z_abs > threshold or stmp == np.nan):
    		sval = np.nan
    		dval = dat2
    	else:
    		sval = stmp
    		dval = dat2
    	stemp_new_m1.append(sval)
    	dat_new_m1.append(dval)	

    #n_dict = {"Date":dat_new_m1,"Soil_Temp":stemp_new_m1}
    #dframe_dt_stemp_m1 = pd.DataFrame(n_dict)

####2 - IQR Method
    Q1 = stemp.quantile(0.25)
    Q3 = stemp.quantile(0.75)
    IQR = Q3-Q1
    fence = IQR*1.5

    stemp_new_m2 = []
    dat_new_m2 = []
    for i in range (0,len(stemp)):
    	stmp = stemp.iloc[i]
    	dat2 = dat_tim[i]
    
    	if(stmp < (Q1 - fence) or stmp > (Q3 + fence) or stmp == np.nan):
    		sval = np.nan
    		dval = dat2
    	else:
    		sval = stmp
    		dval = dat2
    	stemp_new_m2.append(sval)
    	dat_new_m2.append(dval)

    #n_dict2 = {"Date":dat_new_m2,"Soil_Temp":stemp_new_m2} 
    #dframe_dt_stemp_m2 = pd.DataFrame(n_dict2)
    #print(dframe_dt_stemp_m1)
    #print(dframe_dt_stemp_m2)

####Create Outfiles
#Z-Score Method
    ofilZ = ["/mnt/data/users/herringtont/soil_temp/In-Situ/All/depth_level/layers/no_outliers/z_score/",bins,"/site_",sitid,"_depth_",dep2,"_zscore.csv"]
    ofilZ = "".join(ofilZ)
    Z_dict = {"Dataset":dataset,"Date":dat_new_m1,"Lat":lat,"Lon":lon,"Depth_cm":dep2,"Soil_Temp":stemp_new_m1}
    dframe_Z = pd.DataFrame(Z_dict)
    #dframe_Z = dframe_Z[dframe_Z['Soil_Temp'].notna()]
    print(ofilZ)
    dframe_Z.to_csv(ofilZ,na_rep="NaN", header=['Dataset','Date','Lat','Lon','Depth_cm','Soil_Temp'], index=False)
    
#IQR Method
    ofilI = ["/mnt/data/users/herringtont/soil_temp/In-Situ/All/depth_level/layers/no_outliers/IQR/",bins,"/site_",sitid,"_depth_",dep2,"_IQR.csv"]
    ofilI = "".join(ofilI)
    I_dict = {"Dataset":dataset,"Date":dat_new_m2,"Lat":lat,"Lon":lon,"Depth_cm":dep2,"Soil_Temp":stemp_new_m2}
    dframe_I = pd.DataFrame(I_dict)
    #dframe_I = dframe_I[dframe_I['Soil_Temp'].notna()]
    print(ofilI)
    dframe_I.to_csv(ofilI,na_rep="NaN", header=['Dataset','Date','Lat','Lon','Depth_cm','Soil_Temp'], index=False)
    
def main():
#set files and directories
    from pathlib import Path
    directory = "/mnt/data/users/herringtont/soil_temp/In-Situ/All/depth_level/layers/"
    dep = ['0_9.9','10_29.9','30_99.9','100_299.9','300_deeper']
    #dep = ['0_9.9']

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
