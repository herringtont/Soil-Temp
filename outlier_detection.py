# -*- coding: utf-8 -*-

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
    dep2 = str(dep)
##check if depth is integer or float
    a = "." in dep2 #look for decimal in dep
    if ( a == False ):
    	sdep = int(dep2)
    	#sdep2 = sdep
    elif ( a == True ):
    	sdep = float(dep2)
    	#sdep2 = math.trunc(sdep)   				    

#separate files into bins by depth
    if ( sdep < 0 ):
    	bins = "above_ground"
    elif ( sdep == "0" ):
    	bins = "0_4.9"
    elif ( 0 <= sdep < 5 ):
    	bins = "0_4.9"
    elif ( 5 <= sdep < 10 ):
    	bins = "5_9.9"
    elif ( 10 <= sdep < 15 ):
    	bins = "10_14.9"
    elif ( 15 <= sdep < 20 ):
    	bins = "15_19.9"
    elif ( 20 <= sdep < 30 ):
    	bins = "20_29.9"
    elif ( 30 <= sdep < 50 ):
    	bins = "30_49.9"
    elif ( 50 <= sdep < 70 ): 
    	bins = "50_69.9"
    elif ( 70 <= sdep < 100 ):
    	bins = "70_99.9"
    elif ( 100 <= sdep < 150 ):
    	bins = "100_149.9"
    elif ( 150 <= sdep < 200 ):
    	bins = "150_199.9"
    elif ( 200 <= sdep < 300 ):
    	bins = "200_299.9"
    elif ( sdep >= 300 ):
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
    sitid = file_name.split("site_")[1].split("_depth")[0] #locate site id within filename
    sdepth = file_name.split("_depth_")[1].split(".csv")[0]
# read values from csv files
    dataset = dframe.iloc[1,0] #take value in 2nd row of column 0
    sdep =  dframe.iloc[1,4]
    dat_tim = dframe.iloc[:,1] #grab values from second column
    stemp = dframe.iloc[:,5]
    stemp2 = np.array(stemp)
    
    if ( dataset == "GTN-P" ):
    	date_fmt = "%Y-%m-%d %H:%M:%S"
    elif ( dataset == "Kropp" ):
    	date_fmt = "%Y-%m-%d" 
    #datetime_column = str_to_datetime(col1, date_fmt)

    #print(stemp2)
    stemp3 = stemp2[np.logical_not(np.isnan(stemp2))]
    #print(type(stemp3))
###plot histogram of data 
#    plt.style.use('seaborn-white')
#    fig = plt.hist(stemp3, color = 'blue', edgecolor = 'black')
#    plt.xlabel('value')
#    plt.ylabel('frequency')
#    pfila = ["/mnt/data/users/herringtont/soil_temp/In-Situ/All/histograms/",str(bins),"/","site_",str(sitid),"_depth_",str(dep2),"_histogram.png"]
#    pfila2 = "".join(pfila)
#    print(pfila2)
#    plt.savefig(pfila2)
#    plt.close()
#
#
##plot boxplot of data
#    sns.set_theme(style="whitegrid")
#    bxplt = sns.boxplot(x=stemp, data=stemp)
#    pfilb = ["/mnt/data/users/herringtont/soil_temp/In-Situ/All/boxplots/",str(bins),"/","site_",str(sitid),"_depth_",str(dep2),"_boxplot.png"]
#    pfilb2 = "".join(pfilb)
#    plt.savefig(pfilb2)
#    plt.close() 


####test for outliers (Standard deviation method)
    mean = stemp.mean
    std = stemp.std
    threshold = 3.5 #set outlier threshold to 3.5 std dev above mean
    stemp_sdev = []
    for i in stemp:
    	z = (i-mean)/std
    	if (z > threshold):
    		sval = np.nan
    	else:
    		sval = i
    	stemp_sdev.append(sval)
	
####test for outliers (IQR method)
    Q1 = stemp.quantile(0.25)
    Q3 = stemp.quantile(0.75)
    IQR = Q3 - Q1
    #print(IQR)
    outliers_IQR = (stemp < (Q1 - (1.5*IQR))) | (stemp > (Q3 + (1.5*IQR)))
    olr_len = len(outliers_IQR)
    print(outliers_IQR)

#if stemp is an outlier, replace with Pandas nan value

    stemp_new = []
    stemp_newI = []
    for i in range (0,olr_len):
    	if (outliers[i] == False):
    		stemp_n = stemp[i]
    		stemp_nI = i
    		print(outliers_IQR[i])
    		#print("outliers is False")
    	else:
    		stemp_n = np.nan
    		stemp_nI = np.nan
    		print(outliers_IQR[i])
    		#print("outliers is true")
    	print(stemp[i])
    	print(stemp_n)
    	stemp_new.append(stemp_n)
    	stemp_new.append(stemp_nI)
    	#print(stemp_new)				

    stemp_LOF = []
    stemp_LOFi = []
    stemp_IF = []
    stemp_IFi = []
###impute missing data and reshape to 2d array
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy = 'mean')
    stemp2d = stemp2.reshape(-1,1)
    stemp2i = imputer.fit_transform(stemp2d)    
    #print(stemp2i)    
#####test for outliers - Local Outlier Factor
    from sklearn.neighbors import LocalOutlierFactor
    clf = LocalOutlierFactor()
    outliers_LOF = clf.fit_predict(stemp2i)
    olr_lenLOF = len(outliers_LOF)
    #print(outliers_LOF)
    for i in range (0,olr_lenLOF):
    	if (outliers_LOF == 1):
    		stemp_l3 = stemp[i]
    	elif (outliers_LOF == -1):
    		stemp_l3 = np.nan
    	print(stemp_l3)
	stemp_LOF.append(stemp_l3)
#####test for outliers - Isolation Forest
    from sklearn.ensemble import IsolationForest
    clf = IsolationForest(random_state=0).fit(stemp2d)
    outliers_IF = clf.predict(stemp2d)
    olr_lenIF = len(outliers_IF)   
    print(outliers_IF)
    for i in range (0,olr_lenIF):
    	if (outliers_IF == 1):
    		stemp_n3 = stemp[i]
    	elif (outliers_IF == -1):
    		stemp_n3 = np.nan
    	print(stemp_n3)
    	stemp_IF.append(stemp_n3)

###Drop missing values
    #stemp_IF.dropna(inplace=True)
    #stemp_LOF.dropna(inplace=True)
		
def main():
#set files and directories
    from pathlib import Path
    directory = "/mnt/data/users/herringtont/soil_temp/In-Situ/All/depth_level/"
    dep = ['above_ground','0_4.9','5_9.9','10_14.9','15_19.9','20_29.9','30_49.9','50_69.9','70_99.9','100_149.9','150_199.9','200_299.99','300_deeper']
    #dep = ['0_4.9']

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
