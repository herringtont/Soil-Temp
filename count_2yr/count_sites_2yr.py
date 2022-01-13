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


sites_yr = []
st_dt_yr = []
ed_dt_yr = []
lat_yr = []
lon_yr = []
olr_yr = []
layer_yr = []
thr_yr = []

sites_CFSR = []
st_dt_CFSR = []
ed_dt_CFSR = []
lat_CFSR = []
lon_CFSR = []
olr_CFSR = []
layer_CFSR = []
thr_CFSR = []
obs_list = "/mnt/data/users/herringtont/soil_temp/In-Situ/Master_Lat_Long_Obs.csv"
dframe_obs = pd.read_csv(obs_list)
print(dframe_obs)
############### loop through files ##############
def load_pandas(file_name):
    #print("Loading in-situ file: ", file_name)
    sitid = file_name.split("site_")[1].split(".csv")[0]
    outlr = file_name.split("/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/")[1].split("/")[0]
    #print(outlr)
    layer = file_name.split("/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/"+outlr+"/")[1].split("/")[0]
    #print(layer)
    thresh = file_name.split("/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/"+outlr+"/"+layer+"/monthly_average/thr_")[1].split("/")[0]
    #print(thresh)
    dframe_obs_sit = dframe_obs[dframe_obs["Site_ID"] == int(sitid)]
    #print(dframe_obs_sit)
    #print(dframe_obs_sit)
    lat = dframe_obs_sit["Lat"].tolist()
    lon = dframe_obs_sit["Long"].tolist()   
    dframe = pd.read_csv(file_name)
    dframe.replace(-999,np.nan,inplace=True)
    levels = dframe.columns.values.tolist()
    date = dframe['Date']
    col1 = dframe[levels[0]]
    date_fmt = "%Y-%m-%d"
    datetime_column = str_to_datetime(col1, date_fmt)
    dframe_stemp = dframe['Layer_Avg']
    len_dat = len(date)        
    s_dat = datetime.datetime.strptime(date[0], date_fmt)
    e_dat = datetime.datetime.strptime(date[len(date)-1], date_fmt)
    diff = relativedelta(e_dat,s_dat)
    diff2 = (diff.years*12) + diff.months
    #print("Start Date:",s_dat,", End Date:",e_dat,", Months:",diff2)

    site_nam = "site_"+sitid
    #print(s_dat, e_dat)
    dat_val = date.tolist()
    dat_val2 = [datetime.datetime.strptime(x, date_fmt) for x in dat_val]
    mont = [datetime.datetime.strptime(x,date_fmt).month for x in dat_val]
    mont = np.array(mont)
    #print(mont)
    jan = np.where(mont == 1)[0]
    feb = np.where(mont == 2)[0]
    mar = np.where(mont == 3)[0]
    apr = np.where(mont == 4)[0]
    may = np.where(mont == 5)[0]
    jun = np.where(mont == 6)[0]
    jul = np.where(mont == 7)[0]
    aug = np.where(mont == 8)[0]
    sep = np.where(mont == 9)[0]
    obr = np.where(mont == 10)[0]
    nov = np.where(mont == 11)[0]
    dec = np.where(mont == 12)[0]
    #print( jan, feb, mar, apr, may, jun, jul, aug, sep, obr, nov, dec)
    jan_l = len(jan)
    feb_l = len(feb)
    mar_l = len(mar)
    apr_l = len(apr)
    may_l = len(may)
    jun_l = len(jun)
    jul_l = len(jul)
    aug_l = len(aug)
    sep_l = len(sep)
    obr_l = len(obr)
    nov_l = len(nov)
    dec_l = len(dec)
    
    #if (len(jan) >= 2 and len(feb) >=2 and len(mar) >=2 and len(apr) >= 2 and len(may) >= 2 and len(jun) >= 2 and len(jul) >= 2 and len(aug) >= 2 and len(sep) >= 2 and len(obr) >= 2 and len(nov) >=2 and len(dec) >= 2):
    	#print("number of datapoints for each month:", jan_l,feb_l,mar_l,apr_l,may_l,jun_l,jul_l,aug_l,sep_l,obr_l,nov_l,dec_l)   
    date_set = set(dat_val2)
    one_m = relativedelta(months = +1)
    test_date = dat_val2[0]
    test_date2 = dat_val2[0]
    missing = []
    valid = []
    num_miss = len(missing)

    while test_date < dat_val2[-1]:
    	if test_date not in date_set:
    		missing.append(test_date)
    	test_date += one_m
    while test_date2 < dat_val2[-1]:
    	if test_date in date_set:
    		valid.append(test_date)
    	test_date2 += one_m
   
    
    if (s_dat <= datetime.datetime(2010,12,31) and e_dat >= datetime.datetime(2011,1,1)):
    	#print("site",str(i))
    	sites_CFSR.append(sitid)
    	st_dt_CFSR.append(s_dat)
    	ed_dt_CFSR.append(e_dat)
    	lat_CFSR.append(lat)
    	lon_CFSR.append(lon)
    	olr_CFSR.append(outlr)
    	layer_CFSR.append(layer)
    	thr_CFSR.append(thresh)	
	
    if (len_dat >= 24 and diff2 >= 25 and len(jan) >= 2 and len(feb) >=2 and len(mar) >=2 and len(apr) >= 2 and len(may) >= 2 and len(jun) >= 2 and len(jul) >= 2 and len(aug) >= 2 and len(sep) >= 2 and len(obr) >= 2 and len(nov) >=2 and len(dec) >= 2):
    	sites_yr.append(sitid)
    	st_dt_yr.append(s_dat)
    	ed_dt_yr.append(e_dat)
    	lat_yr.append(lat)
    	lon_yr.append(lon)
    	olr_yr.append(outlr)
    	layer_yr.append(layer)
    	thr_yr.append(thresh)	


def main():
#set files and directories
    from pathlib import Path
    directory = "/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/"
    olr = ['outliers','zscore','IQR']
    dep = ['0_9.9','10_29.9','30_99.9','100_299.9','300_deeper','top_30cm','30_299.9']
    thr = ['0','25','50','75','100']

    for i in olr:
    	#print(i)
    	for j in dep:
    		for k in thr:
    			pthl2 = [directory,str(i),"/",str(j),"/","monthly_average","/","thr_",str(k),"/"]	
    			pthl3 = "".join(pthl2)
    			#print(pthl3)
    			pathlist = Path(pthl3).glob('*.csv')
    			#print(pathlist)		
    			for path in sorted(pathlist, key=lambda path: int(path.stem.rsplit("_",1)[1])):
    				fil = str(path)
    				load_pandas(fil)

main()


sites_CFSR = [j for sub in [sites_CFSR] for j in sub]
sites_yr = [j for sub in [sites_yr] for j in sub]
st_dt_CFSR = [j for sub in [st_dt_CFSR] for j in sub]
st_dt_yr = [j for sub in [st_dt_yr] for j in sub]
ed_dt_CFSR = [j for sub in [ed_dt_CFSR] for j in sub]
ed_dt_yr = [j for sub in [ed_dt_yr] for j in sub]
lat_CFSR = [j for sub in [lat_CFSR] for j in sub]
lat_yr = [j for sub in [lat_yr] for j in sub]
lon_CFSR = [j for sub in [lon_CFSR] for j in sub]
lon_yr = [j for sub in [lon_yr] for j in sub]
olr_CFSR = [j for sub in [olr_CFSR] for j in sub]
olr_yr = [j for sub in [olr_yr] for j in sub]
layer_CFSR = [j for sub in [layer_CFSR] for j in sub]
layer_yr = [j for sub in [layer_yr] for j in sub]
thr_CFSR = [j for sub in [thr_CFSR] for j in sub]
thr_yr = [j for sub in [thr_yr] for j in sub]

sites_CFSR = np.asarray(sites_CFSR)
sites_yr = np.asarray(sites_yr)
st_dt_CFSR = np.asarray(st_dt_CFSR)
st_dt_yr = np.asarray(st_dt_yr)
ed_dt_CFSR = np.asarray(ed_dt_CFSR)
ed_dt_yr = np.asarray(ed_dt_yr)
lat_CFSR = np.asarray(lat_CFSR)
lat_yr = np.asarray(lat_yr)
lon_CFSR = np.asarray(lon_CFSR)
lon_yr = np.asarray(lon_yr)
olr_CFSR = np.asarray(olr_CFSR)
olr_yr = np.asarray(olr_yr)
layer_CFSR = np.asarray(layer_CFSR)
layer_yr = np.asarray(layer_yr)
thr_CFSR = np.asarray(thr_CFSR)
thr_yr = np.asarray(thr_yr)


df_CFSR = pd.DataFrame(data=sites_CFSR, columns=['Sites'])
df_CFSR.insert(1, 'Start_Date', st_dt_CFSR)
df_CFSR.insert(2, 'End_Date', ed_dt_CFSR)
df_CFSR.insert(3, 'Lat', lat_CFSR)
df_CFSR.insert(4, 'Long', lon_CFSR)
df_CFSR.insert(5, 'Outlier', olr_CFSR)
df_CFSR.insert(6, 'Layer', layer_CFSR)
df_CFSR.insert(7, 'Threshold', thr_CFSR)

df_yr = pd.DataFrame(data=sites_yr, columns=['Sites'])
df_yr.insert(1, 'Start_Date', st_dt_yr)
df_yr.insert(2, 'End_Date', ed_dt_yr)
df_yr.insert(3, 'Lat', lat_yr)
df_yr.insert(4, 'Long', lon_yr)
df_yr.insert(5, 'Outlier', olr_yr)
df_yr.insert(6, 'Layer', layer_yr)
df_yr.insert(7, 'Threshold', thr_yr)

print(df_yr)
#print(df_CFSR)

olr = ['outliers','zscore','IQR']
lyr = ['0_9.9','10_29.9','30_99.9','100_299.9','300_deeper','top_30cm','30_299.9']
thr = ['0','25','50','75','100']

for i in olr:
    olri = i 
    #print(df_yr_new)
    for j in lyr:
    	lyrj = j
    	#print(df_yr_new)
    	for k in thr:
    		thrk = k	
    		df_CFSR_new = df_CFSR[(df_CFSR['Outlier'] == i) & (df_CFSR['Layer'] ==j) & (df_CFSR['Threshold'] == k)] 
    		df_yr_new = df_yr[(df_yr['Outlier'] == i) & (df_yr['Layer'] ==j) & (df_yr['Threshold'] == k)]
			
    		#print("Outlier:",olri,", Layer:",lyrj,", Threshold:",thrk)
    		if (len(df_CFSR_new) > 0):
    			#print("Outlier:",i,", Layer:",j,", Threshold:",k)
    			#print("Sites with both CFSR and CFSR2:",len(df_CFSR_new))
    			CFSR_ofil = "".join(["/mnt/data/users/herringtont/soil_temp/CFSR_overlap/CFSR_overlap_sites_",str(olri),"_",str(lyrj),"_",str(thrk),".csv"])
    			df_CFSR_new.to_csv(CFSR_ofil,index=False)
    		if (len(df_yr_new) > 0):
    			print("Outlier:",i,", Layer:",j,", Threshold:",k)
    			print("Sites with at least 2 years:",len(df_yr_new))
    			yr_ofil = "".join(["/mnt/data/users/herringtont/soil_temp/sites_2yr/sites_2yr_",str(olri),"_",str(lyrj),"_",str(thrk),".csv"])
    			df_yr_new.to_csv(yr_ofil,index=False)

#df_CFSR.to_csv("/mnt/data/users/herringtont/soil_temp/CFSR_overlap_sites.csv")
#df_yr.to_csv("/mnt/data/users/herringtont/soil_temp/sites_2yr.csv")    	
    
