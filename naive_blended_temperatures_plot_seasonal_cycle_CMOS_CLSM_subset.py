import os
import glob
import netCDF4
import csv
import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import matplotlib.patches as mpl_patches
import numpy as np
import scipy
import pandas as pd
import xarray as xr
import seaborn as sns
import pytesmo
import math
import cftime
import pathlib
import re
import cdo
import skill_metrics as sm
from cdo import Cdo
from netCDF4 import Dataset,num2date # http://unidata.github.io/netcdf4-python/
from natsort import natsorted
from natsort import os_sorted
from calendar import isleap
from dateutil.relativedelta import *
from pathlib import Path
import seaborn as sn
from calendar import isleap
from dateutil.relativedelta import *
from scipy.stats import pearsonr
from matplotlib.ticker import (MultipleLocator, AutoLocator, AutoMinorLocator)

########## Define Functions ##########

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def calc_bias(y_pred, y_true):
    diff = np.array(y_pred) - np.array(y_true)
    sum_diff = sum(diff)
    N = len(y_true)
    bias = sum_diff/N
    return bias

def SDVnorm(y_pred, y_true):
    SDVp = np.std(y_pred)
    SDVt = np.std(y_true)
    SDVnorml = SDVp/SDVt
    return SDVnorml

def bias(pred,obs):
    """
    Difference of the mean values.

    Parameters
    ----------
    pred : numpy.ndarray
        Predictions.
    obs : numpy.ndarray
        Observations.

    Returns
    -------
    bias : float
        Bias between observations and predictions.
    """
    return np.mean(pred) - np.mean(obs)

def ubrmsd(o, p, ddof=0):
    """
    Unbiased root-mean-square deviation (uRMSD).

    Parameters
    ----------
    o : numpy.ndarray
        Observations.
    p : numpy.ndarray
        Predictions.
    ddof : int, optional
        Delta degree of freedom.The divisor used in calculations is N - ddof,
        where N represents the number of elements. By default ddof is zero.

    Returns
    -------
    urmsd : float
        Unbiased root-mean-square deviation (uRMSD).
    """
    return np.sqrt(np.sum(((o - np.mean(o)) -
                           (p - np.mean(p))) ** 2) / (len(o) - ddof))



def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


# Convert longitudes to -180 to 180 degrees
def LonTo180(d_lon):
   dlon = ((d_lon + 180) % 360) - 180
   return dlon

############# Set Directories ############

naive_type = ['simple_average']
olr = ['zscore']#['outliers','zscore','IQR']
lyr = ['top_30cm','30cm_300cm']
thr = ['100']#['0','25','50','75','100']
rmp_type = ['con']#['nn','bil','con']
tmp_type = ['raw_temp']


ERA5_air = 'Air_Temp'

############# Grab Reanalysis Data ############

for i in rmp_type:
    rmp_type_i = i
    remap_type = ''.join(['remap'+rmp_type_i])

    for j in naive_type:
    	naive_type_j = j

    	for k in olr:
    		olr_k = k

    		for l in lyr:
    			lyr_l = l
    			if (lyr_l == 'top_30cm'):
    				CFSR_layer = "Soil_Temp_TOP30"
    				CFSR2_layer = "Soil_Temp_TOP30"
    				GLDAS_layer = "Soil_Temp_TOP30"
    				GLDAS_CLSM_layer = "Soil_Temp_TOP30"
    				ERA5_layer = "Soil_Temp_TOP30"
    				ERA5_Land_layer = "Soil_Temp_TOP30"
    				ERAI_layer = "Soil_Temp_TOP30"
    				JRA_layer = "Soil_Temp_TOP30"
    				MERRA2_layer = "Soil_Temp_TOP30"
    				Naive_layer = "Soil_Temp_TOP30"

    				in_situ_layer = 'top_30cm'

    			if (lyr_l == '30cm_300cm'):
    				CFSR_layer = "Soil_Temp_30cm_300cm"
    				CFSR2_layer = "Soil_Temp_30cm_300cm"
    				GLDAS_layer = "Soil_Temp_30cm_300cm"
    				GLDAS_CLSM_layer = "Soil_Temp_30cm_300cm"
    				ERA5_layer = "Soil_Temp_30cm_300cm"
    				ERA5_Land_layer = "Soil_Temp_30cm_300cm"
    				ERAI_layer = "Soil_Temp_30cm_300cm"
    				JRA_layer = "Soil_Temp_30cm_300cm"
    				MERRA2_layer = "Soil_Temp_30cm_300cm"
    				Naive_layer = "Soil_Temp_30cm_300cm"

    				in_situ_layer = '30_299.9'


    			print("Remap Type:",remap_type)
    			print("Layer:",lyr_l)

    			for m in thr:
    				thr_m = m
    				scatter_fil =  ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/CLSM_res/subset/'+str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_dframe_scatterplot_ERA5_-2C_CMOS_CLSM_subset.csv'])

    				timeseries_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_timeseries/new_data/CLSM_res/subset/'+str(remap_type)+"_"+str(naive_type_j)+'_'+str(olr_k)+'_'+str(lyr_l)+'_'+str(thr_m)+'_data_CMOS_CLSM_subset.csv'])

    				dframe_scatter = pd.read_csv(scatter_fil)

    				dframe_timeseries = pd.read_csv(timeseries_fil)

##### separate by continent #####

# Eurasia ##
    				dframe_continent_Eur = dframe_scatter[dframe_scatter['Continent'] == 'Eurasia']
    				#print(dframe_continent_Eur)
    				gcell_Eur = dframe_continent_Eur['Grid Cell'].values
    				gcell_Eur_uq = np.unique(gcell_Eur)

## North America ##
    				dframe_continent_NAm = dframe_scatter[dframe_scatter['Continent'] == 'North_America']
    				gcell_NAm = dframe_continent_NAm['Grid Cell'].values
    				gcell_NAm_uq = np.unique(gcell_NAm)
    				print(gcell_NAm_uq)
				
#### loop through grid cells ####

## Eurasia ##
    				start_date_master_Eur = []
    				end_date_master_Eur = []
    				mon_master_Eur = []
    				grid_cell_master_Eur = []
    				lat_master_Eur = []
    				lon_master_Eur = []
    				sites_incl_master_Eur = []
    				station_clim_master_Eur = []
    				CFSR_clim_master_Eur = []
    				ERAI_clim_master_Eur = []
    				ERA5_clim_master_Eur = []
    				ERA5_Land_clim_master_Eur = []
    				JRA_clim_master_Eur = []
    				MERRA2_clim_master_Eur = []
    				GLDAS_clim_master_Eur = []
    				GLDAS_CLSM_clim_master_Eur = []				
    				jan_val_master_Eur = []
    				feb_val_master_Eur = []
    				mar_val_master_Eur = []
    				apr_val_master_Eur = []
    				may_val_master_Eur = []
    				jun_val_master_Eur = []
    				jul_val_master_Eur = []
    				aug_val_master_Eur = []
    				sep_val_master_Eur = []
    				oct_val_master_Eur = []
    				nov_val_master_Eur = []
    				dec_val_master_Eur = []

    				for n in gcell_Eur_uq:
    					gcell_Eur_n = n
    					#print('Grid Cell:',gcell_Eur_n)
# grab corresponding timeseries data #

    					dframe_gcell = dframe_timeseries[dframe_timeseries['Grid Cell'] == gcell_Eur_n]
    					dframe_gcell_nonan = dframe_gcell.dropna(subset = ['Station'])
    					#print(dframe_gcell_nonan)
    					dtime = dframe_gcell_nonan['Date'].values
    					len_dtime = len(dtime)
    					dt_time = [datetime.datetime.strptime(i,'%Y-%m-%d') for i in dtime]
    					stn_temp = dframe_gcell_nonan['Station'].values
    					#print(dt_time)
# grab datetime information #

    					st_dt = dt_time[0]
    					ed_dt = dt_time[len_dtime - 1]

    					beg_mon = st_dt.month
    					beg_yr = st_dt.year
    					fin_mon = ed_dt.month
    					fin_yr = ed_dt.year



    					#print(st_dt)
    					#print(ed_dt)
    					#print(beg_yr)
    					#print(fin_yr)
    					#print(beg_mon)
    					#print(fin_mon)

# create dictionary to keep track of actual number of dates present in timeseries #
# create base dictionary which will count the number of months#

    					monthly_entries_Eur = dict({'Jan':0,'Feb':0,'Mar':0,'Apr':0,'May':0,'Jun':0,'Jul':0,'Aug':0,'Sep':0,'Oct':0,'Nov':0,'Dec':0})

#Now you have a dictionary which keeps track of how many data entries you have per month. 
#Then you can iterate over all of your data rows, and just increment the monthly_entries[month_key] counter by one for each entry.
#In order to generate the month_key from a datetime object, you'd just run:

    					dt_idx = 0
    					dat_new = []

    					for dt in dt_time: 
    						dt2 = dt_time[dt_idx]
    						#print(dt2.month)
    						dt2_mon = dt2.month
    						#print(dt2_mon) 
    						dat_new.append(dt2_mon)
    						dt_idx += 1


    					for dt in dat_new:
    						dt_i = dt
    						if (dt_i == 1):
    							key = 'Jan'
    						elif (dt_i == 2):
    							key = 'Feb'
    						elif (dt_i == 3):
    							key = 'Mar'
    						elif (dt_i == 4):
    							key = 'Apr'
    						elif (dt_i == 5):
    							key = 'May'
    						elif (dt_i == 6):
    							key = 'Jun'
    						elif (dt_i == 7):
    							key = 'Jul'
    						elif (dt_i == 8):
    							key = 'Aug'
    						elif (dt_i == 9):
    							key = 'Sep'
    						elif (dt_i == 10):
    							key = 'Oct'
    						elif (dt_i == 11):
    							key = 'Nov'
    						elif (dt_i == 12):
    							key = 'Dec'
    						monthly_entries_Eur[key] += 1


    					jan_val = int(monthly_entries_Eur['Jan'])
    					feb_val = int(monthly_entries_Eur['Feb'])
    					mar_val = int(monthly_entries_Eur['Mar'])
    					apr_val = int(monthly_entries_Eur['Apr'])
    					may_val = int(monthly_entries_Eur['May'])
    					jun_val = int(monthly_entries_Eur['Jun'])
    					jul_val = int(monthly_entries_Eur['Jul'])
    					aug_val = int(monthly_entries_Eur['Aug'])
    					sep_val = int(monthly_entries_Eur['Sep'])
    					oct_val = int(monthly_entries_Eur['Oct'])
    					nov_val = int(monthly_entries_Eur['Nov'])
    					dec_val = int(monthly_entries_Eur['Dec'])



# if there are at least 20 years worth of data, calculate climatology #
    					if (jan_val >= 20 and feb_val >= 20 and mar_val >= 20 and apr_val >= 20 and may_val >= 20 and jun_val >= 20 and jul_val >= 20 and aug_val >= 20 and sep_val >= 20 and oct_val >= 20 and nov_val >= 20 and dec_val >= 20 and beg_yr <= 1990 and fin_yr >= 2010):

    							#print(gcell_Eur_n)
    							#print(dframe_gcell_nonan)
    							dframe_gcell_nonan2 = dframe_gcell_nonan.set_index('Date')
    							dframe_climatology = dframe_gcell_nonan2[(dframe_gcell_nonan2.index >= '1991-01-01') & (dframe_gcell_nonan2.index <= '2010-12-31')]
    							dtime_new = dframe_climatology.index.values
    							#print(dframe_climatology)
    							dt_time_new = [datetime.datetime.strptime(i,'%Y-%m-%d') for i in dtime_new]
    							mon_new = [i.month for i in dt_time_new]
    							dframe_climatology['DateTime'] = dt_time_new
    							dframe_climatology['mon'] = mon_new
    							#print(dframe_climatology)
    							climatology = dframe_climatology.groupby('mon').mean()
    							#print(climatology) 

    							grid_clim = climatology['Grid Cell'].values.tolist()
    							lat_clim = climatology['Central Lat'].values.tolist()
    							lon_clim = climatology['Central Lon'].values.tolist()
    							mon_clim = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    							sites_clim = climatology['Sites Incl'].values.tolist()
    							station_clim = climatology['Station'].values.tolist()
    							CFSR_clim = climatology['CFSR'].values.tolist()
    							ERAI_clim = climatology['ERA-Interim'].values.tolist()
    							ERA5_clim = climatology['ERA5'].values.tolist()
    							ERA5_Land_clim = climatology['ERA5-Land'].values.tolist()
    							JRA_clim = climatology['JRA-55'].values.tolist()
    							MERRA2_clim = climatology['MERRA2'].values.tolist()
    							GLDAS_clim = climatology['GLDAS-Noah'].values.tolist()
    							GLDAS_CLSM_clim = climatology['GLDAS-CLSM'].values.tolist()

    							mon_master_Eur.append(mon_clim)
    							station_clim_master_Eur.append(station_clim)
    							CFSR_clim_master_Eur.append(CFSR_clim)
    							ERAI_clim_master_Eur.append(ERAI_clim)
    							ERA5_clim_master_Eur.append(ERA5_clim)
    							ERA5_Land_clim_master_Eur.append(ERA5_Land_clim)
    							JRA_clim_master_Eur.append(JRA_clim)
    							MERRA2_clim_master_Eur.append(MERRA2_clim)
    							GLDAS_clim_master_Eur.append(GLDAS_clim)
    							GLDAS_CLSM_clim_master_Eur.append(GLDAS_CLSM_clim)
    							grid_cell_master_Eur.append(grid_clim)
    							lat_master_Eur.append(lat_clim)
    							lon_master_Eur.append(lon_clim)
    							sites_incl_master_Eur.append(sites_clim)


    				mon_master_Eur = [i for sub in mon_master_Eur for i in sub]
    				grid_cell_master_Eur = [i for sub in grid_cell_master_Eur for i in sub]
    				lat_master_Eur = [i for sub in lat_master_Eur for i in sub]
    				lon_master_Eur = [i for sub in lon_master_Eur for i in sub]
    				sites_incl_master_Eur = [i for sub in sites_incl_master_Eur for i in sub]
    				station_clim_master_Eur = [i for sub in station_clim_master_Eur for i in sub]
    				CFSR_clim_master_Eur = [i for sub in CFSR_clim_master_Eur for i in sub]
    				ERAI_clim_master_Eur = [i for sub in ERAI_clim_master_Eur for i in sub]
    				ERA5_Land_clim_master_Eur = [i for sub in ERA5_Land_clim_master_Eur for i in sub]
    				ERA5_clim_master_Eur = [i for sub in ERA5_clim_master_Eur for i in sub]
    				JRA_clim_master_Eur = [i for sub in JRA_clim_master_Eur for i in sub]
    				MERRA2_clim_master_Eur = [i for sub in MERRA2_clim_master_Eur for i in sub]
    				GLDAS_clim_master_Eur = [i for sub in GLDAS_clim_master_Eur for i in sub]
    				GLDAS_CLSM_clim_master_Eur = [i for sub in GLDAS_CLSM_clim_master_Eur for i in sub]
								
    				dframe_master = pd.DataFrame(data= grid_cell_master_Eur, columns=['Grid Cell'])
    				dframe_master['Central Lat'] = lat_master_Eur
    				dframe_master['Central Lon'] = lon_master_Eur
    				dframe_master['Sites Incl'] = sites_incl_master_Eur
    				dframe_master['Month'] = mon_master_Eur
    				dframe_master['Station'] = station_clim_master_Eur
    				dframe_master['CFSR'] = CFSR_clim_master_Eur
    				dframe_master['ERA-Interim'] = ERAI_clim_master_Eur
    				dframe_master['ERA5'] = ERA5_clim_master_Eur
    				dframe_master['ERA5-Land'] = ERA5_Land_clim_master_Eur
    				dframe_master['JRA-55'] = JRA_clim_master_Eur
    				dframe_master['MERRA2'] = MERRA2_clim_master_Eur
    				dframe_master['GLDAS-Noah'] = GLDAS_clim_master_Eur
    				dframe_master['GLDAS-CLSM'] = GLDAS_CLSM_clim_master_Eur

    				seasonal_cycle_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/seasonal_cycle/'+str(remap_type)+"_"+str(naive_type_j)+'_'+str(olr_k)+'_'+str(lyr_l)+'_'+str(thr_m)+'_seasonal_cycle_1991_2010_Eur.csv'])
    				dframe_master.to_csv(seasonal_cycle_fil,na_rep=np.nan,index=False)

#    					year_diff = fin_yr - beg_yr
#
## if there are at least 4 years worth of data, calculate climatology #
#    					if (year_diff >= 4):
#
#    						#print(gcell_Eur_n)
#    						#print(dframe_gcell_nonan)
#    						dframe_gcell_nonan2 = dframe_gcell_nonan.set_index('Date')
#    						year_begin = beg_yr + 1
#    						year_end = fin_yr - 1
#    						start_date = ''.join([str(year_begin)+'-01-01'])
#    						end_date = ''.join([str(year_end)+'12-31'])
#    						dframe_climatology = dframe_gcell_nonan2[(dframe_gcell_nonan2.index >= start_date) & (dframe_gcell_nonan2.index <= end_date)]
#    						dtime_new = dframe_climatology.index.values
#    						#print(dframe_climatology)
#    						dt_time_new = [datetime.datetime.strptime(i,'%Y-%m-%d') for i in dtime_new]
#    						mon_new = [i.month for i in dt_time_new]
#    						dframe_climatology['DateTime'] = dt_time_new
#    						dframe_climatology['mon'] = mon_new
#    						#print(dframe_climatology)
#    						climatology = dframe_climatology.groupby('mon').mean()
#    						#print(climatology) 
#
#    						
#    						grid_clim = climatology['Grid Cell'].values.tolist()
#    						lat_clim = climatology['Central Lat'].values.tolist()
#    						lon_clim = climatology['Central Lon'].values.tolist()
#    						mon_clim = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
#    						sites_clim = climatology['Sites Incl'].values.tolist()
#    						station_clim = climatology['Station'].values.tolist()
#    						CFSR_clim = climatology['CFSR'].values.tolist()
#    						ERAI_clim = climatology['ERA-Interim'].values.tolist()
#    						ERA5_clim = climatology['ERA5'].values.tolist()
#    						ERA5_Land_clim = climatology['ERA5-Land'].values.tolist()
#    						JRA_clim = climatology['JRA-55'].values.tolist()
#    						MERRA2_clim = climatology['MERRA2'].values.tolist()
#    						GLDAS_clim = climatology['GLDAS-Noah'].values.tolist()
#    						GLDAS_CLSM_clim = climatology['GLDAS-CLSM'].values.tolist()
#
#    						for b in range(0,12):
#    							start_date_master_Eur.append(year_begin)
#    							end_date_master_Eur.append(year_end)
#
#    						mon_master_Eur.append(mon_clim)
#    						station_clim_master_Eur.append(station_clim)
#    						CFSR_clim_master_Eur.append(CFSR_clim)
#    						ERAI_clim_master_Eur.append(ERAI_clim)
#    						ERA5_clim_master_Eur.append(ERA5_clim)
#    						ERA5_Land_clim_master_Eur.append(ERA5_Land_clim)
#    						JRA_clim_master_Eur.append(JRA_clim)
#    						MERRA2_clim_master_Eur.append(MERRA2_clim)
#    						GLDAS_clim_master_Eur.append(GLDAS_clim)
#    						GLDAS_CLSM_clim_master_Eur.append(GLDAS_CLSM_clim)
#    						grid_cell_master_Eur.append(grid_clim)
#    						lat_master_Eur.append(lat_clim)
#    						lon_master_Eur.append(lon_clim)
#    						sites_incl_master_Eur.append(sites_clim)
#
#
#    				#start_date_master_Eur = [i for sub in start_date_master_Eur for i in sub]
#    				#end_date_master_Nam = [i for sub in end_date_master_Eur for i in sub]
#    				mon_master_Eur = [i for sub in mon_master_Eur for i in sub]
#    				grid_cell_master_Eur = [i for sub in grid_cell_master_Eur for i in sub]
#    				lat_master_Eur = [i for sub in lat_master_Eur for i in sub]
#    				lon_master_Eur = [i for sub in lon_master_Eur for i in sub]
#    				sites_incl_master_Eur = [i for sub in sites_incl_master_Eur for i in sub]
#    				station_clim_master_Eur = [i for sub in station_clim_master_Eur for i in sub]
#    				CFSR_clim_master_Eur = [i for sub in CFSR_clim_master_Eur for i in sub]
#    				ERAI_clim_master_Eur = [i for sub in ERAI_clim_master_Eur for i in sub]
#    				ERA5_Land_clim_master_Eur = [i for sub in ERA5_Land_clim_master_Eur for i in sub]
#    				ERA5_clim_master_Eur = [i for sub in ERA5_clim_master_Eur for i in sub]
#    				JRA_clim_master_Eur = [i for sub in JRA_clim_master_Eur for i in sub]
#    				MERRA2_clim_master_Eur = [i for sub in MERRA2_clim_master_Eur for i in sub]
#    				GLDAS_clim_master_Eur = [i for sub in GLDAS_clim_master_Eur for i in sub]
#    				GLDAS_CLSM_clim_master_Eur = [i for sub in GLDAS_CLSM_clim_master_Eur for i in sub]
#								
#    				dframe_master = pd.DataFrame(data= grid_cell_master_Eur, columns=['Grid Cell'])
#    				dframe_master['Central Lat'] = lat_master_Eur
#    				dframe_master['Central Lon'] = lon_master_Eur
#    				dframe_master['Sites Incl'] = sites_incl_master_Eur
#    				dframe_master['Start Year'] = start_date_master_Eur
#    				dframe_master['End Year'] = end_date_master_Eur
#    				dframe_master['Month'] = mon_master_Eur
#    				dframe_master['Station'] = station_clim_master_Eur
#    				dframe_master['CFSR'] = CFSR_clim_master_Eur
#    				dframe_master['ERA-Interim'] = ERAI_clim_master_Eur
#    				dframe_master['ERA5'] = ERA5_clim_master_Eur
#    				dframe_master['ERA5-Land'] = ERA5_Land_clim_master_Eur
#    				dframe_master['JRA-55'] = JRA_clim_master_Eur
#    				dframe_master['MERRA2'] = MERRA2_clim_master_Eur
#    				dframe_master['GLDAS-Noah'] = GLDAS_clim_master_Eur
#    				dframe_master['GLDAS-CLSM'] = GLDAS_CLSM_clim_master_Eur
#
#    				seasonal_cycle_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/seasonal_cycle/'+str(remap_type)+"_"+str(naive_type_j)+'_'+str(olr_k)+'_'+str(lyr_l)+'_'+str(thr_m)+'_seasonal_cycle_Eur.csv'])
#    				dframe_master.to_csv(seasonal_cycle_fil,na_rep=np.nan,index=False)
#
#
#
#
#
#
##### North America ##
#
#    				start_date_master_NAm = []
#    				end_date_master_NAm = []
#    				mon_master_NAm = []
#    				grid_cell_master_NAm = []
#    				lat_master_NAm = []
#    				lon_master_NAm = []
#    				sites_incl_master_NAm = []
#    				station_clim_master_NAm = []
#    				CFSR_clim_master_NAm = []
#    				ERAI_clim_master_NAm = []
#    				ERA5_clim_master_NAm = []
#    				ERA5_Land_clim_master_NAm = []
#    				JRA_clim_master_NAm = []
#    				MERRA2_clim_master_NAm = []
#    				GLDAS_clim_master_NAm = []
#    				GLDAS_CLSM_clim_master_NAm = []				
#    				jan_val_master_NAm = []
#    				feb_val_master_NAm = []
#    				mar_val_master_NAm = []
#    				apr_val_master_NAm = []
#    				may_val_master_NAm = []
#    				jun_val_master_NAm = []
#    				jul_val_master_NAm = []
#    				aug_val_master_NAm = []
#    				sep_val_master_NAm = []
#    				oct_val_master_NAm = []
#    				nov_val_master_NAm = []
#    				dec_val_master_NAm = []
#
#    				for n in gcell_NAm_uq:
#    					gcell_NAm_n = n
#    					#print('Grid Cell:',gcell_NAm_n)
## grab corresponding timeseries data #
#
#    					dframe_gcell = dframe_timeseries[dframe_timeseries['Grid Cell'] == gcell_NAm_n]
#    					dframe_gcell_nonan = dframe_gcell.dropna(subset = ['Station'])
#    					#print(dframe_gcell_nonan)
#    					dtime = dframe_gcell_nonan['Date'].values
#    					len_dtime = len(dtime)
#    					dt_time = [datetime.datetime.strptime(i,'%Y-%m-%d') for i in dtime]
#    					stn_temp = dframe_gcell_nonan['Station'].values
#    					#print(dt_time)
## grab datetime information #
#
#    					st_dt = dt_time[0]
#    					ed_dt = dt_time[len_dtime - 1]
#
#    					beg_mon = st_dt.month
#    					beg_yr = st_dt.year
#    					fin_mon = ed_dt.month
#    					fin_yr = ed_dt.year
#
#
#
#    					#print(st_dt)
#    					#print(ed_dt)
#    					#print(beg_yr)
#    					#print(fin_yr)
#    					#print(beg_mon)
#    					#print(fin_mon)
#
## create dictionary to keep track of actual number of dates present in timeseries #
## create base dictionary which will count the number of months#
#
#    					monthly_entries_NAm = dict({'Jan':0,'Feb':0,'Mar':0,'Apr':0,'May':0,'Jun':0,'Jul':0,'Aug':0,'Sep':0,'Oct':0,'Nov':0,'Dec':0})
#
##Now you have a dictionary which keeps track of how many data entries you have per month. 
##Then you can iterate over all of your data rows, and just increment the monthly_entries[month_key] counter by one for each entry.
##In order to generate the month_key from a datetime object, you'd just run:
#
#    					dt_idx = 0
#    					dat_new = []
#
#    					for dt in dt_time: 
#    						dt2 = dt_time[dt_idx]
#    						#print(dt2.month)
#    						dt2_mon = dt2.month
#    						#print(dt2_mon) 
#    						dat_new.append(dt2_mon)
#    						dt_idx += 1
#
#
#    					for dt in dat_new:
#    						dt_i = dt
#    						if (dt_i == 1):
#    							key = 'Jan'
#    						elif (dt_i == 2):
#    							key = 'Feb'
#    						elif (dt_i == 3):
#    							key = 'Mar'
#    						elif (dt_i == 4):
#    							key = 'Apr'
#    						elif (dt_i == 5):
#    							key = 'May'
#    						elif (dt_i == 6):
#    							key = 'Jun'
#    						elif (dt_i == 7):
#    							key = 'Jul'
#    						elif (dt_i == 8):
#    							key = 'Aug'
#    						elif (dt_i == 9):
#    							key = 'Sep'
#    						elif (dt_i == 10):
#    							key = 'Oct'
#    						elif (dt_i == 11):
#    							key = 'Nov'
#    						elif (dt_i == 12):
#    							key = 'Dec'
#    						monthly_entries_NAm[key] += 1
#
#
#    					jan_val = int(monthly_entries_NAm['Jan'])
#    					feb_val = int(monthly_entries_NAm['Feb'])
#    					mar_val = int(monthly_entries_NAm['Mar'])
#    					apr_val = int(monthly_entries_NAm['Apr'])
#    					may_val = int(monthly_entries_NAm['May'])
#    					jun_val = int(monthly_entries_NAm['Jun'])
#    					jul_val = int(monthly_entries_NAm['Jul'])
#    					aug_val = int(monthly_entries_NAm['Aug'])
#    					sep_val = int(monthly_entries_NAm['Sep'])
#    					oct_val = int(monthly_entries_NAm['Oct'])
#    					nov_val = int(monthly_entries_NAm['Nov'])
#    					dec_val = int(monthly_entries_NAm['Dec'])
#
#    					#print('Grid Cell:', gcell_NAm_n)
#    					#print('first year:', beg_yr, ' first month:',beg_mon )
#    					#print('last year:', fin_yr, ' final month:',fin_mon)
#
#    					year_diff = fin_yr - beg_yr
#
## if there are at least 4 years worth of data, calculate climatology #
#    					if (year_diff >= 4):
#
#    						#print(gcell_NAm_n)
#    						#print(dframe_gcell_nonan)
#    						dframe_gcell_nonan2 = dframe_gcell_nonan.set_index('Date')
#    						year_begin = beg_yr + 1
#    						year_end = fin_yr - 1
#    						start_date = ''.join([str(year_begin)+'-01-01'])
#    						end_date = ''.join([str(year_end)+'12-31'])
#    						dframe_climatology = dframe_gcell_nonan2[(dframe_gcell_nonan2.index >= start_date) & (dframe_gcell_nonan2.index <= end_date)]
#    						dtime_new = dframe_climatology.index.values
#    						#print(dframe_climatology)
#    						dt_time_new = [datetime.datetime.strptime(i,'%Y-%m-%d') for i in dtime_new]
#    						mon_new = [i.month for i in dt_time_new]
#    						dframe_climatology['DateTime'] = dt_time_new
#    						dframe_climatology['mon'] = mon_new
#    						#print(dframe_climatology)
#    						climatology = dframe_climatology.groupby('mon').mean()
#    						#print(climatology) 
#
#    						
#    						grid_clim = climatology['Grid Cell'].values.tolist()
#    						lat_clim = climatology['Central Lat'].values.tolist()
#    						lon_clim = climatology['Central Lon'].values.tolist()
#    						mon_clim = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
#    						sites_clim = climatology['Sites Incl'].values.tolist()
#    						station_clim = climatology['Station'].values.tolist()
#    						CFSR_clim = climatology['CFSR'].values.tolist()
#    						ERAI_clim = climatology['ERA-Interim'].values.tolist()
#    						ERA5_clim = climatology['ERA5'].values.tolist()
#    						ERA5_Land_clim = climatology['ERA5-Land'].values.tolist()
#    						JRA_clim = climatology['JRA-55'].values.tolist()
#    						MERRA2_clim = climatology['MERRA2'].values.tolist()
#    						GLDAS_clim = climatology['GLDAS-Noah'].values.tolist()
#    						GLDAS_CLSM_clim = climatology['GLDAS-CLSM'].values.tolist()
#
#    						for b in range(0,12):
#    							start_date_master_NAm.append(year_begin)
#    							end_date_master_NAm.append(year_end)
#
#    						mon_master_NAm.append(mon_clim)
#    						station_clim_master_NAm.append(station_clim)
#    						CFSR_clim_master_NAm.append(CFSR_clim)
#    						ERAI_clim_master_NAm.append(ERAI_clim)
#    						ERA5_clim_master_NAm.append(ERA5_clim)
#    						ERA5_Land_clim_master_NAm.append(ERA5_Land_clim)
#    						JRA_clim_master_NAm.append(JRA_clim)
#    						MERRA2_clim_master_NAm.append(MERRA2_clim)
#    						GLDAS_clim_master_NAm.append(GLDAS_clim)
#    						GLDAS_CLSM_clim_master_NAm.append(GLDAS_CLSM_clim)
#    						grid_cell_master_NAm.append(grid_clim)
#    						lat_master_NAm.append(lat_clim)
#    						lon_master_NAm.append(lon_clim)
#    						sites_incl_master_NAm.append(sites_clim)
#
#
#    				#start_date_master_NAm = [i for sub in start_date_master_NAm for i in sub]
#    				#end_date_master_Nam = [i for sub in end_date_master_NAm for i in sub]
#    				mon_master_NAm = [i for sub in mon_master_NAm for i in sub]
#    				grid_cell_master_NAm = [i for sub in grid_cell_master_NAm for i in sub]
#    				lat_master_NAm = [i for sub in lat_master_NAm for i in sub]
#    				lon_master_NAm = [i for sub in lon_master_NAm for i in sub]
#    				sites_incl_master_NAm = [i for sub in sites_incl_master_NAm for i in sub]
#    				station_clim_master_NAm = [i for sub in station_clim_master_NAm for i in sub]
#    				CFSR_clim_master_NAm = [i for sub in CFSR_clim_master_NAm for i in sub]
#    				ERAI_clim_master_NAm = [i for sub in ERAI_clim_master_NAm for i in sub]
#    				ERA5_Land_clim_master_NAm = [i for sub in ERA5_Land_clim_master_NAm for i in sub]
#    				ERA5_clim_master_NAm = [i for sub in ERA5_clim_master_NAm for i in sub]
#    				JRA_clim_master_NAm = [i for sub in JRA_clim_master_NAm for i in sub]
#    				MERRA2_clim_master_NAm = [i for sub in MERRA2_clim_master_NAm for i in sub]
#    				GLDAS_clim_master_NAm = [i for sub in GLDAS_clim_master_NAm for i in sub]
#    				GLDAS_CLSM_clim_master_NAm = [i for sub in GLDAS_CLSM_clim_master_NAm for i in sub]
#								
#    				dframe_master = pd.DataFrame(data= grid_cell_master_NAm, columns=['Grid Cell'])
#    				dframe_master['Central Lat'] = lat_master_NAm
#    				dframe_master['Central Lon'] = lon_master_NAm
#    				dframe_master['Sites Incl'] = sites_incl_master_NAm
#    				dframe_master['Start Year'] = start_date_master_NAm
#    				dframe_master['End Year'] = end_date_master_NAm
#    				dframe_master['Month'] = mon_master_NAm
#    				dframe_master['Station'] = station_clim_master_NAm
#    				dframe_master['CFSR'] = CFSR_clim_master_NAm
#    				dframe_master['ERA-Interim'] = ERAI_clim_master_NAm
#    				dframe_master['ERA5'] = ERA5_clim_master_NAm
#    				dframe_master['ERA5-Land'] = ERA5_Land_clim_master_NAm
#    				dframe_master['JRA-55'] = JRA_clim_master_NAm
#    				dframe_master['MERRA2'] = MERRA2_clim_master_NAm
#    				dframe_master['GLDAS-Noah'] = GLDAS_clim_master_NAm
#    				dframe_master['GLDAS-CLSM'] = GLDAS_CLSM_clim_master_NAm
#
#    				seasonal_cycle_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/seasonal_cycle/'+str(remap_type)+"_"+str(naive_type_j)+'_'+str(olr_k)+'_'+str(lyr_l)+'_'+str(thr_m)+'_seasonal_cycle_NAm.csv'])
#    				dframe_master.to_csv(seasonal_cycle_fil,na_rep=np.nan,index=False)




### Grid Cells with at least 30 yrs of data:
#top-30cm
#Eurasia: [31109, 31260, 31261, 31406, 31412, 31414, 31415, 31554, 31555, 31566, 31850, 31855, 31857, 31859, 32000, 32147, 32151, 32156, 32295, 32311, 32443, 32457, 32589, 32594, 32597, 32604, 32735, 32740, 32750, 32758, 32887, 32891, 32893, 32905, 33034, 33037, 33040, 33182, 33191, 33198, 33200, 33332, 33333, 33353, 33488, 33489, 33496, 33637, 33649, 33777, 33783, 33924, 33929, 33934, 33948, 34082, 34085, 34235, 34392, 34544, 34696, 34978, 35129, 35133, 35138, 35276, 35289, 35290, 35429, 35430, 35576, 35581, 35723, 35726, 35732, 35877, 35882, 35884, 35886, 36323, 36325, 36471, 36473, 36625, 36627, 36769, 36921, 37068, 37073, 37663, 37814, 38110, 38408, 38409, 38412, 38413, 38418, 38555, 38560, 38712, 38852, 38854, 39000, 39002, 39006, 39009, 39013, 39153, 39156, 39300, 39449, 39600, 39601, 39602, 39747, 39761, 39895, 39909, 40042, 40203, 40344, 40345, 40493, 40497, 40644, 40648, 40942, 41242, 41389, 41537, 41689, 41835, 41985, 41989, 42130, 42132, 42279, 42431, 42575, 42577, 42578, 42581, 42875, 42881, 42884, 43022, 43175, 43177, 43469, 43472, 43487, 43620, 43623, 43627, 43775, 43917, 43927, 44214, 44219, 44365, 44370, 44375, 44662, 44665, 44671, 44817, 44973, 45125, 45261, 45276, 45417, 45558, 45704, 46007, 46150, 46153, 46163, 46616, 46744, 46746, 46757, 47196, 47643, 48101, 48104, 48846, 49290, 49296, 51235, 52126]
#N America: [none]

#30cm - 300cm
#Eurasia: [31109, 31260, 31261, 31406, 31412, 31414, 31415, 31554, 31555, 31566, 31850, 31855, 31857, 31859, 32000, 32147, 32151, 32156, 32295, 32311, 32439, 32443, 32457, 32589, 32594, 32596, 32597, 32604, 32735, 32740, 32750, 32758, 32887, 32891, 32893, 32898, 32905, 33034, 33037, 33040, 33044, 33182, 33191, 33198, 33200, 33332, 33333, 33346, 33353, 33488, 33489, 33496, 33637, 33640, 33649, 33777, 33783, 33924, 33929, 33934, 33940, 33948, 34082, 34085, 34235, 34239, 34392, 34544, 34696, 34833, 34978, 35129, 35133, 35138, 35276, 35289, 35290, 35429, 35430, 35576, 35581, 35723, 35726, 35732, 35877, 35882, 35884, 35886, 36323, 36325, 36471, 36473, 36625, 36627, 36769, 36921, 37068, 37073, 37663, 37814, 38110, 38408, 38409, 38412, 38413, 38418, 38555, 38560, 38712, 38852, 38854, 39000, 39002, 39006, 39009, 39013, 39153, 39156, 39300, 39449, 39595, 39600, 39601, 39602, 39747, 39761, 39895, 39909, 40042, 40203, 40344, 40345, 40493, 40497, 40644, 40648, 40942, 41242, 41389, 41537, 41689, 41835, 41985, 41989, 42130, 42132, 42279, 42431, 42575, 42577, 42578, 42581, 42875, 42881, 42884, 43022, 43175, 43177, 43469, 43472, 43487, 43620, 43623, 43627, 43775, 43917, 43927, 44214, 44219, 44365, 44370, 44375, 44662, 44665, 44671, 44817, 44973, 45125, 45261, 45276, 45417, 45558, 45704, 46007, 46150, 46153, 46163, 46442, 46443, 46616, 46743, 46744, 46746, 46757, 47196, 47643, 47938, 47940, 48101, 48104, 48846, 49290, 49296, 49744, 50776, 51235, 52126]
#N America: [none] 





### Grid cells with at least 20 years of data ###

#top-30cm
#Eurasia: [30961, 31109, 31260, 31261, 31406, 31412, 31414, 31415, 31554, 31555, 31566, 31850, 31855, 31857, 31859, 32000, 32147, 32151, 32156, 32295, 32311, 32443, 32457, 32589, 32594, 32596, 32597, 32604, 32735, 32740, 32750, 32758, 32887, 32891, 32893, 32898, 32905, 33034, 33037, 33040, 33044, 33182, 33191, 33198, 33200, 33332, 33333, 33337, 33346, 33353, 33488, 33489, 33496, 33637, 33640, 33646, 33649, 33777, 33782, 33783, 33924, 33929, 33934, 33940, 33948, 34082, 34085, 34235, 34239, 34392, 34544, 34696, 34833, 34978, 35129, 35133, 35138, 35276, 35289, 35290, 35429, 35430, 35576, 35581, 35723, 35726, 35732, 35877, 35882, 35884, 35886, 36323, 36325, 36471, 36473, 36625, 36627, 36769, 36921, 37068, 37073, 37663, 37814, 38110, 38408, 38409, 38412, 38413, 38418, 38555, 38560, 38712, 38852, 38854, 39000, 39002, 39006, 39009, 39013, 39153, 39156, 39300, 39449, 39453, 39595, 39600, 39601, 39602, 39747, 39761, 39895, 39909, 40042, 40203, 40344, 40345, 40493, 40497, 40644, 40648, 40942, 41242, 41389, 41537, 41689, 41835, 41985, 41989, 42130, 42132, 42279, 42431, 42575, 42577, 42578, 42581, 42875, 42879, 42881, 42884, 43022, 43175, 43177, 43469, 43472, 43487, 43620, 43623, 43627, 43775, 43917, 43927, 44214, 44219, 44365, 44370, 44375, 44662, 44665, 44671, 44817, 44973, 45125, 45261, 45276, 45417, 45558, 45704, 46004, 46007, 46150, 46153, 46163, 46298, 46442, 46443, 46450, 46616, 46743, 46744, 46746, 46757, 47044, 47196, 47643, 47938, 47940, 48101, 48104, 48846, 49290, 49296, 49441, 49744, 50181, 50776, 50785, 51235, 52126]
#N America: [none]

#30cm - 300cm
#Eurasia: [30961, 31109, 31260, 31261, 31406, 31412, 31414, 31415, 31554, 31555, 31566, 31850, 31855, 31857, 31859, 32000, 32147, 32151, 32156, 32295, 32311, 32439, 32443, 32457, 32589, 32594, 32596, 32597, 32604, 32735, 32740, 32750, 32758, 32887, 32891, 32893, 32898, 32905, 33034, 33037, 33040, 33044, 33182, 33191, 33198, 33200, 33332, 33333, 33337, 33346, 33353, 33488, 33489, 33496, 33637, 33640, 33646, 33649, 33777, 33782, 33783, 33924, 33929, 33934, 33940, 33948, 34082, 34085, 34235, 34239, 34392, 34544, 34696, 34833, 34978, 35129, 35133, 35138, 35276, 35289, 35290, 35429, 35430, 35576, 35581, 35723, 35726, 35732, 35877, 35882, 35884, 35886, 36323, 36325, 36471, 36473, 36625, 36627, 36769, 36921, 37068, 37073, 37663, 37814, 38110, 38408, 38409, 38412, 38413, 38418, 38555, 38560, 38712, 38852, 38854, 39000, 39002, 39006, 39009, 39013, 39153, 39156, 39300, 39449, 39453, 39595, 39600, 39601, 39602, 39747, 39761, 39895, 39909, 40042, 40203, 40344, 40345, 40493, 40497, 40644, 40648, 40942, 41242, 41389, 41537, 41689, 41835, 41985, 41989, 42130, 42132, 42140, 42279, 42291, 42431, 42575, 42577, 42578, 42581, 42875, 42879, 42881, 42884, 43022, 43175, 43177, 43469, 43472, 43487, 43620, 43623, 43627, 43775, 43917, 43927, 44214, 44219, 44365, 44370, 44375, 44382, 44662, 44665, 44671, 44817, 44973, 45125, 45261, 45276, 45412, 45417, 45558, 45704, 45725, 46004, 46007, 46150, 46153, 46163, 46298, 46442, 46443, 46450, 46616, 46743, 46744, 46746, 46757, 47044, 47196, 47643, 47938, 47940, 48101, 48104, 48846, 49290, 49296, 49441, 49744, 50181, 50776, 50785, 51235, 52126]
#N AmericaL [None]
