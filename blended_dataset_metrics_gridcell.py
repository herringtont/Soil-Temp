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
import pathlib
from calendar import isleap
from dateutil.relativedelta import *
from pathlib import Path
import seaborn as sn
from calendar import isleap
from dateutil.relativedelta import *
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr


########## Define Functions ##########

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


########## Set directories ##########

olr = ['outliers','zscore','IQR']
lyr = ['0_9.9']
thr = ['0','25','50','75','100']
rmp_type = ['nn','bil']



TC_raw_dir = "/mnt/data/users/herringtont/soil_temp/Blended_Product/collocated/TC_blended/raw/"
TC_anom_dir = "/mnt/data/users/herringtont/soil_temp/Blended_Product/collocated/TC_blended/anom/"

################# loop through files ###############
for h in rmp_type: #loops through remap type
    rmph = h
    if(rmph == "nn"):
    	remap_type = "remapnn"
    elif(rmph == "bil"):
    	remap_type = "remapbil"    	 
    for i in olr: #loops throuh outlier type
    	olri = i
    	for j in lyr: #loops through layer
    		lyrj = j
    		for k in thr: #loops through missing threshold
    			thrk = k

    			if (remap_type == 'remapbil'):
    				stn_gcell_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/geometry_L1_bil.csv'])
    			elif (remap_type == 'remapnn'):
    				stn_gcell_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/geometry_L1_nn.csv']) 
    			stn_gcell_dframe = pd.read_csv(stn_gcell_fil)

    			stn_gcells = stn_gcell_dframe['Grid Cell'].values
    			gcell_uq_stn = np.unique(stn_gcells)

    			TC_raw_fil = ''.join([TC_raw_dir+str(remap_type)+'_'+str(olri)+'_'+str(lyrj)+'_thr'+str(thrk)+'_TC_blended.csv'])
    			TC_anom_fil = ''.join([TC_anom_dir+str(remap_type)+'_'+str(olri)+'_'+str(lyrj)+'_thr'+str(thrk)+'_TC_blended_anom.csv'])
    			
    			station_dir = ''.join(["/mnt/data/users/herringtont/soil_temp/In-Situ/All/spatial_average_anom/"+str(remap_type)+'/no_outliers/'+str(olri)+'/'+str(lyrj)+'/thr_'+str(thrk)+'/'])
################# grab temperatures and anomalies for the different products ##############

##### Date/spatial info #####

    			TC_raw_dframe = pd.read_csv(TC_raw_fil)
    			gcell = TC_raw_dframe['Grid Cell'].values

##### Create Master Arrays #####
    			gcell_master_stn = []
    			gcell_master = []
    			lat_master_stn = []
    			lon_master_stn = []
    			lat_master = []
    			lon_master = []
    			date_master = []

    			naive_bias_master_raw = []
    			TC_bias_master_raw = []
    			CFSR_bias_master_raw = []
    			ERAI_bias_master_raw = []
    			ERA5_bias_master_raw = []
    			JRA_bias_master_raw = []
    			MERRA2_bias_master_raw = []
    			GLDAS_bias_master_raw = []

    			naive_SDV_master_raw = []
    			TC_SDV_master_raw = []
    			CFSR_SDV_master_raw = []
    			ERAI_SDV_master_raw = []
    			ERA5_SDV_master_raw = []
    			JRA_SDV_master_raw = []
    			MERRA2_SDV_master_raw = []
    			GLDAS_SDV_master_raw = []

    			naive_rmse_master_raw = []
    			TC_rmse_master_raw = []
    			CFSR_rmse_master_raw = []
    			ERAI_rmse_master_raw = []
    			ERA5_rmse_master_raw = []
    			JRA_rmse_master_raw = []
    			MERRA2_rmse_master_raw = []
    			GLDAS_rmse_master_raw = []

    			naive_ubrmse_master_raw = []
    			TC_ubrmse_master_raw = []
    			CFSR_ubrmse_master_raw = []
    			ERAI_ubrmse_master_raw = []
    			ERA5_ubrmse_master_raw = []
    			JRA_ubrmse_master_raw = []
    			MERRA2_ubrmse_master_raw = []
    			GLDAS_ubrmse_master_raw = []

    			naive_corr_master_raw = []
    			TC_corr_master_raw = []
    			CFSR_corr_master_raw = []
    			ERAI_corr_master_raw = []
    			ERA5_corr_master_raw = []
    			JRA_corr_master_raw = []
    			MERRA2_corr_master_raw = []
    			GLDAS_corr_master_raw = []
    			delta_corr_master_raw = []

    			naive_bias_master_anom = []
    			TC_bias_master_anom = []
    			CFSR_bias_master_anom = []
    			ERAI_bias_master_anom = []
    			ERA5_bias_master_anom = []
    			JRA_bias_master_anom = []
    			MERRA2_bias_master_anom = []
    			GLDAS_bias_master_anom = []
    			delta_corr_master_raw = []

    			naive_SDV_master_anom = []
    			TC_SDV_master_anom = []
    			CFSR_SDV_master_anom = []
    			ERAI_SDV_master_anom = []
    			ERA5_SDV_master_anom = []
    			JRA_SDV_master_anom = []
    			MERRA2_SDV_master_anom = []
    			GLDAS_SDV_master_anom = []

    			naive_rmse_master_anom = []
    			TC_rmse_master_anom = []
    			CFSR_rmse_master_anom = []
    			ERAI_rmse_master_anom = []
    			ERA5_rmse_master_anom = []
    			JRA_rmse_master_anom = []
    			MERRA2_rmse_master_anom = []
    			GLDAS_rmse_master_anom = []

    			naive_ubrmse_master_anom = []
    			TC_ubrmse_master_anom = []
    			CFSR_ubrmse_master_anom = []
    			ERAI_ubrmse_master_anom = []
    			ERA5_ubrmse_master_anom = []
    			JRA_ubrmse_master_anom = []
    			MERRA2_ubrmse_master_anom = []
    			GLDAS_ubrmse_master_anom = []

    			naive_corr_master_anom = []
    			TC_corr_master_anom = []
    			CFSR_corr_master_anom = []
    			ERAI_corr_master_anom = []
    			ERA5_corr_master_anom = []
    			JRA_corr_master_anom = []
    			MERRA2_corr_master_anom = []
    			GLDAS_corr_master_anom = []
    			delta_corr_master_anom = []

    			print(remap_type,olri,thrk)



###################### REANALYSIS DATA ONLY ########################
#####Climatologies are between 1981-2010####
#
#    			gcell_uq = np.unique(gcell)
#			
#    			TC_anom_dframe = pd.read_csv(TC_anom_fil)
#
#    			TC_raw_dframe = pd.read_csv(TC_raw_fil)
#
#
#    			for l in gcell_uq:
#
#    				gcell_l = l
#    				gcell_master.append(gcell_l)
#    				TC_raw_dframe_gcell = TC_raw_dframe[TC_raw_dframe['Grid Cell'] == gcell_l]
#    				TC_anom_dframe_gcell = TC_anom_dframe[TC_anom_dframe['Grid Cell'] == gcell_l]
#    				lat_cen = TC_raw_dframe_gcell['Central Lat'].iloc[0]
#    				lat_master.append(lat_cen)
#    				lon_cen = TC_raw_dframe_gcell['Central Lon'].iloc[0]
#    				lon_master.append(lon_cen)
#
#    				TC_raw_temp = TC_raw_dframe_gcell['TC Blended']
#    				naive_raw_temp = TC_raw_dframe_gcell['Naive Blended']
#    				CFSR_raw_temp = TC_raw_dframe_gcell['CFSR']
#    				ERAI_raw_temp = TC_raw_dframe_gcell['ERA-Interim']
#    				ERA5_raw_temp = TC_raw_dframe_gcell['ERA5']
#    				JRA_raw_temp = TC_raw_dframe_gcell['JRA-55']
#    				MERRA2_raw_temp = TC_raw_dframe_gcell['MERRA2']
#    				GLDAS_raw_temp = TC_raw_dframe_gcell['GLDAS']
#
#
#    				TC_anom = TC_anom_dframe_gcell['TC Blended']
#    				naive_anom = TC_anom_dframe_gcell['Naive Blended']
#    				CFSR_anom = TC_anom_dframe_gcell['CFSR']
#    				ERAI_anom = TC_anom_dframe_gcell['ERA-Interim']
#    				ERA5_anom = TC_anom_dframe_gcell['ERA5']
#    				JRA_anom = TC_anom_dframe_gcell['JRA-55']
#    				MERRA2_anom = TC_anom_dframe_gcell['MERRA2']
#    				GLDAS_anom = TC_anom_dframe_gcell['GLDAS']
#
#
#
#
################ Performance relative to ERA5 ###################
#
#
################ Calculate Biases ############
#    				
####### Raw Temp #####
#
#    				naive_bias_raw = bias(naive_raw_temp, ERA5_raw_temp)
#    				naive_bias_master_raw.append(naive_bias_raw)
#
#    				TC_bias_raw = bias(TC_raw_temp, ERA5_raw_temp)
#    				TC_bias_master_raw.append(TC_bias_raw)
#
#    				CFSR_bias_raw = bias(CFSR_raw_temp, ERA5_raw_temp)
#    				CFSR_bias_master_raw.append(CFSR_bias_raw)
#
#    				ERAI_bias_raw = bias(ERAI_raw_temp, ERA5_raw_temp)
#    				ERAI_bias_master_raw.append(ERAI_bias_raw)
#
#    				ERA5_bias_raw = bias(ERA5_raw_temp, ERA5_raw_temp)
#    				ERA5_bias_master_raw.append(ERA5_bias_raw)
#
#    				JRA_bias_raw = bias(JRA_raw_temp, ERA5_raw_temp)
#    				JRA_bias_master_raw.append(JRA_bias_raw)
#
#    				MERRA2_bias_raw = bias(MERRA2_raw_temp, ERA5_raw_temp)
#    				MERRA2_bias_master_raw.append(MERRA2_bias_raw)
#
#    				GLDAS_bias_raw = bias(GLDAS_raw_temp, ERA5_raw_temp)
#    				GLDAS_bias_master_raw.append(GLDAS_bias_raw)
#
####### Anomalies #####
#
#    				naive_bias_anom = bias(naive_anom, ERA5_anom)
#    				naive_bias_master_anom.append(naive_bias_anom)
#
#    				TC_bias_anom = bias(TC_anom, ERA5_anom)
#    				TC_bias_master_anom.append(TC_bias_anom)
#
#    				CFSR_bias_anom = bias(CFSR_anom, ERA5_anom)
#    				CFSR_bias_master_anom.append(CFSR_bias_anom)
#
#    				ERAI_bias_anom = bias(ERAI_anom, ERA5_anom)
#    				ERAI_bias_master_anom.append(ERAI_bias_anom)
#
#    				ERA5_bias_anom = bias(ERA5_anom, ERA5_anom)
#    				ERA5_bias_master_anom.append(ERA5_bias_anom)
#
#    				JRA_bias_anom = bias(JRA_anom, ERA5_anom)
#    				JRA_bias_master_anom.append(JRA_bias_anom)
#
#    				MERRA2_bias_anom = bias(MERRA2_anom, ERA5_anom)
#    				MERRA2_bias_master_anom.append(MERRA2_bias_anom)
#
#    				GLDAS_bias_anom = bias(GLDAS_anom, ERA5_anom)
#    				GLDAS_bias_master_anom.append(GLDAS_bias_anom)
#
################ Calculate normalized standard deviations (relative to in-situ) ############
#
####### Raw Temp #####
#
#    				naive_SDV_raw = SDVnorm(naive_raw_temp, ERA5_raw_temp)
#    				naive_SDV_master_raw.append(naive_SDV_raw)
#
#    				TC_SDV_raw = SDVnorm(TC_raw_temp, ERA5_raw_temp)
#    				TC_SDV_master_raw.append(TC_SDV_raw)
#
#    				CFSR_SDV_raw = SDVnorm(CFSR_raw_temp, ERA5_raw_temp)
#    				CFSR_SDV_master_raw.append(CFSR_SDV_raw)
#
#    				ERAI_SDV_raw = SDVnorm(ERAI_raw_temp, ERA5_raw_temp)
#    				ERAI_SDV_master_raw.append(ERAI_SDV_raw)
#
#    				ERA5_SDV_raw = SDVnorm(ERA5_raw_temp, ERA5_raw_temp)
#    				ERA5_SDV_master_raw.append(ERA5_SDV_raw)
#
#    				JRA_SDV_raw = SDVnorm(JRA_raw_temp, ERA5_raw_temp)
#    				JRA_SDV_master_raw.append(JRA_SDV_raw)
#
#    				MERRA2_SDV_raw = SDVnorm(MERRA2_raw_temp, ERA5_raw_temp)
#    				MERRA2_SDV_master_raw.append(MERRA2_SDV_raw)
#
#    				GLDAS_SDV_raw = SDVnorm(GLDAS_raw_temp, ERA5_raw_temp)
#    				GLDAS_SDV_master_raw.append(GLDAS_SDV_raw)
#
####### Anomalies #####
#
#    				naive_SDV_anom = SDVnorm(naive_anom, ERA5_anom)
#    				naive_SDV_master_anom.append(naive_SDV_anom)
#
#    				TC_SDV_anom = SDVnorm(TC_anom, ERA5_anom)
#    				TC_SDV_master_anom.append(TC_SDV_anom)
#
#    				CFSR_SDV_anom = SDVnorm(CFSR_anom, ERA5_anom)
#    				CFSR_SDV_master_anom.append(CFSR_SDV_anom)
#
#    				ERAI_SDV_anom = SDVnorm(ERAI_anom, ERA5_anom)
#    				ERAI_SDV_master_anom.append(ERAI_SDV_anom)
#
#    				ERA5_SDV_anom = SDVnorm(ERA5_anom, ERA5_anom)
#    				ERA5_SDV_master_anom.append(ERA5_SDV_anom)
#
#    				JRA_SDV_anom = SDVnorm(JRA_anom, ERA5_anom)
#    				JRA_SDV_master_anom.append(JRA_SDV_anom)
#
#    				MERRA2_SDV_anom = SDVnorm(MERRA2_anom, ERA5_anom)
#    				MERRA2_SDV_master_anom.append(MERRA2_SDV_anom)
#
#    				GLDAS_SDV_anom = SDVnorm(GLDAS_anom, ERA5_anom)
#    				GLDAS_SDV_master_anom.append(GLDAS_SDV_anom)
#
#												
############### Calculate RMSE and ubRMSE for products ##############
#
####### Raw Temp #####
#
#    				y_naive_raw = naive_raw_temp
#    				y_TC_raw = TC_raw_temp
#    				y_CFSR_raw = CFSR_raw_temp
#    				y_ERAI_raw = ERAI_raw_temp
#    				y_ERA5_raw = ERA5_raw_temp
#    				y_JRA_raw = JRA_raw_temp
#    				y_MERRA2_raw = MERRA2_raw_temp
#    				y_GLDAS_raw = GLDAS_raw_temp   			
#
#    				naive_rmse_raw = mean_squared_error(y_ERA5_raw, y_naive_raw, squared=False)
#    				naive_rmse_master_raw.append(naive_rmse_raw)
#
#    				TC_rmse_raw = mean_squared_error(y_ERA5_raw, y_TC_raw, squared=False)
#    				TC_rmse_master_raw.append(TC_rmse_raw)
#
#    				CFSR_rmse_raw = mean_squared_error(y_ERA5_raw, y_CFSR_raw, squared=False)
#    				CFSR_rmse_master_raw.append(CFSR_rmse_raw)
#
#    				ERAI_rmse_raw = mean_squared_error(y_ERA5_raw, y_ERAI_raw, squared=False)
#    				ERAI_rmse_master_raw.append(ERAI_rmse_raw)
#
#    				ERA5_rmse_raw = mean_squared_error(y_ERA5_raw, y_ERA5_raw, squared=False)
#    				ERA5_rmse_master_raw.append(ERA5_rmse_raw)
#
#    				JRA_rmse_raw = mean_squared_error(y_ERA5_raw, y_JRA_raw, squared=False)
#    				JRA_rmse_master_raw.append(JRA_rmse_raw)
#
#    				MERRA2_rmse_raw = mean_squared_error(y_ERA5_raw, y_MERRA2_raw, squared=False)
#    				MERRA2_rmse_master_raw.append(MERRA2_rmse_raw)
#
#    				GLDAS_rmse_raw = mean_squared_error(y_ERA5_raw, y_GLDAS_raw, squared=False)    			
#    				GLDAS_rmse_master_raw.append(GLDAS_rmse_raw)
#
#
#    				naive_ubrmse_raw = ubrmsd(y_ERA5_raw, y_naive_raw)
#    				naive_ubrmse_master_raw.append(naive_ubrmse_raw)
#
#    				TC_ubrmse_raw = ubrmsd(y_ERA5_raw, y_TC_raw)
#    				TC_ubrmse_master_raw.append(TC_ubrmse_raw)
#    			
#    				CFSR_ubrmse_raw = ubrmsd(y_ERA5_raw, y_CFSR_raw)
#    				CFSR_ubrmse_master_raw.append(CFSR_ubrmse_raw)
#
#    				ERAI_ubrmse_raw = ubrmsd(y_ERA5_raw, y_ERAI_raw)
#    				ERAI_ubrmse_master_raw.append(ERAI_ubrmse_raw)
#
#    				ERA5_ubrmse_raw = ubrmsd(y_ERA5_raw, y_ERA5_raw)
#    				ERA5_ubrmse_master_raw.append(ERA5_ubrmse_raw)
#
#    				JRA_ubrmse_raw = ubrmsd(y_ERA5_raw, y_JRA_raw)
#    				JRA_ubrmse_master_raw.append(JRA_ubrmse_raw)
#
#    				MERRA2_ubrmse_raw = ubrmsd(y_ERA5_raw, y_MERRA2_raw)
#    				MERRA2_ubrmse_master_raw.append(MERRA2_ubrmse_raw)
#
#    				GLDAS_ubrmse_raw = ubrmsd(y_ERA5_raw, y_GLDAS_raw)
#    				GLDAS_ubrmse_master_raw.append(GLDAS_ubrmse_raw) 
#
#
####### Anomalies #####
#
#
#    				y_naive_anom = naive_anom
#    				y_TC_anom = TC_anom
#    				y_CFSR_anom = CFSR_anom
#    				y_ERAI_anom = ERAI_anom
#    				y_ERA5_anom = ERA5_anom
#    				y_JRA_anom = JRA_anom
#    				y_MERRA2_anom = MERRA2_anom
#    				y_GLDAS_anom = GLDAS_anom   			
#
#    				naive_rmse_anom = mean_squared_error(y_ERA5_anom, y_naive_anom, squared=False)
#    				naive_rmse_master_anom.append(naive_rmse_anom)
#
#    				TC_rmse_anom = mean_squared_error(y_ERA5_anom, y_TC_anom, squared=False)
#    				TC_rmse_master_anom.append(TC_rmse_anom)
#
#    				CFSR_rmse_anom = mean_squared_error(y_ERA5_anom, y_CFSR_anom, squared=False)
#    				CFSR_rmse_master_anom.append(CFSR_rmse_anom)
#
#    				ERAI_rmse_anom = mean_squared_error(y_ERA5_anom, y_ERAI_anom, squared=False)
#    				ERAI_rmse_master_anom.append(ERAI_rmse_anom)
#
#    				ERA5_rmse_anom = mean_squared_error(y_ERA5_anom, y_ERA5_anom, squared=False)
#    				ERA5_rmse_master_anom.append(ERA5_rmse_anom)
#
#    				JRA_rmse_anom = mean_squared_error(y_ERA5_anom, y_JRA_anom, squared=False)
#    				JRA_rmse_master_anom.append(JRA_rmse_anom)
#
#    				MERRA2_rmse_anom = mean_squared_error(y_ERA5_anom, y_MERRA2_anom, squared=False)
#    				MERRA2_rmse_master_anom.append(MERRA2_rmse_anom)
#
#    				GLDAS_rmse_anom = mean_squared_error(y_ERA5_anom, y_GLDAS_anom, squared=False)    			
#    				GLDAS_rmse_master_anom.append(GLDAS_rmse_anom)
#
#
#    				naive_ubrmse_anom = ubrmsd(y_ERA5_anom, y_naive_anom)
#    				naive_ubrmse_master_anom.append(naive_ubrmse_anom)
#
#    				TC_ubrmse_anom = ubrmsd(y_ERA5_anom, y_TC_anom)
#    				TC_ubrmse_master_anom.append(TC_ubrmse_anom)
#    			
#    				CFSR_ubrmse_anom = ubrmsd(y_ERA5_anom, y_CFSR_anom)
#    				CFSR_ubrmse_master_anom.append(CFSR_ubrmse_anom)
#
#    				ERAI_ubrmse_anom = ubrmsd(y_ERA5_anom, y_ERAI_anom)
#    				ERAI_ubrmse_master_anom.append(ERAI_ubrmse_anom)
#
#    				ERA5_ubrmse_anom = ubrmsd(y_ERA5_anom, y_ERA5_anom)
#    				ERA5_ubrmse_master_anom.append(ERA5_ubrmse_anom)
#
#    				JRA_ubrmse_anom = ubrmsd(y_ERA5_anom, y_JRA_anom)
#    				JRA_ubrmse_master_anom.append(JRA_ubrmse_anom)
#
#    				MERRA2_ubrmse_anom = ubrmsd(y_ERA5_anom, y_MERRA2_anom)
#    				MERRA2_ubrmse_master_anom.append(MERRA2_ubrmse_anom)
#
#    				GLDAS_ubrmse_anom = ubrmsd(y_ERA5_anom, y_GLDAS_anom)
#    				GLDAS_ubrmse_master_anom.append(GLDAS_ubrmse_anom)
#
#
################### Calculate Pearson Correlations ####################
#
###### Raw Temperatures #####
#    				TC_corr_raw, _ = pearsonr(TC_raw_temp, ERA5_raw_temp)
#    				TC_corr_master_raw.append(TC_corr_raw)
#    				naive_corr_raw, _ = pearsonr(naive_raw_temp, ERA5_raw_temp)
#    				naive_corr_master_raw.append(naive_corr_raw)
#    				CFSR_corr_raw, _ = pearsonr(CFSR_raw_temp, ERA5_raw_temp)
#    				CFSR_corr_master_raw.append(CFSR_corr_raw)
#    				ERAI_corr_raw, _ = pearsonr(ERAI_raw_temp, ERA5_raw_temp)
#    				ERAI_corr_master_raw.append(ERAI_corr_raw)
#    				ERA5_corr_raw, _ = pearsonr(ERA5_raw_temp, ERA5_raw_temp)
#    				ERA5_corr_master_raw.append(ERA5_corr_raw)
#    				JRA_corr_raw, _ = pearsonr(JRA_raw_temp, ERA5_raw_temp)
#    				JRA_corr_master_raw.append(JRA_corr_raw)
#    				MERRA2_corr_raw, _ = pearsonr(MERRA2_raw_temp, ERA5_raw_temp)
#    				MERRA2_corr_master_raw.append(MERRA2_corr_raw)
#    				GLDAS_corr_raw, _ = pearsonr(GLDAS_raw_temp, ERA5_raw_temp)
#    				GLDAS_corr_master_raw.append(GLDAS_corr_raw)
#    				delta_corr_raw = TC_corr_raw - naive_corr_raw
#    				delta_corr_master_raw.append(delta_corr_raw)
#
###### Anomalies #####
#    				TC_corr_anom, _ = pearsonr(TC_anom, ERA5_anom)
#    				TC_corr_master_anom.append(TC_corr_anom)
#    				naive_corr_anom, _ = pearsonr(naive_anom, ERA5_anom)
#    				naive_corr_master_anom.append(naive_corr_anom)
#    				CFSR_corr_anom, _ = pearsonr(CFSR_anom, ERA5_anom)
#    				CFSR_corr_master_anom.append(CFSR_corr_anom)
#    				ERAI_corr_anom, _ = pearsonr(ERAI_anom, ERA5_anom)
#    				ERAI_corr_master_anom.append(ERAI_corr_anom)
#    				ERA5_corr_anom, _ = pearsonr(ERA5_anom, ERA5_anom)
#    				ERA5_corr_master_anom.append(ERA5_corr_anom)
#    				JRA_corr_anom, _ = pearsonr(JRA_anom, ERA5_anom)
#    				JRA_corr_master_anom.append(JRA_corr_anom)
#    				MERRA2_corr_anom, _ = pearsonr(MERRA2_anom, ERA5_anom)
#    				MERRA2_corr_master_anom.append(MERRA2_corr_anom)
#    				GLDAS_corr_anom, _ = pearsonr(GLDAS_anom, ERA5_anom)
#    				GLDAS_corr_master_anom.append(GLDAS_corr_anom)
#    				delta_corr_anom = TC_corr_anom - naive_corr_anom
#    				delta_corr_master_anom.append(delta_corr_anom)  					
#										    					
################### Create Summary Statistics Dataframes ##############
#
#
#    			df_summary_raw = pd.DataFrame(data=gcell_master, columns=['Grid Cell'])
#    			df_summary_raw['Central Lat'] = lat_master
#    			df_summary_raw['Central Lon'] = lon_master
#    			df_summary_raw['Naive Blend Bias'] = naive_bias_master_raw
#    			df_summary_raw['TC Blend Bias'] = TC_bias_master_raw
#    			df_summary_raw['CFSR Bias'] = CFSR_bias_master_raw
#    			df_summary_raw['ERA-Interim Bias'] = ERAI_bias_master_raw
#    			df_summary_raw['ERA5 Bias'] = ERA5_bias_master_raw
#    			df_summary_raw['JRA-55 Bias'] = JRA_bias_master_raw
#    			df_summary_raw['MERRA2 Bias'] = MERRA2_bias_master_raw
#    			df_summary_raw['GLDAS Bias'] = GLDAS_bias_master_raw
#
#    			df_summary_raw['Naive Blend SDV'] = naive_SDV_master_raw
#    			df_summary_raw['TC Blend SDV'] = TC_SDV_master_raw
#    			df_summary_raw['CFSR SDV'] = CFSR_SDV_master_raw
#    			df_summary_raw['ERA-Interim SDV'] = ERAI_SDV_master_raw
#    			df_summary_raw['ERA5 SDV'] = ERA5_SDV_master_raw
#    			df_summary_raw['JRA-55 SDV'] = JRA_SDV_master_raw
#    			df_summary_raw['MERRA2 SDV'] = MERRA2_SDV_master_raw
#    			df_summary_raw['GLDAS SDV'] = GLDAS_SDV_master_raw
#
#    			df_summary_raw['Naive Blend RMSE'] = naive_rmse_master_raw
#    			df_summary_raw['TC Blend RMSE'] = TC_rmse_master_raw
#    			df_summary_raw['CFSR RMSE'] = CFSR_rmse_master_raw
#    			df_summary_raw['ERA-Interim RMSE'] = ERAI_rmse_master_raw
#    			df_summary_raw['ERA5 RMSE'] = ERA5_rmse_master_raw
#    			df_summary_raw['JRA-55 RMSE'] = JRA_rmse_master_raw
#    			df_summary_raw['MERRA2 RMSE'] = MERRA2_rmse_master_raw
#    			df_summary_raw['GLDAS RMSE'] = GLDAS_rmse_master_raw
#
#    			df_summary_raw['Naive Blend ubRMSE'] = naive_ubrmse_master_raw
#    			df_summary_raw['TC Blend ubRMSE'] = TC_ubrmse_master_raw
#    			df_summary_raw['CFSR ubRMSE'] = CFSR_ubrmse_master_raw
#    			df_summary_raw['ERA-Interim ubRMSE'] = ERAI_ubrmse_master_raw
#    			df_summary_raw['ERA5 ubRMSE'] = ERA5_ubrmse_master_raw
#    			df_summary_raw['JRA-55 ubRMSE'] = JRA_ubrmse_master_raw
#    			df_summary_raw['MERRA2 ubRMSE'] = MERRA2_ubrmse_master_raw
#    			df_summary_raw['GLDAS ubRMSE'] = GLDAS_ubrmse_master_raw
#
#    			df_summary_raw['delta corr'] = delta_corr_master_raw
#
#    			#print(df_summary_raw)
#
#    			df_summary_anom = pd.DataFrame(data=gcell_master, columns=['Grid Cell'])
#    			df_summary_anom['Central Lat'] = lat_master
#    			df_summary_anom['Central Lon'] = lon_master
#    			df_summary_anom['Naive Blend Bias'] = naive_bias_master_anom
#    			df_summary_anom['TC Blend Bias'] = TC_bias_master_anom
#    			df_summary_anom['CFSR Bias'] = CFSR_bias_master_anom
#    			df_summary_anom['ERA-Interim Bias'] = ERAI_bias_master_anom
#    			df_summary_anom['ERA5 Bias'] = ERA5_bias_master_anom
#    			df_summary_anom['JRA-55 Bias'] = JRA_bias_master_anom
#    			df_summary_anom['MERRA2 Bias'] = MERRA2_bias_master_anom
#    			df_summary_anom['GLDAS Bias'] = GLDAS_bias_master_anom
#
#    			df_summary_anom['Naive Blend SDV'] = naive_SDV_master_anom
#    			df_summary_anom['TC Blend SDV'] = TC_SDV_master_anom
#    			df_summary_anom['CFSR SDV'] = CFSR_SDV_master_anom
#    			df_summary_anom['ERA-Interim SDV'] = ERAI_SDV_master_anom
#    			df_summary_anom['ERA5 SDV'] = ERA5_SDV_master_anom
#    			df_summary_anom['JRA-55 SDV'] = JRA_SDV_master_anom
#    			df_summary_anom['MERRA2 SDV'] = MERRA2_SDV_master_anom
#    			df_summary_anom['GLDAS SDV'] = GLDAS_SDV_master_anom
#
#    			df_summary_anom['Naive Blend RMSE'] = naive_rmse_master_anom
#    			df_summary_anom['TC Blend RMSE'] = TC_rmse_master_anom
#    			df_summary_anom['CFSR RMSE'] = CFSR_rmse_master_anom
#    			df_summary_anom['ERA-Interim RMSE'] = ERAI_rmse_master_anom
#    			df_summary_anom['ERA5 RMSE'] = ERA5_rmse_master_anom
#    			df_summary_anom['JRA-55 RMSE'] = JRA_rmse_master_anom
#    			df_summary_anom['MERRA2 RMSE'] = MERRA2_rmse_master_anom
#    			df_summary_anom['GLDAS RMSE'] = GLDAS_rmse_master_anom
#
#    			df_summary_anom['Naive Blend ubRMSE'] = naive_ubrmse_master_anom
#    			df_summary_anom['TC Blend ubRMSE'] = TC_ubrmse_master_anom
#    			df_summary_anom['CFSR ubRMSE'] = CFSR_ubrmse_master_anom
#    			df_summary_anom['ERA-Interim ubRMSE'] = ERAI_ubrmse_master_anom
#    			df_summary_anom['ERA5 ubRMSE'] = ERA5_ubrmse_master_anom
#    			df_summary_anom['JRA-55 ubRMSE'] = JRA_ubrmse_master_anom
#    			df_summary_anom['MERRA2 ubRMSE'] = MERRA2_ubrmse_master_anom
#    			df_summary_anom['GLDAS ubRMSE'] = GLDAS_ubrmse_master_anom
#
#    			df_summary_anom ['delta corr'] = delta_corr_master_anom
#    			#print(df_summary_anom)
#
###### Grid Cell Level #####
#
#    			raw_sum_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/blended_metrics/grid_cell/raw_temp/'+str(remap_type)+'_'+str(olri)+'_'+str(lyrj)+'_thr'+str(thrk)+'_summary_statistics_gridcell.csv'])
#    			print(raw_sum_fil)
#    			path = pathlib.Path(raw_sum_fil)
#    			path.parent.mkdir(parents=True, exist_ok=True)			
#    			df_summary_raw.to_csv(raw_sum_fil,index=False)
#
#
#    			anom_sum_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/blended_metrics/grid_cell/anom/'+str(remap_type)+'_'+str(olri)+'_'+str(lyrj)+'_thr'+str(thrk)+'_summary_statistics_anom_gridcell.csv'])
#    			print(anom_sum_fil)
#    			path2 = pathlib.Path(anom_sum_fil)
#    			path2.parent.mkdir(parents=True, exist_ok=True)			
#    			df_summary_anom.to_csv(anom_sum_fil,index=False)






##################### COLLOCATION WITH STATION DATA ###########################

##### Loop Through Grid Cells and grab temps collocated with station data #####

    			#gcell_uq = np.unique(gcell)
			
    			TC_anom_dframe = pd.read_csv(TC_anom_fil)

    			TC_raw_dframe = pd.read_csv(TC_raw_fil)


    			#print(gcell_uq)

    			for l in gcell_uq_stn:


    				gcell_l = l
    				#print(gcell_l)
    				TC_raw_dframe_gcell = TC_raw_dframe[TC_raw_dframe['Grid Cell'] == gcell_l]
    				TC_anom_dframe_gcell = TC_anom_dframe[TC_anom_dframe['Grid Cell'] == gcell_l]
    				len_TC = len(TC_raw_dframe_gcell)
    				#print(TC_raw_dframe_gcell)
    				if (len_TC == 0): #skip grid cell if grid cell doesn't exist in reanalysis data
    					continue
    				station_fil = ''.join([station_dir+'grid_'+str(gcell_l)+'_anom.csv'])
    				if os.path.exists(station_fil):
    					station_dframe_gcell = pd.read_csv(station_fil)
    					#print(station_dframe_gcell)
    					Date = station_dframe_gcell['Date'].values
    					DateTime = [datetime.datetime.strptime(x,'%Y-%m-%d') for x in Date] 
    					rnysis_edate = TC_raw_dframe_gcell['Date'].iloc[len_TC-1] ###locate end date of reanalysis data
    					rnysis_edate = datetime.datetime.strptime(rnysis_edate,'%Y-%m-%d')
    					 
    					#print(TC_raw_dframe_gcell)
###### Raw (Absolute Temps) #####

    					date_raw_gcell = []
    					station_raw_gcell = []
    					station_anom_gcell = []
    					TC_raw_gcell = []
    					TC_anom_gcell = []
    					naive_raw_gcell = []
    					CFSR_raw_gcell = []
    					ERAI_raw_gcell = []
    					ERA5_raw_gcell = []
    					JRA_raw_gcell = []
    					MERRA2_raw_gcell = []
    					GLDAS_raw_gcell = []
   				
    					for m in range(0,len(Date)):
    						datetime_m = DateTime[m]
	    					date_m = Date[m]
    						if(datetime_m > rnysis_edate): #skip all dates beyond last reanalysis date
    							continue
    						station_dframe_dt = station_dframe_gcell[station_dframe_gcell['Date'] == date_m]
    						#print(station_dframe_dt)
    						TC_raw_dframe_dt = TC_raw_dframe_gcell[TC_raw_dframe_gcell['Date'] == date_m]
    						station_anom = station_dframe_dt['Spatial Avg Anom'].values.tolist()
    						station_anom_gcell.append(station_anom)
    						#print(TC_raw_dframe_dt)
    						station_raw_temp = station_dframe_dt['Spatial Avg Temp'].values.tolist()
    						date_raw_gcell.append(date_m)
    						station_raw_gcell.append(station_raw_temp)
    						naive_raw_temp = TC_raw_dframe_dt['Naive Blended'].values.tolist()
    						naive_raw_gcell.append(naive_raw_temp)
    						TC_raw_temp = TC_raw_dframe_dt['TC Blended'].values.tolist()
    						TC_raw_gcell.append(TC_raw_temp)
    						CFSR_raw_temp = TC_raw_dframe_dt['CFSR'].values.tolist()
    						CFSR_raw_gcell.append(CFSR_raw_temp)
    						ERAI_raw_temp = TC_raw_dframe_dt['ERA-Interim'].values.tolist()
    						ERAI_raw_gcell.append(ERAI_raw_temp)
    						ERA5_raw_temp = TC_raw_dframe_dt['ERA5'].values.tolist()
    						ERA5_raw_gcell.append(ERA5_raw_temp)
    						JRA_raw_temp = TC_raw_dframe_dt['JRA-55'].values.tolist()
    						JRA_raw_gcell.append(JRA_raw_temp)
    						MERRA2_raw_temp = TC_raw_dframe_dt['MERRA2'].values.tolist()
    						MERRA2_raw_gcell.append(MERRA2_raw_temp)
    						GLDAS_raw_temp = TC_raw_dframe_dt['GLDAS'].values.tolist()
    						GLDAS_raw_gcell.append(GLDAS_raw_temp)			

    					
    					date_raw_gcell = date_raw_gcell
    					station_anom_gcell = [i for sub in station_anom_gcell for i in sub]
    					station_raw_gcell = [i for sub in station_raw_gcell for i in sub]
    					TC_raw_gcell = [i for sub in TC_raw_gcell for i in sub]   					
    					naive_raw_gcell = [i for sub in naive_raw_gcell for i in sub]				
    					CFSR_raw_gcell = [i for sub in CFSR_raw_gcell for i in sub]
    					ERAI_raw_gcell = [i for sub in ERAI_raw_gcell for i in sub]
    					ERA5_raw_gcell = [i for sub in ERA5_raw_gcell for i in sub]
    					JRA_raw_gcell = [i for sub in JRA_raw_gcell for i in sub]
    					MERRA2_raw_gcell = [i for sub in MERRA2_raw_gcell for i in sub]
    					GLDAS_raw_gcell = [i for sub in GLDAS_raw_gcell for i in sub]

    					if(len(TC_raw_gcell) == 0 or len(station_raw_gcell) == 0): #skip if empty
    						continue 

    					gcell_master_stn.append(gcell_l)
    					lat_cen = station_dframe_gcell['Central Lat'].iloc[0]
    					lat_master_stn.append(lat_cen)
    					lon_cen =  station_dframe_gcell['Central Lon'].iloc[0]
    					lon_master_stn.append(lon_cen)

									
#################### create anomalies for reanalysis files #######################
    					rnysis_anom_master = []
    					rnysis_date_master = []
    					rnysis_name_master = []
    					rnysis_stemp_master = []
    					rnysis = [TC_raw_gcell,naive_raw_gcell,CFSR_raw_gcell,ERAI_raw_gcell,ERA5_raw_gcell,JRA_raw_gcell,MERRA2_raw_gcell,GLDAS_raw_gcell]
    					rnysis_name = ['TC Blend','Naive Blend','CFSR','ERA-Interim','ERA-5','JRA-55','MERRA2','GLDAS']
    					dat_rowlist = [datetime.datetime.strptime(x,'%Y-%m-%d') for x in date_raw_gcell]
    					dat_rowlist2 = date_raw_gcell
    					num_rows = len(dat_rowlist)

    					for m in range(0,8):
    						rnysisi = rnysis[m]
    						rnysis_namei = rnysis_name[m]
    						#print("Reanalysis Product:",rnysis_namei)
    						#print(rnysisi)
    						climatology = dict()
    						clim_averages = dict()
    						stemp_mstr = []
    						stemp_anom_master = []
    						date_mstr = []
    						name_mstr = []
    						for month in range(1,13):
    							month_key = f"{month:02}"
    							climatology[month_key] = list()

    						for n in range(0,num_rows):
					###add month data to list based on key
    							dat_row = dat_rowlist[n]
    							stemp_row = rnysisi[n]
    							month_key = dat_row.strftime("%m")
    							climatology[month_key].append(stemp_row)

    						climatology_keys = list(climatology.keys())
    						climatology_keys2 = np.array(climatology_keys).flatten()
    					#print(climatology)
					
    						for key in climatology:
					###take averages and write to averages dictionary
    							current_total = 0
    							len_current_list = 0
    							current_list = climatology[key]
    							for temp in current_list:
    								if (temp == np.nan):
    									current_total = current_total + 0
    									len_current_list = len_current_list + 0
    								else:
    									current_total = current_total + temp
    									len_current_list = len_current_list + 1
    							if (len_current_list == 0):
    								average = np.nan
    							else:
    								average = current_total/len_current_list
    							clim_averages[key] = average
    							#print(average)
							
    						clim_avg = list(clim_averages.values())
    						#print(clim_averages)
						
    						for o in range (0, num_rows):
    							stemp_rw = rnysisi[o]
    							dat_row = dat_rowlist[o]
    							dat_row_mon = dat_row.month
    							dat_row_mons = f"{dat_row_mon:02}"
    							#print(stemp_rw,dat_row_mon,clim_averages[dat_row_mons])
    							stemp_anom = stemp_rw - clim_averages[dat_row_mons]

    							rnysis_anom_master.append(stemp_anom)
    							rnysis_date_master.append(dat_row)					
    							rnysis_name_master.append(rnysis_namei)
    							rnysis_stemp_master.append(stemp_rw)



####### Station Collocated Anomalies #####

    					dframe_anom_master = pd.DataFrame(data=rnysis_date_master, columns=['Date'])
    					dframe_anom_master['Name'] = rnysis_name_master
    					dframe_anom_master['Raw Temp'] = rnysis_stemp_master
    					dframe_anom_master['Anom'] = rnysis_anom_master

    					station_anom = station_anom_gcell
    					TC_anom = dframe_anom_master[dframe_anom_master['Name'] == 'TC Blend']
    					TC_anom = TC_anom['Anom'].values
    					naive_anom = dframe_anom_master[dframe_anom_master['Name'] == 'Naive Blend']
    					naive_anom = naive_anom['Anom'].values
    					CFSR_anom = dframe_anom_master[dframe_anom_master['Name'] == 'CFSR']
    					CFSR_anom = CFSR_anom['Anom'].values
    					ERAI_anom = dframe_anom_master[dframe_anom_master['Name'] == 'ERA-Interim']
    					ERAI_anom = ERAI_anom['Anom'].values
    					ERA5_anom = dframe_anom_master[dframe_anom_master['Name'] == 'ERA-5']
    					ERA5_anom = ERA5_anom['Anom'].values
    					JRA_anom = dframe_anom_master[dframe_anom_master['Name'] == 'JRA-55']
    					JRA_anom = JRA_anom['Anom'].values					
    					MERRA2_anom = dframe_anom_master[dframe_anom_master['Name'] == 'MERRA2']
    					MERRA2_anom = MERRA2_anom['Anom'].values
    					GLDAS_anom = dframe_anom_master[dframe_anom_master['Name'] == 'GLDAS']
    					GLDAS_anom = GLDAS_anom['Anom'].values

############### Calculate Biases ############
    					station_raw_temp = station_raw_gcell
    					TC_raw_temp = TC_raw_gcell
    					naive_raw_temp = naive_raw_gcell
    					CFSR_raw_temp = CFSR_raw_gcell
    					ERAI_raw_temp = ERAI_raw_gcell
    					ERA5_raw_temp = ERAI_raw_gcell
    					JRA_raw_temp = JRA_raw_gcell
    					MERRA2_raw_temp = MERRA2_raw_gcell
    					GLDAS_raw_temp = GLDAS_raw_gcell

    					#print(station_raw_temp)
    					#print(TC_raw_temp)    				
###### Raw Temp #####

    					naive_bias_raw = bias(naive_raw_temp, station_raw_temp)
    					naive_bias_master_raw.append(naive_bias_raw)

    					TC_bias_raw = bias(TC_raw_temp, station_raw_temp)
    					TC_bias_master_raw.append(TC_bias_raw)

    					CFSR_bias_raw = bias(CFSR_raw_temp, station_raw_temp)
    					CFSR_bias_master_raw.append(CFSR_bias_raw)

    					ERAI_bias_raw = bias(ERAI_raw_temp, station_raw_temp)
    					ERAI_bias_master_raw.append(ERAI_bias_raw)

    					ERA5_bias_raw = bias(ERA5_raw_temp, station_raw_temp)
    					ERA5_bias_master_raw.append(ERA5_bias_raw)

    					JRA_bias_raw = bias(JRA_raw_temp, station_raw_temp)
    					JRA_bias_master_raw.append(JRA_bias_raw)

    					MERRA2_bias_raw = bias(MERRA2_raw_temp, station_raw_temp)
    					MERRA2_bias_master_raw.append(MERRA2_bias_raw)

    					GLDAS_bias_raw = bias(GLDAS_raw_temp, station_raw_temp)
    					GLDAS_bias_master_raw.append(GLDAS_bias_raw)

###### Anomalies #####

    					naive_bias_anom = bias(naive_anom, station_anom)
    					naive_bias_master_anom.append(naive_bias_anom)

    					TC_bias_anom = bias(TC_anom, station_anom)
    					TC_bias_master_anom.append(TC_bias_anom)

    					CFSR_bias_anom = bias(CFSR_anom, station_anom)
    					CFSR_bias_master_anom.append(CFSR_bias_anom)

    					ERAI_bias_anom = bias(ERAI_anom, station_anom)
    					ERAI_bias_master_anom.append(ERAI_bias_anom)

    					ERA5_bias_anom = bias(ERA5_anom, station_anom)
    					ERA5_bias_master_anom.append(ERA5_bias_anom)

    					JRA_bias_anom = bias(JRA_anom, station_anom)
    					JRA_bias_master_anom.append(JRA_bias_anom)

    					MERRA2_bias_anom = bias(MERRA2_anom, station_anom)
    					MERRA2_bias_master_anom.append(MERRA2_bias_anom)

    					GLDAS_bias_anom = bias(GLDAS_anom, station_anom)
    					GLDAS_bias_master_anom.append(GLDAS_bias_anom)

############### Calculate normalized standard deviations (relative to in-situ) ############

###### Raw Temp #####

    					naive_SDV_raw = SDVnorm(naive_raw_temp, station_raw_temp)
    					naive_SDV_master_raw.append(naive_SDV_raw)

    					TC_SDV_raw = SDVnorm(TC_raw_temp, station_raw_temp)
    					TC_SDV_master_raw.append(TC_SDV_raw)

    					CFSR_SDV_raw = SDVnorm(CFSR_raw_temp, station_raw_temp)
    					CFSR_SDV_master_raw.append(CFSR_SDV_raw)

    					ERAI_SDV_raw = SDVnorm(ERAI_raw_temp, station_raw_temp)
    					ERAI_SDV_master_raw.append(ERAI_SDV_raw)

    					ERA5_SDV_raw = SDVnorm(ERA5_raw_temp, station_raw_temp)
    					ERA5_SDV_master_raw.append(ERA5_SDV_raw)

    					JRA_SDV_raw = SDVnorm(JRA_raw_temp, station_raw_temp)
    					JRA_SDV_master_raw.append(JRA_SDV_raw)

    					MERRA2_SDV_raw = SDVnorm(MERRA2_raw_temp, station_raw_temp)
    					MERRA2_SDV_master_raw.append(MERRA2_SDV_raw)

    					GLDAS_SDV_raw = SDVnorm(GLDAS_raw_temp, station_raw_temp)
    					GLDAS_SDV_master_raw.append(GLDAS_SDV_raw)

###### Anomalies #####

    					naive_SDV_anom = SDVnorm(naive_anom, station_anom)
    					naive_SDV_master_anom.append(naive_SDV_anom)

    					TC_SDV_anom = SDVnorm(TC_anom, station_anom)
    					TC_SDV_master_anom.append(TC_SDV_anom)

    					CFSR_SDV_anom = SDVnorm(CFSR_anom, station_anom)
    					CFSR_SDV_master_anom.append(CFSR_SDV_anom)

    					ERAI_SDV_anom = SDVnorm(ERAI_anom, station_anom)
    					ERAI_SDV_master_anom.append(ERAI_SDV_anom)

    					ERA5_SDV_anom = SDVnorm(ERA5_anom, station_anom)
    					ERA5_SDV_master_anom.append(ERA5_SDV_anom)

    					JRA_SDV_anom = SDVnorm(JRA_anom, station_anom)
    					JRA_SDV_master_anom.append(JRA_SDV_anom)

    					MERRA2_SDV_anom = SDVnorm(MERRA2_anom, station_anom)
    					MERRA2_SDV_master_anom.append(MERRA2_SDV_anom)

    					GLDAS_SDV_anom = SDVnorm(GLDAS_anom, station_anom)
    					GLDAS_SDV_master_anom.append(GLDAS_SDV_anom)

												
############## Calculate RMSE and ubRMSE for products ##############

###### Raw Temp #####
    					y_true_raw = station_raw_temp
    					y_naive_raw = naive_raw_temp
    					y_TC_raw = TC_raw_temp
    					y_CFSR_raw = CFSR_raw_temp
    					y_ERAI_raw = ERAI_raw_temp
    					y_ERA5_raw = ERA5_raw_temp
    					y_JRA_raw = JRA_raw_temp
    					y_MERRA2_raw = MERRA2_raw_temp
    					y_GLDAS_raw = GLDAS_raw_temp   			

    					naive_rmse_raw = mean_squared_error(y_true_raw, y_naive_raw, squared=False)
    					naive_rmse_master_raw.append(naive_rmse_raw)

    					TC_rmse_raw = mean_squared_error(y_true_raw, y_TC_raw, squared=False)
    					TC_rmse_master_raw.append(TC_rmse_raw)

    					CFSR_rmse_raw = mean_squared_error(y_true_raw, y_CFSR_raw, squared=False)
    					CFSR_rmse_master_raw.append(CFSR_rmse_raw)

    					ERAI_rmse_raw = mean_squared_error(y_true_raw, y_ERAI_raw, squared=False)
    					ERAI_rmse_master_raw.append(ERAI_rmse_raw)

    					ERA5_rmse_raw = mean_squared_error(y_true_raw, y_ERA5_raw, squared=False)
    					ERA5_rmse_master_raw.append(ERA5_rmse_raw)

    					JRA_rmse_raw = mean_squared_error(y_true_raw, y_JRA_raw, squared=False)
    					JRA_rmse_master_raw.append(JRA_rmse_raw)

    					MERRA2_rmse_raw = mean_squared_error(y_true_raw, y_MERRA2_raw, squared=False)
    					MERRA2_rmse_master_raw.append(MERRA2_rmse_raw)

    					GLDAS_rmse_raw = mean_squared_error(y_true_raw, y_GLDAS_raw, squared=False)    			
    					GLDAS_rmse_master_raw.append(GLDAS_rmse_raw)


    					naive_ubrmse_raw = ubrmsd(y_true_raw, y_naive_raw)
    					naive_ubrmse_master_raw.append(naive_ubrmse_raw)

    					TC_ubrmse_raw = ubrmsd(y_true_raw, y_TC_raw)
    					TC_ubrmse_master_raw.append(TC_ubrmse_raw)
    			
    					CFSR_ubrmse_raw = ubrmsd(y_true_raw, y_CFSR_raw)
    					CFSR_ubrmse_master_raw.append(CFSR_ubrmse_raw)

    					ERAI_ubrmse_raw = ubrmsd(y_true_raw, y_ERAI_raw)
    					ERAI_ubrmse_master_raw.append(ERAI_ubrmse_raw)

    					ERA5_ubrmse_raw = ubrmsd(y_true_raw, y_ERA5_raw)
    					ERA5_ubrmse_master_raw.append(ERA5_ubrmse_raw)

    					JRA_ubrmse_raw = ubrmsd(y_true_raw, y_JRA_raw)
    					JRA_ubrmse_master_raw.append(JRA_ubrmse_raw)

    					MERRA2_ubrmse_raw = ubrmsd(y_true_raw, y_MERRA2_raw)
    					MERRA2_ubrmse_master_raw.append(MERRA2_ubrmse_raw)

    					GLDAS_ubrmse_raw = ubrmsd(y_true_raw, y_GLDAS_raw)
    					GLDAS_ubrmse_master_raw.append(GLDAS_ubrmse_raw) 


###### Anomalies #####

    					y_true_anom = station_anom
    					y_naive_anom = naive_anom
    					y_TC_anom = TC_anom
    					y_CFSR_anom = CFSR_anom
    					y_ERAI_anom = ERAI_anom
    					y_ERA5_anom = ERA5_anom
    					y_JRA_anom = JRA_anom
    					y_MERRA2_anom = MERRA2_anom
    					y_GLDAS_anom = GLDAS_anom   			

    					naive_rmse_anom = mean_squared_error(y_true_anom, y_naive_anom, squared=False)
    					naive_rmse_master_anom.append(naive_rmse_anom)

    					TC_rmse_anom = mean_squared_error(y_true_anom, y_TC_anom, squared=False)
    					TC_rmse_master_anom.append(TC_rmse_anom)

    					CFSR_rmse_anom = mean_squared_error(y_true_anom, y_CFSR_anom, squared=False)
    					CFSR_rmse_master_anom.append(CFSR_rmse_anom)

    					ERAI_rmse_anom = mean_squared_error(y_true_anom, y_ERAI_anom, squared=False)
    					ERAI_rmse_master_anom.append(ERAI_rmse_anom)

    					ERA5_rmse_anom = mean_squared_error(y_true_anom, y_ERA5_anom, squared=False)
    					ERA5_rmse_master_anom.append(ERA5_rmse_anom)

    					JRA_rmse_anom = mean_squared_error(y_true_anom, y_JRA_anom, squared=False)
    					JRA_rmse_master_anom.append(JRA_rmse_anom)

    					MERRA2_rmse_anom = mean_squared_error(y_true_anom, y_MERRA2_anom, squared=False)
    					MERRA2_rmse_master_anom.append(MERRA2_rmse_anom)

    					GLDAS_rmse_anom = mean_squared_error(y_true_anom, y_GLDAS_anom, squared=False)    			
    					GLDAS_rmse_master_anom.append(GLDAS_rmse_anom)


    					naive_ubrmse_anom = ubrmsd(y_true_anom, y_naive_anom)
    					naive_ubrmse_master_anom.append(naive_ubrmse_anom)

    					TC_ubrmse_anom = ubrmsd(y_true_anom, y_TC_anom)
    					TC_ubrmse_master_anom.append(TC_ubrmse_anom)
    			
    					CFSR_ubrmse_anom = ubrmsd(y_true_anom, y_CFSR_anom)
    					CFSR_ubrmse_master_anom.append(CFSR_ubrmse_anom)

    					ERAI_ubrmse_anom = ubrmsd(y_true_anom, y_ERAI_anom)
    					ERAI_ubrmse_master_anom.append(ERAI_ubrmse_anom)

    					ERA5_ubrmse_anom = ubrmsd(y_true_anom, y_ERA5_anom)
    					ERA5_ubrmse_master_anom.append(ERA5_ubrmse_anom)

    					JRA_ubrmse_anom = ubrmsd(y_true_anom, y_JRA_anom)
    					JRA_ubrmse_master_anom.append(JRA_ubrmse_anom)

    					MERRA2_ubrmse_anom = ubrmsd(y_true_anom, y_MERRA2_anom)
    					MERRA2_ubrmse_master_anom.append(MERRA2_ubrmse_anom)

    					GLDAS_ubrmse_anom = ubrmsd(y_true_anom, y_GLDAS_anom)
    					GLDAS_ubrmse_master_anom.append(GLDAS_ubrmse_anom)


################## Calculate Pearson Correlations ####################

##### Raw Temperatures #####
    					TC_corr_raw, _ = pearsonr(TC_raw_temp, station_raw_temp)
    					TC_corr_master_raw.append(TC_corr_raw)
    					naive_corr_raw, _ = pearsonr(naive_raw_temp, station_raw_temp)
    					naive_corr_master_raw.append(naive_corr_raw)
    					CFSR_corr_raw, _ = pearsonr(CFSR_raw_temp, station_raw_temp)
    					CFSR_corr_master_raw.append(CFSR_corr_raw)
    					ERAI_corr_raw, _ = pearsonr(ERAI_raw_temp, station_raw_temp)
    					ERAI_corr_master_raw.append(ERAI_corr_raw)
    					ERA5_corr_raw, _ = pearsonr(station_raw_temp, station_raw_temp)
    					ERA5_corr_master_raw.append(ERA5_corr_raw)
    					JRA_corr_raw, _ = pearsonr(JRA_raw_temp, station_raw_temp)
    					JRA_corr_master_raw.append(JRA_corr_raw)
    					MERRA2_corr_raw, _ = pearsonr(MERRA2_raw_temp, station_raw_temp)
    					MERRA2_corr_master_raw.append(MERRA2_corr_raw)
    					GLDAS_corr_raw, _ = pearsonr(GLDAS_raw_temp, station_raw_temp)
    					GLDAS_corr_master_raw.append(GLDAS_corr_raw)
    					delta_corr_raw = TC_corr_raw - naive_corr_raw
    					delta_corr_master_raw.append(delta_corr_raw)

##### Anomalies #####
    					TC_corr_anom, _ = pearsonr(TC_anom, station_anom)
    					TC_corr_master_anom.append(TC_corr_anom)
    					naive_corr_anom, _ = pearsonr(naive_anom, station_anom)
    					naive_corr_master_anom.append(naive_corr_anom)
    					CFSR_corr_anom, _ = pearsonr(CFSR_anom, station_anom)
    					CFSR_corr_master_anom.append(CFSR_corr_anom)
    					ERAI_corr_anom, _ = pearsonr(ERAI_anom, station_anom)
    					ERAI_corr_master_anom.append(ERAI_corr_anom)
    					ERA5_corr_anom, _ = pearsonr(station_anom, station_anom)
    					ERA5_corr_master_anom.append(ERA5_corr_anom)
    					JRA_corr_anom, _ = pearsonr(JRA_anom, station_anom)
    					JRA_corr_master_anom.append(JRA_corr_anom)
    					MERRA2_corr_anom, _ = pearsonr(MERRA2_anom, station_anom)
    					MERRA2_corr_master_anom.append(MERRA2_corr_anom)
    					GLDAS_corr_anom, _ = pearsonr(GLDAS_anom, station_anom)
    					GLDAS_corr_master_anom.append(GLDAS_corr_anom)
    					delta_corr_anom = TC_corr_anom - naive_corr_anom
    					delta_corr_master_anom.append(delta_corr_anom)   					
										    					
################## Create Summary Statistics Dataframes ##############


    			df_summary_raw = pd.DataFrame(data=gcell_master_stn, columns=['Grid Cell'])
    			df_summary_raw['Central Lat'] = lat_master_stn
    			df_summary_raw['Central Lon'] = lon_master_stn
    			df_summary_raw['Naive Blend Bias'] = naive_bias_master_raw
    			df_summary_raw['TC Blend Bias'] = TC_bias_master_raw
    			df_summary_raw['CFSR Bias'] = CFSR_bias_master_raw
    			df_summary_raw['ERA-Interim Bias'] = ERAI_bias_master_raw
    			df_summary_raw['ERA5 Bias'] = ERA5_bias_master_raw
    			df_summary_raw['JRA-55 Bias'] = JRA_bias_master_raw
    			df_summary_raw['MERRA2 Bias'] = MERRA2_bias_master_raw
    			df_summary_raw['GLDAS Bias'] = GLDAS_bias_master_raw

    			df_summary_raw['Naive Blend SDV'] = naive_SDV_master_raw
    			df_summary_raw['TC Blend SDV'] = TC_SDV_master_raw
    			df_summary_raw['CFSR SDV'] = CFSR_SDV_master_raw
    			df_summary_raw['ERA-Interim SDV'] = ERAI_SDV_master_raw
    			df_summary_raw['ERA5 SDV'] = ERA5_SDV_master_raw
    			df_summary_raw['JRA-55 SDV'] = JRA_SDV_master_raw
    			df_summary_raw['MERRA2 SDV'] = MERRA2_SDV_master_raw
    			df_summary_raw['GLDAS SDV'] = GLDAS_SDV_master_raw

    			df_summary_raw['Naive Blend RMSE'] = naive_rmse_master_raw
    			df_summary_raw['TC Blend RMSE'] = TC_rmse_master_raw
    			df_summary_raw['CFSR RMSE'] = CFSR_rmse_master_raw
    			df_summary_raw['ERA-Interim RMSE'] = ERAI_rmse_master_raw
    			df_summary_raw['ERA5 RMSE'] = ERA5_rmse_master_raw
    			df_summary_raw['JRA-55 RMSE'] = JRA_rmse_master_raw
    			df_summary_raw['MERRA2 RMSE'] = MERRA2_rmse_master_raw
    			df_summary_raw['GLDAS RMSE'] = GLDAS_rmse_master_raw

    			df_summary_raw['Naive Blend ubRMSE'] = naive_ubrmse_master_raw
    			df_summary_raw['TC Blend ubRMSE'] = TC_ubrmse_master_raw
    			df_summary_raw['CFSR ubRMSE'] = CFSR_ubrmse_master_raw
    			df_summary_raw['ERA-Interim ubRMSE'] = ERAI_ubrmse_master_raw
    			df_summary_raw['ERA5 ubRMSE'] = ERA5_ubrmse_master_raw
    			df_summary_raw['JRA-55 ubRMSE'] = JRA_ubrmse_master_raw
    			df_summary_raw['MERRA2 ubRMSE'] = MERRA2_ubrmse_master_raw
    			df_summary_raw['GLDAS ubRMSE'] = GLDAS_ubrmse_master_raw

    			df_summary_raw['delta corr'] = delta_corr_master_raw

    			#print(df_summary_raw)

    			df_summary_anom = pd.DataFrame(data=gcell_master_stn, columns=['Grid Cell'])
    			df_summary_anom['Central Lat'] = lat_master_stn
    			df_summary_anom['Central Lon'] = lon_master_stn
    			df_summary_anom['Naive Blend Bias'] = naive_bias_master_anom
    			df_summary_anom['TC Blend Bias'] = TC_bias_master_anom
    			df_summary_anom['CFSR Bias'] = CFSR_bias_master_anom
    			df_summary_anom['ERA-Interim Bias'] = ERAI_bias_master_anom
    			df_summary_anom['ERA5 Bias'] = ERA5_bias_master_anom
    			df_summary_anom['JRA-55 Bias'] = JRA_bias_master_anom
    			df_summary_anom['MERRA2 Bias'] = MERRA2_bias_master_anom
    			df_summary_anom['GLDAS Bias'] = GLDAS_bias_master_anom

    			df_summary_anom['Naive Blend SDV'] = naive_SDV_master_anom
    			df_summary_anom['TC Blend SDV'] = TC_SDV_master_anom
    			df_summary_anom['CFSR SDV'] = CFSR_SDV_master_anom
    			df_summary_anom['ERA-Interim SDV'] = ERAI_SDV_master_anom
    			df_summary_anom['ERA5 SDV'] = ERA5_SDV_master_anom
    			df_summary_anom['JRA-55 SDV'] = JRA_SDV_master_anom
    			df_summary_anom['MERRA2 SDV'] = MERRA2_SDV_master_anom
    			df_summary_anom['GLDAS SDV'] = GLDAS_SDV_master_anom

    			df_summary_anom['Naive Blend RMSE'] = naive_rmse_master_anom
    			df_summary_anom['TC Blend RMSE'] = TC_rmse_master_anom
    			df_summary_anom['CFSR RMSE'] = CFSR_rmse_master_anom
    			df_summary_anom['ERA-Interim RMSE'] = ERAI_rmse_master_anom
    			df_summary_anom['ERA5 RMSE'] = ERA5_rmse_master_anom
    			df_summary_anom['JRA-55 RMSE'] = JRA_rmse_master_anom
    			df_summary_anom['MERRA2 RMSE'] = MERRA2_rmse_master_anom
    			df_summary_anom['GLDAS RMSE'] = GLDAS_rmse_master_anom

    			df_summary_anom['Naive Blend ubRMSE'] = naive_ubrmse_master_anom
    			df_summary_anom['TC Blend ubRMSE'] = TC_ubrmse_master_anom
    			df_summary_anom['CFSR ubRMSE'] = CFSR_ubrmse_master_anom
    			df_summary_anom['ERA-Interim ubRMSE'] = ERAI_ubrmse_master_anom
    			df_summary_anom['ERA5 ubRMSE'] = ERA5_ubrmse_master_anom
    			df_summary_anom['JRA-55 ubRMSE'] = JRA_ubrmse_master_anom
    			df_summary_anom['MERRA2 ubRMSE'] = MERRA2_ubrmse_master_anom
    			df_summary_anom['GLDAS ubRMSE'] = GLDAS_ubrmse_master_anom

    			df_summary_anom ['delta corr'] = delta_corr_master_anom
    			#print(df_summary_anom)

##### Grid Cell Level #####

    			raw_sum_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/blended_metrics/grid_cell_stn/raw_temp/'+str(remap_type)+'_'+str(olri)+'_'+str(lyrj)+'_thr'+str(thrk)+'_summary_statistics_gridcell_stn.csv'])
    			print(raw_sum_fil)
    			path = pathlib.Path(raw_sum_fil)
    			path.parent.mkdir(parents=True, exist_ok=True)			
    			df_summary_raw.to_csv(raw_sum_fil,index=False)


    			anom_sum_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/blended_metrics/grid_cell_stn/anom/'+str(remap_type)+'_'+str(olri)+'_'+str(lyrj)+'_thr'+str(thrk)+'_summary_statistics_anom_gridcell_stn.csv'])
    			print(anom_sum_fil)
    			path2 = pathlib.Path(anom_sum_fil)
    			path2.parent.mkdir(parents=True, exist_ok=True)			
    			df_summary_anom.to_csv(anom_sum_fil,index=False)
















