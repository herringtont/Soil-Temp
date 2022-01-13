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

naive_raw_dir = "/mnt/data/users/herringtont/soil_temp/Blended_Product/collocated/Naive/raw/"
naive_anom_dir = "/mnt/data/users/herringtont/soil_temp/Blended_Product/collocated/Naive/anom/"
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

    			naive_raw_fil = ''.join([naive_raw_dir+str(remap_type)+'_'+str(olri)+'_'+str(lyrj)+'_thr'+str(thrk)+'_naive_blending.csv'])
    			naive_anom_fil = ''.join([naive_anom_dir+str(remap_type)+'_'+str(olri)+'_'+str(lyrj)+'_thr'+str(thrk)+'_naive_blending_anom.csv'])
    			TC_raw_fil = ''.join([TC_raw_dir+str(remap_type)+'_'+str(olri)+'_'+str(lyrj)+'_thr'+str(thrk)+'_TC_blended.csv'])
    			TC_anom_fil = ''.join([TC_anom_dir+str(remap_type)+'_'+str(olri)+'_'+str(lyrj)+'_thr'+str(thrk)+'_TC_blended_anom.csv'])


################# grab temperatures and anomalies for the different products ##############

##### Date/spatial info #####

    			TC_raw_dframe = pd.read_csv(TC_raw_fil)
    			Date = TC_raw_dframe['Date'].values
    			DateTime = [datetime.datetime.strptime(x,'%Y-%m-%d') for x in Date] 
    			gcell = TC_raw_dframe['Grid Cell'].values
    			lat_cen = TC_raw_dframe['Central Lat'].values
    			lon_cen = TC_raw_dframe['Central Lon'].values


##### Loop Through Grid Cells #####

    			gcell_uq = np.unique(gcell)

##### Raw (Absolute Temps) #####

    			TC_raw_temp = TC_raw_dframe['TC Blended'].values 
    			station_raw_temp = TC_raw_dframe['In-Situ'].values
    			CFSR_raw_temp = TC_raw_dframe['CFSR'].values
    			ERAI_raw_temp = TC_raw_dframe['ERA-Interim'].values
    			ERA5_raw_temp = TC_raw_dframe['ERA5'].values
    			JRA_raw_temp = TC_raw_dframe['JRA-55'].values
    			MERRA2_raw_temp = TC_raw_dframe['MERRA2'].values
    			GLDAS_raw_temp = TC_raw_dframe['GLDAS'].values
			
    			naive_raw_dframe = pd.read_csv(naive_raw_fil)
    			naive_raw_temp = naive_raw_dframe['Naive Blending'].values

##### Anomalies #####

    			TC_anom_dframe = pd.read_csv(TC_anom_fil)
    			TC_anom = TC_anom_dframe['TC Blended'].values 
    			station_anom = TC_anom_dframe['In-Situ'].values
    			CFSR_anom = TC_anom_dframe['CFSR'].values
    			ERAI_anom = TC_anom_dframe['ERA-Interim'].values
    			ERA5_anom = TC_anom_dframe['ERA5'].values
    			JRA_anom = TC_anom_dframe['JRA-55'].values
    			MERRA2_anom = TC_anom_dframe['MERRA2'].values
    			GLDAS_anom = TC_anom_dframe['GLDAS'].values
			
    			naive_anom_dframe = pd.read_csv(naive_anom_fil)
    			naive_anom = naive_anom_dframe['Naive Blending'].values


############# Calculate Biases ############

##### Raw Temp #####

    			naive_bias_raw = bias(naive_raw_temp, station_raw_temp)
    			TC_bias_raw = bias(TC_raw_temp, station_raw_temp)
    			CFSR_bias_raw = bias(CFSR_raw_temp, station_raw_temp)
    			ERAI_bias_raw = bias(ERAI_raw_temp, station_raw_temp)
    			ERA5_bias_raw = bias(ERA5_raw_temp, station_raw_temp)
    			JRA_bias_raw = bias(JRA_raw_temp, station_raw_temp)
    			MERRA2_bias_raw = bias(MERRA2_raw_temp, station_raw_temp)
    			GLDAS_bias_raw = bias(GLDAS_raw_temp, station_raw_temp)

##### Anomalies #####

    			naive_bias_anom = bias(naive_anom, station_anom)
    			TC_bias_anom = bias(TC_anom, station_anom)
    			CFSR_bias_anom = bias(CFSR_anom, station_anom)
    			ERAI_bias_anom = bias(ERAI_anom, station_anom)
    			ERA5_bias_anom = bias(ERA5_anom, station_anom)
    			JRA_bias_anom = bias(JRA_anom, station_anom)
    			MERRA2_bias_anom = bias(MERRA2_anom, station_anom)
    			GLDAS_bias_anom = bias(GLDAS_anom, station_anom)
    			print("Bias:")			
    			print(naive_bias_anom, TC_bias_anom)


############## Calculate normalized standard deviations (relative to in-situ) ############

##### Raw Temp #####

    			naive_SDV_raw = SDVnorm(naive_raw_temp, station_raw_temp)
    			TC_SDV_raw = SDVnorm(TC_raw_temp, station_raw_temp)
    			CFSR_SDV_raw = SDVnorm(CFSR_raw_temp, station_raw_temp)
    			ERAI_SDV_raw = SDVnorm(ERAI_raw_temp, station_raw_temp)
    			ERA5_SDV_raw = SDVnorm(ERA5_raw_temp, station_raw_temp)
    			JRA_SDV_raw = SDVnorm(JRA_raw_temp, station_raw_temp)
    			MERRA2_SDV_raw = SDVnorm(MERRA2_raw_temp, station_raw_temp)
    			GLDAS_SDV_raw = SDVnorm(GLDAS_raw_temp, station_raw_temp)

##### Anomalies #####

    			naive_SDV_anom = SDVnorm(naive_anom, station_anom)
    			TC_SDV_anom = SDVnorm(TC_anom, station_anom)
    			CFSR_SDV_anom = SDVnorm(CFSR_anom, station_anom)
    			ERAI_SDV_anom = SDVnorm(ERAI_anom, station_anom)
    			ERA5_SDV_anom = SDVnorm(ERA5_anom, station_anom)
    			JRA_SDV_anom = SDVnorm(JRA_anom, station_anom)
    			MERRA2_SDV_anom = SDVnorm(MERRA2_anom, station_anom)
    			GLDAS_SDV_anom = SDVnorm(GLDAS_anom, station_anom)


    			print('SDVnorm:')
    			print(naive_SDV_anom,TC_SDV_anom)												
############# Calculate RMSE and ubRMSE for products ##############

##### Raw Temp #####
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
    			TC_rmse_raw = mean_squared_error(y_true_raw, y_TC_raw, squared=False)
    			CFSR_rmse_raw = mean_squared_error(y_true_raw, y_CFSR_raw, squared=False)
    			ERAI_rmse_raw = mean_squared_error(y_true_raw, y_ERAI_raw, squared=False)
    			ERA5_rmse_raw = mean_squared_error(y_true_raw, y_ERA5_raw, squared=False)
    			JRA_rmse_raw = mean_squared_error(y_true_raw, y_JRA_raw, squared=False)
    			MERRA2_rmse_raw = mean_squared_error(y_true_raw, y_MERRA2_raw, squared=False)
    			GLDAS_rmse_raw = mean_squared_error(y_true_raw, y_GLDAS_raw, squared=False)    			

    			naive_ubrmse_raw = ubrmsd(y_true_raw, y_naive_raw)
    			TC_ubrmse_raw = ubrmsd(y_true_raw, y_TC_raw)    			
    			CFSR_ubrmse_raw = ubrmsd(y_true_raw, y_CFSR_raw)
    			ERAI_ubrmse_raw = ubrmsd(y_true_raw, y_ERAI_raw)
    			ERA5_ubrmse_raw = ubrmsd(y_true_raw, y_ERA5_raw)
    			JRA_ubrmse_raw = ubrmsd(y_true_raw, y_JRA_raw)
    			MERRA2_ubrmse_raw = ubrmsd(y_true_raw, y_MERRA2_raw)
    			GLDAS_ubrmse_raw = ubrmsd(y_true_raw, y_GLDAS_raw)


##### Anomalies #####
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
    			TC_rmse_anom = mean_squared_error(y_true_anom, y_TC_anom, squared=False)
    			CFSR_rmse_anom = mean_squared_error(y_true_anom, y_CFSR_anom, squared=False)
    			ERAI_rmse_anom = mean_squared_error(y_true_anom, y_ERAI_anom, squared=False)
    			ERA5_rmse_anom = mean_squared_error(y_true_anom, y_ERA5_anom, squared=False)
    			JRA_rmse_anom = mean_squared_error(y_true_anom, y_JRA_anom, squared=False)
    			MERRA2_rmse_anom = mean_squared_error(y_true_anom, y_MERRA2_anom, squared=False)
    			GLDAS_rmse_anom = mean_squared_error(y_true_anom, y_GLDAS_anom, squared=False)    			

    			naive_ubrmse_anom = ubrmsd(y_true_anom, y_naive_anom)
    			TC_ubrmse_anom = ubrmsd(y_true_anom, y_TC_anom)    			
    			CFSR_ubrmse_anom = ubrmsd(y_true_anom, y_CFSR_anom)
    			ERAI_ubrmse_anom = ubrmsd(y_true_anom, y_ERAI_anom)
    			ERA5_ubrmse_anom = ubrmsd(y_true_anom, y_ERA5_anom)
    			MERRA2_ubrmse_anom = ubrmsd(y_true_anom, y_MERRA2_anom)
    			JRA_ubrmse_anom = ubrmsd(y_true_anom, y_JRA_anom)
    			GLDAS_ubrmse_anom = ubrmsd(y_true_anom, y_GLDAS_anom)


    			print("RMSE:")
    			print(naive_rmse_anom, TC_rmse_anom)


    			print("ubRMSE:")
    			print(naive_ubrmse_anom,TC_ubrmse_anom)

################## Create Summary Statistics Dataframes ##############

    			raw_dict = { 'Dataset':['Naive Blended', 'TC Blended', 'CFSR', 'ERA-Interim', 'ERA5', 'JRA-55','MERRA2','GLDAS'],'Bias':[naive_bias_raw,TC_bias_raw,CFSR_bias_raw,ERAI_bias_raw,ERA5_bias_raw,JRA_bias_raw,MERRA2_bias_raw,GLDAS_bias_raw],'RMSE':[naive_rmse_raw,TC_rmse_raw,CFSR_rmse_raw,ERAI_rmse_raw,ERA5_rmse_raw,JRA_rmse_raw,MERRA2_rmse_raw,GLDAS_rmse_raw],'ubRMSE':[naive_ubrmse_raw,TC_ubrmse_raw,CFSR_ubrmse_raw,ERAI_ubrmse_raw,ERA5_ubrmse_raw,JRA_ubrmse_raw,MERRA2_ubrmse_raw,GLDAS_ubrmse_raw],'SDVnorm':[naive_SDV_raw,TC_SDV_raw,CFSR_SDV_raw,ERAI_SDV_raw,ERA5_SDV_raw,JRA_SDV_raw,MERRA2_SDV_raw,GLDAS_SDV_raw]}
    			df_summary_raw = pd.DataFrame(raw_dict)
    			raw_sum_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/blended_metrics/raw_temp/'+str(remap_type)+'_'+str(olri)+'_'+str(lyrj)+'_thr'+str(thrk)+'_summary_statistics.csv'])
    			print(raw_sum_fil)
    			path = pathlib.Path(raw_sum_fil)
    			path.parent.mkdir(parents=True, exist_ok=True)			
    			df_summary_raw.to_csv(raw_sum_fil,index=False)

    			anom_dict = { 'Dataset':['Naive Blended', 'TC Blended', 'CFSR', 'ERA-Interim', 'ERA5', 'JRA-55','MERRA2','GLDAS'],'Bias':[naive_bias_anom,TC_bias_anom,CFSR_bias_anom,ERAI_bias_anom,ERA5_bias_anom,JRA_bias_anom,MERRA2_bias_anom,GLDAS_bias_anom],'RMSE':[naive_rmse_anom,TC_rmse_anom,CFSR_rmse_anom,ERAI_rmse_anom,ERA5_rmse_anom,JRA_rmse_anom,MERRA2_rmse_anom,GLDAS_rmse_anom],'ubRMSE':[naive_ubrmse_anom,TC_ubrmse_anom,CFSR_ubrmse_anom,ERAI_ubrmse_anom,ERA5_ubrmse_anom,JRA_ubrmse_anom,MERRA2_ubrmse_anom,GLDAS_ubrmse_anom],'SDVnorm':[naive_SDV_anom,TC_SDV_anom,CFSR_SDV_anom,ERAI_SDV_anom,ERA5_SDV_anom,JRA_SDV_anom,MERRA2_SDV_anom,GLDAS_SDV_anom]}
    			df_summary_anom = pd.DataFrame(anom_dict)
    			anom_sum_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/blended_metrics/anom/'+str(remap_type)+'_'+str(olri)+'_'+str(lyrj)+'_thr'+str(thrk)+'_summary_statistics_anom.csv'])
    			print(anom_sum_fil)
    			path2 = pathlib.Path(raw_sum_fil)
    			path2.parent.mkdir(parents=True, exist_ok=True)			
    			df_summary_anom.to_csv(anom_sum_fil,index=False)





################## Correlation Matrix ###################

##### Raw Temp #####

    			df_corr_raw = pd.DataFrame(data=station_raw_temp, columns=['Station'])
    			df_corr_raw['Naive Blended'] = naive_raw_temp
    			df_corr_raw['TC Blended'] = TC_raw_temp		
    			df_corr_raw['CFSR'] = CFSR_raw_temp
    			df_corr_raw['ERA-Interim'] = ERAI_raw_temp
    			df_corr_raw['ERA5'] = ERA5_raw_temp
    			df_corr_raw['JRA-55'] = JRA_raw_temp
    			df_corr_raw['MERRA2'] = MERRA2_raw_temp
    			df_corr_raw['GLDAS'] = GLDAS_raw_temp
    			df_corr_raw['Date'] = DateTime
    			df_corr_raw = df_corr_raw.set_index('Date')

### Seasonal ###

    			df_corr_raw_DJF = df_corr_raw[(df_corr_raw.index.month == 12) | (df_corr_raw.index.month == 1) | (df_corr_raw.index.month == 2)]
    			station_DJF_r = df_corr_raw_DJF['Station'].values.tolist()
    			Naive_DJF_r = df_corr_raw_DJF['Naive Blended'].values.tolist()
    			TC_DJF_r = df_corr_raw_DJF['TC Blended'].values.tolist()
    			CFSR_DJF_r = df_corr_raw_DJF['CFSR'].values.tolist()
    			ERAI_DJF_r = df_corr_raw_DJF['ERA-Interim'].values.tolist()
    			ERA5_DJF_r = df_corr_raw_DJF['ERA5'].values.tolist()
    			JRA_DJF_r = df_corr_raw_DJF['JRA-55'].values.tolist()
    			MERRA2_DJF_r = df_corr_raw_DJF['MERRA2'].values.tolist()
    			GLDAS_DJF_r = df_corr_raw_DJF['GLDAS'].values.tolist()

    			data_corr_raw_DJF = {'Station':station_DJF_r,'Naive Blend':Naive_DJF_r,'TC Blend':TC_DJF_r,'CFSR':CFSR_DJF_r,'ERA-Interim':ERAI_DJF_r,'ERA5':ERA5_DJF_r,'JRA-55':JRA_DJF_r,'MERRA2':MERRA2_DJF_r,'GLDAS':GLDAS_DJF_r}
    			df_corr_raw_DJF = pd.DataFrame(data_corr_raw_DJF,columns=['Station','Naive Blend','TC Blend','CFSR','ERA-Interim','ERA5','JRA-55','MERRA2','GLDAS'])
    			corrMatrix_raw_DJF = df_corr_raw_DJF.corr()

    			df_corr_raw_MAM = df_corr_raw[(df_corr_raw.index.month == 3) | (df_corr_raw.index.month == 4) | (df_corr_raw.index.month == 5)]
    			station_MAM_r = df_corr_raw_MAM['Station'].values.tolist()
    			Naive_MAM_r = df_corr_raw_MAM['Naive Blended'].values.tolist()
    			TC_MAM_r = df_corr_raw_MAM['TC Blended'].values.tolist()
    			CFSR_MAM_r = df_corr_raw_MAM['CFSR'].values.tolist()
    			ERAI_MAM_r = df_corr_raw_MAM['ERA-Interim'].values.tolist()
    			ERA5_MAM_r = df_corr_raw_MAM['ERA5'].values.tolist()
    			JRA_MAM_r = df_corr_raw_MAM['JRA-55'].values.tolist()
    			MERRA2_MAM_r = df_corr_raw_MAM['MERRA2'].values.tolist()
    			GLDAS_MAM_r = df_corr_raw_MAM['GLDAS'].values.tolist()

    			data_corr_raw_MAM = {'Station':station_MAM_r,'Naive Blend':Naive_MAM_r,'TC Blend':TC_MAM_r,'CFSR':CFSR_MAM_r,'ERA-Interim':ERAI_MAM_r,'ERA5':ERA5_MAM_r,'JRA-55':JRA_MAM_r,'MERRA2':MERRA2_MAM_r,'GLDAS':GLDAS_MAM_r}
    			df_corr_raw_MAM = pd.DataFrame(data_corr_raw_MAM,columns=['Station','Naive Blend','TC Blend','CFSR','ERA-Interim','ERA5','JRA-55','MERRA2','GLDAS'])
    			corrMatrix_raw_MAM = df_corr_raw_MAM.corr()

    			df_corr_raw_JJA = df_corr_raw[(df_corr_raw.index.month == 6) | (df_corr_raw.index.month == 7) | (df_corr_raw.index.month == 8)]
    			station_JJA_r = df_corr_raw_JJA['Station'].values.tolist()
    			Naive_JJA_r = df_corr_raw_JJA['Naive Blended'].values.tolist()
    			TC_JJA_r = df_corr_raw_JJA['TC Blended'].values.tolist()
    			CFSR_JJA_r = df_corr_raw_JJA['CFSR'].values.tolist()
    			ERAI_JJA_r = df_corr_raw_JJA['ERA-Interim'].values.tolist()
    			ERA5_JJA_r = df_corr_raw_JJA['ERA5'].values.tolist()
    			JRA_JJA_r = df_corr_raw_JJA['JRA-55'].values.tolist()
    			MERRA2_JJA_r = df_corr_raw_JJA['MERRA2'].values.tolist()
    			GLDAS_JJA_r = df_corr_raw_JJA['GLDAS'].values.tolist()

    			data_corr_raw_JJA = {'Station':station_JJA_r,'Naive Blend':Naive_JJA_r,'TC Blend':TC_JJA_r,'CFSR':CFSR_JJA_r,'ERA-Interim':ERAI_JJA_r,'ERA5':ERA5_JJA_r,'JRA-55':JRA_JJA_r,'MERRA2':MERRA2_JJA_r,'GLDAS':GLDAS_JJA_r}
    			df_corr_raw_JJA = pd.DataFrame(data_corr_raw_JJA,columns=['Station','Naive Blend','TC Blend','CFSR','ERA-Interim','ERA5','JRA-55','MERRA2','GLDAS'])
    			corrMatrix_raw_JJA = df_corr_raw_JJA.corr() 

    			df_corr_raw_SON = df_corr_raw[(df_corr_raw.index.month == 9) | (df_corr_raw.index.month == 10) | (df_corr_raw.index.month == 11)]
    			station_SON_r = df_corr_raw_SON['Station'].values.tolist()
    			Naive_SON_r = df_corr_raw_SON['Naive Blended'].values.tolist()
    			TC_SON_r = df_corr_raw_SON['TC Blended'].values.tolist()
    			CFSR_SON_r = df_corr_raw_SON['CFSR'].values.tolist()
    			ERAI_SON_r = df_corr_raw_SON['ERA-Interim'].values.tolist()
    			ERA5_SON_r = df_corr_raw_SON['ERA5'].values.tolist()
    			JRA_SON_r = df_corr_raw_SON['JRA-55'].values.tolist()
    			MERRA2_SON_r = df_corr_raw_SON['MERRA2'].values.tolist()
    			GLDAS_SON_r = df_corr_raw_SON['GLDAS'].values.tolist()

    			data_corr_raw_SON = {'Station':station_SON_r,'Naive Blend':Naive_SON_r,'TC Blend':TC_SON_r,'CFSR':CFSR_SON_r,'ERA-Interim':ERAI_SON_r,'ERA5':ERA5_SON_r,'JRA-55':JRA_SON_r,'MERRA2':MERRA2_SON_r,'GLDAS':GLDAS_SON_r}
    			df_corr_raw_SON = pd.DataFrame(data_corr_raw_SON,columns=['Station','Naive Blend','TC Blend','CFSR','ERA-Interim','ERA5','JRA-55','MERRA2','GLDAS'])
    			corrMatrix_raw_SON = df_corr_raw_SON.corr()

##### Anomalies #####

    			df_corr_anom = pd.DataFrame(data=station_anom, columns=['Station'])
    			df_corr_anom['Naive Blended'] = naive_anom
    			df_corr_anom['TC Blended'] = TC_anom	
    			df_corr_anom['CFSR'] = CFSR_anom
    			df_corr_anom['ERA-Interim'] = ERAI_anom
    			df_corr_anom['ERA5'] = ERA5_anom
    			df_corr_anom['JRA-55'] = JRA_anom
    			df_corr_anom['MERRA2'] = MERRA2_anom
    			df_corr_anom['GLDAS'] = GLDAS_anom
    			df_corr_anom['Date'] = DateTime
    			df_corr_anom = df_corr_anom.set_index('Date')

### Seasonal ###

    			df_corr_anom_DJF = df_corr_anom[(df_corr_anom.index.month == 12) | (df_corr_anom.index.month == 1) | (df_corr_anom.index.month == 2)]
    			station_DJF_a = df_corr_anom_DJF['Station'].values.tolist()
    			Naive_DJF_a = df_corr_anom_DJF['Naive Blended'].values.tolist()
    			TC_DJF_a = df_corr_anom_DJF['TC Blended'].values.tolist()
    			CFSR_DJF_a = df_corr_anom_DJF['CFSR'].values.tolist()
    			ERAI_DJF_a = df_corr_anom_DJF['ERA-Interim'].values.tolist()
    			ERA5_DJF_a = df_corr_anom_DJF['ERA5'].values.tolist()
    			JRA_DJF_a = df_corr_anom_DJF['JRA-55'].values.tolist()
    			MERRA2_DJF_a = df_corr_anom_DJF['MERRA2'].values.tolist()
    			GLDAS_DJF_a = df_corr_anom_DJF['GLDAS'].values.tolist()

    			data_corr_anom_DJF = {'Station':station_DJF_a,'Naive Blend':Naive_DJF_a,'TC Blend':TC_DJF_a,'CFSR':CFSR_DJF_a,'ERA-Interim':ERAI_DJF_a,'ERA5':ERA5_DJF_a,'JRA-55':JRA_DJF_a,'MERRA2':MERRA2_DJF_a,'GLDAS':GLDAS_DJF_a}
    			df_corr_anom_DJF = pd.DataFrame(data_corr_anom_DJF,columns=['Station','Naive Blend','TC Blend','CFSR','ERA-Interim','ERA5','JRA-55','MERRA2','GLDAS'])
    			corrMatrix_anom_DJF = df_corr_anom_DJF.corr()

    			df_corr_anom_MAM = df_corr_anom[(df_corr_anom.index.month == 3) | (df_corr_anom.index.month == 4) | (df_corr_anom.index.month == 5)]
    			station_MAM_a = df_corr_anom_MAM['Station'].values.tolist()
    			Naive_MAM_a = df_corr_anom_MAM['Naive Blended'].values.tolist()
    			TC_MAM_a = df_corr_anom_MAM['TC Blended'].values.tolist()
    			CFSR_MAM_a = df_corr_anom_MAM['CFSR'].values.tolist()
    			ERAI_MAM_a = df_corr_anom_MAM['ERA-Interim'].values.tolist()
    			ERA5_MAM_a = df_corr_anom_MAM['ERA5'].values.tolist()
    			JRA_MAM_a = df_corr_anom_MAM['JRA-55'].values.tolist()
    			MERRA2_MAM_a = df_corr_anom_MAM['MERRA2'].values.tolist()
    			GLDAS_MAM_a = df_corr_anom_MAM['GLDAS'].values.tolist()

    			data_corr_anom_MAM = {'Station':station_MAM_a,'Naive Blend':Naive_MAM_a,'TC Blend':TC_MAM_a,'CFSR':CFSR_MAM_a,'ERA-Interim':ERAI_MAM_a,'ERA5':ERA5_MAM_a,'JRA-55':JRA_MAM_a,'MERRA2':MERRA2_MAM_a,'GLDAS':GLDAS_MAM_a}
    			df_corr_anom_MAM = pd.DataFrame(data_corr_anom_MAM,columns=['Station','Naive Blend','TC Blend','CFSR','ERA-Interim','ERA5','JRA-55','MERRA2','GLDAS'])
    			corrMatrix_anom_MAM = df_corr_anom_MAM.corr()

    			df_corr_anom_JJA = df_corr_anom[(df_corr_anom.index.month == 6) | (df_corr_anom.index.month == 7) | (df_corr_anom.index.month == 8)]
    			station_JJA_a = df_corr_anom_JJA['Station'].values.tolist()
    			Naive_JJA_a = df_corr_anom_JJA['Naive Blended'].values.tolist()
    			TC_JJA_a = df_corr_anom_JJA['TC Blended'].values.tolist()
    			CFSR_JJA_a = df_corr_anom_JJA['CFSR'].values.tolist()
    			ERAI_JJA_a = df_corr_anom_JJA['ERA-Interim'].values.tolist()
    			ERA5_JJA_a = df_corr_anom_JJA['ERA5'].values.tolist()
    			JRA_JJA_a = df_corr_anom_JJA['JRA-55'].values.tolist()
    			MERRA2_JJA_a = df_corr_anom_JJA['MERRA2'].values.tolist()
    			GLDAS_JJA_a = df_corr_anom_JJA['GLDAS'].values.tolist()

    			data_corr_anom_JJA = {'Station':station_JJA_a,'Naive Blend':Naive_JJA_a,'TC Blend':TC_JJA_a,'CFSR':CFSR_JJA_a,'ERA-Interim':ERAI_JJA_a,'ERA5':ERA5_JJA_a,'JRA-55':JRA_JJA_a,'MERRA2':MERRA2_JJA_a,'GLDAS':GLDAS_JJA_a}
    			df_corr_anom_JJA = pd.DataFrame(data_corr_anom_JJA,columns=['Station','Naive Blend','TC Blend','CFSR','ERA-Interim','ERA5','JRA-55','MERRA2','GLDAS'])
    			corrMatrix_anom_JJA = df_corr_anom_JJA.corr()

    			df_corr_anom_SON = df_corr_anom[(df_corr_anom.index.month == 9) | (df_corr_anom.index.month == 10) | (df_corr_anom.index.month == 11)]
    			station_SON_a = df_corr_anom_SON['Station'].values.tolist()
    			Naive_SON_a = df_corr_anom_SON['Naive Blended'].values.tolist()
    			TC_SON_a = df_corr_anom_SON['TC Blended'].values.tolist()
    			CFSR_SON_a = df_corr_anom_SON['CFSR'].values.tolist()
    			ERAI_SON_a = df_corr_anom_SON['ERA-Interim'].values.tolist()
    			ERA5_SON_a = df_corr_anom_SON['ERA5'].values.tolist()
    			JRA_SON_a = df_corr_anom_SON['JRA-55'].values.tolist()
    			MERRA2_SON_a = df_corr_anom_SON['MERRA2'].values.tolist()
    			GLDAS_SON_a = df_corr_anom_SON['GLDAS'].values.tolist()

    			data_corr_anom_SON = {'Station':station_SON_a,'Naive Blend':Naive_SON_a,'TC Blend':TC_SON_a,'CFSR':CFSR_SON_a,'ERA-Interim':ERAI_SON_a,'ERA5':ERA5_SON_a,'JRA-55':JRA_SON_a,'MERRA2':MERRA2_SON_a,'GLDAS':GLDAS_SON_a}
    			df_corr_anom_SON = pd.DataFrame(data_corr_anom_SON,columns=['Station','Naive Blend','TC Blend','CFSR','ERA-Interim','ERA5','JRA-55','MERRA2','GLDAS'])
    			corrMatrix_anom_SON = df_corr_anom_SON.corr()

    			#print(corrMatrix_anom_SON) 			

######## Create correlation figures #########

##### Raw Temp #####

    			corr_fil_raw = ''.join(['/mnt/data/users/herringtont/soil_temp/blended_metrics/corr_matrix_seasonal/raw_temp/'+str(remap_type)+'_'+str(olri)+'_'+str(lyrj)+'_thr'+str(thrk)+'_corr_matrix_seasonal.png'])
    			print(corr_fil_raw)

    			fig,axes = plt.subplots(2,2, figsize=(20,20), sharey=True)
    			fig.suptitle('Pearsons R Correlation Matrix',fontweight='bold',fontsize='large')
    			plt.subplot(221)
    			corrDJFr = sn.heatmap(corrMatrix_raw_DJF, annot=True, square=True, vmin=0, vmax=1).set_title('DJF Correlations')
    			plt.subplot(222)
    			corrMAMr = sn.heatmap(corrMatrix_raw_MAM, annot=True, square=True, vmin=0, vmax=1).set_title('MAM Correlations')
    			plt.subplot(223)
    			corrJJAr = sn.heatmap(corrMatrix_raw_JJA, annot=True, square=True, vmin=0, vmax=1).set_title('JJA Correlations')
    			plt.subplot(224)
    			corrSONr = sn.heatmap(corrMatrix_raw_SON, annot=True, square=True, vmin=0, vmax=1).set_title('SON Correlations')
    			plt.savefig(corr_fil_raw)
    			plt.close()

##### Anom #####

    			corr_fil_anom = ''.join(['/mnt/data/users/herringtont/soil_temp/blended_metrics/corr_matrix_seasonal/anom/'+str(remap_type)+'_'+str(olri)+'_'+str(lyrj)+'_thr'+str(thrk)+'_corr_matrix_seasonal_anom.png'])
    			print(corr_fil_anom)

    			fig,axes = plt.subplots(2,2, figsize=(20,20), sharey=True)
    			fig.suptitle('Pearsons R Correlation Matrix',fontweight='bold',fontsize='large')
    			plt.subplot(221)
    			corrDJFa = sn.heatmap(corrMatrix_anom_DJF, annot=True, square=True, vmin=0, vmax=1).set_title('DJF Correlations')
    			plt.subplot(222)
    			corrMAMa = sn.heatmap(corrMatrix_anom_MAM, annot=True, square=True, vmin=0, vmax=1).set_title('MAM Correlations')
    			plt.subplot(223)
    			corrJJAa = sn.heatmap(corrMatrix_anom_JJA, annot=True, square=True, vmin=0, vmax=1).set_title('JJA Correlations')
    			plt.subplot(224)
    			corrSONa = sn.heatmap(corrMatrix_anom_SON, annot=True, square=True, vmin=0, vmax=1).set_title('SON Correlations')
    			plt.savefig(corr_fil_anom)
    			plt.close()
