import os
import glob
import netCDF4
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
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from dateutil.relativedelta import *



#/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/by_date/remapnn_simple_average_top_30cm_thr_75_warm_season_date_summary.csv
#/home/herringtont/anaconda3/envs/SoilTemp/lib/python3.8/site-packages/numpy/core/_methods.py:233: RuntimeWarning: Degrees of freedom <= 0 for slice
#  ret = _var(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
#/home/herringtont/anaconda3/envs/SoilTemp/lib/python3.8/site-packages/numpy/core/_methods.py:226: RuntimeWarning: invalid value encountered in double_scalars
#  ret = ret.dtype.type(ret / rcount)
#Traceback (most recent call last):
#  File "naive_blended_taylor_diagram.py", line 803, in <module>
#    naive_corr_cold_temp, _ = scipy.stats.pearsonr(station_cold_temp,naive_cold_temp)
#  File "/home/herringtont/anaconda3/envs/SoilTemp/lib/python3.8/site-packages/scipy/stats/stats.py", line 3501, in pearsonr
#    raise ValueError('x and y must have length at least 2.')
#ValueError: x and y must have length at least 2.






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


############# Set Directories ############

naive_type = ['simple_average']
olr = ['zscore']#['outliers','zscore','IQR']
lyr = ['top_30cm','30cm_100cm','100cm_300cm']
thr = ['100']#['0','25','50','75','100']
rmp_type = ['nn','bil']
temp_thr = ['0C','-2C','-5C','-10C']

############# Grab Data ##############

for i in rmp_type:
    rmp_type_i = i
    remap_type = ''.join(['remap'+rmp_type_i])
    
    for j in naive_type:
    	naive_type_j = j

    	for k in olr:
    		olr_k = k

    		for l in lyr:
    			lyr_l = l

    			for m in thr:
    				thr_m = m
    
    				for n in temp_thr:
    					temp_thr_n = n
    					cold_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/common_grid/'+str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_dframe_cold_season_temp_master_ERA5_'+str(temp_thr_n)+'_common_grid.csv'])
    					warm_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/common_grid/'+str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_dframe_warm_season_temp_master_ERA5_'+str(temp_thr_n)+'_common_grid.csv'])    			
    					scatter_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/common_grid/'+str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_dframe_scatterplot_ERA5_'+str(temp_thr_n)+'_common_grid.csv'])


    					dframe_cold = pd.read_csv(cold_fil)
    					station_cold = dframe_cold['Station'].values
    					station_cold = dframe_cold['Naive Blend'].values
    					CFSR_cold = dframe_cold['CFSR'].values
    					ERAI_cold = dframe_cold['ERA-Interim'].values
    					ERA5_cold = dframe_cold['ERA5'].values
    					ERA5_Land_cold = dframe_cold['ERA5-Land'].values
    					JRA_cold = dframe_cold['JRA55'].values
    					MERRA2_cold = dframe_cold['MERRA2'].values
    					GLDAS_cold = dframe_cold['GLDAS-Noah'].values
    					GLDAS_CLSM_cold = dframe_cold['GLDAS-CLSM'].values

    					dframe_warm = pd.read_csv(warm_fil)
    					station_warm = dframe_warm['Station'].values
    					station_warm = dframe_warm['Naive Blend'].values
    					CFSR_warm = dframe_warm['CFSR'].values
    					ERAI_warm = dframe_warm['ERA-Interim'].values
    					ERA5_warm = dframe_warm['ERA5'].values
    					ERA5_Land_warm = dframe_warm['ERA5-Land'].values
    					JRA_warm = dframe_warm['JRA55'].values
    					MERRA2_warm = dframe_warm['MERRA2'].values
    					GLDAS_warm = dframe_warm['GLDAS-Noah'].values
    					GLDAS_CLSM_warm = dframe_warm['GLDAS-CLSM'].values

    					dframe_scatter = pd.read_csv(scatter_fil)
    					station_scatter = dframe_scatter['Station'].values
    					station_scatter = dframe_scatter['Naive Blend'].values
    					CFSR_scatter = dframe_scatter['CFSR'].values
    					ERAI_scatter = dframe_scatter['ERA-Interim'].values
    					ERA5_scatter = dframe_scatter['ERA5'].values
    					ERA5_Land_scatter = dframe_scatter['ERA5-Land'].values
    					JRA_scatter = dframe_scatter['JRA55'].values
    					MERRA2_scatter = dframe_scatter['MERRA2'].values
    					GLDAS_scatter = dframe_scatter['GLDAS-Noah'].values
    					GLDAS_CLSM_scatter = dframe_scatter['GLDAS-CLSM'].values


###################### Calculate Correlation Matrices #########################

    					fig,axs = plt.subplots(nrows=1, ncols=2, sharex='col', sharey='row', figsize=(15,20))
    					cbar_kws = {"orientation":"horizontal","shrink":1}
###### Cold Season (Temp) #######

    					dframe_cold_season_temp_corr = dframe_cold[['Station','Naive Blend','CFSR','ERA-Interim','ERA5','ERA5-Land','JRA55','MERRA2','GLDAS-Noah','GLDAS-CLSM']]
    					cold_season_temp_corrMatrix = dframe_cold_season_temp_corr.corr()

    					ax1 = plt.subplot(121)
    					corr1 = sn.heatmap(cold_season_temp_corrMatrix,annot=True,square='True',vmin=0,vmax=1,cbar_kws=cbar_kws)
    					ax1.set_title('Cold Season Correlation Matrix')

###### Warm Season (Temp) #######
    					dframe_warm_season_temp_corr = dframe_warm[['Station','Naive Blend','CFSR','ERA-Interim','ERA5','ERA5-Land','JRA55','MERRA2','GLDAS-Noah','GLDAS-CLSM']]
    					warm_season_temp_corrMatrix = dframe_warm_season_temp_corr.corr()

    					ax2 = plt.subplot(122)
    					corr2 = sn.heatmap(warm_season_temp_corrMatrix,annot=True,square='True',vmin=0,vmax=1,cbar_kws=cbar_kws)
    					ax2.set_title('Warm Season Correlation Matrix')

    					plt.savefig('/mnt/data/users/herringtont/soil_temp/plots/naive_blend_correlation/common_grid/'+str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_correlation_byseason_ERA5_'+str(temp_thr_n)+'_common_grid.png')
    					plt.close()


###################### Create Scatterplot Matrices ####################

###### Cold Season (Temp) ######
    					scatter1 = sn.pairplot(dframe_scatter, hue='Season')
    					for ax in scatter1.axes.flat:
    						ax.set_xlim(-40,30)
    						ax.set_ylim(-40,30)
    					plt.savefig('/mnt/data/users/herringtont/soil_temp/plots/naive_blend_scatterplots/common_grid/'+str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_scatterplot_all_temp_ERA5_'+str(temp_thr_n)+'_common_grid.png')
    					plt.close()		



##################### Create Taylor Diagrams ####################

###### Cold Season (Temp) #####
    					station_cold_season_temp = dframe_cold['Station'].values
    					naive_cold_season_temp = dframe_cold['Naive Blend'].values
    					CFSR_cold_season_temp = dframe_cold['CFSR'].values    				
    					ERAI_cold_season_temp = dframe_cold['ERA-Interim'].values
    					ERA5_cold_season_temp = dframe_cold['ERA5'].values
    					ERA5_Land_cold_season_temp = dframe_cold['ERA5-Land'].values
    					JRA_cold_season_temp = dframe_cold['JRA55'].values
    					MERRA2_cold_season_temp = dframe_cold['MERRA2'].values
    					GLDAS_cold_season_temp = dframe_cold['GLDAS-Noah'].values
    					GLDAS_CLSM_cold_season_temp = dframe_cold['GLDAS-CLSM'].values

    					taylor_stats_naive_cold_temp = sm.taylor_statistics(naive_cold_season_temp,station_cold_season_temp)
    					taylor_stats_CFSR_cold_temp = sm.taylor_statistics(CFSR_cold_season_temp,station_cold_season_temp)			
    					taylor_stats_ERAI_cold_temp = sm.taylor_statistics(ERAI_cold_season_temp,station_cold_season_temp)
    					taylor_stats_ERA5_cold_temp = sm.taylor_statistics(ERA5_cold_season_temp,station_cold_season_temp)
    					taylor_stats_ERA5_Land_cold_temp = sm.taylor_statistics(ERA5_Land_cold_season_temp,station_cold_season_temp)
    					taylor_stats_JRA_cold_temp = sm.taylor_statistics(JRA_cold_season_temp,station_cold_season_temp)
    					taylor_stats_MERRA2_cold_temp = sm.taylor_statistics(MERRA2_cold_season_temp,station_cold_season_temp)
    					taylor_stats_GLDAS_cold_temp = sm.taylor_statistics(GLDAS_cold_season_temp,station_cold_season_temp)
    					taylor_stats_GLDAS_CLSM_cold_temp = sm.taylor_statistics(GLDAS_CLSM_cold_season_temp,station_cold_season_temp)


    					print('Remap Style:',remap_type)
    					print('Layer:',lyr_l)
    					print('Temp Threshold:',temp_thr_n)


    					taylor_dir = '/mnt/data/users/herringtont/soil_temp/plots/taylor_diagrams/common_grid/'
    					taylor_plt_fil_cold = ''.join([taylor_dir,str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_ERA5_'+str(temp_thr_n)+'_taylor_diagram_cold_common_grid.png'])



    					sdev_naive = np.array([taylor_stats_naive_cold_temp['sdev'][0],taylor_stats_naive_cold_temp['sdev'][1]])
    					crmsd_naive = np.array([taylor_stats_naive_cold_temp['crmsd'][0],taylor_stats_naive_cold_temp['crmsd'][1]])
    					ccoef_naive = np.array([taylor_stats_naive_cold_temp['ccoef'][0],taylor_stats_naive_cold_temp['ccoef'][1]])

    					sdev_CFSR = np.array([taylor_stats_CFSR_cold_temp['sdev'][0],taylor_stats_CFSR_cold_temp['sdev'][1]])
    					crmsd_CFSR = np.array([taylor_stats_CFSR_cold_temp['crmsd'][0],taylor_stats_CFSR_cold_temp['crmsd'][1]])
    					ccoef_CFSR = np.array([taylor_stats_CFSR_cold_temp['ccoef'][0],taylor_stats_CFSR_cold_temp['ccoef'][1]])

    					sdev_ERAI = np.array([taylor_stats_ERAI_cold_temp['sdev'][0],taylor_stats_ERAI_cold_temp['sdev'][1]])
    					crmsd_ERAI = np.array([taylor_stats_ERAI_cold_temp['crmsd'][0],taylor_stats_ERAI_cold_temp['crmsd'][1]])
    					ccoef_ERAI = np.array([taylor_stats_ERAI_cold_temp['ccoef'][0],taylor_stats_ERAI_cold_temp['ccoef'][1]])

    					sdev_ERA5 = np.array([taylor_stats_ERA5_cold_temp['sdev'][0],taylor_stats_ERA5_cold_temp['sdev'][1]])
    					crmsd_ERA5 = np.array([taylor_stats_ERA5_cold_temp['crmsd'][0],taylor_stats_ERA5_cold_temp['crmsd'][1]])
    					ccoef_ERA5 = np.array([taylor_stats_ERA5_cold_temp['ccoef'][0],taylor_stats_ERA5_cold_temp['ccoef'][1]])

    					sdev_ERA5_Land = np.array([taylor_stats_ERA5_Land_cold_temp['sdev'][0],taylor_stats_ERA5_Land_cold_temp['sdev'][1]])
    					crmsd_ERA5_Land = np.array([taylor_stats_ERA5_Land_cold_temp['crmsd'][0],taylor_stats_ERA5_Land_cold_temp['crmsd'][1]])
    					ccoef_ERA5_Land = np.array([taylor_stats_ERA5_Land_cold_temp['ccoef'][0],taylor_stats_ERA5_Land_cold_temp['ccoef'][1]])

    					sdev_JRA = np.array([taylor_stats_JRA_cold_temp['sdev'][0],taylor_stats_JRA_cold_temp['sdev'][1]])
    					crmsd_JRA = np.array([taylor_stats_JRA_cold_temp['crmsd'][0],taylor_stats_JRA_cold_temp['crmsd'][1]])
    					ccoef_JRA = np.array([taylor_stats_JRA_cold_temp['ccoef'][0],taylor_stats_JRA_cold_temp['ccoef'][1]])

    					sdev_MERRA2 = np.array([taylor_stats_MERRA2_cold_temp['sdev'][0],taylor_stats_MERRA2_cold_temp['sdev'][1]])
    					crmsd_MERRA2 = np.array([taylor_stats_MERRA2_cold_temp['crmsd'][0],taylor_stats_MERRA2_cold_temp['crmsd'][1]])
    					ccoef_MERRA2 = np.array([taylor_stats_MERRA2_cold_temp['ccoef'][0],taylor_stats_MERRA2_cold_temp['ccoef'][1]])

    					sdev_GLDAS = np.array([taylor_stats_GLDAS_cold_temp['sdev'][0],taylor_stats_GLDAS_cold_temp['sdev'][1]])
    					crmsd_GLDAS = np.array([taylor_stats_GLDAS_cold_temp['crmsd'][0],taylor_stats_GLDAS_cold_temp['crmsd'][1]])
    					ccoef_GLDAS = np.array([taylor_stats_GLDAS_cold_temp['ccoef'][0],taylor_stats_GLDAS_cold_temp['ccoef'][1]])

    					sdev_GLDAS_CLSM = np.array([taylor_stats_GLDAS_CLSM_cold_temp['sdev'][0],taylor_stats_GLDAS_CLSM_cold_temp['sdev'][1]])
    					crmsd_GLDAS_CLSM = np.array([taylor_stats_GLDAS_CLSM_cold_temp['crmsd'][0],taylor_stats_GLDAS_CLSM_cold_temp['crmsd'][1]])
    					ccoef_GLDAS_CLSM = np.array([taylor_stats_GLDAS_CLSM_cold_temp['ccoef'][0],taylor_stats_GLDAS_CLSM_cold_temp['ccoef'][1]])

    					label = {'Station':'dimgrey','Naive Blend': 'dodgerblue', 'CFSR': 'm','ERA-Interim': 'g', 'ERA5': 'c','ERA5-Land': 'lightblue', 'JRA55': 'r', 'MERRA2': 'y', 'GLDAS-Noah': 'k','GLDAS-CLSM': 'gainsboro'}
								
    					sm.taylor_diagram(sdev_naive,crmsd_naive,ccoef_naive,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 110.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5,colOBS = 'dimgrey',markerobs = 'o',titleOBS = 'Station', markercolor='dodgerblue')
    					sm.taylor_diagram(sdev_CFSR,crmsd_CFSR,ccoef_CFSR,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 110.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='m',overlay='on')
    					sm.taylor_diagram(sdev_ERAI,crmsd_ERAI,ccoef_ERAI,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 110.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='g',overlay='on')
    					sm.taylor_diagram(sdev_ERA5,crmsd_ERA5,ccoef_ERA5,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 110.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='c',overlay='on')
    					sm.taylor_diagram(sdev_ERA5_Land,crmsd_ERA5_Land,ccoef_ERA5_Land,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 110.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='lightblue',overlay='on')
    					sm.taylor_diagram(sdev_JRA,crmsd_JRA,ccoef_JRA,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 70.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='r',overlay='on')
    					sm.taylor_diagram(sdev_MERRA2,crmsd_MERRA2,ccoef_MERRA2,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 110.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='y',overlay='on')
    					sm.taylor_diagram(sdev_GLDAS,crmsd_GLDAS,ccoef_GLDAS,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 110.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='k',overlay='on')
    					sm.taylor_diagram(sdev_GLDAS_CLSM,crmsd_GLDAS_CLSM,ccoef_GLDAS_CLSM,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 110.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='gainsboro',overlay='on',markerLabel=label)
    					plt.savefig(taylor_plt_fil_cold)
    					plt.close()
    					print(taylor_plt_fil_cold)


###### Warm Season (Temp) #####


    					station_warm_season_temp = dframe_warm['Station'].values
    					naive_warm_season_temp = dframe_warm['Naive Blend'].values
    					CFSR_warm_season_temp = dframe_warm['CFSR'].values    				
    					ERAI_warm_season_temp = dframe_warm['ERA-Interim'].values
    					ERA5_warm_season_temp = dframe_warm['ERA5'].values
    					ERA5_Land_warm_season_temp = dframe_warm['ERA5-Land'].values
    					JRA_warm_season_temp = dframe_warm['JRA55'].values
    					MERRA2_warm_season_temp = dframe_warm['MERRA2'].values
    					GLDAS_warm_season_temp = dframe_warm['GLDAS-Noah'].values
    					GLDAS_CLSM_warm_season_temp = dframe_warm['GLDAS-CLSM'].values

    					taylor_stats_naive_warm_temp = sm.taylor_statistics(naive_warm_season_temp,station_warm_season_temp)
    					taylor_stats_CFSR_warm_temp = sm.taylor_statistics(CFSR_warm_season_temp,station_warm_season_temp)			
    					taylor_stats_ERAI_warm_temp = sm.taylor_statistics(ERAI_warm_season_temp,station_warm_season_temp)
    					taylor_stats_ERA5_warm_temp = sm.taylor_statistics(ERA5_warm_season_temp,station_warm_season_temp)
    					taylor_stats_ERA5_Land_warm_temp = sm.taylor_statistics(ERA5_Land_warm_season_temp,station_warm_season_temp)
    					taylor_stats_JRA_warm_temp = sm.taylor_statistics(JRA_warm_season_temp,station_warm_season_temp)
    					taylor_stats_MERRA2_warm_temp = sm.taylor_statistics(MERRA2_warm_season_temp,station_warm_season_temp)
    					taylor_stats_GLDAS_warm_temp = sm.taylor_statistics(GLDAS_warm_season_temp,station_warm_season_temp)
    					taylor_stats_GLDAS_CLSM_warm_temp = sm.taylor_statistics(GLDAS_CLSM_warm_season_temp,station_warm_season_temp)


    					taylor_dir = '/mnt/data/users/herringtont/soil_temp/plots/taylor_diagrams/common_grid/'
    					taylor_plt_fil_warm = ''.join([taylor_dir,str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_ERA5_'+str(temp_thr_n)+'_taylor_diagram_warm_common_grid.png'])



    					sdev_naive = np.array([taylor_stats_naive_warm_temp['sdev'][0],taylor_stats_naive_warm_temp['sdev'][1]])
    					crmsd_naive = np.array([taylor_stats_naive_warm_temp['crmsd'][0],taylor_stats_naive_warm_temp['crmsd'][1]])
    					ccoef_naive = np.array([taylor_stats_naive_warm_temp['ccoef'][0],taylor_stats_naive_warm_temp['ccoef'][1]])

    					sdev_CFSR = np.array([taylor_stats_CFSR_warm_temp['sdev'][0],taylor_stats_CFSR_warm_temp['sdev'][1]])
    					crmsd_CFSR = np.array([taylor_stats_CFSR_warm_temp['crmsd'][0],taylor_stats_CFSR_warm_temp['crmsd'][1]])
    					ccoef_CFSR = np.array([taylor_stats_CFSR_warm_temp['ccoef'][0],taylor_stats_CFSR_warm_temp['ccoef'][1]])

    					sdev_ERAI = np.array([taylor_stats_ERAI_warm_temp['sdev'][0],taylor_stats_ERAI_warm_temp['sdev'][1]])
    					crmsd_ERAI = np.array([taylor_stats_ERAI_warm_temp['crmsd'][0],taylor_stats_ERAI_warm_temp['crmsd'][1]])
    					ccoef_ERAI = np.array([taylor_stats_ERAI_warm_temp['ccoef'][0],taylor_stats_ERAI_warm_temp['ccoef'][1]])

    					sdev_ERA5 = np.array([taylor_stats_ERA5_warm_temp['sdev'][0],taylor_stats_ERA5_warm_temp['sdev'][1]])
    					crmsd_ERA5 = np.array([taylor_stats_ERA5_warm_temp['crmsd'][0],taylor_stats_ERA5_warm_temp['crmsd'][1]])
    					ccoef_ERA5 = np.array([taylor_stats_ERA5_warm_temp['ccoef'][0],taylor_stats_ERA5_warm_temp['ccoef'][1]])

    					sdev_ERA5_Land = np.array([taylor_stats_ERA5_Land_warm_temp['sdev'][0],taylor_stats_ERA5_Land_warm_temp['sdev'][1]])
    					crmsd_ERA5_Land = np.array([taylor_stats_ERA5_Land_warm_temp['crmsd'][0],taylor_stats_ERA5_Land_warm_temp['crmsd'][1]])
    					ccoef_ERA5_Land = np.array([taylor_stats_ERA5_Land_warm_temp['ccoef'][0],taylor_stats_ERA5_Land_warm_temp['ccoef'][1]])

    					sdev_JRA = np.array([taylor_stats_JRA_warm_temp['sdev'][0],taylor_stats_JRA_warm_temp['sdev'][1]])
    					crmsd_JRA = np.array([taylor_stats_JRA_warm_temp['crmsd'][0],taylor_stats_JRA_warm_temp['crmsd'][1]])
    					ccoef_JRA = np.array([taylor_stats_JRA_warm_temp['ccoef'][0],taylor_stats_JRA_warm_temp['ccoef'][1]])

    					sdev_MERRA2 = np.array([taylor_stats_MERRA2_warm_temp['sdev'][0],taylor_stats_MERRA2_warm_temp['sdev'][1]])
    					crmsd_MERRA2 = np.array([taylor_stats_MERRA2_warm_temp['crmsd'][0],taylor_stats_MERRA2_warm_temp['crmsd'][1]])
    					ccoef_MERRA2 = np.array([taylor_stats_MERRA2_warm_temp['ccoef'][0],taylor_stats_MERRA2_warm_temp['ccoef'][1]])

    					sdev_GLDAS = np.array([taylor_stats_GLDAS_warm_temp['sdev'][0],taylor_stats_GLDAS_warm_temp['sdev'][1]])
    					crmsd_GLDAS = np.array([taylor_stats_GLDAS_warm_temp['crmsd'][0],taylor_stats_GLDAS_warm_temp['crmsd'][1]])
    					ccoef_GLDAS = np.array([taylor_stats_GLDAS_warm_temp['ccoef'][0],taylor_stats_GLDAS_warm_temp['ccoef'][1]])

    					sdev_GLDAS_CLSM = np.array([taylor_stats_GLDAS_CLSM_warm_temp['sdev'][0],taylor_stats_GLDAS_CLSM_warm_temp['sdev'][1]])
    					crmsd_GLDAS_CLSM = np.array([taylor_stats_GLDAS_CLSM_warm_temp['crmsd'][0],taylor_stats_GLDAS_CLSM_warm_temp['crmsd'][1]])
    					ccoef_GLDAS_CLSM = np.array([taylor_stats_GLDAS_CLSM_warm_temp['ccoef'][0],taylor_stats_GLDAS_CLSM_warm_temp['ccoef'][1]])

    					label = {'Station':'dimgrey','Naive Blend': 'dodgerblue', 'CFSR': 'm','ERA-Interim': 'g', 'ERA5': 'c','ERA5-Land': 'lightblue', 'JRA55': 'r', 'MERRA2': 'y', 'GLDAS-Noah': 'k','GLDAS-CLSM': 'silver'}
								
    					sm.taylor_diagram(sdev_naive,crmsd_naive,ccoef_naive,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 70.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5,colOBS = 'dimgrey',markerobs = 'o',titleOBS = 'Station', markercolor='dodgerblue')
    					sm.taylor_diagram(sdev_CFSR,crmsd_CFSR,ccoef_CFSR,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 70.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='m',overlay='on')
    					sm.taylor_diagram(sdev_ERAI,crmsd_ERAI,ccoef_ERAI,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 70.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='g',overlay='on')
    					sm.taylor_diagram(sdev_ERA5,crmsd_ERA5,ccoef_ERA5,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 70.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='c',overlay='on')
    					sm.taylor_diagram(sdev_ERA5_Land,crmsd_ERA5_Land,ccoef_ERA5_Land,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 70.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='lightblue',overlay='on')
    					sm.taylor_diagram(sdev_JRA,crmsd_JRA,ccoef_JRA,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 70.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='r',overlay='on')
    					sm.taylor_diagram(sdev_MERRA2,crmsd_MERRA2,ccoef_MERRA2,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 70.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='y',overlay='on')
    					sm.taylor_diagram(sdev_GLDAS,crmsd_GLDAS,ccoef_GLDAS,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 70.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='k',overlay='on')
    					sm.taylor_diagram(sdev_GLDAS_CLSM,crmsd_GLDAS_CLSM,ccoef_GLDAS_CLSM,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 70.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='gainsboro',overlay='on',markerLabel=label)
    					plt.savefig(taylor_plt_fil_warm)
    					plt.close()
    					print(taylor_plt_fil_warm)
