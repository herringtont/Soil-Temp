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
lyr = ['top_30cm','30cm_300cm']
thr = ['100']#['0','25','50','75','100']
rmp_type = ['con']#['nn','bil','con']
temp_thr = ['-2C']#['0C','-2C','-5C','-10C']

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
    					cold_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/CLSM_res/subset/'+str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_dframe_cold_season_temp_master_ERA5_'+str(temp_thr_n)+'_CMOS_CLSM_subset.csv'])
    					warm_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/CLSM_res/subset/'+str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_dframe_warm_season_temp_master_ERA5_'+str(temp_thr_n)+'_CMOS_CLSM_subset.csv'])    			
    					scatter_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/CLSM_res/subset/'+str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_dframe_scatterplot_ERA5_'+str(temp_thr_n)+'_CMOS_CLSM_subset.csv'])


    					dframe_cold = pd.read_csv(cold_fil)
    					dframe_cold_continent = dframe_cold[dframe_cold['Continent'] == 'North_America']
    					station_cold = dframe_cold_continent['Station'].values
    					naive_cold = dframe_cold_continent['Naive Blend'].values
    					naive_noJRA_cold = dframe_cold_continent['Naive Blend no JRA55'].values
    					naive_noJRAold_cold = dframe_cold_continent['Naive Blend no JRA55 Old'].values
    					naive_all_cold = dframe_cold_continent['Naive Blend All'].values
    					CFSR_cold = dframe_cold_continent['CFSR'].values
    					ERAI_cold = dframe_cold_continent['ERA-Interim'].values
    					ERA5_cold = dframe_cold_continent['ERA5'].values
    					ERA5_Land_cold = dframe_cold_continent['ERA5-Land'].values
    					JRA_cold = dframe_cold_continent['JRA55'].values
    					MERRA2_cold = dframe_cold_continent['MERRA2'].values
    					GLDAS_cold = dframe_cold_continent['GLDAS-Noah'].values
    					GLDAS_CLSM_cold = dframe_cold_continent['GLDAS-CLSM'].values

    					dframe_warm = pd.read_csv(warm_fil)
    					dframe_warm_continent = dframe_warm[dframe_warm['Continent'] == 'North_America']
    					station_warm = dframe_warm_continent['Station'].values
    					naive_warm = dframe_warm_continent['Naive Blend'].values
    					naive_noJRA_warm = dframe_warm_continent['Naive Blend no JRA55'].values
    					naive_noJRAold_warm = dframe_warm_continent['Naive Blend no JRA55 Old'].values
    					naive_all_warm = dframe_warm_continent['Naive Blend All'].values
    					CFSR_warm = dframe_warm_continent['CFSR'].values
    					ERAI_warm = dframe_warm_continent['ERA-Interim'].values
    					ERA5_warm = dframe_warm_continent['ERA5'].values
    					ERA5_Land_warm = dframe_warm_continent['ERA5-Land'].values
    					JRA_warm = dframe_warm_continent['JRA55'].values
    					MERRA2_warm = dframe_warm_continent['MERRA2'].values
    					GLDAS_warm = dframe_warm_continent['GLDAS-Noah'].values
    					GLDAS_CLSM_warm = dframe_warm_continent['GLDAS-CLSM'].values

    					dframe_scatter = pd.read_csv(scatter_fil)
    					dframe_scatter_continent = dframe_scatter[dframe_scatter['Continent'] == 'North_America']
    					station_scatter = dframe_scatter_continent['Station'].values
    					naive_scatter = dframe_scatter_continent['Naive Blend'].values
    					naive_noJRA_scatter = dframe_scatter_continent['Naive Blend no JRA55'].values
    					naive_noJRAold_scatter = dframe_scatter_continent['Naive Blend no JRA55 Old'].values
    					naive_all_scatter = dframe_scatter_continent['Naive Blend All'].values
    					CFSR_scatter = dframe_scatter_continent['CFSR'].values
    					ERAI_scatter = dframe_scatter_continent['ERA-Interim'].values
    					ERA5_scatter = dframe_scatter_continent['ERA5'].values
    					ERA5_Land_scatter = dframe_scatter_continent['ERA5-Land'].values
    					JRA_scatter = dframe_scatter_continent['JRA55'].values
    					MERRA2_scatter = dframe_scatter_continent['MERRA2'].values
    					GLDAS_scatter = dframe_scatter_continent['GLDAS-Noah'].values
    					GLDAS_CLSM_scatter = dframe_scatter_continent['GLDAS-CLSM'].values
    					season_scatter = dframe_scatter_continent['Season'].values

    					#dframe_scatter2 = pd.DataFrame({'Station':station_scatter,'Season':season_scatter,'Naive Blend A':naive_scatter,'Naive Blend B':naive_noJRAold_scatter,'Naive Blend C':naive_all_scatter,'Naive Blend D':naive_noJRA_scatter,'CFSR':CFSR_scatter,'ERA-Interim':ERAI_scatter,'ERA5':ERA5_scatter,'ERA5-Land':ERA5_Land_scatter,'JRA55':JRA_scatter,'MERRA2':MERRA2_scatter,'GLDAS-Noah':GLDAS_scatter,'GLDAS-CLSM':GLDAS_CLSM_scatter})
    					#dframe_cold2 = pd.DataFrame({'Station':station_cold,'Naive Blend A':naive_cold,'Naive Blend B':naive_noJRAold_cold,'Naive Blend C':naive_all_cold,'Naive Blend D':naive_noJRA_cold,'CFSR':CFSR_cold,'ERA-Interim':ERAI_cold,'ERA5':ERA5_cold,'ERA5-Land':ERA5_Land_cold,'JRA55':JRA_cold,'MERRA2':MERRA2_cold,'GLDAS-Noah':GLDAS_cold,'GLDAS-CLSM':GLDAS_CLSM_cold})
    					#dframe_warm2 = pd.DataFrame({'Station':station_warm,'Naive Blend A':naive_warm,'Naive Blend B':naive_noJRAold_warm,'Naive Blend C':naive_all_warm,'Naive Blend D':naive_noJRA_warm,'CFSR':CFSR_warm,'ERA-Interim':ERAI_warm,'ERA5':ERA5_warm,'ERA5-Land':ERA5_Land_warm,'JRA55':JRA_warm,'MERRA2':MERRA2_warm,'GLDAS-Noah':GLDAS_warm,'GLDAS-CLSM':GLDAS_CLSM_warm})

    					dframe_scatter2 = pd.DataFrame({'Station':station_scatter,'Season':season_scatter,'Ensemble Mean':naive_all_scatter,'CFSR':CFSR_scatter,'ERA-Interim':ERAI_scatter,'ERA5':ERA5_scatter,'ERA5-Land':ERA5_Land_scatter,'JRA55':JRA_scatter,'MERRA2':MERRA2_scatter,'GLDAS-Noah':GLDAS_scatter,'GLDAS-CLSM':GLDAS_CLSM_scatter})
    					dframe_cold2 = pd.DataFrame({'Station':station_cold,'Ensemble Mean':naive_all_cold,'CFSR':CFSR_cold,'ERA-Interim':ERAI_cold,'ERA5':ERA5_cold,'ERA5-Land':ERA5_Land_cold,'JRA55':JRA_cold,'MERRA2':MERRA2_cold,'GLDAS-Noah':GLDAS_cold,'GLDAS-CLSM':GLDAS_CLSM_cold})
    					dframe_warm2 = pd.DataFrame({'Station':station_warm,'Ensemble Mean':naive_all_warm,'CFSR':CFSR_warm,'ERA-Interim':ERAI_warm,'ERA5':ERA5_warm,'ERA5-Land':ERA5_Land_warm,'JRA55':JRA_warm,'MERRA2':MERRA2_warm,'GLDAS-Noah':GLDAS_warm,'GLDAS-CLSM':GLDAS_CLSM_warm})

    					print(dframe_scatter2)
    					print(dframe_cold2)
    					print(dframe_warm2)

###################### Calculate Correlation Matrices #########################

    					fig,axs = plt.subplots(nrows=1, ncols=2, sharex='col', sharey='row', figsize=(15,20))
    					cbar_kws = {"orientation":"horizontal","shrink":1}
###### Cold Season (Temp) #######

    					#dframe_cold_season_temp_corr = dframe_cold[['Station','Naive Blend A','Naive Blend B','Naive Blend C','Naive Blend D','CFSR','ERA-Interim','ERA5','JRA55','MERRA2','GLDAS-Noah','GLDAS-CLSM']]
    					cold_season_temp_corrMatrix = dframe_cold2.corr()

    					ax1 = plt.subplot(121)
    					corr1 = sn.heatmap(cold_season_temp_corrMatrix,annot=True,square='True',vmin=0,vmax=1,cbar_kws=cbar_kws)
    					ax1.set_title('Cold Season Correlation Matrix')

###### Warm Season (Temp) #######
    					#dframe_warm_season_temp_corr = dframe_warm[['Station','Naive Blend A','Naive Blend B','Naive Blend C','Naive Blend D','CFSR','ERA-Interim','ERA5','JRA55','MERRA2','GLDAS-Noah','GLDAS-CLSM']]
    					warm_season_temp_corrMatrix = dframe_warm2.corr()

    					ax2 = plt.subplot(122)
    					corr2 = sn.heatmap(warm_season_temp_corrMatrix,annot=True,square='True',vmin=0,vmax=1,cbar_kws=cbar_kws)
    					ax2.set_title('Warm Season Correlation Matrix')

    					plt.savefig('/mnt/data/users/herringtont/soil_temp/plots/naive_blend_correlation/new_data/CMOS_poster/CLSM_res/'+str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_correlation_byseason_ERA5_'+str(temp_thr_n)+'_new_data_CMOS_CLSM_NAm.png')
    					plt.close()


###################### Create Scatterplot Matrices ####################

###### Cold Season (Temp) ######
    					scatter1 = sn.pairplot(dframe_scatter2,hue='Season',)
    					sns.set_context(rc={"axes.labelsize":20}, font_scale=1.0)
    					sns.set_context(rc={"legend.fontsize":18}, font_scale=1.0)
    					#plt.setp(scatter1.get_legend().get_texts(), fontsize='14') #set legend text size
    					#plt.setp(scatter1.get_legend().get_title(), fontsize='18') #set legend title size
    					#plt.setp(scatter1.get_legend().get_texts(), fontsize='16') # for legend text
    					#plt.setp(scatter1.get_legend().get_title(), fontsize='20') # for legend title					
    					for ax in scatter1.axes.flat:
    						ax.set_xlim(-40,40)
    						ax.set_ylim(-40,40)
    					scatter1.add_legend(title="legend",title_fontsize = 20)
    					plt.savefig('/mnt/data/users/herringtont/soil_temp/plots/naive_blend_scatterplots/new_data/CMOS_poster/CLSM_res/'+str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_scatterplot_all_temp_ERA5_'+str(temp_thr_n)+'_new_data_CMOS_CLSM_NAm.png')
    					plt.close()		




##################### Create Taylor Diagrams ####################

###### Cold Season (Temp) #####
    					station_cold_season_temp = dframe_cold_continent['Station'].values
    					naive_cold_season_temp = dframe_cold_continent['Naive Blend'].values
    					naive_noJRA_cold_season_temp = dframe_cold_continent['Naive Blend no JRA55'].values
    					naive_noJRAold_cold_season_temp = dframe_cold_continent['Naive Blend no JRA55 Old'].values
    					naive_all_cold_season_temp = dframe_cold_continent['Naive Blend All'].values
    					CFSR_cold_season_temp = dframe_cold_continent['CFSR'].values    				
    					ERAI_cold_season_temp = dframe_cold_continent['ERA-Interim'].values
    					ERA5_cold_season_temp = dframe_cold_continent['ERA5'].values
    					ERA5_Land_cold_season_temp = dframe_cold_continent['ERA5-Land'].values
    					JRA_cold_season_temp = dframe_cold_continent['JRA55'].values
    					MERRA2_cold_season_temp = dframe_cold_continent['MERRA2'].values
    					GLDAS_cold_season_temp = dframe_cold_continent['GLDAS-Noah'].values
    					GLDAS_CLSM_cold_season_temp = dframe_cold_continent['GLDAS-CLSM'].values

    					taylor_stats_naive_cold_temp = sm.taylor_statistics(naive_cold_season_temp,station_cold_season_temp)
    					taylor_stats_naive_noJRA_cold_temp = sm.taylor_statistics(naive_noJRA_cold_season_temp,station_cold_season_temp)
    					taylor_stats_naive_noJRAold_cold_temp = sm.taylor_statistics(naive_noJRAold_cold_season_temp,station_cold_season_temp)
    					taylor_stats_naive_all_cold_temp = sm.taylor_statistics(naive_all_cold_season_temp,station_cold_season_temp)
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


    					taylor_dir = '/mnt/data/users/herringtont/soil_temp/plots/taylor_diagrams/new_data/CMOS_poster/CLSM_res/'
    					taylor_plt_fil_cold = ''.join([taylor_dir,str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_ERA5_'+str(temp_thr_n)+'_taylor_diagram_cold_new_data_CMOS_CLSM_NAm.png'])

    					sdev_naive = np.array([taylor_stats_naive_cold_temp['sdev'][0],taylor_stats_naive_cold_temp['sdev'][1]])
    					crmsd_naive = np.array([taylor_stats_naive_cold_temp['crmsd'][0],taylor_stats_naive_cold_temp['crmsd'][1]])
    					ccoef_naive = np.array([taylor_stats_naive_cold_temp['ccoef'][0],taylor_stats_naive_cold_temp['ccoef'][1]])

    					sdev_naive_noJRA = np.array([taylor_stats_naive_noJRA_cold_temp['sdev'][0],taylor_stats_naive_noJRA_cold_temp['sdev'][1]])
    					crmsd_naive_noJRA = np.array([taylor_stats_naive_noJRA_cold_temp['crmsd'][0],taylor_stats_naive_noJRA_cold_temp['crmsd'][1]])
    					ccoef_naive_noJRA = np.array([taylor_stats_naive_noJRA_cold_temp['ccoef'][0],taylor_stats_naive_noJRA_cold_temp['ccoef'][1]])

    					sdev_naive_noJRAold = np.array([taylor_stats_naive_noJRAold_cold_temp['sdev'][0],taylor_stats_naive_noJRAold_cold_temp['sdev'][1]])
    					crmsd_naive_noJRAold = np.array([taylor_stats_naive_noJRAold_cold_temp['crmsd'][0],taylor_stats_naive_noJRAold_cold_temp['crmsd'][1]])
    					ccoef_naive_noJRAold = np.array([taylor_stats_naive_noJRAold_cold_temp['ccoef'][0],taylor_stats_naive_noJRAold_cold_temp['ccoef'][1]])

    					sdev_naive_all = np.array([taylor_stats_naive_all_cold_temp['sdev'][0],taylor_stats_naive_all_cold_temp['sdev'][1]])
    					crmsd_naive_all = np.array([taylor_stats_naive_all_cold_temp['crmsd'][0],taylor_stats_naive_all_cold_temp['crmsd'][1]])
    					ccoef_naive_all = np.array([taylor_stats_naive_all_cold_temp['ccoef'][0],taylor_stats_naive_all_cold_temp['ccoef'][1]])

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

    					label = {'Station':'dimgrey','Ensemble Mean':'fuchsia','CFSR': 'm','ERA-Interim': 'limegreen', 'ERA5': 'cyan','ERA5-Land':'skyblue','JRA55': 'red', 'MERRA2': 'goldenrod', 'GLDAS-Noah': 'black','GLDAS-CLSM':'darkgrey'}
								
    					#sm.taylor_diagram(sdev_naive,crmsd_naive,ccoef_naive,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5,colOBS = 'dimgrey',markerobs = 'o',titleOBS = 'Station', markercolor='dodgerblue', markerLabelcolor='dodgerblue')
    					sm.taylor_diagram(sdev_naive_all,crmsd_naive_all,ccoef_naive_all,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5,colOBS = 'dimgrey',markerobs = 'o',titleOBS = 'Station', markercolor='fuchsia', markerLabelcolor='fuchsia')
    					#sm.taylor_diagram(sdev_naive_noJRAold,crmsd_naive_noJRAold,ccoef_naive_noJRAold,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='mediumslateblue', markerLabelcolor='mediumslateblue',overlay='on')
    					#sm.taylor_diagram(sdev_naive_all,crmsd_naive_all,ccoef_naive_all,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='fuchsia', markerLabelcolor='fuchsia',overlay='on')
    					#sm.taylor_diagram(sdev_naive_noJRA,crmsd_naive_noJRA,ccoef_naive_noJRA,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='coral', markerLabelcolor='coral',overlay='on')
    					sm.taylor_diagram(sdev_CFSR,crmsd_CFSR,ccoef_CFSR,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='m', markerLabelcolor='m', overlay='on')
    					sm.taylor_diagram(sdev_ERAI,crmsd_ERAI,ccoef_ERAI,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='limegreen', markerLabelcolor='limegreen',overlay='on')
    					sm.taylor_diagram(sdev_ERA5,crmsd_ERA5,ccoef_ERA5,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='cyan', markerLabelcolor='cyan',overlay='on')
    					sm.taylor_diagram(sdev_ERA5_Land,crmsd_ERA5_Land,ccoef_ERA5_Land,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='skyblue',markerLabelcolor='skyblue',overlay='on')
    					sm.taylor_diagram(sdev_JRA,crmsd_JRA,ccoef_JRA,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='red',markerLabelcolor='red',overlay='on')
    					sm.taylor_diagram(sdev_MERRA2,crmsd_MERRA2,ccoef_MERRA2,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='goldenrod',markerLabelcolor='goldenrod',overlay='on')
    					sm.taylor_diagram(sdev_GLDAS,crmsd_GLDAS,ccoef_GLDAS,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='black',markerLabelcolor='black',overlay='on', markerLabel=label)
    					sm.taylor_diagram(sdev_GLDAS_CLSM,crmsd_GLDAS_CLSM,ccoef_GLDAS_CLSM,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='darkgrey',markerLabelcolor='darkgrey',overlay='on',markerLabel=label)

    					plt.savefig(taylor_plt_fil_cold)
    					plt.close()
    					print(taylor_plt_fil_cold)


###### Warm Season (Temp) #####

    					station_warm_season_temp = dframe_warm_continent['Station'].values
    					naive_warm_season_temp = dframe_warm_continent['Naive Blend'].values
    					naive_noJRA_warm_season_temp = dframe_warm_continent['Naive Blend no JRA55'].values
    					naive_noJRAold_warm_season_temp = dframe_warm_continent['Naive Blend no JRA55 Old'].values
    					naive_all_warm_season_temp = dframe_warm_continent['Naive Blend All'].values
    					CFSR_warm_season_temp = dframe_warm_continent['CFSR'].values    				
    					ERAI_warm_season_temp = dframe_warm_continent['ERA-Interim'].values
    					ERA5_warm_season_temp = dframe_warm_continent['ERA5'].values
    					ERA5_Land_warm_season_temp = dframe_warm_continent['ERA5-Land'].values
    					JRA_warm_season_temp = dframe_warm_continent['JRA55'].values
    					MERRA2_warm_season_temp = dframe_warm_continent['MERRA2'].values
    					GLDAS_warm_season_temp = dframe_warm_continent['GLDAS-Noah'].values
    					GLDAS_CLSM_warm_season_temp = dframe_warm_continent['GLDAS-CLSM'].values

    					taylor_stats_naive_warm_temp = sm.taylor_statistics(naive_warm_season_temp,station_warm_season_temp)
    					taylor_stats_naive_noJRA_warm_temp = sm.taylor_statistics(naive_noJRA_warm_season_temp,station_warm_season_temp)
    					taylor_stats_naive_noJRAold_warm_temp = sm.taylor_statistics(naive_noJRAold_warm_season_temp,station_warm_season_temp)
    					taylor_stats_naive_all_warm_temp = sm.taylor_statistics(naive_all_warm_season_temp,station_warm_season_temp)
    					taylor_stats_CFSR_warm_temp = sm.taylor_statistics(CFSR_warm_season_temp,station_warm_season_temp)			
    					taylor_stats_ERAI_warm_temp = sm.taylor_statistics(ERAI_warm_season_temp,station_warm_season_temp)
    					taylor_stats_ERA5_warm_temp = sm.taylor_statistics(ERA5_warm_season_temp,station_warm_season_temp)
    					taylor_stats_ERA5_Land_warm_temp = sm.taylor_statistics(ERA5_Land_warm_season_temp,station_warm_season_temp)
    					taylor_stats_JRA_warm_temp = sm.taylor_statistics(JRA_warm_season_temp,station_warm_season_temp)
    					taylor_stats_MERRA2_warm_temp = sm.taylor_statistics(MERRA2_warm_season_temp,station_warm_season_temp)
    					taylor_stats_GLDAS_warm_temp = sm.taylor_statistics(GLDAS_warm_season_temp,station_warm_season_temp)
    					taylor_stats_GLDAS_CLSM_warm_temp = sm.taylor_statistics(GLDAS_CLSM_warm_season_temp,station_warm_season_temp)


    					print('Remap Style:',remap_type)
    					print('Layer:',lyr_l)
    					print('Temp Threshold:',temp_thr_n)


    					taylor_dir = '/mnt/data/users/herringtont/soil_temp/plots/taylor_diagrams/new_data/CMOS_poster/CLSM_res/'
    					taylor_plt_fil_warm = ''.join([taylor_dir,str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_ERA5_'+str(temp_thr_n)+'_taylor_diagram_warm_new_data_CMOS_CLSM_NAm.png'])

    					sdev_naive = np.array([taylor_stats_naive_warm_temp['sdev'][0],taylor_stats_naive_warm_temp['sdev'][1]])
    					crmsd_naive = np.array([taylor_stats_naive_warm_temp['crmsd'][0],taylor_stats_naive_warm_temp['crmsd'][1]])
    					ccoef_naive = np.array([taylor_stats_naive_warm_temp['ccoef'][0],taylor_stats_naive_warm_temp['ccoef'][1]])

    					sdev_naive_noJRA = np.array([taylor_stats_naive_noJRA_warm_temp['sdev'][0],taylor_stats_naive_noJRA_warm_temp['sdev'][1]])
    					crmsd_naive_noJRA = np.array([taylor_stats_naive_noJRA_warm_temp['crmsd'][0],taylor_stats_naive_noJRA_warm_temp['crmsd'][1]])
    					ccoef_naive_noJRA = np.array([taylor_stats_naive_noJRA_warm_temp['ccoef'][0],taylor_stats_naive_noJRA_warm_temp['ccoef'][1]])

    					sdev_naive_noJRAold = np.array([taylor_stats_naive_noJRAold_warm_temp['sdev'][0],taylor_stats_naive_noJRAold_warm_temp['sdev'][1]])
    					crmsd_naive_noJRAold = np.array([taylor_stats_naive_noJRAold_warm_temp['crmsd'][0],taylor_stats_naive_noJRAold_warm_temp['crmsd'][1]])
    					ccoef_naive_noJRAold = np.array([taylor_stats_naive_noJRAold_warm_temp['ccoef'][0],taylor_stats_naive_noJRAold_warm_temp['ccoef'][1]])

    					sdev_naive_all = np.array([taylor_stats_naive_all_warm_temp['sdev'][0],taylor_stats_naive_all_warm_temp['sdev'][1]])
    					crmsd_naive_all = np.array([taylor_stats_naive_all_warm_temp['crmsd'][0],taylor_stats_naive_all_warm_temp['crmsd'][1]])
    					ccoef_naive_all = np.array([taylor_stats_naive_all_warm_temp['ccoef'][0],taylor_stats_naive_all_warm_temp['ccoef'][1]])

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

    					label = {'Station':'dimgrey','Ensemble Mean':'fuchsia','CFSR': 'm','ERA-Interim': 'limegreen', 'ERA5': 'cyan','ERA5-Land':'skyblue','JRA55': 'red', 'MERRA2': 'goldenrod', 'GLDAS-Noah': 'black','GLDAS-CLSM':'darkgrey'}
								
    					#sm.taylor_diagram(sdev_naive,crmsd_naive,ccoef_naive,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5,colOBS = 'dimgrey',markerobs = 'o',titleOBS = 'Station', markercolor='dodgerblue', markerLabelcolor='dodgerblue')
    					sm.taylor_diagram(sdev_naive_all,crmsd_naive_all,ccoef_naive_all,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5,colOBS = 'dimgrey',markerobs = 'o',titleOBS = 'Station', markercolor='fuchsia', markerLabelcolor='fuchsia')
    					#sm.taylor_diagram(sdev_naive_noJRAold,crmsd_naive_noJRAold,ccoef_naive_noJRAold,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='mediumslateblue', markerLabelcolor='mediumslateblue',overlay='on')
    					#sm.taylor_diagram(sdev_naive_all,crmsd_naive_all,ccoef_naive_all,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='fuchsia', markerLabelcolor='fuchsia',overlay='on')
    					#sm.taylor_diagram(sdev_naive_noJRA,crmsd_naive_noJRA,ccoef_naive_noJRA,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='coral', markerLabelcolor='coral',overlay='on')
    					sm.taylor_diagram(sdev_CFSR,crmsd_CFSR,ccoef_CFSR,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='m', markerLabelcolor='m', overlay='on')
    					sm.taylor_diagram(sdev_ERAI,crmsd_ERAI,ccoef_ERAI,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='limegreen', markerLabelcolor='limegreen',overlay='on')
    					sm.taylor_diagram(sdev_ERA5,crmsd_ERA5,ccoef_ERA5,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='cyan', markerLabelcolor='cyan',overlay='on')
    					sm.taylor_diagram(sdev_ERA5_Land,crmsd_ERA5_Land,ccoef_ERA5_Land,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='skyblue',markerLabelcolor='skyblue',overlay='on')
    					sm.taylor_diagram(sdev_JRA,crmsd_JRA,ccoef_JRA,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='red',markerLabelcolor='red',overlay='on')
    					sm.taylor_diagram(sdev_MERRA2,crmsd_MERRA2,ccoef_MERRA2,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='goldenrod',markerLabelcolor='goldenrod',overlay='on')
    					sm.taylor_diagram(sdev_GLDAS,crmsd_GLDAS,ccoef_GLDAS,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='black',markerLabelcolor='black',overlay='on', markerLabel=label)
    					sm.taylor_diagram(sdev_GLDAS_CLSM,crmsd_GLDAS_CLSM,ccoef_GLDAS_CLSM,tickSTD=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10],axismax=10.0,styleSTD = '-.', widthSTD = 0.5,styleOBS = '-',tickRMS=np.arange(0,8,1),tickRMSangle = 90.0,styleRMS = ':', widthRMS = 0.5,widthCOR = 0.5, markercolor='darkgrey',markerLabelcolor='darkgrey',overlay='on',markerLabel=label)
    					plt.savefig(taylor_plt_fil_warm)
    					plt.close()
    					print(taylor_plt_fil_warm)






























