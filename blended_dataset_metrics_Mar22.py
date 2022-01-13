import os
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


#/mnt/data/users/herringtont/soil_temp/In-Situ/All/spatial_average_anom/remapnn/no_outliers/outliers/0_9.9/thr_0/grid_82140_anom.csv
#finished extracting grid cell level values for this combo
#Traceback (most recent call last):
#  File "blended_dataset_metrics_Mar22.py", line 723, in <module>
#    dframe_anom_master2['GLDAS'] = GLDAS_anom
#  File "/home/herringtont/anaconda3/envs/SoilTemp/lib/python3.8/site-packages/pandas/core/frame.py", line 3040, in __setitem__
#    self._set_item(key, value)
#  File "/home/herringtont/anaconda3/envs/SoilTemp/lib/python3.8/site-packages/pandas/core/frame.py", line 3116, in _set_item
#    value = self._sanitize_column(key, value)
#  File "/home/herringtont/anaconda3/envs/SoilTemp/lib/python3.8/site-packages/pandas/core/frame.py", line 3764, in _sanitize_column
#    value = sanitize_index(value, self.index)
#  File "/home/herringtont/anaconda3/envs/SoilTemp/lib/python3.8/site-packages/pandas/core/internals/construction.py", line 747, in sanitize_index
#    raise ValueError(
#ValueError: Length of values (0) does not match length of index (60)


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



def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

########## Set directories ##########
TC_dir = ['global_triple_collocation_no_rescaling']#'global_triple_collocation','global_triple_collocationB','global_triple_collocationC','global_triple_collocationD','global_triple_collocationE']
olr = ['outliers','zscore','IQR']
lyr = ['0_9.9']
thr = ['0','25','50','75','100']
rmp_type = ['nn','bil']
tmp_type = ['raw_temp']


CFSR_layer = "Soil_Temp_L1"
CFSR2_layer = "Soil_Temp_L1"
GLDAS_layer = "Soil_Temp_L1"
ERA5_layer = "Soil_Temp_L1"
ERAI_layer = "Soil_Temp_L1"
JRA_layer = "Soil_Temp"
MERRA2_layer = "Soil_Temp_L1"


########### Grab Reanalysis Data ##########
for i in tmp_type: #loop through data type (absolute temp, anomalies)
    tmp_type_i = i
    if (tmp_type_i == 'raw_temp'):
    	temp_type = 'Absolute Temps'
    	bldsfx = 'raw'
    	bldvar = 'TC_blended_stemp'
    	nvar = 'naive_blended_stemp'
    	s_nam = 'spatial_average'
    	stemp_nam = 'Spatial Avg Temp'
    if (tmp_type_i == 'anom'):
    	temp_type = 'Anomalies'
    	bldsfx = 'anom'
    	bldvar = 'TC_blended_anom'
    	nvar = 'naive_blended_anom'
    	s_nam = 'spatial_average_anom'
    	stemp_nam = 'Spatial Avg Anom'   	
    for j in rmp_type:
    	rmp_type_j = j
    	remap_type = ''.join(['remap'+rmp_type_j])
    	rnys_dir = ''.join(['/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/common_grid/'+str(remap_type)+'/common_date/'])
    	rnys_gcell_dir = ''.join([rnys_dir,'grid_cell_level/'])
    	if(tmp_type_i == 'raw_temp'):
    		CFSR_fi = "".join([rnys_dir,"CFSR_all.nc"])
    		MERRA2_fi = "".join([rnys_dir,"MERRA2.nc"])
    		ERA5_fi = "".join([rnys_dir,"ERA5.nc"])
    		ERAI_fi = "".join([rnys_dir,"ERA-Interim.nc"])
    		JRA_fi = "".join([rnys_dir,"JRA55.nc"])
    		GLDAS_fi = "".join([rnys_dir,"GLDAS.nc"])

    	elif(tmp_type_i == 'anom'):
    		CFSR_fi = "".join([rnys_dir,"CFSR_anom.nc"])
    		MERRA2_fi = "".join([rnys_dir,"MERRA2_anom.nc"])
    		ERA5_fi = "".join([rnys_dir,"ERA5_anom.nc"])
    		ERAI_fi = "".join([rnys_dir,"ERA-Interim_anom.nc"])
    		JRA_fi = "".join([rnys_dir,"JRA55_anom.nc"])
    		GLDAS_fi = "".join([rnys_dir,"GLDAS_anom.nc"])


################# Grab Blended Soil Temperature Data ############### 
    	for k in TC_dir:
    		TC_dir_k = k   	
    		TC_basedir = ''.join(['/mnt/data/users/herringtont/soil_temp/'+str(TC_dir_k)+'/'+str(tmp_type_i)+'/'+str(remap_type)+'/blended_products/'])
    		TC_fi = ''.join([TC_basedir+str(remap_type)+'_TC_blended_'+str(bldsfx)+'.nc']) 
    		naive_fi = ''.join([TC_basedir+str(remap_type)+'_naive_blended_'+str(bldsfx)+'.nc'])

    		for l in olr:
    			olr_l = l

    			for m in thr:
    				thr_m = m
    				insitu_dir =  ''.join(['/mnt/data/users/herringtont/soil_temp/In-Situ/All/spatial_average_anom/'+str(remap_type)+'/no_outliers/'+str(olr_l)+'/0_9.9/thr_'+str(thr_m)+'/'])

##### Create Master Arrays #####
    				gcell_master_stn = []
    				gcell_master = []
    				lat_master_stn = []
    				lon_master_stn = []
    				lat_master = []
    				lon_master = []
    				date_master = []
    				len_anom_master = []
    				len_raw_master = []

    				model1_nam_master = []
    				model2_nam_master = []
    				model3_nam_master = []

    				model1_err_var_raw_master = []
    				model2_err_var_raw_master = []
    				model3_err_var_raw_master = []

    				model1_err_var_anom_master = []
    				model2_err_var_anom_master = []
    				model3_err_var_anom_master = []

    				naive_temp_master_raw = []
    				TC_temp_master_raw = []
    				CFSR_temp_master_raw = []
    				ERAI_temp_master_raw = []
    				ERA5_temp_master_raw = []
    				JRA_temp_master_raw = []
    				MERRA2_temp_master_raw = []
    				GLDAS_temp_master_raw = []

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

################# loop through in-situ files ########################
    				cdo = Cdo()
    				#pathlist = "/mnt/data/users/herringtont/soil_temp/In-Situ/All/spatial_average_anom/remapnn/no_outliers/outliers/0_9.9/thr_0/grid_92836_anom.csv"
    				pathlist = os_sorted(os.listdir(insitu_dir))
    				for path in pathlist:
    					#insitu_fil = path
    					insitu_fil = ''.join([insitu_dir,path])
    					dframe_insitu = pd.read_csv(insitu_fil)
    					print(insitu_fil)
    					dattim = dframe_insitu['Date'].values
    					DateTime = [datetime.datetime.strptime(x,'%Y-%m-%d') for x in dattim]
    					soil_temp = dframe_insitu[stemp_nam]
    					gcell = dframe_insitu['Grid Cell'].iloc[0]
    					#gcell_master_stn.append(gcell)
    					lat_cen = dframe_insitu['Central Lat'].iloc[0]
    					#lat_master_stn.append(lat_cen)
    					lon_cen = dframe_insitu['Central Lon'].iloc[0]
    					#lon_master_stn.append(lon_cen)
    					if(tmp_type_i == 'raw_temp'):
    						CFSR_gcell_fi = "".join([rnys_gcell_dir+"CFSR_"+str(remap_type)+'_'+str(olr_l)+'_thr_'+str(thr_m)+'_'+str(TC_dir_k)+'_grid_'+str(gcell)+'.nc'])
    						MERRA2_gcell_fi = "".join([rnys_gcell_dir+"MERRA2_"+str(remap_type)+'_'+str(olr_l)+'_thr_'+str(thr_m)+'_'+str(TC_dir_k)+'_grid_'+str(gcell)+'.nc'])
    						ERA5_gcell_fi = "".join([rnys_gcell_dir,"ERA5_"+str(remap_type)+'_'+str(olr_l)+'_thr_'+str(thr_m)+'_'+str(TC_dir_k)+'_grid_'+str(gcell)+'.nc'])
    						ERAI_gcell_fi = "".join([rnys_gcell_dir,"ERA-Interim_"+str(remap_type)+'_'+str(olr_l)+'_thr_'+str(thr_m)+'_'+str(TC_dir_k)+'_grid_'+str(gcell)+'.nc'])
    						JRA_gcell_fi = "".join([rnys_gcell_dir,"JRA55_"+str(remap_type)+'_'+str(olr_l)+'_thr_'+str(thr_m)+'_'+str(TC_dir_k)+'_grid_'+str(gcell)+'.nc'])
    						GLDAS_gcell_fi = "".join([rnys_gcell_dir,"GLDAS_"+str(remap_type)+'_'+str(olr_l)+'_thr_'+str(thr_m)+'_'+str(TC_dir_k)+'_grid_'+str(gcell)+'.nc'])
    						err_var_dir = ''.join(['/mnt/data/users/herringtont/soil_temp/'+str(TC_dir_k)+'/raw_temp/'+str(remap_type)+'/'])
    						err_var_dir_gcell = ''.join(['/mnt/data/users/herringtont/soil_temp/'+str(TC_dir_k)+'/raw_temp/'+str(remap_type)+'/grid_cell_level/'])

    					elif(tmp_type_i == 'anom'):
    						CFSR_gcell_fi = "".join([rnys_gcell_dir,"CFSR_anom_"+str(remap_type)+'_'+str(olr_l)+'_thr_'+str(thr_m)+'_'+str(TC_dir_k)+'_grid_'+str(gcell)+'.nc'])
    						MERRA2_gcell_fi = "".join([rnys_gcell_dir,"MERRA2_anom_"+str(remap_type)+'_'+str(olr_l)+'_thr_'+str(thr_m)+'_'+str(TC_dir_k)+'_grid_'+str(gcell)+'.nc'])
    						ERA5_gcell_fi = "".join([rnys_gcell_dir,"ERA5_anom_"+str(remap_type)+'_'+str(olr_l)+'_thr_'+str(thr_m)+'_'+str(TC_dir_k)+'_grid_'+str(gcell)+'.nc'])
    						ERAI_gcell_fi = "".join([rnys_gcell_dir,"ERA-Interim_anom_"+str(remap_type)+'_'+str(olr_l)+'_thr'+str(thr_m)+'_'+str(TC_dir_k)+'_grid_'+str(gcell)+'.nc'])
    						JRA_gcell_fi = "".join([rnys_gcell_dir,"JRA55_anom_"+str(remap_type)+'_'+str(olr_l)+'_thr_'+str(thr_m)+'_'+str(TC_dir_k)+'_grid_'+str(gcell)+'.nc'])
    						GLDAS_gcell_fi = "".join([rnys_gcell_dir,"GLDAS_anom_"+str(remap_type)+'_'+str(olr_l)+'_thr_'+str(thr_m)+'_'+str(TC_dir_k)+'_grid_'+str(gcell)+'.nc'])
    						err_var_dir = ''.join(['/mnt/data/users/herringtont/soil_temp/'+str(TC_dir_k)+'/anom/'+str(remap_type)+'/'])    					
    						err_var_dir_gcell = ''.join(['/mnt/data/users/herringtont/soil_temp/'+str(TC_dir_k)+'/anom/'+str(remap_type)+'/grid_cell_level'])

    					if (TC_dir_k == 'global_triple_collocation' or TC_dir_k == 'global_triple_collocation_no_rescaling'):
    						model1_nam = 'JRA55 err_var'
    						model2_nam = 'MERRA2 err_var'
    						model3_nam = 'GLDAS err_var'
    						model1 = 'JRA55'
    						model2 = 'MERRA2'
    						model3 = 'GLDAS'

    					if (TC_dir_k == 'global_triple_collocationB'):
    						model1_nam = 'JRA55 err_var'
    						model2_nam = 'MERRA2 err_var'
    						model3_nam = 'ERA-Interim err_var'
    						model1 = 'JRA55'
    						model2 = 'MERRA2'
    						model3 = 'ERA-Interim'

    					if (TC_dir_k == 'global_triple_collocationC'):
    						model1_nam = 'JRA55 err_var'
    						model2_nam = 'MERRA2 err_var'
    						model3_nam = 'ERA5 err_var'
    						model1 = 'JRA55'
    						model2 = 'MERRA2'
    						model3 = 'ERA5'


    					if (TC_dir_k == 'global_triple_collocationD'):
    						model1_nam = 'CFSR err_var'
    						model2_nam = 'MERRA2 err_var'
    						model3_nam = 'ERA5 err_var'
    						model1 = 'CFSR'
    						model2 = 'MERRA2'
    						model3 = 'ERA5'

    					if (TC_dir_k == 'global_triple_collocationE'):
    						model1_nam = 'JRA55 err_var'
    						model2_nam = 'ERA-Interim err_var'
    						model3_nam = 'ERA5 err_var'
    						model1 = 'JRA55'
    						model2 = 'ERA-Interim'
    						model3 = 'ERA5'


    					err_var_dir_raw = ''.join(['/mnt/data/users/herringtont/soil_temp/'+str(TC_dir_k)+'/raw_temp/remap'+str(rmp_type_j)+'/'])
    					err_var_dir_anom = ''.join(['/mnt/data/users/herringtont/soil_temp/'+str(TC_dir_k)+'/anom/remap'+str(rmp_type_j)+'/'])
    					err_var_dir_raw_gcell = ''.join(['/mnt/data/users/herringtont/soil_temp/'+str(TC_dir_k)+'/raw_temp/remap'+str(rmp_type_j)+'/grid_cell_level/'])    	
    					err_var_dir_anom_gcell = ''.join(['/mnt/data/users/herringtont/soil_temp/'+str(TC_dir_k)+'/anom/remap'+str(rmp_type_j)+'/grid_cell_level'])
						
    					err_var_model1_fi = ''.join([err_var_dir_raw+'remap'+str(rmp_type_j)+'_'+str(model1)+'_err_var_cov.nc'])
    					err_var_model2_fi = ''.join([err_var_dir_raw+'remap'+str(rmp_type_j)+'_'+str(model2)+'_err_var_cov.nc'])
    					err_var_model3_fi = ''.join([err_var_dir_raw+'remap'+str(rmp_type_j)+'_'+str(model3)+'_err_var_cov.nc'])

    					err_var_model1_fi_anom = ''.join([err_var_dir_anom+'remap'+str(rmp_type_j)+'_'+str(model1)+'_err_var_cov.nc'])
    					err_var_model2_fi_anom = ''.join([err_var_dir_anom+'remap'+str(rmp_type_j)+'_'+str(model2)+'_err_var_cov.nc'])
    					err_var_model3_fi_anom = ''.join([err_var_dir_anom+'remap'+str(rmp_type_j)+'_'+str(model3)+'_err_var_cov.nc'])

    					err_var_model1_gcell_fi = ''.join([err_var_dir_raw+'remap'+str(rmp_type_j)+'_'+str(model1)+'_err_var_cov_grid_'+str(gcell)+'.nc'])
    					err_var_model2_gcell_fi = ''.join([err_var_dir_raw+'remap'+str(rmp_type_j)+'_'+str(model2)+'_err_var_cov_grid_'+str(gcell)+'.nc'])
    					err_var_model3_gcell_fi = ''.join([err_var_dir_raw+'remap'+str(rmp_type_j)+'_'+str(model3)+'_err_var_cov_grid_'+str(gcell)+'.nc'])

    					err_var_model1_gcell_fi_anom = ''.join([err_var_dir_anom+'remap'+str(rmp_type_j)+'_'+str(model1)+'_err_var_cov_grid_'+str(gcell)+'.nc'])
    					err_var_model2_gcell_fi_anom = ''.join([err_var_dir_anom+'remap'+str(rmp_type_j)+'_'+str(model2)+'_err_var_cov_grid_'+str(gcell)+'.nc'])
    					err_var_model3_gcell_fi_anom = ''.join([err_var_dir_anom+'remap'+str(rmp_type_j)+'_'+str(model3)+'_err_var_cov_grid_'+str(gcell)+'.nc'])

    					Path(err_var_dir_raw_gcell).mkdir(parents=True,exist_ok=True)
    					Path(err_var_dir_anom_gcell).mkdir(parents=True,exist_ok=True)
					
    					TC_gcell_dir = ''.join([TC_basedir,'grid_cell_level/'])
    					Path(TC_gcell_dir).mkdir(parents=True,exist_ok=True)
    					TC_gcell_fi = ''.join([TC_gcell_dir,'TC_blended_'+str(remap_type)+'_'+str(olr_l)+'_thr'+str(thr_m)+'_'+str(bldsfx)+'_'+str(TC_dir_k)+'_grid_'+str(gcell)+'.nc'])
    					naive_gcell_fi = ''.join([TC_gcell_dir,'naive_blended_'+str(remap_type)+'_'+str(olr_l)+'_thr_'+str(thr_m)+'_'+str(bldsfx)+'_'+str(TC_dir_k)+'_grid_'+str(gcell)+'.nc'])
    					cdo.remapnn('lon='+str(lon_cen)+'/lat='+str(lat_cen), input=CFSR_fi, output=CFSR_gcell_fi, options = '-f nc')    					
    					cdo.remapnn('lon='+str(lon_cen)+'/lat='+str(lat_cen), input=MERRA2_fi, output=MERRA2_gcell_fi, options = '-f nc') 
    					cdo.remapnn('lon='+str(lon_cen)+'/lat='+str(lat_cen), input=ERA5_fi, output=ERA5_gcell_fi, options = '-f nc') 
    					cdo.remapnn('lon='+str(lon_cen)+'/lat='+str(lat_cen), input=ERAI_fi, output=ERAI_gcell_fi, options = '-f nc')
    					cdo.remapnn('lon='+str(lon_cen)+'/lat='+str(lat_cen), input=JRA_fi, output=JRA_gcell_fi, options = '-f nc')
    					cdo.remapnn('lon='+str(lon_cen)+'/lat='+str(lat_cen), input=GLDAS_fi, output=GLDAS_gcell_fi, options = '-f nc') 
    					cdo.remapnn('lon='+str(lon_cen)+'/lat='+str(lat_cen), input=TC_fi, output=TC_gcell_fi, options = '-f nc')
    					cdo.remapnn('lon='+str(lon_cen)+'/lat='+str(lat_cen), input=naive_fi, output=naive_gcell_fi, options = '-f nc')

    					cdo.remapnn('lon='+str(lon_cen)+'/lat='+str(lat_cen), input=err_var_model1_fi, output=err_var_model1_gcell_fi, options = '-f nc')
    					cdo.remapnn('lon='+str(lon_cen)+'/lat='+str(lat_cen), input=err_var_model2_fi, output=err_var_model2_gcell_fi, options = '-f nc')
    					cdo.remapnn('lon='+str(lon_cen)+'/lat='+str(lat_cen), input=err_var_model3_fi, output=err_var_model3_gcell_fi, options = '-f nc')

    					cdo.remapnn('lon='+str(lon_cen)+'/lat='+str(lat_cen), input=err_var_model1_fi_anom, output=err_var_model1_gcell_fi_anom, options = '-f nc')
    					cdo.remapnn('lon='+str(lon_cen)+'/lat='+str(lat_cen), input=err_var_model2_fi_anom, output=err_var_model2_gcell_fi_anom, options = '-f nc')
    					cdo.remapnn('lon='+str(lon_cen)+'/lat='+str(lat_cen), input=err_var_model3_fi_anom, output=err_var_model3_gcell_fi_anom, options = '-f nc')


    					print("finished extracting grid cell level values for this combo")
  					
    					GLDAS_fil = xr.open_dataset(GLDAS_gcell_fi).isel(lat=0,lon=0,drop=True)
    					JRA_fil = xr.open_dataset(JRA_gcell_fi).isel(lat=0,lon=0,drop=True)
    					ERAI_fil = xr.open_dataset(ERAI_gcell_fi).isel(lat=0,lon=0,drop=True)
    					ERA5_fil = xr.open_dataset(ERA5_gcell_fi).isel(lat=0,lon=0,drop=True)
    					MERRA2_fil = xr.open_dataset(MERRA2_gcell_fi).isel(lat=0,lon=0,drop=True)
    					CFSR_fil = xr.open_dataset(CFSR_gcell_fi).isel(lat=0,lon=0,drop=True)
    					TC_fil = xr.open_dataset(TC_gcell_fi).isel(lat=0,lon=0,drop=True)
    					naive_fil = xr.open_dataset(naive_gcell_fi).isel(lat=0,lon=0,drop=True)
    					

    					err_var_model1_fil = xr.open_dataset(err_var_model1_gcell_fi).isel(lat=0,lon=0,drop=True)
    					err_var_model2_fil = xr.open_dataset(err_var_model2_gcell_fi).isel(lat=0,lon=0,drop=True)
    					err_var_model3_fil = xr.open_dataset(err_var_model3_gcell_fi).isel(lat=0,lon=0,drop=True)

    					err_var_model1_fil_anom = xr.open_dataset(err_var_model1_gcell_fi_anom).isel(lat=0,lon=0,drop=True)
    					err_var_model2_fil_anom = xr.open_dataset(err_var_model2_gcell_fi_anom).isel(lat=0,lon=0,drop=True)
    					err_var_model3_fil_anom = xr.open_dataset(err_var_model3_gcell_fi_anom).isel(lat=0,lon=0,drop=True)

    					err_var_model1 = err_var_model1_fil['err_var_cov']
    					err_var_model2 = err_var_model2_fil['err_var_cov']
    					err_var_model3 = err_var_model3_fil['err_var_cov']

    					model1_err_var = err_var_model1.values.tolist()
    					model2_err_var = err_var_model2.values.tolist()
    					model3_err_var = err_var_model3.values.tolist()

    					err_var_model1_anom = err_var_model1_fil_anom['err_var_cov']
    					err_var_model2_anom = err_var_model2_fil_anom['err_var_cov']
    					err_var_model3_anom = err_var_model3_fil_anom['err_var_cov']

    					model1_err_var_anom = err_var_model1_anom.values.tolist()
    					model2_err_var_anom = err_var_model2_anom.values.tolist()
    					model3_err_var_anom = err_var_model3_anom.values.tolist()
					

    					rnys_dattim = TC_fil['time']
    					rnys_datetime = rnys_dattim.dt.strftime('%Y-%m-%d')    					
    					len_rnys_dattim = len(rnys_dattim) - 1
    					rnys_edate = rnys_dattim.isel(time=len_rnys_dattim).values
    					rnys_edate_str = str(rnys_edate)
    					rnys_edate_dt = datetime.datetime.strptime(rnys_edate_str[0:10],'%Y-%m-%d')

    					CFSR_dattim = CFSR_fil['time']
    					CFSR_datetime = CFSR_dattim.dt.strftime('%Y-%m-%d')
    					len_CFSR_dattim = len(CFSR_dattim) - 1 
    					CFSR_sdate = CFSR_dattim.isel(time=0).values
    					CFSR_sdate_str = str(CFSR_sdate)
    					CFSR_sdate_dt = datetime.datetime.strptime(CFSR_sdate_str[0:10],'%Y-%m-%d')
    					CFSR_edate = CFSR_dattim.isel(time=len_CFSR_dattim).values
    					CFSR_edate_str = str(CFSR_edate)
    					CFSR_edate_dt = datetime.datetime.strptime(CFSR_edate_str[0:10],'%Y-%m-%d')

    					if(tmp_type_i == 'raw_temp'):
    						GLDAS_temp = GLDAS_fil[GLDAS_layer] - 273.15
    						JRA_temp = JRA_fil[JRA_layer] - 273.15
    						ERAI_temp = ERAI_fil[ERAI_layer] - 273.15
    						ERA5_temp = ERA5_fil[ERA5_layer] - 273.15
    						MERRA2_temp = MERRA2_fil[MERRA2_layer] - 273.15 #convert from Kelvin to Celsius
    						CFSR_temp = CFSR_fil[CFSR_layer] - 273.15
    						TC_temp = TC_fil['TC_blended_stemp']
    						naive_temp = naive_fil['naive_blended_stemp']

    					elif(tmp_type_i == 'anom'):
    						GLDAS_temp = GLDAS_fil[GLDAS_layer]
    						JRA_temp = JRA_fil[JRA_layer]
    						ERAI_temp = ERAI_fil[ERAI_layer]
    						ERA5_temp = ERA5_fil[ERA5_layer]
    						MERRA2_temp = MERRA2_fil[MERRA2_layer]
    						CFSR_temp = CFSR_fil[CFSR_layer]
    						TC_temp = TC_fil['TC_blended_anom']
    						naive_temp = naive_fil['naive_blended_anom']



#################### grab collocated temperature data from reanalysis files #######################

    					CFSR_temp_master = []
    					JRA_temp_master = []
    					ERAI_temp_master = []
    					ERA5_temp_master = []
    					MERRA2_temp_master = []
    					GLDAS_temp_master = []
    					TC_temp_master = []
    					naive_temp_master = []
    					station_temp_master = []
    					station_anom_master = []
    					date_temp_master = []


    					for n in range(0,len(DateTime)):
    						DateTime_m = DateTime[n]
    						dattim_m = dattim[n]
    						if(DateTime_m > rnys_edate_dt): #skip all dates beyond last reanalysis date
    							continue
    						TC_temp_dt = TC_temp.sel(time=DateTime_m).values.tolist()
    						if(str(TC_temp_dt) == "nan"):
    							TC_temp_dt = np.nan
    						naive_temp_dt = naive_temp.sel(time=DateTime_m).values.tolist()
    						if(str(naive_temp_dt) == "nan"):
    							naive_temp_dt = np.nan  						
    						dframe_insitu_dt = dframe_insitu[dframe_insitu['Date'] == dattim_m]
    						station_temp_dt = dframe_insitu_dt[stemp_nam].values.tolist()
    						if(str(station_temp_dt) == "nan"):
    							station_temp_dt = np.nan
    						station_temp_master.append(station_temp_dt)
    						station_anom_dt = dframe_insitu_dt['Spatial Avg Anom'].tolist()
    						if(str(station_anom_dt) == "nan"):
    							station_anom_dt = np.nan
    						station_anom_master.append(station_anom_dt)									
    						CFSR_temp_dt = CFSR_temp.sel(time=DateTime_m).values.tolist()
    						if(str(CFSR_temp_dt) == "nan"):
    							CFSR_temp_dt = np.nan
    						CFSR_temp_master.append(CFSR_temp_dt)    						
    						JRA_temp_dt = JRA_temp.sel(time=DateTime_m).values.tolist()
    						if(str(JRA_temp_dt) == "nan"):
    							JRA_temp_dt = np.nan
    						JRA_temp_master.append(JRA_temp_dt)      							
    						ERAI_temp_dt = ERAI_temp.sel(time=DateTime_m).values.tolist()
    						if(str(ERAI_temp_dt) == "nan"):
    							ERAI_temp_dt = np.nan
    						ERAI_temp_master.append(ERAI_temp_dt)
    						ERA5_temp_dt = ERA5_temp.sel(time=DateTime_m).values.tolist()
    						if(str(ERA5_temp_dt) == "nan"):
    							ERA5_temp_dt = np.nan
    						ERA5_temp_master.append(ERA5_temp_dt)
    						MERRA2_temp_dt = MERRA2_temp.sel(time=DateTime_m).values.tolist()
    						if(str(MERRA2_temp_dt) == "nan"):
    							MERRA2_temp_dt = np.nan
    						MERRA2_temp_master.append(MERRA2_temp_dt)
    						GLDAS_temp_dt = GLDAS_temp.sel(time=DateTime_m).values.tolist()
    						if(str(GLDAS_temp_dt) == "nan"):
    							GLDAS_temp_dt = np.nan
    						GLDAS_temp_master.append(GLDAS_temp_dt)
    						TC_temp_master.append(TC_temp_dt)
    						date_temp_master.append(dattim_m)    						
    						naive_temp_master.append(naive_temp_dt)            							    						


    					station_temp_master = [i for sub in station_temp_master for i in sub]
    					station_anom_master = [i for sub in station_anom_master for i in sub]
    					station_temp_master = np.array(station_temp_master)
    					station_anom_master = np.array(station_anom_master)
    					date_temp_master = np.array(date_temp_master)
    					CFSR_temp_master = np.array(CFSR_temp_master)
    					ERAI_temp_master = np.array(ERAI_temp_master)
    					ERA5_temp_master = np.array(ERA5_temp_master)
    					JRA_temp_master = np.array(JRA_temp_master)
    					MERRA2_temp_master = np.array(MERRA2_temp_master)
    					GLDAS_temp_master = np.array(GLDAS_temp_master)
    					TC_temp_master = np.array(TC_temp_master)
    					naive_temp_master = np.array(naive_temp_master)
	
#################### create anomalies for reanalysis files #######################
    					rnysis_anom_master = []
    					rnysis_date_master = []
    					rnysis_name_master = []
    					rnysis_stemp_master = []
    					rnysis = [TC_temp_master,naive_temp_master,CFSR_temp_master,ERAI_temp_master,ERA5_temp_master,JRA_temp_master,MERRA2_temp_master,GLDAS_temp_master]
    					rnysis_name = ['TC Blend','Naive Blend','CFSR','ERA-Interim','ERA-5','JRA-55','MERRA2','GLDAS']
    					dat_rowlist = [datetime.datetime.strptime(x,'%Y-%m-%d') for x in date_temp_master]
    					dat_rowlist2 = date_temp_master
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

    						#print(rnysis_stemp_master)

    					naive_no_nan = naive_temp_master[~np.isnan(naive_temp_master)]

    					TC_no_nan = TC_temp_master[~np.isnan(TC_temp_master)]

    					#print(naive_no_nan,TC_no_nan)

    					CFSR_no_nan = CFSR_temp_master[~np.isnan(CFSR_temp_master)]
    					#print(CFSR_no_nan)

    					if(DateTime[0]>CFSR_edate_dt or DateTime[len(DateTime) -1] < CFSR_sdate_dt): #skip if the CFSR dates and station dates do not overlap
    						continue
    					
    					if(len(naive_no_nan) == 0 or len(TC_no_nan) == 0 or len(CFSR_no_nan) == 0): #skip if there are NaN values in blended data
    						continue

####### Station Collocated Anomalies #####

    					dframe_anom_master = pd.DataFrame(data=rnysis_date_master, columns=['Date'])
    					dframe_anom_master['Name'] = rnysis_name_master
    					dframe_anom_master['Raw Temp'] = rnysis_stemp_master
    					dframe_anom_master['Anom'] = rnysis_anom_master
    					dframe_anom_master.dropna(inplace=True)
    					len_dframe_anom = len(dframe_anom_master)

    					station_anom = station_anom_master
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

    					if (len(CFSR_anom) == 0):
    						continue

    					dframe_anom_master2 = pd.DataFrame(data=date_temp_master, columns=['Date'])
    					dframe_anom_master2['Station'] = station_anom_master
    					dframe_anom_master2['TC Blend'] = TC_anom					
    					dframe_anom_master2['Naive Blend'] = naive_anom
    					dframe_anom_master2['CFSR'] = CFSR_anom
    					dframe_anom_master2['ERA-Interim'] = ERAI_anom
    					dframe_anom_master2['ERA5'] = ERA5_anom
    					dframe_anom_master2['JRA55'] = JRA_anom
    					dframe_anom_master2['MERRA2'] = MERRA2_anom
    					dframe_anom_master2['GLDAS'] = GLDAS_anom
    					dframe_anom_master2.dropna(inplace=True)
    					len_dframe_anom = len(dframe_anom_master2)
    					print(dframe_anom_master2)
    					if (len_dframe_anom == 0): #skip if length of non-NaN dframe is 0
    						continue

    					dframe_raw_master = pd.DataFrame(data=date_temp_master, columns=['Date'])
    					dframe_raw_master['Station'] = station_temp_master
    					dframe_raw_master['TC Blend'] = TC_temp_master
    					dframe_raw_master['Naive Blend'] = naive_temp_master
    					dframe_raw_master['CFSR'] = CFSR_temp_master
    					dframe_raw_master['ERA-Interim'] = ERAI_temp_master
    					dframe_raw_master['ERA5'] = ERA5_temp_master
    					dframe_raw_master['JRA55'] = JRA_temp_master
    					dframe_raw_master['MERRA2'] = MERRA2_temp_master
    					dframe_raw_master['GLDAS'] = GLDAS_temp_master
    					dframe_raw_master.dropna(inplace=True)
    					len_dframe_raw = len(dframe_raw_master)
    					if (len_dframe_raw == 0): #skip if length of non-NaN dframe is 0
    						continue	

############### Store Error Variances ############
    					model1_nam_master.append(str(model1)+' Err Var')
    					model2_nam_master.append(str(model2)+' Err Var')
    					model3_nam_master.append(str(model3)+' Err Var')

    					model1_err_var_raw_master.append(model1_err_var)
    					model2_err_var_raw_master.append(model2_err_var)
    					model3_err_var_raw_master.append(model3_err_var)

    					model1_err_var_anom_master.append(model1_err_var_anom)
    					model2_err_var_anom_master.append(model2_err_var_anom)
    					model3_err_var_anom_master.append(model3_err_var_anom)

									
############### Calculate Biases ############
    					station_raw_temp = dframe_raw_master['Station'].values
    					TC_raw_temp = dframe_raw_master['TC Blend'].values
    					naive_raw_temp = dframe_raw_master['Naive Blend'].values
    					CFSR_raw_temp = dframe_raw_master['CFSR'].values
    					ERAI_raw_temp = dframe_raw_master['ERA-Interim'].values
    					ERA5_raw_temp = dframe_raw_master['ERA5'].values
    					JRA_raw_temp = dframe_raw_master['JRA55'].values
    					MERRA2_raw_temp = dframe_raw_master['MERRA2'].values
    					GLDAS_raw_temp = dframe_raw_master['GLDAS'].values

    					station_anom = dframe_anom_master2['Station'].values
    					TC_anom = dframe_anom_master2['TC Blend'].values
    					naive_anom = dframe_anom_master2['Naive Blend'].values
    					CFSR_anom = dframe_anom_master2['CFSR'].values
    					ERAI_anom = dframe_anom_master2['ERA-Interim'].values
    					ERA5_anom = dframe_anom_master2['ERA5'].values
    					JRA_anom = dframe_anom_master2['JRA55'].values
    					MERRA2_anom = dframe_anom_master2['MERRA2'].values
    					GLDAS_anom = dframe_anom_master2['GLDAS'].values


    					gcell = dframe_insitu['Grid Cell'].iloc[0]
    					gcell_master_stn.append(gcell)
    					lat_cen = dframe_insitu['Central Lat'].iloc[0]
    					lat_master_stn.append(lat_cen)
    					lon_cen = dframe_insitu['Central Lon'].iloc[0]
    					lon_master_stn.append(lon_cen)
    					len_raw_master.append(len_dframe_raw)
    					len_anom_master.append(len_dframe_anom)
    				
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
    					#print("Station Data:")
    					#print(DateTime)    					
    					#print(len(y_true_raw))
    					#print("CFSR Data:")
    					#print(CFSR_datetime)
    					#print(len(y_CFSR_raw))

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

    					#print(TC_raw_temp)
    					#print(station_raw_temp)
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

    				model1_name = model1_nam_master[0]
    				model2_name = model2_nam_master[0]
    				model3_name = model3_nam_master[0]

    				df_summary_raw = pd.DataFrame(data=gcell_master_stn, columns=['Grid Cell'])
    				df_summary_raw['Central Lat'] = lat_master_stn
    				df_summary_raw['Central Lon'] = lon_master_stn
    				df_summary_raw['N'] = len_raw_master
    				df_summary_raw[model1_name] = model1_err_var_raw_master
    				df_summary_raw[model2_name] = model2_err_var_raw_master
    				df_summary_raw[model3_name] = model3_err_var_raw_master
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

    				print(df_summary_raw)

    				df_summary_anom = pd.DataFrame(data=gcell_master_stn, columns=['Grid Cell'])
    				df_summary_anom['Central Lat'] = lat_master_stn
    				df_summary_anom['Central Lon'] = lon_master_stn
    				df_summary_anom['N'] = len_anom_master
    				df_summary_anom[model1_name] = model1_err_var_anom_master
    				df_summary_anom[model2_name] = model2_err_var_anom_master
    				df_summary_anom[model3_name] = model3_err_var_anom_master
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
    				print(df_summary_anom)

##### create CSV files #####

    				raw_sum_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/blended_metrics/grid_cell_stn/raw_temp/'+str(TC_dir_k)+'/'+str(remap_type)+'_'+str(olr_l)+'_0_9.9_thr'+str(thr_m)+'_summary_statistics_gridcell_stn.csv'])
    				print(raw_sum_fil)
    				path = pathlib.Path(raw_sum_fil)
    				path.parent.mkdir(parents=True, exist_ok=True)			
    				df_summary_raw.to_csv(raw_sum_fil,index=False)


    				anom_sum_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/blended_metrics/grid_cell_stn/anom/'+str(TC_dir_k)+'/'+str(remap_type)+'_'+str(olr_l)+'_0_9.9_thr'+str(thr_m)+'_summary_statistics_anom_gridcell_stn_.csv'])
    				print(anom_sum_fil)
    				path2 = pathlib.Path(anom_sum_fil)
    				path2.parent.mkdir(parents=True, exist_ok=True)			
    				df_summary_anom.to_csv(anom_sum_fil,index=False)


