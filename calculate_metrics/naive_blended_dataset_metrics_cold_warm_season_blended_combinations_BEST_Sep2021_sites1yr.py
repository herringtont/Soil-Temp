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
import math
from statistics import mean
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
from itertools import combinations


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
tmp_type = ['raw_temp']
temp_thr = ['-2C']#['0C','-2C','-5C','-10C']


############# Loop through data ###########

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


    				for o in temp_thr:
    					temp_thr_o = o

    					if (temp_thr_o == "0C"):
    						tmp_val = 0

    					if (temp_thr_o == "-2C"):
    						tmp_val = -2

    					if (temp_thr_o == "-5C"):
    						tmp_val = -5

    					if (temp_thr_o == "-10C"):
    						tmp_val = -10

    					print("Remap Type:",remap_type)
    					print("Layer:",lyr_l)
    					print("Temp Threshold:", temp_thr_o)


###### Overall (across all validation grid cells) ######

    					fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/CLSM_res/subset/sites_1yr/'+str(remap_type)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_dframe_scatterplot_CMOS_CLSM_subset_permafrost_cold_warm_BEST_Sep2021_sites_1yr.csv'])
    					dframe = pd.read_csv(fil)
    					dframe_cold_season = dframe[dframe['Season'] == 'Cold']

    					dframe_warm_season = dframe[dframe['Season'] == 'Warm']



#### Calculate the Combinatorics ####

    					combos_1_model = combinations(['CFSR','ERA-Interim','ERA5','ERA5-Land','JRA55','MERRA2','GLDAS-Noah','GLDAS-CLSM'],1)
    					CFSR_array = []
    					ERAI_array = []
    					ERA5_array = []
    					ERA5_Land_array = []
    					JRA_array = []
    					MERRA2_array = []
    					GLDAS_array = []
    					GLDAS_CLSM_array = []

    					for i in combos_1_model:
    						#print(i[0])
    						temp_array = []
    						if (i[0] == 'CFSR'):
    							temp_array.append(i[0])
    							CFSR_array.append(temp_array)
    						elif (i[0] == 'ERA-Interim'):
    							temp_array.append(i[0])
    							ERAI_array.append(temp_array)
    						elif (i[0] == 'ERA5'):
    							temp_array.append(i[0])
    							ERA5_array.append(temp_array)
    						elif (i[0] == 'ERA5-Land'):
    							temp_array.append(i[0])
    							ERA5_Land_array.append(temp_array)
    						elif (i[0] == 'JRA55'):
    							temp_array.append(i[0])
    							JRA_array.append(temp_array)
    						elif (i[0] == 'MERRA2'):
    							temp_array.append(i[0])
    							MERRA2_array.append(temp_array)
    						elif (i[0] == 'GLDAS-Noah'):
    							temp_array.append(i[0])
    							GLDAS_array.append(temp_array)
    						elif (i[0] == 'GLDAS-CLSM'):
    							temp_array.append(i[0])
    							GLDAS_CLSM_array.append(temp_array)
						

					
    					combos_2_model = combinations(['CFSR','ERA-Interim','ERA5','ERA5-Land','JRA55','MERRA2','GLDAS-Noah','GLDAS-CLSM'],2)

    					for i in combos_2_model:

    						temp_array = []
    						if (i[0] == 'CFSR' or i[1] == 'CFSR'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							CFSR_array.append(temp_array)
    						elif (i[0] == 'ERA-Interim' or i[1] == 'ERA-Interim'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							ERAI_array.append(temp_array)
    						elif (i[0] == 'ERA5' or i[1] == 'ERA5'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							ERA5_array.append(temp_array)
    						elif (i[0] == 'ERA5-Land' or i[1] == 'ERA5-Land'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							ERA5_Land_array.append(temp_array)
    						elif (i[0] == 'JRA55' or i[1] == 'JRA55'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							JRA_array.append(temp_array)
    						elif (i[0] == 'MERRA2' or i[1] == 'MERRA2'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							MERRA2_array.append(temp_array)
    						elif (i[0] == 'GLDAS-Noah' or i[1] == 'GLDAS-Noah'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							GLDAS_array.append(temp_array)
    						elif (i[0] == 'GLDAS-CLSM' or i[1] == 'GLDAS-CLSM'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							GLDAS_CLSM_array.append(temp_array)


    					combos_3_model = combinations(['CFSR','ERA-Interim','ERA5','ERA5-Land','JRA55','MERRA2','GLDAS-Noah','GLDAS-CLSM'],3)

    					for i in combos_3_model:

    						temp_array = []
    						if (i[0] == 'CFSR' or i[1] == 'CFSR' or i[2] == 'CFSR'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							temp_array.append(i[2])
    							CFSR_array.append(temp_array)
    						elif (i[0] == 'ERA-Interim' or i[1] == 'ERA-Interim' or i[2] == 'ERA-Interim'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							temp_array.append(i[2])
    							ERAI_array.append(temp_array)
    						elif (i[0] == 'ERA5' or i[1] == 'ERA5' or i[2] == 'ERA5'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							temp_array.append(i[2])
    							ERA5_array.append(temp_array)
    						elif (i[0] == 'ERA5-Land' or i[1] == 'ERA5-Land' or i[2] == 'ERA5-Land'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							temp_array.append(i[2])
    							ERA5_Land_array.append(temp_array)
    						elif (i[0] == 'JRA55' or i[1] == 'JRA55' or i[2] == 'JRA55'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							temp_array.append(i[2])
    							JRA_array.append(temp_array)
    						elif (i[0] == 'MERRA2' or i[1] == 'MERRA2' or i[2] == 'MERRA2'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							temp_array.append(i[2])
    							MERRA2_array.append(temp_array)
    						elif (i[0] == 'GLDAS-Noah' or i[1] == 'GLDAS-Noah' or i[2] == 'GLDAS-Noah'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							temp_array.append(i[2])
    							GLDAS_array.append(temp_array)
    						elif (i[0] == 'GLDAS-CLSM' or i[1] == 'GLDAS-CLSM' or i[2] == 'GLDAS-CLSM'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							temp_array.append(i[2])
    							GLDAS_CLSM_array.append(temp_array)


    					combos_4_model = combinations(['CFSR','ERA-Interim','ERA5','ERA5-Land','JRA55','MERRA2','GLDAS-Noah','GLDAS-CLSM'],4)

    					for i in combos_4_model:

    						temp_array = []
    						if (i[0] == 'CFSR' or i[1] == 'CFSR' or i[2] == 'CFSR' or i[3] == 'CFSR'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							temp_array.append(i[2])
    							temp_array.append(i[3])
    							CFSR_array.append(temp_array)
    						elif (i[0] == 'ERA-Interim' or i[1] == 'ERA-Interim' or i[2] == 'ERA-Interim' or i[3] == 'ERA-Interim'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							temp_array.append(i[2])
    							temp_array.append(i[3])
    							ERAI_array.append(temp_array)
    						elif (i[0] == 'ERA5' or i[1] == 'ERA5' or i[2] == 'ERA5' or i[3] == 'ERA5'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							temp_array.append(i[2])
    							temp_array.append(i[3])
    							ERA5_array.append(temp_array)
    						elif (i[0] == 'ERA5-Land' or i[1] == 'ERA5-Land' or i[2] == 'ERA5-Land' or i[3] == 'ERA5-Land'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							temp_array.append(i[2])
    							temp_array.append(i[3])
    							ERA5_Land_array.append(temp_array)
    						elif (i[0] == 'JRA55' or i[1] == 'JRA55' or i[2] == 'JRA55' or i[3] == 'JRA55'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							temp_array.append(i[2])
    							temp_array.append(i[3])
    							JRA_array.append(temp_array)
    						elif (i[0] == 'MERRA2' or i[1] == 'MERRA2' or i[2] == 'MERRA2' or i[3] == 'MERRA2'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							temp_array.append(i[2])
    							temp_array.append(i[3])
    							MERRA2_array.append(temp_array)
    						elif (i[0] == 'GLDAS-Noah' or i[1] == 'GLDAS-Noah' or i[2] == 'GLDAS-Noah' or i[3] == 'GLDAS-Noah'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							temp_array.append(i[2])
    							temp_array.append(i[3])
    							GLDAS_array.append(temp_array)
    						elif (i[0] == 'GLDAS-CLSM' or i[1] == 'GLDAS-CLSM' or i[2] == 'GLDAS-CLSM' or i[3] == 'GLDAS-CLSM'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							temp_array.append(i[2])
    							temp_array.append(i[3])
    							GLDAS_CLSM_array.append(temp_array)

    						#print(CFSR_array)

    					combos_5_model = combinations(['CFSR','ERA-Interim','ERA5','ERA5-Land','JRA55','MERRA2','GLDAS-Noah','GLDAS-CLSM'],5)

    					for i in combos_5_model:

    						temp_array = []
    						if (i[0] == 'CFSR' or i[1] == 'CFSR' or i[2] == 'CFSR' or i[3] == 'CFSR' or i[4] == 'CFSR'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							temp_array.append(i[2])
    							temp_array.append(i[3])
    							temp_array.append(i[4])
    							CFSR_array.append(temp_array)
    						elif (i[0] == 'ERA-Interim' or i[1] == 'ERA-Interim' or i[2] == 'ERA-Interim' or i[3] == 'ERA-Interim' or i[4] == 'ERA-Interim'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							temp_array.append(i[2])
    							temp_array.append(i[3])
    							temp_array.append(i[4])
    							ERAI_array.append(temp_array)
    						elif (i[0] == 'ERA5' or i[1] == 'ERA5' or i[2] == 'ERA5' or i[3] == 'ERA5' or i[4] == 'ERA5'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							temp_array.append(i[2])
    							temp_array.append(i[3])
    							temp_array.append(i[4])
    							ERA5_array.append(temp_array)
    						elif (i[0] == 'ERA5-Land' or i[1] == 'ERA5-Land' or i[2] == 'ERA5-Land' or i[3] == 'ERA5-Land' or i[4] == 'ERA5-Land'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							temp_array.append(i[2])
    							temp_array.append(i[3])
    							temp_array.append(i[4])
    							ERA5_Land_array.append(temp_array)
    						elif (i[0] == 'JRA55' or i[1] == 'JRA55' or i[2] == 'JRA55' or i[3] == 'JRA55' or i[4] == 'JRA55'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							temp_array.append(i[2])
    							temp_array.append(i[3])
    							temp_array.append(i[4])
    							JRA_array.append(temp_array)
    						elif (i[0] == 'MERRA2' or i[1] == 'MERRA2' or i[2] == 'MERRA2' or i[3] == 'MERRA2' or i[4] == 'MERRA2'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							temp_array.append(i[2])
    							temp_array.append(i[3])
    							temp_array.append(i[4])
    							MERRA2_array.append(temp_array)
    						elif (i[0] == 'GLDAS-Noah' or i[1] == 'GLDAS-Noah' or i[2] == 'GLDAS-Noah' or i[3] == 'GLDAS-Noah' or i[4] == 'GLDAS-Noah'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							temp_array.append(i[2])
    							temp_array.append(i[3])
    							temp_array.append(i[4])
    							GLDAS_array.append(temp_array)
    						elif (i[0] == 'GLDAS-CLSM' or i[1] == 'GLDAS-CLSM' or i[2] == 'GLDAS-CLSM' or i[3] == 'GLDAS-CLSM' or i[4] == 'GLDAS-CLSM'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							temp_array.append(i[2])
    							temp_array.append(i[3])
    							temp_array.append(i[4])
    							GLDAS_CLSM_array.append(temp_array)

    					combos_6_model = combinations(['CFSR','ERA-Interim','ERA5','ERA5-Land','JRA55','MERRA2','GLDAS-Noah','GLDAS-CLSM'],6)

    					for i in combos_6_model:

    						temp_array = []
    						if (i[0] == 'CFSR' or i[1] == 'CFSR' or i[2] == 'CFSR' or i[3] == 'CFSR' or i[4] == 'CFSR' or i[5] == 'CFSR'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							temp_array.append(i[2])
    							temp_array.append(i[3])
    							temp_array.append(i[4])
    							temp_array.append(i[5])
    							CFSR_array.append(temp_array)
    						elif (i[0] == 'ERA-Interim' or i[1] == 'ERA-Interim' or i[2] == 'ERA-Interim' or i[3] == 'ERA-Interim' or i[4] == 'ERA-Interim' or i[5] == 'ERA-Interim'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							temp_array.append(i[2])
    							temp_array.append(i[3])
    							temp_array.append(i[4])
    							temp_array.append(i[5])
    							ERAI_array.append(temp_array)
    						elif (i[0] == 'ERA5' or i[1] == 'ERA5' or i[2] == 'ERA5' or i[3] == 'ERA5' or i[4] == 'ERA5' or i[5] == 'ERA5'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							temp_array.append(i[2])
    							temp_array.append(i[3])
    							temp_array.append(i[4])
    							temp_array.append(i[5])
    							ERA5_array.append(temp_array)
    						elif (i[0] == 'ERA5-Land' or i[1] == 'ERA5-Land' or i[2] == 'ERA5-Land' or i[3] == 'ERA5-Land' or i[4] == 'ERA5-Land' or i[5] == 'ERA5-Land'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							temp_array.append(i[2])
    							temp_array.append(i[3])
    							temp_array.append(i[4])
    							temp_array.append(i[5])
    							ERA5_Land_array.append(temp_array)
    						elif (i[0] == 'JRA55' or i[1] == 'JRA55' or i[2] == 'JRA55' or i[3] == 'JRA55' or i[4] == 'JRA55' or i[5] == 'JRA55'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							temp_array.append(i[2])
    							temp_array.append(i[3])
    							temp_array.append(i[4])
    							temp_array.append(i[5])
    							JRA_array.append(temp_array)
    						elif (i[0] == 'MERRA2' or i[1] == 'MERRA2' or i[2] == 'MERRA2' or i[3] == 'MERRA2' or i[4] == 'MERRA2' or i[5] == 'MERRA2'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							temp_array.append(i[2])
    							temp_array.append(i[3])
    							temp_array.append(i[4])
    							temp_array.append(i[5])
    							MERRA2_array.append(temp_array)
    						elif (i[0] == 'GLDAS-Noah' or i[1] == 'GLDAS-Noah' or i[2] == 'GLDAS-Noah' or i[3] == 'GLDAS-Noah' or i[4] == 'GLDAS-Noah' or i[5] == 'GLDAS-Noah'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							temp_array.append(i[2])
    							temp_array.append(i[3])
    							temp_array.append(i[4])
    							temp_array.append(i[5])
    							GLDAS_array.append(temp_array)
    						elif (i[0] == 'GLDAS-CLSM' or i[1] == 'GLDAS-CLSM' or i[2] == 'GLDAS-CLSM' or i[3] == 'GLDAS-CLSM' or i[4] == 'GLDAS-CLSM' or i[5] == 'GLDAS-CLSM'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							temp_array.append(i[2])
    							temp_array.append(i[3])
    							temp_array.append(i[4])
    							temp_array.append(i[5])
    							GLDAS_CLSM_array.append(temp_array)


    					combos_7_model = combinations(['CFSR','ERA-Interim','ERA5','ERA5-Land','JRA55','MERRA2','GLDAS-Noah','GLDAS-CLSM'],7)

    					for i in combos_7_model:

    						temp_array = []
    						if (i[0] == 'CFSR' or i[1] == 'CFSR' or i[2] == 'CFSR' or i[3] == 'CFSR' or i[4] == 'CFSR' or i[5] == 'CFSR' or i[6] == 'CFSR'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							temp_array.append(i[2])
    							temp_array.append(i[3])
    							temp_array.append(i[4])
    							temp_array.append(i[5])
    							temp_array.append(i[6])
    							CFSR_array.append(temp_array)
    						elif (i[0] == 'ERA-Interim' or i[1] == 'ERA-Interim' or i[2] == 'ERA-Interim' or i[3] == 'ERA-Interim' or i[4] == 'ERA-Interim' or i[5] == 'ERA-Interim' or i[6] == 'ERA-Interim'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							temp_array.append(i[2])
    							temp_array.append(i[3])
    							temp_array.append(i[4])
    							temp_array.append(i[5])
    							temp_array.append(i[6])
    							ERAI_array.append(temp_array)
    						elif (i[0] == 'ERA5' or i[1] == 'ERA5' or i[2] == 'ERA5' or i[3] == 'ERA5' or i[4] == 'ERA5' or i[5] == 'ERA5' or i[6] == 'ERA5'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							temp_array.append(i[2])
    							temp_array.append(i[3])
    							temp_array.append(i[4])
    							temp_array.append(i[5])
    							temp_array.append(i[6])
    							ERA5_array.append(temp_array)
    						elif (i[0] == 'ERA5-Land' or i[1] == 'ERA5-Land' or i[2] == 'ERA5-Land' or i[3] == 'ERA5-Land' or i[4] == 'ERA5-Land' or i[5] == 'ERA5-Land' or i[6] == 'ERA5-Land'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							temp_array.append(i[2])
    							temp_array.append(i[3])
    							temp_array.append(i[4])
    							temp_array.append(i[5])
    							temp_array.append(i[6])
    							ERA5_Land_array.append(temp_array)
    						elif (i[0] == 'JRA55' or i[1] == 'JRA55' or i[2] == 'JRA55' or i[3] == 'JRA55' or i[4] == 'JRA55' or i[5] == 'JRA55' or i[6] == 'JRA55'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							temp_array.append(i[2])
    							temp_array.append(i[3])
    							temp_array.append(i[4])
    							temp_array.append(i[5])
    							temp_array.append(i[6])
    							JRA_array.append(temp_array)
    						elif (i[0] == 'MERRA2' or i[1] == 'MERRA2' or i[2] == 'MERRA2' or i[3] == 'MERRA2' or i[4] == 'MERRA2' or i[5] == 'MERRA2' or i[6] == 'MERRA2'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							temp_array.append(i[2])
    							temp_array.append(i[3])
    							temp_array.append(i[4])
    							temp_array.append(i[5])
    							temp_array.append(i[6])
    							MERRA2_array.append(temp_array)
    						elif (i[0] == 'GLDAS-Noah' or i[1] == 'GLDAS-Noah' or i[2] == 'GLDAS-Noah' or i[3] == 'GLDAS-Noah' or i[4] == 'GLDAS-Noah' or i[5] == 'GLDAS-Noah' or i[6] == 'GLDAS-Noah'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							temp_array.append(i[2])
    							temp_array.append(i[3])
    							temp_array.append(i[4])
    							temp_array.append(i[5])
    							temp_array.append(i[6])
    							GLDAS_array.append(temp_array)
    						elif (i[0] == 'GLDAS-CLSM' or i[1] == 'GLDAS-CLSM' or i[2] == 'GLDAS-CLSM' or i[3] == 'GLDAS-CLSM' or i[4] == 'GLDAS-CLSM' or i[5] == 'GLDAS-CLSM' or i[6] == 'GLDAS-CLSM'):
    							temp_array.append(i[0])
    							temp_array.append(i[1])
    							temp_array.append(i[2])
    							temp_array.append(i[3])
    							temp_array.append(i[4])
    							temp_array.append(i[5])
    							temp_array.append(i[6])
    							GLDAS_CLSM_array.append(temp_array)


    					combos_8_model = combinations(['CFSR','ERA-Interim','ERA5','ERA5-Land','JRA55','MERRA2','GLDAS-Noah','GLDAS-CLSM'],8)

    					CFSR_array.append(['CFSR','ERA-Interim','ERA5','ERA5-Land','JRA55','MERRA2','GLDAS-Noah','GLDAS-CLSM'])
    					ERAI_array.append(['CFSR','ERA-Interim','ERA5','ERA5-Land','JRA55','MERRA2','GLDAS-Noah','GLDAS-CLSM'])
    					ERA5_array.append(['CFSR','ERA-Interim','ERA5','ERA5-Land','JRA55','MERRA2','GLDAS-Noah','GLDAS-CLSM'])
    					ERA5_Land_array.append(['CFSR','ERA-Interim','ERA5','ERA5-Land','JRA55','MERRA2','GLDAS-Noah','GLDAS-CLSM'])
    					JRA_array.append(['CFSR','ERA-Interim','ERA5','ERA5-Land','JRA55','MERRA2','GLDAS-Noah','GLDAS-CLSM'])
    					MERRA2_array.append(['CFSR','ERA-Interim','ERA5','ERA5-Land','JRA55','MERRA2','GLDAS-Noah','GLDAS-CLSM'])
    					GLDAS_array.append(['CFSR','ERA-Interim','ERA5','ERA5-Land','JRA55','MERRA2','GLDAS-Noah','GLDAS-CLSM'])
    					GLDAS_CLSM_array.append(['CFSR','ERA-Interim','ERA5','ERA5-Land','JRA55','MERRA2','GLDAS-Noah','GLDAS-CLSM'])


#### Cold Season ####

## Calculate All Possible Combinations ##

    					gcell_cold = dframe_cold_season['Grid Cell'].values
    					gcell_cold_uq = np.unique(gcell_cold)


    					bias_naive_cold_gcell_master = []
    					stdev_naive_cold_gcell_master = []
    					rmse_naive_cold_gcell_master = []
    					corr_naive_cold_gcell_master = []
    					stdev_stn_cold_gcell_master = []
    					for p in gcell_cold_uq:
    						if (p == 33777):
    							continue
    						dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]

    						if (len(dframe_cold_season_gcell) < 2):
    							continue

    						station_temp_cold = dframe_cold_season_gcell['Station'].values
    						naive_temp_cold = dframe_cold_season_gcell['Naive Blend All'].values
    						bias_naive_cold = bias(naive_temp_cold,station_temp_cold)
    						bias_naive_cold_gcell_master.append(bias_naive_cold)
    						stdev_naive_cold = np.var(naive_temp_cold)
    						stdev_naive_cold_gcell_master.append(stdev_naive_cold)
    						stdev_station_cold = np.var(station_temp_cold)
    						stdev_stn_cold_gcell_master.append(stdev_station_cold)
    						rmse_naive_cold = mean_squared_error(station_temp_cold,naive_temp_cold,squared=False)
    						rmse_naive_cold_gcell_master.append(rmse_naive_cold)
    						corr_naive_cold,_ = pearsonr(naive_temp_cold,station_temp_cold)
    						corr_naive_cold_gcell_master.append(corr_naive_cold)

    					bias_naive_cold_mean = mean(bias_naive_cold_gcell_master)
    					stdev_naive_cold_mean = mean(stdev_naive_cold_gcell_master)
    					stdev_naive_cold_mean = math.sqrt(stdev_naive_cold_mean)
    					stdev_stn_cold_mean = mean(stdev_stn_cold_gcell_master)
    					stdev_station_cold = math.sqrt(stdev_stn_cold_mean)
    					SDV_naive_cold_mean = stdev_stn_cold_mean/stdev_station_cold
    					rmse_naive_cold_mean = mean(rmse_naive_cold_gcell_master)
    					corr_naive_cold_mean = mean(corr_naive_cold_gcell_master) 						


					
# 1 model combos #

    					bias_1_model_cold_master = []
    					rmse_1_model_cold_master = []
    					stdev_1_model_cold_master = []
    					SDV_1_model_cold_master = []
    					corr_1_model_cold_master = []

    					for i in ['CFSR','ERA-Interim','ERA5','ERA5-Land','JRA55','MERRA2','GLDAS-Noah','GLDAS-CLSM']:
    						blend_1_model_temp_cold_gcell_master = []
    						bias_1_model_cold_gcell_master = []
    						stdev_1_model_cold_gcell_master = []
    						rmse_1_model_cold_gcell_master = []
    						corr_1_model_cold_gcell_master = []
    						for p in gcell_cold_uq:
    							if (p == 3377):
    								continue
    							dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]


    							if (len(dframe_cold_season_gcell) < 2):
    								continue
    							station_temp_cold = dframe_cold_season_gcell['Station'].values
    							blend_1_model_temp_cold = dframe_cold_season_gcell[i].values
    							bias_1_model_cold = bias(blend_1_model_temp_cold,station_temp_cold)
    							bias_1_model_cold_gcell_master.append(bias_1_model_cold)
    							stdev_1_model_cold = np.var(blend_1_model_temp_cold)
    							stdev_1_model_cold_gcell_master.append(stdev_1_model_cold)
    							rmse_1_model_cold = mean_squared_error(station_temp_cold,blend_1_model_temp_cold,squared=False)
    							rmse_1_model_cold_gcell_master.append(rmse_1_model_cold)
    							corr_1_model_cold,_ = pearsonr(blend_1_model_temp_cold,station_temp_cold)
    							corr_1_model_cold_gcell_master.append(corr_1_model_cold)
    						bias_1_model_cold_gcell_mean = mean(bias_1_model_cold_gcell_master)
    						bias_1_model_cold_master.append(bias_1_model_cold_gcell_mean)
    						stdev_1_model_cold_gcell_mean = mean(stdev_1_model_cold_gcell_master)
    						stdev_1_model_cold_master.append(stdev_1_model_cold_gcell_mean)
    						rmse_1_model_cold_gcell_mean = mean(rmse_1_model_cold_gcell_master)
    						rmse_1_model_cold_master.append(rmse_1_model_cold_gcell_mean)
    						corr_1_model_cold_gcell_mean = mean(corr_1_model_cold_gcell_master)
    						corr_1_model_cold_master.append(corr_1_model_cold_gcell_mean)						    												
    					bias_1_model_cold_mean = mean(bias_1_model_cold_master)
    					stdev_1_model_cold_mean = mean(stdev_1_model_cold_master)
    					stdev_1_model_cold_mean2 = math.sqrt(stdev_1_model_cold_mean)
    					SDV_1_model_cold_mean = stdev_1_model_cold_mean2/stdev_station_cold
    					rmse_1_model_cold_mean = mean(rmse_1_model_cold_master)
    					corr_1_model_cold_mean = mean(corr_1_model_cold_master)    					

    						
# 2 model combos #

    					bias_2_model_cold_master = []
    					rmse_2_model_cold_master = []
    					stdev_2_model_cold_master = []
    					SDV_2_model_cold_master = []
    					corr_2_model_cold_master = []

    					blend_combos_2 = combinations(['CFSR','ERA-Interim','ERA5','ERA5-Land','JRA55','MERRA2','GLDAS-Noah','GLDAS-CLSM'],2)
    					for i in blend_combos_2:
    						combo = i

    						model_1 = i[0]
    						model_2 = i[1]

    						blend_2_model_temp_cold_gcell_master = []
    						bias_2_model_cold_gcell_master = []
    						stdev_2_model_cold_gcell_master = []
    						rmse_2_model_cold_gcell_master = []
    						corr_2_model_cold_gcell_master = []
    						for p in gcell_cold_uq:
    							dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    							if (len(dframe_cold_season_gcell) < 2):
    								continue
    							station_temp_cold = dframe_cold_season_gcell['Station'].values
    							model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    							model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    							dframe_2_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    							dframe_2_model[model_2] = model_2_temp_cold
    							dframe_2_model_avg = dframe_2_model.mean(axis=1)
    							blend_2_model_temp_cold = dframe_2_model_avg.values
    							bias_2_model_cold = bias(blend_2_model_temp_cold,station_temp_cold)
    							bias_2_model_cold_gcell_master.append(bias_2_model_cold)
    							stdev_2_model_cold = np.std(blend_2_model_temp_cold)
    							stdev_2_model_cold_gcell_master.append(stdev_2_model_cold)
    							rmse_2_model_cold = mean_squared_error(station_temp_cold,blend_2_model_temp_cold,squared=False)
    							rmse_2_model_cold_gcell_master.append(rmse_2_model_cold)
    							corr_2_model_cold,_ = pearsonr(blend_2_model_temp_cold,station_temp_cold)
    							corr_2_model_cold_gcell_master.append(corr_2_model_cold)
    						bias_2_model_cold_gcell_mean = mean(bias_2_model_cold_gcell_master)
    						bias_2_model_cold_master.append(bias_2_model_cold_gcell_mean)
    						stdev_2_model_cold_gcell_mean = mean(stdev_2_model_cold_gcell_master)
    						stdev_2_model_cold_master.append(stdev_2_model_cold_gcell_mean)
    						rmse_2_model_cold_gcell_mean = mean(rmse_2_model_cold_gcell_master)
    						rmse_2_model_cold_master.append(rmse_2_model_cold_gcell_mean)
    						corr_2_model_cold_gcell_mean = mean(corr_2_model_cold_gcell_master)
    						corr_2_model_cold_master.append(corr_2_model_cold_gcell_mean)						    												
    					bias_2_model_cold_mean = mean(bias_2_model_cold_master)
    					stdev_2_model_cold_mean = mean(stdev_2_model_cold_master)
    					stdev_2_model_cold_mean2 = math.sqrt(stdev_2_model_cold_mean)
    					SDV_2_model_cold_mean = stdev_2_model_cold_mean2/stdev_station_cold
    					rmse_2_model_cold_mean = mean(rmse_2_model_cold_master)
    					corr_2_model_cold_mean = mean(corr_2_model_cold_master) 



# 3 model combos #

    					bias_3_model_cold_master = []
    					rmse_3_model_cold_master = []
    					stdev_3_model_cold_master = []
    					SDV_3_model_cold_master = []
    					corr_3_model_cold_master = []

    					blend_combos_3 = combinations(['CFSR','ERA-Interim','ERA5','ERA5-Land','JRA55','MERRA2','GLDAS-Noah','GLDAS-CLSM'],3)
    					for i in blend_combos_3:
    						combo = i

    						model_1 = i[0]
    						model_2 = i[1]
    						model_3 = i[2]

    						blend_3_model_temp_cold_gcell_master = []
    						bias_3_model_cold_gcell_master = []
    						stdev_3_model_cold_gcell_master = []
    						rmse_3_model_cold_gcell_master = []
    						corr_3_model_cold_gcell_master = []
    						for p in gcell_cold_uq:
    							if (p == 3377):
    								continue
    							dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    							if (len(dframe_cold_season_gcell) < 2):
    								continue
    							station_temp_cold = dframe_cold_season_gcell['Station'].values
    							model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    							model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    							model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    							dframe_3_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    							dframe_3_model[model_2] = model_2_temp_cold
    							dframe_3_model[model_3] = model_3_temp_cold						
    							dframe_3_model_avg = dframe_3_model.mean(axis=1)
    							blend_3_model_temp_cold = dframe_3_model_avg.values
    							bias_3_model_cold = bias(blend_3_model_temp_cold,station_temp_cold)
    							bias_3_model_cold_gcell_master.append(bias_3_model_cold)
    							stdev_3_model_cold = np.std(blend_3_model_temp_cold)
    							stdev_3_model_cold_gcell_master.append(stdev_3_model_cold)
    							rmse_3_model_cold = mean_squared_error(station_temp_cold,blend_3_model_temp_cold,squared=False)
    							rmse_3_model_cold_gcell_master.append(rmse_3_model_cold)
    							corr_3_model_cold,_ = pearsonr(blend_3_model_temp_cold,station_temp_cold)
    							corr_3_model_cold_gcell_master.append(corr_3_model_cold)

    						bias_3_model_cold_gcell_mean = mean(bias_3_model_cold_gcell_master)
    						bias_3_model_cold_master.append(bias_3_model_cold_gcell_mean)
    						stdev_3_model_cold_gcell_mean = mean(stdev_3_model_cold_gcell_master)
    						stdev_3_model_cold_master.append(stdev_3_model_cold_gcell_mean)
    						rmse_3_model_cold_gcell_mean = mean(rmse_3_model_cold_gcell_master)
    						rmse_3_model_cold_master.append(rmse_3_model_cold_gcell_mean)
    						corr_3_model_cold_gcell_mean = mean(corr_3_model_cold_gcell_master)
    						corr_3_model_cold_master.append(corr_3_model_cold_gcell_mean)						    												
    					bias_3_model_cold_mean = mean(bias_3_model_cold_master)
    					stdev_3_model_cold_mean = mean(stdev_3_model_cold_master)
    					stdev_3_model_cold_mean2 = math.sqrt(stdev_3_model_cold_mean)
    					SDV_3_model_cold_mean = stdev_3_model_cold_mean2/stdev_station_cold
    					rmse_3_model_cold_mean = mean(rmse_3_model_cold_master)
    					corr_3_model_cold_mean = mean(corr_3_model_cold_master) 



# 4 model combos #

    					bias_4_model_cold_master = []
    					rmse_4_model_cold_master = []
    					stdev_4_model_cold_master = []
    					SDV_4_model_cold_master = []
    					corr_4_model_cold_master = []

    					blend_combos_4 = combinations(['CFSR','ERA-Interim','ERA5','ERA5-Land','JRA55','MERRA2','GLDAS-Noah','GLDAS-CLSM'],4)
    					for i in blend_combos_4:
    						combo = i

    						model_1 = i[0]
    						model_2 = i[1]
    						model_3 = i[2]
    						model_4 = i[3]

    						blend_4_model_temp_cold_gcell_master = []
    						bias_4_model_cold_gcell_master = []
    						stdev_4_model_cold_gcell_master = []
    						rmse_4_model_cold_gcell_master = []
    						corr_4_model_cold_gcell_master = []
    						for p in gcell_cold_uq:
    							if (p == 3377):
    								continue
    							dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    							if (len(dframe_cold_season_gcell) < 2):
    								continue
    							station_temp_cold = dframe_cold_season_gcell['Station'].values
    							model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    							model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    							model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    							model_4_temp_cold = dframe_cold_season_gcell[model_4].values						
    							dframe_4_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    							dframe_4_model[model_2] = model_2_temp_cold
    							dframe_4_model[model_3] = model_3_temp_cold
    							dframe_4_model[model_4] = model_4_temp_cold
    							dframe_4_model_avg = dframe_4_model.mean(axis=1)
    							blend_4_model_temp_cold = dframe_4_model_avg.values
    							bias_4_model_cold = bias(blend_4_model_temp_cold,station_temp_cold)
    							bias_4_model_cold_gcell_master.append(bias_4_model_cold)
    							stdev_4_model_cold = np.std(blend_4_model_temp_cold)
    							stdev_4_model_cold_gcell_master.append(stdev_4_model_cold)
    							rmse_4_model_cold = mean_squared_error(station_temp_cold,blend_4_model_temp_cold,squared=False)
    							rmse_4_model_cold_gcell_master.append(rmse_4_model_cold)
    							corr_4_model_cold,_ = pearsonr(blend_4_model_temp_cold,station_temp_cold)
    							corr_4_model_cold_gcell_master.append(corr_4_model_cold)


    						bias_4_model_cold_gcell_mean = mean(bias_4_model_cold_gcell_master)
    						bias_4_model_cold_master.append(bias_4_model_cold_gcell_mean)
    						stdev_4_model_cold_gcell_mean = mean(stdev_4_model_cold_gcell_master)
    						stdev_4_model_cold_master.append(stdev_4_model_cold_gcell_mean)
    						rmse_4_model_cold_gcell_mean = mean(rmse_4_model_cold_gcell_master)
    						rmse_4_model_cold_master.append(rmse_4_model_cold_gcell_mean)
    						corr_4_model_cold_gcell_mean = mean(corr_4_model_cold_gcell_master)
    						corr_4_model_cold_master.append(corr_4_model_cold_gcell_mean)						    												
    					bias_4_model_cold_mean = mean(bias_4_model_cold_master)
    					stdev_4_model_cold_mean = mean(stdev_4_model_cold_master)
    					stdev_4_model_cold_mean2 = math.sqrt(stdev_4_model_cold_mean)
    					SDV_4_model_cold_mean = stdev_4_model_cold_mean2/stdev_station_cold
    					rmse_4_model_cold_mean = mean(rmse_4_model_cold_master)
    					corr_4_model_cold_mean = mean(corr_4_model_cold_master)

# 5 model combos #

    					bias_5_model_cold_master = []
    					rmse_5_model_cold_master = []
    					stdev_5_model_cold_master = []
    					SDV_5_model_cold_master = []
    					corr_5_model_cold_master = []

    					blend_combos_5 = combinations(['CFSR','ERA-Interim','ERA5','ERA5-Land','JRA55','MERRA2','GLDAS-Noah','GLDAS-CLSM'],5)
    					for i in blend_combos_5:
    						combo = i

    						model_1 = i[0]
    						model_2 = i[1]
    						model_3 = i[2]
    						model_4 = i[3]
    						model_5 = i[4]
    						blend_5_model_temp_cold_gcell_master = []
    						bias_5_model_cold_gcell_master = []
    						stdev_5_model_cold_gcell_master = []
    						rmse_5_model_cold_gcell_master = []
    						corr_5_model_cold_gcell_master = []
    						for p in gcell_cold_uq:
    							if (p == 3377):
    								continue
    							dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    							if (len(dframe_cold_season_gcell) < 2):
    								continue
    							station_temp_cold = dframe_cold_season_gcell['Station'].values
    							model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    							model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    							model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    							model_4_temp_cold = dframe_cold_season_gcell[model_4].values
    							model_5_temp_cold = dframe_cold_season_gcell[model_5].values
    							dframe_5_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    							dframe_5_model[model_2] = model_2_temp_cold
    							dframe_5_model[model_3] = model_3_temp_cold
    							dframe_5_model[model_4] = model_4_temp_cold
    							dframe_5_model[model_5] = model_5_temp_cold
    							dframe_5_model_avg = dframe_5_model.mean(axis=1)
    							blend_5_model_temp_cold = dframe_5_model_avg.values

    							bias_5_model_cold = bias(blend_5_model_temp_cold,station_temp_cold)
    							bias_5_model_cold_gcell_master.append(bias_5_model_cold)
    							stdev_5_model_cold = np.std(blend_5_model_temp_cold)
    							stdev_5_model_cold_gcell_master.append(stdev_5_model_cold)
    							rmse_5_model_cold = mean_squared_error(station_temp_cold,blend_5_model_temp_cold,squared=False)
    							rmse_5_model_cold_gcell_master.append(rmse_5_model_cold)
    							corr_5_model_cold,_ = pearsonr(blend_5_model_temp_cold,station_temp_cold)
    							corr_5_model_cold_gcell_master.append(corr_5_model_cold)

    						bias_5_model_cold_gcell_mean = mean(bias_5_model_cold_gcell_master)
    						bias_5_model_cold_master.append(bias_5_model_cold_gcell_mean)
    						stdev_5_model_cold_gcell_mean = mean(stdev_5_model_cold_gcell_master)
    						stdev_5_model_cold_master.append(stdev_5_model_cold_gcell_mean)
    						rmse_5_model_cold_gcell_mean = mean(rmse_5_model_cold_gcell_master)
    						rmse_5_model_cold_master.append(rmse_5_model_cold_gcell_mean)
    						corr_5_model_cold_gcell_mean = mean(corr_5_model_cold_gcell_master)
    						corr_5_model_cold_master.append(corr_5_model_cold_gcell_mean)						    												
    					bias_5_model_cold_mean = mean(bias_5_model_cold_master)
    					stdev_5_model_cold_mean = mean(stdev_5_model_cold_master)
    					stdev_5_model_cold_mean2 = math.sqrt(stdev_5_model_cold_mean)
    					SDV_5_model_cold_mean = stdev_5_model_cold_mean2/stdev_station_cold
    					rmse_5_model_cold_mean = mean(rmse_5_model_cold_master)
    					corr_5_model_cold_mean = mean(corr_5_model_cold_master) 

# 6 model combos #

    					bias_6_model_cold_master = []
    					rmse_6_model_cold_master = []
    					stdev_6_model_cold_master = []
    					SDV_6_model_cold_master = []
    					corr_6_model_cold_master = []

    					blend_combos_6 = combinations(['CFSR','ERA-Interim','ERA5','ERA5-Land','JRA55','MERRA2','GLDAS-Noah','GLDAS-CLSM'],6)
    					for i in blend_combos_6:
    						combo = i

    						model_1 = i[0]
    						model_2 = i[1]
    						model_3 = i[2]
    						model_4 = i[3]
    						model_5 = i[4]
    						model_6 = i[5]

    						blend_6_model_temp_cold_gcell_master = []
    						bias_6_model_cold_gcell_master = []
    						stdev_6_model_cold_gcell_master = []
    						rmse_6_model_cold_gcell_master = []
    						corr_6_model_cold_gcell_master = []
    						for p in gcell_cold_uq:
    							if (p == 3377):
    								continue
    							dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    							if (len(dframe_cold_season_gcell) < 2):
    								continue
    							station_temp_cold = dframe_cold_season_gcell['Station'].values
    							model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    							model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    							model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    							model_4_temp_cold = dframe_cold_season_gcell[model_4].values
    							model_5_temp_cold = dframe_cold_season_gcell[model_5].values
    							model_6_temp_cold = dframe_cold_season_gcell[model_6].values							
    							dframe_6_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    							dframe_6_model[model_2] = model_2_temp_cold
    							dframe_6_model[model_3] = model_3_temp_cold
    							dframe_6_model[model_4] = model_4_temp_cold
    							dframe_6_model[model_5] = model_5_temp_cold
    							dframe_6_model[model_6] = model_6_temp_cold
    							dframe_6_model_avg = dframe_6_model.mean(axis=1)
    							blend_6_model_temp_cold = dframe_6_model_avg.values
    							bias_6_model_cold = bias(blend_6_model_temp_cold,station_temp_cold)
    							bias_6_model_cold_gcell_master.append(bias_6_model_cold)
    							stdev_6_model_cold = np.std(blend_6_model_temp_cold)
    							stdev_6_model_cold_gcell_master.append(stdev_6_model_cold)
    							rmse_6_model_cold = mean_squared_error(station_temp_cold,blend_6_model_temp_cold,squared=False)
    							rmse_6_model_cold_gcell_master.append(rmse_6_model_cold)
    							corr_6_model_cold,_ = pearsonr(blend_6_model_temp_cold,station_temp_cold)
    							corr_6_model_cold_gcell_master.append(corr_6_model_cold)

    						bias_6_model_cold_gcell_mean = mean(bias_6_model_cold_gcell_master)
    						bias_6_model_cold_master.append(bias_6_model_cold_gcell_mean)
    						stdev_6_model_cold_gcell_mean = mean(stdev_6_model_cold_gcell_master)
    						stdev_6_model_cold_master.append(stdev_6_model_cold_gcell_mean)
    						rmse_6_model_cold_gcell_mean = mean(rmse_6_model_cold_gcell_master)
    						rmse_6_model_cold_master.append(rmse_6_model_cold_gcell_mean)
    						corr_6_model_cold_gcell_mean = mean(corr_6_model_cold_gcell_master)
    						corr_6_model_cold_master.append(corr_6_model_cold_gcell_mean)						    												
    					bias_6_model_cold_mean = mean(bias_6_model_cold_master)
    					stdev_6_model_cold_mean = mean(stdev_6_model_cold_master)
    					stdev_6_model_cold_mean2 = math.sqrt(stdev_6_model_cold_mean)
    					SDV_6_model_cold_mean = stdev_6_model_cold_mean2/stdev_station_cold
    					rmse_6_model_cold_mean = mean(rmse_6_model_cold_master)
    					corr_6_model_cold_mean = mean(corr_6_model_cold_master)




# 7 model combos #

    					bias_7_model_cold_master = []
    					rmse_7_model_cold_master = []
    					stdev_7_model_cold_master = []
    					SDV_7_model_cold_master = []
    					corr_7_model_cold_master = []

    					blend_combos_7 = combinations(['CFSR','ERA-Interim','ERA5','ERA5-Land','JRA55','MERRA2','GLDAS-Noah','GLDAS-CLSM'],7)
    					for i in blend_combos_7:
    						combo = i

    						model_1 = i[0]
    						model_2 = i[1]
    						model_3 = i[2]
    						model_4 = i[3]
    						model_5 = i[4]
    						model_6 = i[5]
    						model_7 = i[6]

    						blend_7_model_temp_cold_gcell_master = []
    						bias_7_model_cold_gcell_master = []
    						stdev_7_model_cold_gcell_master = []
    						rmse_7_model_cold_gcell_master = []
    						corr_7_model_cold_gcell_master = []
    						for p in gcell_cold_uq:
    							if (p == 3377):
    								continue
    							dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    							if (len(dframe_cold_season_gcell) < 2):
    								continue
    							station_temp_cold = dframe_cold_season_gcell['Station'].values
    							model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    							model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    							model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    							model_4_temp_cold = dframe_cold_season_gcell[model_4].values
    							model_5_temp_cold = dframe_cold_season_gcell[model_5].values
    							model_6_temp_cold = dframe_cold_season_gcell[model_6].values
    							model_7_temp_cold = dframe_cold_season_gcell[model_7].values

    							dframe_7_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    							dframe_7_model[model_2] = model_2_temp_cold
    							dframe_7_model[model_3] = model_3_temp_cold
    							dframe_7_model[model_4] = model_4_temp_cold
    							dframe_7_model[model_5] = model_5_temp_cold
    							dframe_7_model[model_6] = model_6_temp_cold
    							dframe_7_model[model_7] = model_7_temp_cold
    							dframe_7_model_avg = dframe_7_model.mean(axis=1)
    							blend_7_model_temp_cold = dframe_7_model_avg.values

    							bias_7_model_cold = bias(blend_7_model_temp_cold,station_temp_cold)
    							bias_7_model_cold_gcell_master.append(bias_7_model_cold)
    							stdev_7_model_cold = np.std(blend_7_model_temp_cold)
    							stdev_7_model_cold_gcell_master.append(stdev_7_model_cold)
    							rmse_7_model_cold = mean_squared_error(station_temp_cold,blend_7_model_temp_cold,squared=False)
    							rmse_7_model_cold_gcell_master.append(rmse_7_model_cold)
    							corr_7_model_cold,_ = pearsonr(blend_7_model_temp_cold,station_temp_cold)
    							corr_7_model_cold_gcell_master.append(corr_7_model_cold)


    						bias_7_model_cold_gcell_mean = mean(bias_7_model_cold_gcell_master)
    						bias_7_model_cold_master.append(bias_7_model_cold_gcell_mean)
    						stdev_7_model_cold_gcell_mean = mean(stdev_7_model_cold_gcell_master)
    						stdev_7_model_cold_master.append(stdev_7_model_cold_gcell_mean)
    						rmse_7_model_cold_gcell_mean = mean(rmse_7_model_cold_gcell_master)
    						rmse_7_model_cold_master.append(rmse_7_model_cold_gcell_mean)
    						corr_7_model_cold_gcell_mean = mean(corr_7_model_cold_gcell_master)
    						corr_7_model_cold_master.append(corr_7_model_cold_gcell_mean)						    												
    					bias_7_model_cold_mean = mean(bias_7_model_cold_master)
    					stdev_7_model_cold_mean = mean(stdev_7_model_cold_master)
    					stdev_7_model_cold_mean2 = math.sqrt(stdev_7_model_cold_mean)
    					SDV_7_model_cold_mean = stdev_7_model_cold_mean2/stdev_station_cold
    					rmse_7_model_cold_mean = mean(rmse_7_model_cold_master)
    					corr_7_model_cold_mean = mean(corr_7_model_cold_master) 

# 8 model combo #


    					bias_8_model_cold_mean = bias_naive_cold_mean
    					stdev_8_model_cold_mean = stdev_naive_cold_mean
    					SDV_8_model_cold_mean = SDV_naive_cold_mean
    					rmse_8_model_cold_mean = rmse_naive_cold_mean
    					corr_8_model_cold_mean = corr_naive_cold_mean


## Calculate Combinations Associated With A Particular Model ##

    					


## CFSR Model ##

    					bias_CFSR_combo_cold_master = []
    					rmse_CFSR_combo_cold_master = []
    					stdev_CFSR_combo_cold_master = []
    					SDV_CFSR_combo_cold_master = []
    					corr_CFSR_combo_cold_master = []

    					for i in CFSR_array:
    						len_i = len(i)
    						if (len_i == 1):
    							blend_CFSR_combo_temp_cold_gcell_master = []
    							bias_CFSR_combo_cold_gcell_master = []
    							stdev_CFSR_combo_cold_gcell_master = []
    							rmse_CFSR_combo_cold_gcell_master = []
    							corr_CFSR_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								blend_1_model_temp_cold = dframe_cold_season_gcell[i[0]].values
    								print(blend_1_model_temp_cold)
    								print(station_temp_cold)
    								bias_1_model_cold = bias(blend_1_model_temp_cold,station_temp_cold)
    								bias_CFSR_combo_cold_gcell_master.append(bias_1_model_cold)
    								stdev_1_model_cold = np.std(blend_1_model_temp_cold)
    								stdev_CFSR_combo_cold_gcell_master.append(stdev_1_model_cold)
    								rmse_1_model_cold = mean_squared_error(station_temp_cold,blend_1_model_temp_cold,squared=False)
    								rmse_CFSR_combo_cold_gcell_master.append(rmse_1_model_cold)
    								corr_1_model_cold,_ = pearsonr(blend_1_model_temp_cold,station_temp_cold)
    								corr_CFSR_combo_cold_gcell_master.append(corr_1_model_cold)    							

    							bias_CFSR_combo_cold_gcell_mean = mean(bias_CFSR_combo_cold_gcell_master)
    							bias_CFSR_combo_cold_master.append(bias_CFSR_combo_cold_gcell_mean)
    							stdev_CFSR_combo_cold_gcell_mean = mean(stdev_CFSR_combo_cold_gcell_master)
    							stdev_CFSR_combo_cold_master.append(stdev_CFSR_combo_cold_gcell_mean)
    							rmse_CFSR_combo_cold_gcell_mean = mean(rmse_CFSR_combo_cold_gcell_master)
    							rmse_CFSR_combo_cold_master.append(rmse_CFSR_combo_cold_gcell_mean)
    							corr_CFSR_combo_cold_gcell_mean = mean(corr_CFSR_combo_cold_gcell_master)
    							corr_CFSR_combo_cold_master.append(corr_CFSR_combo_cold_gcell_mean)

    						elif (len_i == 2):
    							model_1 = i[0]
    							model_2 = i[1]

    							blend_CFSR_combo_temp_cold_gcell_master = []
    							bias_CFSR_combo_cold_gcell_master = []
    							stdev_CFSR_combo_cold_gcell_master = []
    							rmse_CFSR_combo_cold_gcell_master = []
    							corr_CFSR_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								dframe_2_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_2_model[model_2] = model_2_temp_cold
    								dframe_2_model_avg = dframe_2_model.mean(axis=1)
    								blend_2_model_temp_cold = dframe_2_model_avg
    								bias_2_model_cold = bias(blend_2_model_temp_cold,station_temp_cold)
    								bias_CFSR_combo_cold_gcell_master.append(bias_2_model_cold)
    								stdev_2_model_cold = np.std(blend_2_model_temp_cold)
    								stdev_CFSR_combo_cold_gcell_master.append(stdev_2_model_cold)
    								rmse_2_model_cold = mean_squared_error(station_temp_cold,blend_2_model_temp_cold,squared=False)
    								rmse_CFSR_combo_cold_gcell_master.append(rmse_2_model_cold)
    								corr_2_model_cold,_ = pearsonr(blend_2_model_temp_cold,station_temp_cold)
    								corr_CFSR_combo_cold_gcell_master.append(corr_2_model_cold)    							

    							bias_CFSR_combo_cold_gcell_mean = mean(bias_CFSR_combo_cold_gcell_master)
    							bias_CFSR_combo_cold_master.append(bias_CFSR_combo_cold_gcell_mean)
    							stdev_CFSR_combo_cold_gcell_mean = mean(stdev_CFSR_combo_cold_gcell_master)
    							stdev_CFSR_combo_cold_master.append(stdev_CFSR_combo_cold_gcell_mean)
    							rmse_CFSR_combo_cold_gcell_mean = mean(rmse_CFSR_combo_cold_gcell_master)
    							rmse_CFSR_combo_cold_master.append(rmse_CFSR_combo_cold_gcell_mean)
    							corr_CFSR_combo_cold_gcell_mean = mean(corr_CFSR_combo_cold_gcell_master)
    							corr_CFSR_combo_cold_master.append(corr_CFSR_combo_cold_gcell_mean)


    						elif (len_i == 3):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]

    							blend_CFSR_combo_temp_cold_gcell_master = []
    							bias_CFSR_combo_cold_gcell_master = []
    							stdev_CFSR_combo_cold_gcell_master = []
    							rmse_CFSR_combo_cold_gcell_master = []
    							corr_CFSR_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    								dframe_3_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_3_model[model_2] = model_2_temp_cold
    								dframe_3_model[model_3] = model_3_temp_cold
    								dframe_3_model_avg = dframe_3_model.mean(axis=1)
    								blend_3_model_temp_cold = dframe_3_model_avg
    								bias_3_model_cold = bias(blend_3_model_temp_cold,station_temp_cold)
    								bias_CFSR_combo_cold_gcell_master.append(bias_3_model_cold)
    								stdev_3_model_cold = np.std(blend_3_model_temp_cold)
    								stdev_CFSR_combo_cold_gcell_master.append(stdev_3_model_cold)
    								rmse_3_model_cold = mean_squared_error(station_temp_cold,blend_3_model_temp_cold,squared=False)
    								rmse_CFSR_combo_cold_gcell_master.append(rmse_3_model_cold)
    								corr_3_model_cold,_ = pearsonr(blend_3_model_temp_cold,station_temp_cold)
    								corr_CFSR_combo_cold_gcell_master.append(corr_3_model_cold)    							

    							bias_CFSR_combo_cold_gcell_mean = mean(bias_CFSR_combo_cold_gcell_master)
    							bias_CFSR_combo_cold_master.append(bias_CFSR_combo_cold_gcell_mean)
    							stdev_CFSR_combo_cold_gcell_mean = mean(stdev_CFSR_combo_cold_gcell_master)
    							stdev_CFSR_combo_cold_master.append(stdev_CFSR_combo_cold_gcell_mean)
    							rmse_CFSR_combo_cold_gcell_mean = mean(rmse_CFSR_combo_cold_gcell_master)
    							rmse_CFSR_combo_cold_master.append(rmse_CFSR_combo_cold_gcell_mean)
    							corr_CFSR_combo_cold_gcell_mean = mean(corr_CFSR_combo_cold_gcell_master)
    							corr_CFSR_combo_cold_master.append(corr_CFSR_combo_cold_gcell_mean)


    						elif (len_i == 4):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]

    							blend_CFSR_combo_temp_cold_gcell_master = []
    							bias_CFSR_combo_cold_gcell_master = []
    							stdev_CFSR_combo_cold_gcell_master = []
    							rmse_CFSR_combo_cold_gcell_master = []
    							corr_CFSR_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    								model_4_temp_cold = dframe_cold_season_gcell[model_4].values
    								dframe_4_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_4_model[model_2] = model_2_temp_cold
    								dframe_4_model[model_3] = model_3_temp_cold
    								dframe_4_model[model_4] = model_4_temp_cold
    								dframe_4_model_avg = dframe_4_model.mean(axis=1)
    								blend_4_model_temp_cold = dframe_4_model_avg
    								bias_4_model_cold = bias(blend_4_model_temp_cold,station_temp_cold)
    								bias_CFSR_combo_cold_gcell_master.append(bias_4_model_cold)
    								stdev_4_model_cold = np.std(blend_4_model_temp_cold)
    								stdev_CFSR_combo_cold_gcell_master.append(stdev_4_model_cold)
    								rmse_4_model_cold = mean_squared_error(station_temp_cold,blend_4_model_temp_cold,squared=False)
    								rmse_CFSR_combo_cold_gcell_master.append(rmse_4_model_cold)
    								corr_4_model_cold,_ = pearsonr(blend_4_model_temp_cold,station_temp_cold)
    								corr_CFSR_combo_cold_gcell_master.append(corr_4_model_cold)    							

    							bias_CFSR_combo_cold_gcell_mean = mean(bias_CFSR_combo_cold_gcell_master)
    							bias_CFSR_combo_cold_master.append(bias_CFSR_combo_cold_gcell_mean)
    							stdev_CFSR_combo_cold_gcell_mean = mean(stdev_CFSR_combo_cold_gcell_master)
    							stdev_CFSR_combo_cold_master.append(stdev_CFSR_combo_cold_gcell_mean)
    							rmse_CFSR_combo_cold_gcell_mean = mean(rmse_CFSR_combo_cold_gcell_master)
    							rmse_CFSR_combo_cold_master.append(rmse_CFSR_combo_cold_gcell_mean)
    							corr_CFSR_combo_cold_gcell_mean = mean(corr_CFSR_combo_cold_gcell_master)
    							corr_CFSR_combo_cold_master.append(corr_CFSR_combo_cold_gcell_mean)



    						elif (len_i == 5):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]

    							blend_CFSR_combo_temp_cold_gcell_master = []
    							bias_CFSR_combo_cold_gcell_master = []
    							stdev_CFSR_combo_cold_gcell_master = []
    							rmse_CFSR_combo_cold_gcell_master = []
    							corr_CFSR_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    								model_4_temp_cold = dframe_cold_season_gcell[model_4].values
    								model_5_temp_cold = dframe_cold_season_gcell[model_5].values
    								dframe_5_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_5_model[model_2] = model_2_temp_cold
    								dframe_5_model[model_3] = model_3_temp_cold
    								dframe_5_model[model_4] = model_4_temp_cold
    								dframe_5_model[model_5] = model_5_temp_cold
    								dframe_5_model_avg = dframe_5_model.mean(axis=1)
    								blend_5_model_temp_cold = dframe_5_model_avg
    								bias_5_model_cold = bias(blend_5_model_temp_cold,station_temp_cold)
    								bias_CFSR_combo_cold_gcell_master.append(bias_5_model_cold)
    								stdev_5_model_cold = np.std(blend_5_model_temp_cold)
    								stdev_CFSR_combo_cold_gcell_master.append(stdev_5_model_cold)
    								rmse_5_model_cold = mean_squared_error(station_temp_cold,blend_5_model_temp_cold,squared=False)
    								rmse_CFSR_combo_cold_gcell_master.append(rmse_5_model_cold)
    								corr_5_model_cold,_ = pearsonr(blend_5_model_temp_cold,station_temp_cold)
    								corr_CFSR_combo_cold_gcell_master.append(corr_5_model_cold)    							

    							bias_CFSR_combo_cold_gcell_mean = mean(bias_CFSR_combo_cold_gcell_master)
    							bias_CFSR_combo_cold_master.append(bias_CFSR_combo_cold_gcell_mean)
    							stdev_CFSR_combo_cold_gcell_mean = mean(stdev_CFSR_combo_cold_gcell_master)
    							stdev_CFSR_combo_cold_master.append(stdev_CFSR_combo_cold_gcell_mean)
    							rmse_CFSR_combo_cold_gcell_mean = mean(rmse_CFSR_combo_cold_gcell_master)
    							rmse_CFSR_combo_cold_master.append(rmse_CFSR_combo_cold_gcell_mean)
    							corr_CFSR_combo_cold_gcell_mean = mean(corr_CFSR_combo_cold_gcell_master)
    							corr_CFSR_combo_cold_master.append(corr_CFSR_combo_cold_gcell_mean)



    						elif (len_i == 6):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]
    							model_6 = i[5]

    							blend_CFSR_combo_temp_cold_gcell_master = []
    							bias_CFSR_combo_cold_gcell_master = []
    							stdev_CFSR_combo_cold_gcell_master = []
    							rmse_CFSR_combo_cold_gcell_master = []
    							corr_CFSR_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    								model_4_temp_cold = dframe_cold_season_gcell[model_4].values
    								model_5_temp_cold = dframe_cold_season_gcell[model_5].values
    								model_6_temp_cold = dframe_cold_season_gcell[model_6].values
    								dframe_6_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_6_model[model_2] = model_2_temp_cold
    								dframe_6_model[model_3] = model_3_temp_cold
    								dframe_6_model[model_4] = model_4_temp_cold
    								dframe_6_model[model_5] = model_5_temp_cold
    								dframe_6_model[model_6] = model_6_temp_cold
    								dframe_6_model_avg = dframe_6_model.mean(axis=1)
    								blend_6_model_temp_cold = dframe_6_model_avg
    								bias_6_model_cold = bias(blend_6_model_temp_cold,station_temp_cold)
    								bias_CFSR_combo_cold_gcell_master.append(bias_6_model_cold)
    								stdev_6_model_cold = np.std(blend_6_model_temp_cold)
    								stdev_CFSR_combo_cold_gcell_master.append(stdev_6_model_cold)
    								rmse_6_model_cold = mean_squared_error(station_temp_cold,blend_6_model_temp_cold,squared=False)
    								rmse_CFSR_combo_cold_gcell_master.append(rmse_6_model_cold)
    								corr_6_model_cold,_ = pearsonr(blend_6_model_temp_cold,station_temp_cold)
    								corr_CFSR_combo_cold_gcell_master.append(corr_6_model_cold)    							

    							bias_CFSR_combo_cold_gcell_mean = mean(bias_CFSR_combo_cold_gcell_master)
    							bias_CFSR_combo_cold_master.append(bias_CFSR_combo_cold_gcell_mean)
    							stdev_CFSR_combo_cold_gcell_mean = mean(stdev_CFSR_combo_cold_gcell_master)
    							stdev_CFSR_combo_cold_master.append(stdev_CFSR_combo_cold_gcell_mean)
    							rmse_CFSR_combo_cold_gcell_mean = mean(rmse_CFSR_combo_cold_gcell_master)
    							rmse_CFSR_combo_cold_master.append(rmse_CFSR_combo_cold_gcell_mean)
    							corr_CFSR_combo_cold_gcell_mean = mean(corr_CFSR_combo_cold_gcell_master)
    							corr_CFSR_combo_cold_master.append(corr_CFSR_combo_cold_gcell_mean)



    						elif (len_i == 7):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]
    							model_6 = i[5]
    							model_7 = i[6]

    							blend_CFSR_combo_temp_cold_gcell_master = []
    							bias_CFSR_combo_cold_gcell_master = []
    							stdev_CFSR_combo_cold_gcell_master = []
    							rmse_CFSR_combo_cold_gcell_master = []
    							corr_CFSR_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    								model_4_temp_cold = dframe_cold_season_gcell[model_4].values
    								model_5_temp_cold = dframe_cold_season_gcell[model_5].values
    								model_6_temp_cold = dframe_cold_season_gcell[model_6].values
    								model_7_temp_cold = dframe_cold_season_gcell[model_7].values
    								dframe_7_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_7_model[model_2] = model_2_temp_cold
    								dframe_7_model[model_3] = model_3_temp_cold
    								dframe_7_model[model_4] = model_4_temp_cold
    								dframe_7_model[model_5] = model_5_temp_cold
    								dframe_7_model[model_6] = model_6_temp_cold
    								dframe_7_model[model_7] = model_7_temp_cold
    								dframe_7_model_avg = dframe_7_model.mean(axis=1)
    								blend_7_model_temp_cold = dframe_7_model_avg
    								bias_7_model_cold = bias(blend_7_model_temp_cold,station_temp_cold)
    								bias_CFSR_combo_cold_gcell_master.append(bias_7_model_cold)
    								stdev_7_model_cold = np.std(blend_7_model_temp_cold)
    								stdev_CFSR_combo_cold_gcell_master.append(stdev_7_model_cold)
    								rmse_7_model_cold = mean_squared_error(station_temp_cold,blend_7_model_temp_cold,squared=False)
    								rmse_CFSR_combo_cold_gcell_master.append(rmse_7_model_cold)
    								corr_7_model_cold,_ = pearsonr(blend_7_model_temp_cold,station_temp_cold)
    								corr_CFSR_combo_cold_gcell_master.append(corr_7_model_cold)    							

    							bias_CFSR_combo_cold_gcell_mean = mean(bias_CFSR_combo_cold_gcell_master)
    							bias_CFSR_combo_cold_master.append(bias_CFSR_combo_cold_gcell_mean)
    							stdev_CFSR_combo_cold_gcell_mean = mean(stdev_CFSR_combo_cold_gcell_master)
    							stdev_CFSR_combo_cold_master.append(stdev_CFSR_combo_cold_gcell_mean)
    							rmse_CFSR_combo_cold_gcell_mean = mean(rmse_CFSR_combo_cold_gcell_master)
    							rmse_CFSR_combo_cold_master.append(rmse_CFSR_combo_cold_gcell_mean)
    							corr_CFSR_combo_cold_gcell_mean = mean(corr_CFSR_combo_cold_gcell_master)
    							corr_CFSR_combo_cold_master.append(corr_CFSR_combo_cold_gcell_mean)



    						elif (len_i == 8):
    							bias_8_model_cold = bias_naive_cold_mean
    							bias_CFSR_combo_cold_master.append(bias_8_model_cold)
    							stdev_8_model_cold = stdev_naive_cold_mean
    							stdev_CFSR_combo_cold_master.append(stdev_8_model_cold)
    							rmse_8_model_cold = rmse_naive_cold_mean 
    							rmse_CFSR_combo_cold_master.append(rmse_8_model_cold)
    							corr_8_model_cold = corr_naive_cold_mean
    							corr_CFSR_combo_cold_master.append(corr_8_model_cold)

    					bias_CFSR_combo_cold_mean = mean(bias_CFSR_combo_cold_master)
    					stdev_CFSR_combo_cold_mean = mean(stdev_CFSR_combo_cold_master)
    					SDV_CFSR_combo_cold_mean = stdev_CFSR_combo_cold_mean/stdev_station_cold
    					rmse_CFSR_combo_cold_mean = mean(rmse_CFSR_combo_cold_master)
    					corr_CFSR_combo_cold_mean = mean(corr_CFSR_combo_cold_master)



## ERA-Interim Model ##

    					bias_ERAI_combo_cold_master = []
    					rmse_ERAI_combo_cold_master = []
    					stdev_ERAI_combo_cold_master = []
    					SDV_ERAI_combo_cold_master = []
    					corr_ERAI_combo_cold_master = []

    					for i in ERAI_array:
    						len_i = len(i)
    						if (len_i == 1):
    							blend_ERAI_combo_temp_cold_gcell_master = []
    							bias_ERAI_combo_cold_gcell_master = []
    							stdev_ERAI_combo_cold_gcell_master = []
    							rmse_ERAI_combo_cold_gcell_master = []
    							corr_ERAI_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								blend_1_model_temp_cold = dframe_cold_season_gcell[i[0]].values
    								print(blend_1_model_temp_cold)
    								print(station_temp_cold)
    								bias_1_model_cold = bias(blend_1_model_temp_cold,station_temp_cold)
    								bias_ERAI_combo_cold_gcell_master.append(bias_1_model_cold)
    								stdev_1_model_cold = np.std(blend_1_model_temp_cold)
    								stdev_ERAI_combo_cold_gcell_master.append(stdev_1_model_cold)
    								rmse_1_model_cold = mean_squared_error(station_temp_cold,blend_1_model_temp_cold,squared=False)
    								rmse_ERAI_combo_cold_gcell_master.append(rmse_1_model_cold)
    								corr_1_model_cold,_ = pearsonr(blend_1_model_temp_cold,station_temp_cold)
    								corr_ERAI_combo_cold_gcell_master.append(corr_1_model_cold)    							

    							bias_ERAI_combo_cold_gcell_mean = mean(bias_ERAI_combo_cold_gcell_master)
    							bias_ERAI_combo_cold_master.append(bias_ERAI_combo_cold_gcell_mean)
    							stdev_ERAI_combo_cold_gcell_mean = mean(stdev_ERAI_combo_cold_gcell_master)
    							stdev_ERAI_combo_cold_master.append(stdev_ERAI_combo_cold_gcell_mean)
    							rmse_ERAI_combo_cold_gcell_mean = mean(rmse_ERAI_combo_cold_gcell_master)
    							rmse_ERAI_combo_cold_master.append(rmse_ERAI_combo_cold_gcell_mean)
    							corr_ERAI_combo_cold_gcell_mean = mean(corr_ERAI_combo_cold_gcell_master)
    							corr_ERAI_combo_cold_master.append(corr_ERAI_combo_cold_gcell_mean)

    						elif (len_i == 2):
    							model_1 = i[0]
    							model_2 = i[1]

    							blend_ERAI_combo_temp_cold_gcell_master = []
    							bias_ERAI_combo_cold_gcell_master = []
    							stdev_ERAI_combo_cold_gcell_master = []
    							rmse_ERAI_combo_cold_gcell_master = []
    							corr_ERAI_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								dframe_2_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_2_model[model_2] = model_2_temp_cold
    								dframe_2_model_avg = dframe_2_model.mean(axis=1)
    								blend_2_model_temp_cold = dframe_2_model_avg
    								bias_2_model_cold = bias(blend_2_model_temp_cold,station_temp_cold)
    								bias_ERAI_combo_cold_gcell_master.append(bias_2_model_cold)
    								stdev_2_model_cold = np.std(blend_2_model_temp_cold)
    								stdev_ERAI_combo_cold_gcell_master.append(stdev_2_model_cold)
    								rmse_2_model_cold = mean_squared_error(station_temp_cold,blend_2_model_temp_cold,squared=False)
    								rmse_ERAI_combo_cold_gcell_master.append(rmse_2_model_cold)
    								corr_2_model_cold,_ = pearsonr(blend_2_model_temp_cold,station_temp_cold)
    								corr_ERAI_combo_cold_gcell_master.append(corr_2_model_cold)    							

    							bias_ERAI_combo_cold_gcell_mean = mean(bias_ERAI_combo_cold_gcell_master)
    							bias_ERAI_combo_cold_master.append(bias_ERAI_combo_cold_gcell_mean)
    							stdev_ERAI_combo_cold_gcell_mean = mean(stdev_ERAI_combo_cold_gcell_master)
    							stdev_ERAI_combo_cold_master.append(stdev_ERAI_combo_cold_gcell_mean)
    							rmse_ERAI_combo_cold_gcell_mean = mean(rmse_ERAI_combo_cold_gcell_master)
    							rmse_ERAI_combo_cold_master.append(rmse_ERAI_combo_cold_gcell_mean)
    							corr_ERAI_combo_cold_gcell_mean = mean(corr_ERAI_combo_cold_gcell_master)
    							corr_ERAI_combo_cold_master.append(corr_ERAI_combo_cold_gcell_mean)


    						elif (len_i == 3):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]

    							blend_ERAI_combo_temp_cold_gcell_master = []
    							bias_ERAI_combo_cold_gcell_master = []
    							stdev_ERAI_combo_cold_gcell_master = []
    							rmse_ERAI_combo_cold_gcell_master = []
    							corr_ERAI_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    								dframe_3_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_3_model[model_2] = model_2_temp_cold
    								dframe_3_model[model_3] = model_3_temp_cold
    								dframe_3_model_avg = dframe_3_model.mean(axis=1)
    								blend_3_model_temp_cold = dframe_3_model_avg
    								bias_3_model_cold = bias(blend_3_model_temp_cold,station_temp_cold)
    								bias_ERAI_combo_cold_gcell_master.append(bias_3_model_cold)
    								stdev_3_model_cold = np.std(blend_3_model_temp_cold)
    								stdev_ERAI_combo_cold_gcell_master.append(stdev_3_model_cold)
    								rmse_3_model_cold = mean_squared_error(station_temp_cold,blend_3_model_temp_cold,squared=False)
    								rmse_ERAI_combo_cold_gcell_master.append(rmse_3_model_cold)
    								corr_3_model_cold,_ = pearsonr(blend_3_model_temp_cold,station_temp_cold)
    								corr_ERAI_combo_cold_gcell_master.append(corr_3_model_cold)    							

    							bias_ERAI_combo_cold_gcell_mean = mean(bias_ERAI_combo_cold_gcell_master)
    							bias_ERAI_combo_cold_master.append(bias_ERAI_combo_cold_gcell_mean)
    							stdev_ERAI_combo_cold_gcell_mean = mean(stdev_ERAI_combo_cold_gcell_master)
    							stdev_ERAI_combo_cold_master.append(stdev_ERAI_combo_cold_gcell_mean)
    							rmse_ERAI_combo_cold_gcell_mean = mean(rmse_ERAI_combo_cold_gcell_master)
    							rmse_ERAI_combo_cold_master.append(rmse_ERAI_combo_cold_gcell_mean)
    							corr_ERAI_combo_cold_gcell_mean = mean(corr_ERAI_combo_cold_gcell_master)
    							corr_ERAI_combo_cold_master.append(corr_ERAI_combo_cold_gcell_mean)


    						elif (len_i == 4):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]

    							blend_ERAI_combo_temp_cold_gcell_master = []
    							bias_ERAI_combo_cold_gcell_master = []
    							stdev_ERAI_combo_cold_gcell_master = []
    							rmse_ERAI_combo_cold_gcell_master = []
    							corr_ERAI_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    								model_4_temp_cold = dframe_cold_season_gcell[model_4].values
    								dframe_4_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_4_model[model_2] = model_2_temp_cold
    								dframe_4_model[model_3] = model_3_temp_cold
    								dframe_4_model[model_4] = model_4_temp_cold
    								dframe_4_model_avg = dframe_4_model.mean(axis=1)
    								blend_4_model_temp_cold = dframe_4_model_avg
    								bias_4_model_cold = bias(blend_4_model_temp_cold,station_temp_cold)
    								bias_ERAI_combo_cold_gcell_master.append(bias_4_model_cold)
    								stdev_4_model_cold = np.std(blend_4_model_temp_cold)
    								stdev_ERAI_combo_cold_gcell_master.append(stdev_4_model_cold)
    								rmse_4_model_cold = mean_squared_error(station_temp_cold,blend_4_model_temp_cold,squared=False)
    								rmse_ERAI_combo_cold_gcell_master.append(rmse_4_model_cold)
    								corr_4_model_cold,_ = pearsonr(blend_4_model_temp_cold,station_temp_cold)
    								corr_ERAI_combo_cold_gcell_master.append(corr_4_model_cold)    							

    							bias_ERAI_combo_cold_gcell_mean = mean(bias_ERAI_combo_cold_gcell_master)
    							bias_ERAI_combo_cold_master.append(bias_ERAI_combo_cold_gcell_mean)
    							stdev_ERAI_combo_cold_gcell_mean = mean(stdev_ERAI_combo_cold_gcell_master)
    							stdev_ERAI_combo_cold_master.append(stdev_ERAI_combo_cold_gcell_mean)
    							rmse_ERAI_combo_cold_gcell_mean = mean(rmse_ERAI_combo_cold_gcell_master)
    							rmse_ERAI_combo_cold_master.append(rmse_ERAI_combo_cold_gcell_mean)
    							corr_ERAI_combo_cold_gcell_mean = mean(corr_ERAI_combo_cold_gcell_master)
    							corr_ERAI_combo_cold_master.append(corr_ERAI_combo_cold_gcell_mean)



    						elif (len_i == 5):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]

    							blend_ERAI_combo_temp_cold_gcell_master = []
    							bias_ERAI_combo_cold_gcell_master = []
    							stdev_ERAI_combo_cold_gcell_master = []
    							rmse_ERAI_combo_cold_gcell_master = []
    							corr_ERAI_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    								model_4_temp_cold = dframe_cold_season_gcell[model_4].values
    								model_5_temp_cold = dframe_cold_season_gcell[model_5].values
    								dframe_5_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_5_model[model_2] = model_2_temp_cold
    								dframe_5_model[model_3] = model_3_temp_cold
    								dframe_5_model[model_4] = model_4_temp_cold
    								dframe_5_model[model_5] = model_5_temp_cold
    								dframe_5_model_avg = dframe_5_model.mean(axis=1)
    								blend_5_model_temp_cold = dframe_5_model_avg
    								bias_5_model_cold = bias(blend_5_model_temp_cold,station_temp_cold)
    								bias_ERAI_combo_cold_gcell_master.append(bias_5_model_cold)
    								stdev_5_model_cold = np.std(blend_5_model_temp_cold)
    								stdev_ERAI_combo_cold_gcell_master.append(stdev_5_model_cold)
    								rmse_5_model_cold = mean_squared_error(station_temp_cold,blend_5_model_temp_cold,squared=False)
    								rmse_ERAI_combo_cold_gcell_master.append(rmse_5_model_cold)
    								corr_5_model_cold,_ = pearsonr(blend_5_model_temp_cold,station_temp_cold)
    								corr_ERAI_combo_cold_gcell_master.append(corr_5_model_cold)    							

    							bias_ERAI_combo_cold_gcell_mean = mean(bias_ERAI_combo_cold_gcell_master)
    							bias_ERAI_combo_cold_master.append(bias_ERAI_combo_cold_gcell_mean)
    							stdev_ERAI_combo_cold_gcell_mean = mean(stdev_ERAI_combo_cold_gcell_master)
    							stdev_ERAI_combo_cold_master.append(stdev_ERAI_combo_cold_gcell_mean)
    							rmse_ERAI_combo_cold_gcell_mean = mean(rmse_ERAI_combo_cold_gcell_master)
    							rmse_ERAI_combo_cold_master.append(rmse_ERAI_combo_cold_gcell_mean)
    							corr_ERAI_combo_cold_gcell_mean = mean(corr_ERAI_combo_cold_gcell_master)
    							corr_ERAI_combo_cold_master.append(corr_ERAI_combo_cold_gcell_mean)



    						elif (len_i == 6):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]
    							model_6 = i[5]

    							blend_ERAI_combo_temp_cold_gcell_master = []
    							bias_ERAI_combo_cold_gcell_master = []
    							stdev_ERAI_combo_cold_gcell_master = []
    							rmse_ERAI_combo_cold_gcell_master = []
    							corr_ERAI_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    								model_4_temp_cold = dframe_cold_season_gcell[model_4].values
    								model_5_temp_cold = dframe_cold_season_gcell[model_5].values
    								model_6_temp_cold = dframe_cold_season_gcell[model_6].values
    								dframe_6_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_6_model[model_2] = model_2_temp_cold
    								dframe_6_model[model_3] = model_3_temp_cold
    								dframe_6_model[model_4] = model_4_temp_cold
    								dframe_6_model[model_5] = model_5_temp_cold
    								dframe_6_model[model_6] = model_6_temp_cold
    								dframe_6_model_avg = dframe_6_model.mean(axis=1)
    								blend_6_model_temp_cold = dframe_6_model_avg
    								bias_6_model_cold = bias(blend_6_model_temp_cold,station_temp_cold)
    								bias_ERAI_combo_cold_gcell_master.append(bias_6_model_cold)
    								stdev_6_model_cold = np.std(blend_6_model_temp_cold)
    								stdev_ERAI_combo_cold_gcell_master.append(stdev_6_model_cold)
    								rmse_6_model_cold = mean_squared_error(station_temp_cold,blend_6_model_temp_cold,squared=False)
    								rmse_ERAI_combo_cold_gcell_master.append(rmse_6_model_cold)
    								corr_6_model_cold,_ = pearsonr(blend_6_model_temp_cold,station_temp_cold)
    								corr_ERAI_combo_cold_gcell_master.append(corr_6_model_cold)    							

    							bias_ERAI_combo_cold_gcell_mean = mean(bias_ERAI_combo_cold_gcell_master)
    							bias_ERAI_combo_cold_master.append(bias_ERAI_combo_cold_gcell_mean)
    							stdev_ERAI_combo_cold_gcell_mean = mean(stdev_ERAI_combo_cold_gcell_master)
    							stdev_ERAI_combo_cold_master.append(stdev_ERAI_combo_cold_gcell_mean)
    							rmse_ERAI_combo_cold_gcell_mean = mean(rmse_ERAI_combo_cold_gcell_master)
    							rmse_ERAI_combo_cold_master.append(rmse_ERAI_combo_cold_gcell_mean)
    							corr_ERAI_combo_cold_gcell_mean = mean(corr_ERAI_combo_cold_gcell_master)
    							corr_ERAI_combo_cold_master.append(corr_ERAI_combo_cold_gcell_mean)



    						elif (len_i == 7):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]
    							model_6 = i[5]
    							model_7 = i[6]

    							blend_ERAI_combo_temp_cold_gcell_master = []
    							bias_ERAI_combo_cold_gcell_master = []
    							stdev_ERAI_combo_cold_gcell_master = []
    							rmse_ERAI_combo_cold_gcell_master = []
    							corr_ERAI_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    								model_4_temp_cold = dframe_cold_season_gcell[model_4].values
    								model_5_temp_cold = dframe_cold_season_gcell[model_5].values
    								model_6_temp_cold = dframe_cold_season_gcell[model_6].values
    								model_7_temp_cold = dframe_cold_season_gcell[model_7].values
    								dframe_7_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_7_model[model_2] = model_2_temp_cold
    								dframe_7_model[model_3] = model_3_temp_cold
    								dframe_7_model[model_4] = model_4_temp_cold
    								dframe_7_model[model_5] = model_5_temp_cold
    								dframe_7_model[model_6] = model_6_temp_cold
    								dframe_7_model[model_7] = model_7_temp_cold
    								dframe_7_model_avg = dframe_7_model.mean(axis=1)
    								blend_7_model_temp_cold = dframe_7_model_avg
    								bias_7_model_cold = bias(blend_7_model_temp_cold,station_temp_cold)
    								bias_ERAI_combo_cold_gcell_master.append(bias_7_model_cold)
    								stdev_7_model_cold = np.std(blend_7_model_temp_cold)
    								stdev_ERAI_combo_cold_gcell_master.append(stdev_7_model_cold)
    								rmse_7_model_cold = mean_squared_error(station_temp_cold,blend_7_model_temp_cold,squared=False)
    								rmse_ERAI_combo_cold_gcell_master.append(rmse_7_model_cold)
    								corr_7_model_cold,_ = pearsonr(blend_7_model_temp_cold,station_temp_cold)
    								corr_ERAI_combo_cold_gcell_master.append(corr_7_model_cold)    							

    							bias_ERAI_combo_cold_gcell_mean = mean(bias_ERAI_combo_cold_gcell_master)
    							bias_ERAI_combo_cold_master.append(bias_ERAI_combo_cold_gcell_mean)
    							stdev_ERAI_combo_cold_gcell_mean = mean(stdev_ERAI_combo_cold_gcell_master)
    							stdev_ERAI_combo_cold_master.append(stdev_ERAI_combo_cold_gcell_mean)
    							rmse_ERAI_combo_cold_gcell_mean = mean(rmse_ERAI_combo_cold_gcell_master)
    							rmse_ERAI_combo_cold_master.append(rmse_ERAI_combo_cold_gcell_mean)
    							corr_ERAI_combo_cold_gcell_mean = mean(corr_ERAI_combo_cold_gcell_master)
    							corr_ERAI_combo_cold_master.append(corr_ERAI_combo_cold_gcell_mean)



    						elif (len_i == 8):
    							bias_8_model_cold = bias_naive_cold_mean
    							bias_ERAI_combo_cold_master.append(bias_8_model_cold)
    							stdev_8_model_cold = stdev_naive_cold_mean
    							stdev_ERAI_combo_cold_master.append(stdev_8_model_cold)
    							rmse_8_model_cold = rmse_naive_cold_mean 
    							rmse_ERAI_combo_cold_master.append(rmse_8_model_cold)
    							corr_8_model_cold = corr_naive_cold_mean
    							corr_ERAI_combo_cold_master.append(corr_8_model_cold)

    					bias_ERAI_combo_cold_mean = mean(bias_ERAI_combo_cold_master)
    					stdev_ERAI_combo_cold_mean = mean(stdev_ERAI_combo_cold_master)
    					SDV_ERAI_combo_cold_mean = stdev_ERAI_combo_cold_mean/stdev_station_cold
    					rmse_ERAI_combo_cold_mean = mean(rmse_ERAI_combo_cold_master)
    					corr_ERAI_combo_cold_mean = mean(corr_ERAI_combo_cold_master)


## ERA5 Model ##


    					bias_ERA5_combo_cold_master = []
    					rmse_ERA5_combo_cold_master = []
    					stdev_ERA5_combo_cold_master = []
    					SDV_ERA5_combo_cold_master = []
    					corr_ERA5_combo_cold_master = []

    					for i in ERA5_array:
    						len_i = len(i)
    						if (len_i == 1):
    							blend_ERA5_combo_temp_cold_gcell_master = []
    							bias_ERA5_combo_cold_gcell_master = []
    							stdev_ERA5_combo_cold_gcell_master = []
    							rmse_ERA5_combo_cold_gcell_master = []
    							corr_ERA5_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								blend_1_model_temp_cold = dframe_cold_season_gcell[i[0]].values
    								print(blend_1_model_temp_cold)
    								print(station_temp_cold)
    								bias_1_model_cold = bias(blend_1_model_temp_cold,station_temp_cold)
    								bias_ERA5_combo_cold_gcell_master.append(bias_1_model_cold)
    								stdev_1_model_cold = np.std(blend_1_model_temp_cold)
    								stdev_ERA5_combo_cold_gcell_master.append(stdev_1_model_cold)
    								rmse_1_model_cold = mean_squared_error(station_temp_cold,blend_1_model_temp_cold,squared=False)
    								rmse_ERA5_combo_cold_gcell_master.append(rmse_1_model_cold)
    								corr_1_model_cold,_ = pearsonr(blend_1_model_temp_cold,station_temp_cold)
    								corr_ERA5_combo_cold_gcell_master.append(corr_1_model_cold)    							

    							bias_ERA5_combo_cold_gcell_mean = mean(bias_ERA5_combo_cold_gcell_master)
    							bias_ERA5_combo_cold_master.append(bias_ERA5_combo_cold_gcell_mean)
    							stdev_ERA5_combo_cold_gcell_mean = mean(stdev_ERA5_combo_cold_gcell_master)
    							stdev_ERA5_combo_cold_master.append(stdev_ERA5_combo_cold_gcell_mean)
    							rmse_ERA5_combo_cold_gcell_mean = mean(rmse_ERA5_combo_cold_gcell_master)
    							rmse_ERA5_combo_cold_master.append(rmse_ERA5_combo_cold_gcell_mean)
    							corr_ERA5_combo_cold_gcell_mean = mean(corr_ERA5_combo_cold_gcell_master)
    							corr_ERA5_combo_cold_master.append(corr_ERA5_combo_cold_gcell_mean)

    						elif (len_i == 2):
    							model_1 = i[0]
    							model_2 = i[1]

    							blend_ERA5_combo_temp_cold_gcell_master = []
    							bias_ERA5_combo_cold_gcell_master = []
    							stdev_ERA5_combo_cold_gcell_master = []
    							rmse_ERA5_combo_cold_gcell_master = []
    							corr_ERA5_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								dframe_2_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_2_model[model_2] = model_2_temp_cold
    								dframe_2_model_avg = dframe_2_model.mean(axis=1)
    								blend_2_model_temp_cold = dframe_2_model_avg
    								bias_2_model_cold = bias(blend_2_model_temp_cold,station_temp_cold)
    								bias_ERA5_combo_cold_gcell_master.append(bias_2_model_cold)
    								stdev_2_model_cold = np.std(blend_2_model_temp_cold)
    								stdev_ERA5_combo_cold_gcell_master.append(stdev_2_model_cold)
    								rmse_2_model_cold = mean_squared_error(station_temp_cold,blend_2_model_temp_cold,squared=False)
    								rmse_ERA5_combo_cold_gcell_master.append(rmse_2_model_cold)
    								corr_2_model_cold,_ = pearsonr(blend_2_model_temp_cold,station_temp_cold)
    								corr_ERA5_combo_cold_gcell_master.append(corr_2_model_cold)    							

    							bias_ERA5_combo_cold_gcell_mean = mean(bias_ERA5_combo_cold_gcell_master)
    							bias_ERA5_combo_cold_master.append(bias_ERA5_combo_cold_gcell_mean)
    							stdev_ERA5_combo_cold_gcell_mean = mean(stdev_ERA5_combo_cold_gcell_master)
    							stdev_ERA5_combo_cold_master.append(stdev_ERA5_combo_cold_gcell_mean)
    							rmse_ERA5_combo_cold_gcell_mean = mean(rmse_ERA5_combo_cold_gcell_master)
    							rmse_ERA5_combo_cold_master.append(rmse_ERA5_combo_cold_gcell_mean)
    							corr_ERA5_combo_cold_gcell_mean = mean(corr_ERA5_combo_cold_gcell_master)
    							corr_ERA5_combo_cold_master.append(corr_ERA5_combo_cold_gcell_mean)


    						elif (len_i == 3):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]

    							blend_ERA5_combo_temp_cold_gcell_master = []
    							bias_ERA5_combo_cold_gcell_master = []
    							stdev_ERA5_combo_cold_gcell_master = []
    							rmse_ERA5_combo_cold_gcell_master = []
    							corr_ERA5_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    								dframe_3_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_3_model[model_2] = model_2_temp_cold
    								dframe_3_model[model_3] = model_3_temp_cold
    								dframe_3_model_avg = dframe_3_model.mean(axis=1)
    								blend_3_model_temp_cold = dframe_3_model_avg
    								bias_3_model_cold = bias(blend_3_model_temp_cold,station_temp_cold)
    								bias_ERA5_combo_cold_gcell_master.append(bias_3_model_cold)
    								stdev_3_model_cold = np.std(blend_3_model_temp_cold)
    								stdev_ERA5_combo_cold_gcell_master.append(stdev_3_model_cold)
    								rmse_3_model_cold = mean_squared_error(station_temp_cold,blend_3_model_temp_cold,squared=False)
    								rmse_ERA5_combo_cold_gcell_master.append(rmse_3_model_cold)
    								corr_3_model_cold,_ = pearsonr(blend_3_model_temp_cold,station_temp_cold)
    								corr_ERA5_combo_cold_gcell_master.append(corr_3_model_cold)    							

    							bias_ERA5_combo_cold_gcell_mean = mean(bias_ERA5_combo_cold_gcell_master)
    							bias_ERA5_combo_cold_master.append(bias_ERA5_combo_cold_gcell_mean)
    							stdev_ERA5_combo_cold_gcell_mean = mean(stdev_ERA5_combo_cold_gcell_master)
    							stdev_ERA5_combo_cold_master.append(stdev_ERA5_combo_cold_gcell_mean)
    							rmse_ERA5_combo_cold_gcell_mean = mean(rmse_ERA5_combo_cold_gcell_master)
    							rmse_ERA5_combo_cold_master.append(rmse_ERA5_combo_cold_gcell_mean)
    							corr_ERA5_combo_cold_gcell_mean = mean(corr_ERA5_combo_cold_gcell_master)
    							corr_ERA5_combo_cold_master.append(corr_ERA5_combo_cold_gcell_mean)


    						elif (len_i == 4):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]

    							blend_ERA5_combo_temp_cold_gcell_master = []
    							bias_ERA5_combo_cold_gcell_master = []
    							stdev_ERA5_combo_cold_gcell_master = []
    							rmse_ERA5_combo_cold_gcell_master = []
    							corr_ERA5_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    								model_4_temp_cold = dframe_cold_season_gcell[model_4].values
    								dframe_4_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_4_model[model_2] = model_2_temp_cold
    								dframe_4_model[model_3] = model_3_temp_cold
    								dframe_4_model[model_4] = model_4_temp_cold
    								dframe_4_model_avg = dframe_4_model.mean(axis=1)
    								blend_4_model_temp_cold = dframe_4_model_avg
    								bias_4_model_cold = bias(blend_4_model_temp_cold,station_temp_cold)
    								bias_ERA5_combo_cold_gcell_master.append(bias_4_model_cold)
    								stdev_4_model_cold = np.std(blend_4_model_temp_cold)
    								stdev_ERA5_combo_cold_gcell_master.append(stdev_4_model_cold)
    								rmse_4_model_cold = mean_squared_error(station_temp_cold,blend_4_model_temp_cold,squared=False)
    								rmse_ERA5_combo_cold_gcell_master.append(rmse_4_model_cold)
    								corr_4_model_cold,_ = pearsonr(blend_4_model_temp_cold,station_temp_cold)
    								corr_ERA5_combo_cold_gcell_master.append(corr_4_model_cold)    							

    							bias_ERA5_combo_cold_gcell_mean = mean(bias_ERA5_combo_cold_gcell_master)
    							bias_ERA5_combo_cold_master.append(bias_ERA5_combo_cold_gcell_mean)
    							stdev_ERA5_combo_cold_gcell_mean = mean(stdev_ERA5_combo_cold_gcell_master)
    							stdev_ERA5_combo_cold_master.append(stdev_ERA5_combo_cold_gcell_mean)
    							rmse_ERA5_combo_cold_gcell_mean = mean(rmse_ERA5_combo_cold_gcell_master)
    							rmse_ERA5_combo_cold_master.append(rmse_ERA5_combo_cold_gcell_mean)
    							corr_ERA5_combo_cold_gcell_mean = mean(corr_ERA5_combo_cold_gcell_master)
    							corr_ERA5_combo_cold_master.append(corr_ERA5_combo_cold_gcell_mean)



    						elif (len_i == 5):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]

    							blend_ERA5_combo_temp_cold_gcell_master = []
    							bias_ERA5_combo_cold_gcell_master = []
    							stdev_ERA5_combo_cold_gcell_master = []
    							rmse_ERA5_combo_cold_gcell_master = []
    							corr_ERA5_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    								model_4_temp_cold = dframe_cold_season_gcell[model_4].values
    								model_5_temp_cold = dframe_cold_season_gcell[model_5].values
    								dframe_5_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_5_model[model_2] = model_2_temp_cold
    								dframe_5_model[model_3] = model_3_temp_cold
    								dframe_5_model[model_4] = model_4_temp_cold
    								dframe_5_model[model_5] = model_5_temp_cold
    								dframe_5_model_avg = dframe_5_model.mean(axis=1)
    								blend_5_model_temp_cold = dframe_5_model_avg
    								bias_5_model_cold = bias(blend_5_model_temp_cold,station_temp_cold)
    								bias_ERA5_combo_cold_gcell_master.append(bias_5_model_cold)
    								stdev_5_model_cold = np.std(blend_5_model_temp_cold)
    								stdev_ERA5_combo_cold_gcell_master.append(stdev_5_model_cold)
    								rmse_5_model_cold = mean_squared_error(station_temp_cold,blend_5_model_temp_cold,squared=False)
    								rmse_ERA5_combo_cold_gcell_master.append(rmse_5_model_cold)
    								corr_5_model_cold,_ = pearsonr(blend_5_model_temp_cold,station_temp_cold)
    								corr_ERA5_combo_cold_gcell_master.append(corr_5_model_cold)    							

    							bias_ERA5_combo_cold_gcell_mean = mean(bias_ERA5_combo_cold_gcell_master)
    							bias_ERA5_combo_cold_master.append(bias_ERA5_combo_cold_gcell_mean)
    							stdev_ERA5_combo_cold_gcell_mean = mean(stdev_ERA5_combo_cold_gcell_master)
    							stdev_ERA5_combo_cold_master.append(stdev_ERA5_combo_cold_gcell_mean)
    							rmse_ERA5_combo_cold_gcell_mean = mean(rmse_ERA5_combo_cold_gcell_master)
    							rmse_ERA5_combo_cold_master.append(rmse_ERA5_combo_cold_gcell_mean)
    							corr_ERA5_combo_cold_gcell_mean = mean(corr_ERA5_combo_cold_gcell_master)
    							corr_ERA5_combo_cold_master.append(corr_ERA5_combo_cold_gcell_mean)



    						elif (len_i == 6):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]
    							model_6 = i[5]

    							blend_ERA5_combo_temp_cold_gcell_master = []
    							bias_ERA5_combo_cold_gcell_master = []
    							stdev_ERA5_combo_cold_gcell_master = []
    							rmse_ERA5_combo_cold_gcell_master = []
    							corr_ERA5_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    								model_4_temp_cold = dframe_cold_season_gcell[model_4].values
    								model_5_temp_cold = dframe_cold_season_gcell[model_5].values
    								model_6_temp_cold = dframe_cold_season_gcell[model_6].values
    								dframe_6_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_6_model[model_2] = model_2_temp_cold
    								dframe_6_model[model_3] = model_3_temp_cold
    								dframe_6_model[model_4] = model_4_temp_cold
    								dframe_6_model[model_5] = model_5_temp_cold
    								dframe_6_model[model_6] = model_6_temp_cold
    								dframe_6_model_avg = dframe_6_model.mean(axis=1)
    								blend_6_model_temp_cold = dframe_6_model_avg
    								bias_6_model_cold = bias(blend_6_model_temp_cold,station_temp_cold)
    								bias_ERA5_combo_cold_gcell_master.append(bias_6_model_cold)
    								stdev_6_model_cold = np.std(blend_6_model_temp_cold)
    								stdev_ERA5_combo_cold_gcell_master.append(stdev_6_model_cold)
    								rmse_6_model_cold = mean_squared_error(station_temp_cold,blend_6_model_temp_cold,squared=False)
    								rmse_ERA5_combo_cold_gcell_master.append(rmse_6_model_cold)
    								corr_6_model_cold,_ = pearsonr(blend_6_model_temp_cold,station_temp_cold)
    								corr_ERA5_combo_cold_gcell_master.append(corr_6_model_cold)    							

    							bias_ERA5_combo_cold_gcell_mean = mean(bias_ERA5_combo_cold_gcell_master)
    							bias_ERA5_combo_cold_master.append(bias_ERA5_combo_cold_gcell_mean)
    							stdev_ERA5_combo_cold_gcell_mean = mean(stdev_ERA5_combo_cold_gcell_master)
    							stdev_ERA5_combo_cold_master.append(stdev_ERA5_combo_cold_gcell_mean)
    							rmse_ERA5_combo_cold_gcell_mean = mean(rmse_ERA5_combo_cold_gcell_master)
    							rmse_ERA5_combo_cold_master.append(rmse_ERA5_combo_cold_gcell_mean)
    							corr_ERA5_combo_cold_gcell_mean = mean(corr_ERA5_combo_cold_gcell_master)
    							corr_ERA5_combo_cold_master.append(corr_ERA5_combo_cold_gcell_mean)



    						elif (len_i == 7):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]
    							model_6 = i[5]
    							model_7 = i[6]

    							blend_ERA5_combo_temp_cold_gcell_master = []
    							bias_ERA5_combo_cold_gcell_master = []
    							stdev_ERA5_combo_cold_gcell_master = []
    							rmse_ERA5_combo_cold_gcell_master = []
    							corr_ERA5_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    								model_4_temp_cold = dframe_cold_season_gcell[model_4].values
    								model_5_temp_cold = dframe_cold_season_gcell[model_5].values
    								model_6_temp_cold = dframe_cold_season_gcell[model_6].values
    								model_7_temp_cold = dframe_cold_season_gcell[model_7].values
    								dframe_7_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_7_model[model_2] = model_2_temp_cold
    								dframe_7_model[model_3] = model_3_temp_cold
    								dframe_7_model[model_4] = model_4_temp_cold
    								dframe_7_model[model_5] = model_5_temp_cold
    								dframe_7_model[model_6] = model_6_temp_cold
    								dframe_7_model[model_7] = model_7_temp_cold
    								dframe_7_model_avg = dframe_7_model.mean(axis=1)
    								blend_7_model_temp_cold = dframe_7_model_avg
    								bias_7_model_cold = bias(blend_7_model_temp_cold,station_temp_cold)
    								bias_ERA5_combo_cold_gcell_master.append(bias_7_model_cold)
    								stdev_7_model_cold = np.std(blend_7_model_temp_cold)
    								stdev_ERA5_combo_cold_gcell_master.append(stdev_7_model_cold)
    								rmse_7_model_cold = mean_squared_error(station_temp_cold,blend_7_model_temp_cold,squared=False)
    								rmse_ERA5_combo_cold_gcell_master.append(rmse_7_model_cold)
    								corr_7_model_cold,_ = pearsonr(blend_7_model_temp_cold,station_temp_cold)
    								corr_ERA5_combo_cold_gcell_master.append(corr_7_model_cold)    							

    							bias_ERA5_combo_cold_gcell_mean = mean(bias_ERA5_combo_cold_gcell_master)
    							bias_ERA5_combo_cold_master.append(bias_ERA5_combo_cold_gcell_mean)
    							stdev_ERA5_combo_cold_gcell_mean = mean(stdev_ERA5_combo_cold_gcell_master)
    							stdev_ERA5_combo_cold_master.append(stdev_ERA5_combo_cold_gcell_mean)
    							rmse_ERA5_combo_cold_gcell_mean = mean(rmse_ERA5_combo_cold_gcell_master)
    							rmse_ERA5_combo_cold_master.append(rmse_ERA5_combo_cold_gcell_mean)
    							corr_ERA5_combo_cold_gcell_mean = mean(corr_ERA5_combo_cold_gcell_master)
    							corr_ERA5_combo_cold_master.append(corr_ERA5_combo_cold_gcell_mean)



    						elif (len_i == 8):
    							bias_8_model_cold = bias_naive_cold_mean
    							bias_ERA5_combo_cold_master.append(bias_8_model_cold)
    							stdev_8_model_cold = stdev_naive_cold_mean
    							stdev_ERA5_combo_cold_master.append(stdev_8_model_cold)
    							rmse_8_model_cold = rmse_naive_cold_mean 
    							rmse_ERA5_combo_cold_master.append(rmse_8_model_cold)
    							corr_8_model_cold = corr_naive_cold_mean
    							corr_ERA5_combo_cold_master.append(corr_8_model_cold)

    					bias_ERA5_combo_cold_mean = mean(bias_ERA5_combo_cold_master)
    					stdev_ERA5_combo_cold_mean = mean(stdev_ERA5_combo_cold_master)
    					SDV_ERA5_combo_cold_mean = stdev_ERA5_combo_cold_mean/stdev_station_cold
    					rmse_ERA5_combo_cold_mean = mean(rmse_ERA5_combo_cold_master)
    					corr_ERA5_combo_cold_mean = mean(corr_ERA5_combo_cold_master)


## ERA5-Land Model ##

    					bias_ERA5_Land_combo_cold_master = []
    					rmse_ERA5_Land_combo_cold_master = []
    					stdev_ERA5_Land_combo_cold_master = []
    					SDV_ERA5_Land_combo_cold_master = []
    					corr_ERA5_Land_combo_cold_master = []

    					for i in ERA5_Land_array:
    						len_i = len(i)
    						if (len_i == 1):
    							blend_ERA5_Land_combo_temp_cold_gcell_master = []
    							bias_ERA5_Land_combo_cold_gcell_master = []
    							stdev_ERA5_Land_combo_cold_gcell_master = []
    							rmse_ERA5_Land_combo_cold_gcell_master = []
    							corr_ERA5_Land_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								blend_1_model_temp_cold = dframe_cold_season_gcell[i[0]].values
    								print(blend_1_model_temp_cold)
    								print(station_temp_cold)
    								bias_1_model_cold = bias(blend_1_model_temp_cold,station_temp_cold)
    								bias_ERA5_Land_combo_cold_gcell_master.append(bias_1_model_cold)
    								stdev_1_model_cold = np.std(blend_1_model_temp_cold)
    								stdev_ERA5_Land_combo_cold_gcell_master.append(stdev_1_model_cold)
    								rmse_1_model_cold = mean_squared_error(station_temp_cold,blend_1_model_temp_cold,squared=False)
    								rmse_ERA5_Land_combo_cold_gcell_master.append(rmse_1_model_cold)
    								corr_1_model_cold,_ = pearsonr(blend_1_model_temp_cold,station_temp_cold)
    								corr_ERA5_Land_combo_cold_gcell_master.append(corr_1_model_cold)    							

    							bias_ERA5_Land_combo_cold_gcell_mean = mean(bias_ERA5_Land_combo_cold_gcell_master)
    							bias_ERA5_Land_combo_cold_master.append(bias_ERA5_Land_combo_cold_gcell_mean)
    							stdev_ERA5_Land_combo_cold_gcell_mean = mean(stdev_ERA5_Land_combo_cold_gcell_master)
    							stdev_ERA5_Land_combo_cold_master.append(stdev_ERA5_Land_combo_cold_gcell_mean)
    							rmse_ERA5_Land_combo_cold_gcell_mean = mean(rmse_ERA5_Land_combo_cold_gcell_master)
    							rmse_ERA5_Land_combo_cold_master.append(rmse_ERA5_Land_combo_cold_gcell_mean)
    							corr_ERA5_Land_combo_cold_gcell_mean = mean(corr_ERA5_Land_combo_cold_gcell_master)
    							corr_ERA5_Land_combo_cold_master.append(corr_ERA5_Land_combo_cold_gcell_mean)

    						elif (len_i == 2):
    							model_1 = i[0]
    							model_2 = i[1]

    							blend_ERA5_Land_combo_temp_cold_gcell_master = []
    							bias_ERA5_Land_combo_cold_gcell_master = []
    							stdev_ERA5_Land_combo_cold_gcell_master = []
    							rmse_ERA5_Land_combo_cold_gcell_master = []
    							corr_ERA5_Land_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								dframe_2_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_2_model[model_2] = model_2_temp_cold
    								dframe_2_model_avg = dframe_2_model.mean(axis=1)
    								blend_2_model_temp_cold = dframe_2_model_avg
    								bias_2_model_cold = bias(blend_2_model_temp_cold,station_temp_cold)
    								bias_ERA5_Land_combo_cold_gcell_master.append(bias_2_model_cold)
    								stdev_2_model_cold = np.std(blend_2_model_temp_cold)
    								stdev_ERA5_Land_combo_cold_gcell_master.append(stdev_2_model_cold)
    								rmse_2_model_cold = mean_squared_error(station_temp_cold,blend_2_model_temp_cold,squared=False)
    								rmse_ERA5_Land_combo_cold_gcell_master.append(rmse_2_model_cold)
    								corr_2_model_cold,_ = pearsonr(blend_2_model_temp_cold,station_temp_cold)
    								corr_ERA5_Land_combo_cold_gcell_master.append(corr_2_model_cold)    							

    							bias_ERA5_Land_combo_cold_gcell_mean = mean(bias_ERA5_Land_combo_cold_gcell_master)
    							bias_ERA5_Land_combo_cold_master.append(bias_ERA5_Land_combo_cold_gcell_mean)
    							stdev_ERA5_Land_combo_cold_gcell_mean = mean(stdev_ERA5_Land_combo_cold_gcell_master)
    							stdev_ERA5_Land_combo_cold_master.append(stdev_ERA5_Land_combo_cold_gcell_mean)
    							rmse_ERA5_Land_combo_cold_gcell_mean = mean(rmse_ERA5_Land_combo_cold_gcell_master)
    							rmse_ERA5_Land_combo_cold_master.append(rmse_ERA5_Land_combo_cold_gcell_mean)
    							corr_ERA5_Land_combo_cold_gcell_mean = mean(corr_ERA5_Land_combo_cold_gcell_master)
    							corr_ERA5_Land_combo_cold_master.append(corr_ERA5_Land_combo_cold_gcell_mean)


    						elif (len_i == 3):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]

    							blend_ERA5_Land_combo_temp_cold_gcell_master = []
    							bias_ERA5_Land_combo_cold_gcell_master = []
    							stdev_ERA5_Land_combo_cold_gcell_master = []
    							rmse_ERA5_Land_combo_cold_gcell_master = []
    							corr_ERA5_Land_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    								dframe_3_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_3_model[model_2] = model_2_temp_cold
    								dframe_3_model[model_3] = model_3_temp_cold
    								dframe_3_model_avg = dframe_3_model.mean(axis=1)
    								blend_3_model_temp_cold = dframe_3_model_avg
    								bias_3_model_cold = bias(blend_3_model_temp_cold,station_temp_cold)
    								bias_ERA5_Land_combo_cold_gcell_master.append(bias_3_model_cold)
    								stdev_3_model_cold = np.std(blend_3_model_temp_cold)
    								stdev_ERA5_Land_combo_cold_gcell_master.append(stdev_3_model_cold)
    								rmse_3_model_cold = mean_squared_error(station_temp_cold,blend_3_model_temp_cold,squared=False)
    								rmse_ERA5_Land_combo_cold_gcell_master.append(rmse_3_model_cold)
    								corr_3_model_cold,_ = pearsonr(blend_3_model_temp_cold,station_temp_cold)
    								corr_ERA5_Land_combo_cold_gcell_master.append(corr_3_model_cold)    							

    							bias_ERA5_Land_combo_cold_gcell_mean = mean(bias_ERA5_Land_combo_cold_gcell_master)
    							bias_ERA5_Land_combo_cold_master.append(bias_ERA5_Land_combo_cold_gcell_mean)
    							stdev_ERA5_Land_combo_cold_gcell_mean = mean(stdev_ERA5_Land_combo_cold_gcell_master)
    							stdev_ERA5_Land_combo_cold_master.append(stdev_ERA5_Land_combo_cold_gcell_mean)
    							rmse_ERA5_Land_combo_cold_gcell_mean = mean(rmse_ERA5_Land_combo_cold_gcell_master)
    							rmse_ERA5_Land_combo_cold_master.append(rmse_ERA5_Land_combo_cold_gcell_mean)
    							corr_ERA5_Land_combo_cold_gcell_mean = mean(corr_ERA5_Land_combo_cold_gcell_master)
    							corr_ERA5_Land_combo_cold_master.append(corr_ERA5_Land_combo_cold_gcell_mean)


    						elif (len_i == 4):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]

    							blend_ERA5_Land_combo_temp_cold_gcell_master = []
    							bias_ERA5_Land_combo_cold_gcell_master = []
    							stdev_ERA5_Land_combo_cold_gcell_master = []
    							rmse_ERA5_Land_combo_cold_gcell_master = []
    							corr_ERA5_Land_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    								model_4_temp_cold = dframe_cold_season_gcell[model_4].values
    								dframe_4_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_4_model[model_2] = model_2_temp_cold
    								dframe_4_model[model_3] = model_3_temp_cold
    								dframe_4_model[model_4] = model_4_temp_cold
    								dframe_4_model_avg = dframe_4_model.mean(axis=1)
    								blend_4_model_temp_cold = dframe_4_model_avg
    								bias_4_model_cold = bias(blend_4_model_temp_cold,station_temp_cold)
    								bias_ERA5_Land_combo_cold_gcell_master.append(bias_4_model_cold)
    								stdev_4_model_cold = np.std(blend_4_model_temp_cold)
    								stdev_ERA5_Land_combo_cold_gcell_master.append(stdev_4_model_cold)
    								rmse_4_model_cold = mean_squared_error(station_temp_cold,blend_4_model_temp_cold,squared=False)
    								rmse_ERA5_Land_combo_cold_gcell_master.append(rmse_4_model_cold)
    								corr_4_model_cold,_ = pearsonr(blend_4_model_temp_cold,station_temp_cold)
    								corr_ERA5_Land_combo_cold_gcell_master.append(corr_4_model_cold)    							

    							bias_ERA5_Land_combo_cold_gcell_mean = mean(bias_ERA5_Land_combo_cold_gcell_master)
    							bias_ERA5_Land_combo_cold_master.append(bias_ERA5_Land_combo_cold_gcell_mean)
    							stdev_ERA5_Land_combo_cold_gcell_mean = mean(stdev_ERA5_Land_combo_cold_gcell_master)
    							stdev_ERA5_Land_combo_cold_master.append(stdev_ERA5_Land_combo_cold_gcell_mean)
    							rmse_ERA5_Land_combo_cold_gcell_mean = mean(rmse_ERA5_Land_combo_cold_gcell_master)
    							rmse_ERA5_Land_combo_cold_master.append(rmse_ERA5_Land_combo_cold_gcell_mean)
    							corr_ERA5_Land_combo_cold_gcell_mean = mean(corr_ERA5_Land_combo_cold_gcell_master)
    							corr_ERA5_Land_combo_cold_master.append(corr_ERA5_Land_combo_cold_gcell_mean)



    						elif (len_i == 5):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]

    							blend_ERA5_Land_combo_temp_cold_gcell_master = []
    							bias_ERA5_Land_combo_cold_gcell_master = []
    							stdev_ERA5_Land_combo_cold_gcell_master = []
    							rmse_ERA5_Land_combo_cold_gcell_master = []
    							corr_ERA5_Land_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    								model_4_temp_cold = dframe_cold_season_gcell[model_4].values
    								model_5_temp_cold = dframe_cold_season_gcell[model_5].values
    								dframe_5_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_5_model[model_2] = model_2_temp_cold
    								dframe_5_model[model_3] = model_3_temp_cold
    								dframe_5_model[model_4] = model_4_temp_cold
    								dframe_5_model[model_5] = model_5_temp_cold
    								dframe_5_model_avg = dframe_5_model.mean(axis=1)
    								blend_5_model_temp_cold = dframe_5_model_avg
    								bias_5_model_cold = bias(blend_5_model_temp_cold,station_temp_cold)
    								bias_ERA5_Land_combo_cold_gcell_master.append(bias_5_model_cold)
    								stdev_5_model_cold = np.std(blend_5_model_temp_cold)
    								stdev_ERA5_Land_combo_cold_gcell_master.append(stdev_5_model_cold)
    								rmse_5_model_cold = mean_squared_error(station_temp_cold,blend_5_model_temp_cold,squared=False)
    								rmse_ERA5_Land_combo_cold_gcell_master.append(rmse_5_model_cold)
    								corr_5_model_cold,_ = pearsonr(blend_5_model_temp_cold,station_temp_cold)
    								corr_ERA5_Land_combo_cold_gcell_master.append(corr_5_model_cold)    							

    							bias_ERA5_Land_combo_cold_gcell_mean = mean(bias_ERA5_Land_combo_cold_gcell_master)
    							bias_ERA5_Land_combo_cold_master.append(bias_ERA5_Land_combo_cold_gcell_mean)
    							stdev_ERA5_Land_combo_cold_gcell_mean = mean(stdev_ERA5_Land_combo_cold_gcell_master)
    							stdev_ERA5_Land_combo_cold_master.append(stdev_ERA5_Land_combo_cold_gcell_mean)
    							rmse_ERA5_Land_combo_cold_gcell_mean = mean(rmse_ERA5_Land_combo_cold_gcell_master)
    							rmse_ERA5_Land_combo_cold_master.append(rmse_ERA5_Land_combo_cold_gcell_mean)
    							corr_ERA5_Land_combo_cold_gcell_mean = mean(corr_ERA5_Land_combo_cold_gcell_master)
    							corr_ERA5_Land_combo_cold_master.append(corr_ERA5_Land_combo_cold_gcell_mean)



    						elif (len_i == 6):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]
    							model_6 = i[5]

    							blend_ERA5_Land_combo_temp_cold_gcell_master = []
    							bias_ERA5_Land_combo_cold_gcell_master = []
    							stdev_ERA5_Land_combo_cold_gcell_master = []
    							rmse_ERA5_Land_combo_cold_gcell_master = []
    							corr_ERA5_Land_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    								model_4_temp_cold = dframe_cold_season_gcell[model_4].values
    								model_5_temp_cold = dframe_cold_season_gcell[model_5].values
    								model_6_temp_cold = dframe_cold_season_gcell[model_6].values
    								dframe_6_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_6_model[model_2] = model_2_temp_cold
    								dframe_6_model[model_3] = model_3_temp_cold
    								dframe_6_model[model_4] = model_4_temp_cold
    								dframe_6_model[model_5] = model_5_temp_cold
    								dframe_6_model[model_6] = model_6_temp_cold
    								dframe_6_model_avg = dframe_6_model.mean(axis=1)
    								blend_6_model_temp_cold = dframe_6_model_avg
    								bias_6_model_cold = bias(blend_6_model_temp_cold,station_temp_cold)
    								bias_ERA5_Land_combo_cold_gcell_master.append(bias_6_model_cold)
    								stdev_6_model_cold = np.std(blend_6_model_temp_cold)
    								stdev_ERA5_Land_combo_cold_gcell_master.append(stdev_6_model_cold)
    								rmse_6_model_cold = mean_squared_error(station_temp_cold,blend_6_model_temp_cold,squared=False)
    								rmse_ERA5_Land_combo_cold_gcell_master.append(rmse_6_model_cold)
    								corr_6_model_cold,_ = pearsonr(blend_6_model_temp_cold,station_temp_cold)
    								corr_ERA5_Land_combo_cold_gcell_master.append(corr_6_model_cold)    							

    							bias_ERA5_Land_combo_cold_gcell_mean = mean(bias_ERA5_Land_combo_cold_gcell_master)
    							bias_ERA5_Land_combo_cold_master.append(bias_ERA5_Land_combo_cold_gcell_mean)
    							stdev_ERA5_Land_combo_cold_gcell_mean = mean(stdev_ERA5_Land_combo_cold_gcell_master)
    							stdev_ERA5_Land_combo_cold_master.append(stdev_ERA5_Land_combo_cold_gcell_mean)
    							rmse_ERA5_Land_combo_cold_gcell_mean = mean(rmse_ERA5_Land_combo_cold_gcell_master)
    							rmse_ERA5_Land_combo_cold_master.append(rmse_ERA5_Land_combo_cold_gcell_mean)
    							corr_ERA5_Land_combo_cold_gcell_mean = mean(corr_ERA5_Land_combo_cold_gcell_master)
    							corr_ERA5_Land_combo_cold_master.append(corr_ERA5_Land_combo_cold_gcell_mean)



    						elif (len_i == 7):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]
    							model_6 = i[5]
    							model_7 = i[6]

    							blend_ERA5_Land_combo_temp_cold_gcell_master = []
    							bias_ERA5_Land_combo_cold_gcell_master = []
    							stdev_ERA5_Land_combo_cold_gcell_master = []
    							rmse_ERA5_Land_combo_cold_gcell_master = []
    							corr_ERA5_Land_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    								model_4_temp_cold = dframe_cold_season_gcell[model_4].values
    								model_5_temp_cold = dframe_cold_season_gcell[model_5].values
    								model_6_temp_cold = dframe_cold_season_gcell[model_6].values
    								model_7_temp_cold = dframe_cold_season_gcell[model_7].values
    								dframe_7_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_7_model[model_2] = model_2_temp_cold
    								dframe_7_model[model_3] = model_3_temp_cold
    								dframe_7_model[model_4] = model_4_temp_cold
    								dframe_7_model[model_5] = model_5_temp_cold
    								dframe_7_model[model_6] = model_6_temp_cold
    								dframe_7_model[model_7] = model_7_temp_cold
    								dframe_7_model_avg = dframe_7_model.mean(axis=1)
    								blend_7_model_temp_cold = dframe_7_model_avg
    								bias_7_model_cold = bias(blend_7_model_temp_cold,station_temp_cold)
    								bias_ERA5_Land_combo_cold_gcell_master.append(bias_7_model_cold)
    								stdev_7_model_cold = np.std(blend_7_model_temp_cold)
    								stdev_ERA5_Land_combo_cold_gcell_master.append(stdev_7_model_cold)
    								rmse_7_model_cold = mean_squared_error(station_temp_cold,blend_7_model_temp_cold,squared=False)
    								rmse_ERA5_Land_combo_cold_gcell_master.append(rmse_7_model_cold)
    								corr_7_model_cold,_ = pearsonr(blend_7_model_temp_cold,station_temp_cold)
    								corr_ERA5_Land_combo_cold_gcell_master.append(corr_7_model_cold)    							

    							bias_ERA5_Land_combo_cold_gcell_mean = mean(bias_ERA5_Land_combo_cold_gcell_master)
    							bias_ERA5_Land_combo_cold_master.append(bias_ERA5_Land_combo_cold_gcell_mean)
    							stdev_ERA5_Land_combo_cold_gcell_mean = mean(stdev_ERA5_Land_combo_cold_gcell_master)
    							stdev_ERA5_Land_combo_cold_master.append(stdev_ERA5_Land_combo_cold_gcell_mean)
    							rmse_ERA5_Land_combo_cold_gcell_mean = mean(rmse_ERA5_Land_combo_cold_gcell_master)
    							rmse_ERA5_Land_combo_cold_master.append(rmse_ERA5_Land_combo_cold_gcell_mean)
    							corr_ERA5_Land_combo_cold_gcell_mean = mean(corr_ERA5_Land_combo_cold_gcell_master)
    							corr_ERA5_Land_combo_cold_master.append(corr_ERA5_Land_combo_cold_gcell_mean)



    						elif (len_i == 8):
    							bias_8_model_cold = bias_naive_cold_mean
    							bias_ERA5_Land_combo_cold_master.append(bias_8_model_cold)
    							stdev_8_model_cold = stdev_naive_cold_mean
    							stdev_ERA5_Land_combo_cold_master.append(stdev_8_model_cold)
    							rmse_8_model_cold = rmse_naive_cold_mean 
    							rmse_ERA5_Land_combo_cold_master.append(rmse_8_model_cold)
    							corr_8_model_cold = corr_naive_cold_mean
    							corr_ERA5_Land_combo_cold_master.append(corr_8_model_cold)

    					bias_ERA5_Land_combo_cold_mean = mean(bias_ERA5_Land_combo_cold_master)
    					stdev_ERA5_Land_combo_cold_mean = mean(stdev_ERA5_Land_combo_cold_master)
    					SDV_ERA5_Land_combo_cold_mean = stdev_ERA5_Land_combo_cold_mean/stdev_station_cold
    					rmse_ERA5_Land_combo_cold_mean = mean(rmse_ERA5_Land_combo_cold_master)
    					corr_ERA5_Land_combo_cold_mean = mean(corr_ERA5_Land_combo_cold_master)


## JRA-55 Model ##

    					bias_JRA_combo_cold_master = []
    					rmse_JRA_combo_cold_master = []
    					stdev_JRA_combo_cold_master = []
    					SDV_JRA_combo_cold_master = []
    					corr_JRA_combo_cold_master = []

    					for i in JRA_array:
    						len_i = len(i)
    						if (len_i == 1):
    							blend_JRA_combo_temp_cold_gcell_master = []
    							bias_JRA_combo_cold_gcell_master = []
    							stdev_JRA_combo_cold_gcell_master = []
    							rmse_JRA_combo_cold_gcell_master = []
    							corr_JRA_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								blend_1_model_temp_cold = dframe_cold_season_gcell[i[0]].values
    								print(blend_1_model_temp_cold)
    								print(station_temp_cold)
    								bias_1_model_cold = bias(blend_1_model_temp_cold,station_temp_cold)
    								bias_JRA_combo_cold_gcell_master.append(bias_1_model_cold)
    								stdev_1_model_cold = np.std(blend_1_model_temp_cold)
    								stdev_JRA_combo_cold_gcell_master.append(stdev_1_model_cold)
    								rmse_1_model_cold = mean_squared_error(station_temp_cold,blend_1_model_temp_cold,squared=False)
    								rmse_JRA_combo_cold_gcell_master.append(rmse_1_model_cold)
    								corr_1_model_cold,_ = pearsonr(blend_1_model_temp_cold,station_temp_cold)
    								corr_JRA_combo_cold_gcell_master.append(corr_1_model_cold)    							

    							bias_JRA_combo_cold_gcell_mean = mean(bias_JRA_combo_cold_gcell_master)
    							bias_JRA_combo_cold_master.append(bias_JRA_combo_cold_gcell_mean)
    							stdev_JRA_combo_cold_gcell_mean = mean(stdev_JRA_combo_cold_gcell_master)
    							stdev_JRA_combo_cold_master.append(stdev_JRA_combo_cold_gcell_mean)
    							rmse_JRA_combo_cold_gcell_mean = mean(rmse_JRA_combo_cold_gcell_master)
    							rmse_JRA_combo_cold_master.append(rmse_JRA_combo_cold_gcell_mean)
    							corr_JRA_combo_cold_gcell_mean = mean(corr_JRA_combo_cold_gcell_master)
    							corr_JRA_combo_cold_master.append(corr_JRA_combo_cold_gcell_mean)

    						elif (len_i == 2):
    							model_1 = i[0]
    							model_2 = i[1]

    							blend_JRA_combo_temp_cold_gcell_master = []
    							bias_JRA_combo_cold_gcell_master = []
    							stdev_JRA_combo_cold_gcell_master = []
    							rmse_JRA_combo_cold_gcell_master = []
    							corr_JRA_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								dframe_2_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_2_model[model_2] = model_2_temp_cold
    								dframe_2_model_avg = dframe_2_model.mean(axis=1)
    								blend_2_model_temp_cold = dframe_2_model_avg
    								bias_2_model_cold = bias(blend_2_model_temp_cold,station_temp_cold)
    								bias_JRA_combo_cold_gcell_master.append(bias_2_model_cold)
    								stdev_2_model_cold = np.std(blend_2_model_temp_cold)
    								stdev_JRA_combo_cold_gcell_master.append(stdev_2_model_cold)
    								rmse_2_model_cold = mean_squared_error(station_temp_cold,blend_2_model_temp_cold,squared=False)
    								rmse_JRA_combo_cold_gcell_master.append(rmse_2_model_cold)
    								corr_2_model_cold,_ = pearsonr(blend_2_model_temp_cold,station_temp_cold)
    								corr_JRA_combo_cold_gcell_master.append(corr_2_model_cold)    							

    							bias_JRA_combo_cold_gcell_mean = mean(bias_JRA_combo_cold_gcell_master)
    							bias_JRA_combo_cold_master.append(bias_JRA_combo_cold_gcell_mean)
    							stdev_JRA_combo_cold_gcell_mean = mean(stdev_JRA_combo_cold_gcell_master)
    							stdev_JRA_combo_cold_master.append(stdev_JRA_combo_cold_gcell_mean)
    							rmse_JRA_combo_cold_gcell_mean = mean(rmse_JRA_combo_cold_gcell_master)
    							rmse_JRA_combo_cold_master.append(rmse_JRA_combo_cold_gcell_mean)
    							corr_JRA_combo_cold_gcell_mean = mean(corr_JRA_combo_cold_gcell_master)
    							corr_JRA_combo_cold_master.append(corr_JRA_combo_cold_gcell_mean)


    						elif (len_i == 3):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]

    							blend_JRA_combo_temp_cold_gcell_master = []
    							bias_JRA_combo_cold_gcell_master = []
    							stdev_JRA_combo_cold_gcell_master = []
    							rmse_JRA_combo_cold_gcell_master = []
    							corr_JRA_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    								dframe_3_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_3_model[model_2] = model_2_temp_cold
    								dframe_3_model[model_3] = model_3_temp_cold
    								dframe_3_model_avg = dframe_3_model.mean(axis=1)
    								blend_3_model_temp_cold = dframe_3_model_avg
    								bias_3_model_cold = bias(blend_3_model_temp_cold,station_temp_cold)
    								bias_JRA_combo_cold_gcell_master.append(bias_3_model_cold)
    								stdev_3_model_cold = np.std(blend_3_model_temp_cold)
    								stdev_JRA_combo_cold_gcell_master.append(stdev_3_model_cold)
    								rmse_3_model_cold = mean_squared_error(station_temp_cold,blend_3_model_temp_cold,squared=False)
    								rmse_JRA_combo_cold_gcell_master.append(rmse_3_model_cold)
    								corr_3_model_cold,_ = pearsonr(blend_3_model_temp_cold,station_temp_cold)
    								corr_JRA_combo_cold_gcell_master.append(corr_3_model_cold)    							

    							bias_JRA_combo_cold_gcell_mean = mean(bias_JRA_combo_cold_gcell_master)
    							bias_JRA_combo_cold_master.append(bias_JRA_combo_cold_gcell_mean)
    							stdev_JRA_combo_cold_gcell_mean = mean(stdev_JRA_combo_cold_gcell_master)
    							stdev_JRA_combo_cold_master.append(stdev_JRA_combo_cold_gcell_mean)
    							rmse_JRA_combo_cold_gcell_mean = mean(rmse_JRA_combo_cold_gcell_master)
    							rmse_JRA_combo_cold_master.append(rmse_JRA_combo_cold_gcell_mean)
    							corr_JRA_combo_cold_gcell_mean = mean(corr_JRA_combo_cold_gcell_master)
    							corr_JRA_combo_cold_master.append(corr_JRA_combo_cold_gcell_mean)


    						elif (len_i == 4):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]

    							blend_JRA_combo_temp_cold_gcell_master = []
    							bias_JRA_combo_cold_gcell_master = []
    							stdev_JRA_combo_cold_gcell_master = []
    							rmse_JRA_combo_cold_gcell_master = []
    							corr_JRA_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    								model_4_temp_cold = dframe_cold_season_gcell[model_4].values
    								dframe_4_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_4_model[model_2] = model_2_temp_cold
    								dframe_4_model[model_3] = model_3_temp_cold
    								dframe_4_model[model_4] = model_4_temp_cold
    								dframe_4_model_avg = dframe_4_model.mean(axis=1)
    								blend_4_model_temp_cold = dframe_4_model_avg
    								bias_4_model_cold = bias(blend_4_model_temp_cold,station_temp_cold)
    								bias_JRA_combo_cold_gcell_master.append(bias_4_model_cold)
    								stdev_4_model_cold = np.std(blend_4_model_temp_cold)
    								stdev_JRA_combo_cold_gcell_master.append(stdev_4_model_cold)
    								rmse_4_model_cold = mean_squared_error(station_temp_cold,blend_4_model_temp_cold,squared=False)
    								rmse_JRA_combo_cold_gcell_master.append(rmse_4_model_cold)
    								corr_4_model_cold,_ = pearsonr(blend_4_model_temp_cold,station_temp_cold)
    								corr_JRA_combo_cold_gcell_master.append(corr_4_model_cold)    							

    							bias_JRA_combo_cold_gcell_mean = mean(bias_JRA_combo_cold_gcell_master)
    							bias_JRA_combo_cold_master.append(bias_JRA_combo_cold_gcell_mean)
    							stdev_JRA_combo_cold_gcell_mean = mean(stdev_JRA_combo_cold_gcell_master)
    							stdev_JRA_combo_cold_master.append(stdev_JRA_combo_cold_gcell_mean)
    							rmse_JRA_combo_cold_gcell_mean = mean(rmse_JRA_combo_cold_gcell_master)
    							rmse_JRA_combo_cold_master.append(rmse_JRA_combo_cold_gcell_mean)
    							corr_JRA_combo_cold_gcell_mean = mean(corr_JRA_combo_cold_gcell_master)
    							corr_JRA_combo_cold_master.append(corr_JRA_combo_cold_gcell_mean)



    						elif (len_i == 5):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]

    							blend_JRA_combo_temp_cold_gcell_master = []
    							bias_JRA_combo_cold_gcell_master = []
    							stdev_JRA_combo_cold_gcell_master = []
    							rmse_JRA_combo_cold_gcell_master = []
    							corr_JRA_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    								model_4_temp_cold = dframe_cold_season_gcell[model_4].values
    								model_5_temp_cold = dframe_cold_season_gcell[model_5].values
    								dframe_5_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_5_model[model_2] = model_2_temp_cold
    								dframe_5_model[model_3] = model_3_temp_cold
    								dframe_5_model[model_4] = model_4_temp_cold
    								dframe_5_model[model_5] = model_5_temp_cold
    								dframe_5_model_avg = dframe_5_model.mean(axis=1)
    								blend_5_model_temp_cold = dframe_5_model_avg
    								bias_5_model_cold = bias(blend_5_model_temp_cold,station_temp_cold)
    								bias_JRA_combo_cold_gcell_master.append(bias_5_model_cold)
    								stdev_5_model_cold = np.std(blend_5_model_temp_cold)
    								stdev_JRA_combo_cold_gcell_master.append(stdev_5_model_cold)
    								rmse_5_model_cold = mean_squared_error(station_temp_cold,blend_5_model_temp_cold,squared=False)
    								rmse_JRA_combo_cold_gcell_master.append(rmse_5_model_cold)
    								corr_5_model_cold,_ = pearsonr(blend_5_model_temp_cold,station_temp_cold)
    								corr_JRA_combo_cold_gcell_master.append(corr_5_model_cold)    							

    							bias_JRA_combo_cold_gcell_mean = mean(bias_JRA_combo_cold_gcell_master)
    							bias_JRA_combo_cold_master.append(bias_JRA_combo_cold_gcell_mean)
    							stdev_JRA_combo_cold_gcell_mean = mean(stdev_JRA_combo_cold_gcell_master)
    							stdev_JRA_combo_cold_master.append(stdev_JRA_combo_cold_gcell_mean)
    							rmse_JRA_combo_cold_gcell_mean = mean(rmse_JRA_combo_cold_gcell_master)
    							rmse_JRA_combo_cold_master.append(rmse_JRA_combo_cold_gcell_mean)
    							corr_JRA_combo_cold_gcell_mean = mean(corr_JRA_combo_cold_gcell_master)
    							corr_JRA_combo_cold_master.append(corr_JRA_combo_cold_gcell_mean)



    						elif (len_i == 6):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]
    							model_6 = i[5]

    							blend_JRA_combo_temp_cold_gcell_master = []
    							bias_JRA_combo_cold_gcell_master = []
    							stdev_JRA_combo_cold_gcell_master = []
    							rmse_JRA_combo_cold_gcell_master = []
    							corr_JRA_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    								model_4_temp_cold = dframe_cold_season_gcell[model_4].values
    								model_5_temp_cold = dframe_cold_season_gcell[model_5].values
    								model_6_temp_cold = dframe_cold_season_gcell[model_6].values
    								dframe_6_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_6_model[model_2] = model_2_temp_cold
    								dframe_6_model[model_3] = model_3_temp_cold
    								dframe_6_model[model_4] = model_4_temp_cold
    								dframe_6_model[model_5] = model_5_temp_cold
    								dframe_6_model[model_6] = model_6_temp_cold
    								dframe_6_model_avg = dframe_6_model.mean(axis=1)
    								blend_6_model_temp_cold = dframe_6_model_avg
    								bias_6_model_cold = bias(blend_6_model_temp_cold,station_temp_cold)
    								bias_JRA_combo_cold_gcell_master.append(bias_6_model_cold)
    								stdev_6_model_cold = np.std(blend_6_model_temp_cold)
    								stdev_JRA_combo_cold_gcell_master.append(stdev_6_model_cold)
    								rmse_6_model_cold = mean_squared_error(station_temp_cold,blend_6_model_temp_cold,squared=False)
    								rmse_JRA_combo_cold_gcell_master.append(rmse_6_model_cold)
    								corr_6_model_cold,_ = pearsonr(blend_6_model_temp_cold,station_temp_cold)
    								corr_JRA_combo_cold_gcell_master.append(corr_6_model_cold)    							

    							bias_JRA_combo_cold_gcell_mean = mean(bias_JRA_combo_cold_gcell_master)
    							bias_JRA_combo_cold_master.append(bias_JRA_combo_cold_gcell_mean)
    							stdev_JRA_combo_cold_gcell_mean = mean(stdev_JRA_combo_cold_gcell_master)
    							stdev_JRA_combo_cold_master.append(stdev_JRA_combo_cold_gcell_mean)
    							rmse_JRA_combo_cold_gcell_mean = mean(rmse_JRA_combo_cold_gcell_master)
    							rmse_JRA_combo_cold_master.append(rmse_JRA_combo_cold_gcell_mean)
    							corr_JRA_combo_cold_gcell_mean = mean(corr_JRA_combo_cold_gcell_master)
    							corr_JRA_combo_cold_master.append(corr_JRA_combo_cold_gcell_mean)



    						elif (len_i == 7):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]
    							model_6 = i[5]
    							model_7 = i[6]

    							blend_JRA_combo_temp_cold_gcell_master = []
    							bias_JRA_combo_cold_gcell_master = []
    							stdev_JRA_combo_cold_gcell_master = []
    							rmse_JRA_combo_cold_gcell_master = []
    							corr_JRA_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    								model_4_temp_cold = dframe_cold_season_gcell[model_4].values
    								model_5_temp_cold = dframe_cold_season_gcell[model_5].values
    								model_6_temp_cold = dframe_cold_season_gcell[model_6].values
    								model_7_temp_cold = dframe_cold_season_gcell[model_7].values
    								dframe_7_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_7_model[model_2] = model_2_temp_cold
    								dframe_7_model[model_3] = model_3_temp_cold
    								dframe_7_model[model_4] = model_4_temp_cold
    								dframe_7_model[model_5] = model_5_temp_cold
    								dframe_7_model[model_6] = model_6_temp_cold
    								dframe_7_model[model_7] = model_7_temp_cold
    								dframe_7_model_avg = dframe_7_model.mean(axis=1)
    								blend_7_model_temp_cold = dframe_7_model_avg
    								bias_7_model_cold = bias(blend_7_model_temp_cold,station_temp_cold)
    								bias_JRA_combo_cold_gcell_master.append(bias_7_model_cold)
    								stdev_7_model_cold = np.std(blend_7_model_temp_cold)
    								stdev_JRA_combo_cold_gcell_master.append(stdev_7_model_cold)
    								rmse_7_model_cold = mean_squared_error(station_temp_cold,blend_7_model_temp_cold,squared=False)
    								rmse_JRA_combo_cold_gcell_master.append(rmse_7_model_cold)
    								corr_7_model_cold,_ = pearsonr(blend_7_model_temp_cold,station_temp_cold)
    								corr_JRA_combo_cold_gcell_master.append(corr_7_model_cold)    							

    							bias_JRA_combo_cold_gcell_mean = mean(bias_JRA_combo_cold_gcell_master)
    							bias_JRA_combo_cold_master.append(bias_JRA_combo_cold_gcell_mean)
    							stdev_JRA_combo_cold_gcell_mean = mean(stdev_JRA_combo_cold_gcell_master)
    							stdev_JRA_combo_cold_master.append(stdev_JRA_combo_cold_gcell_mean)
    							rmse_JRA_combo_cold_gcell_mean = mean(rmse_JRA_combo_cold_gcell_master)
    							rmse_JRA_combo_cold_master.append(rmse_JRA_combo_cold_gcell_mean)
    							corr_JRA_combo_cold_gcell_mean = mean(corr_JRA_combo_cold_gcell_master)
    							corr_JRA_combo_cold_master.append(corr_JRA_combo_cold_gcell_mean)



    						elif (len_i == 8):
    							bias_8_model_cold = bias_naive_cold_mean
    							bias_JRA_combo_cold_master.append(bias_8_model_cold)
    							stdev_8_model_cold = stdev_naive_cold_mean
    							stdev_JRA_combo_cold_master.append(stdev_8_model_cold)
    							rmse_8_model_cold = rmse_naive_cold_mean 
    							rmse_JRA_combo_cold_master.append(rmse_8_model_cold)
    							corr_8_model_cold = corr_naive_cold_mean
    							corr_JRA_combo_cold_master.append(corr_8_model_cold)

    					bias_JRA_combo_cold_mean = mean(bias_JRA_combo_cold_master)
    					stdev_JRA_combo_cold_mean = mean(stdev_JRA_combo_cold_master)
    					SDV_JRA_combo_cold_mean = stdev_JRA_combo_cold_mean/stdev_station_cold
    					rmse_JRA_combo_cold_mean = mean(rmse_JRA_combo_cold_master)
    					corr_JRA_combo_cold_mean = mean(corr_JRA_combo_cold_master)



## MERRA2 Model ##

    					bias_MERRA2_combo_cold_master = []
    					rmse_MERRA2_combo_cold_master = []
    					stdev_MERRA2_combo_cold_master = []
    					SDV_MERRA2_combo_cold_master = []
    					corr_MERRA2_combo_cold_master = []

    					for i in MERRA2_array:
    						len_i = len(i)
    						if (len_i == 1):
    							blend_MERRA2_combo_temp_cold_gcell_master = []
    							bias_MERRA2_combo_cold_gcell_master = []
    							stdev_MERRA2_combo_cold_gcell_master = []
    							rmse_MERRA2_combo_cold_gcell_master = []
    							corr_MERRA2_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								blend_1_model_temp_cold = dframe_cold_season_gcell[i[0]].values
    								print(blend_1_model_temp_cold)
    								print(station_temp_cold)
    								bias_1_model_cold = bias(blend_1_model_temp_cold,station_temp_cold)
    								bias_MERRA2_combo_cold_gcell_master.append(bias_1_model_cold)
    								stdev_1_model_cold = np.std(blend_1_model_temp_cold)
    								stdev_MERRA2_combo_cold_gcell_master.append(stdev_1_model_cold)
    								rmse_1_model_cold = mean_squared_error(station_temp_cold,blend_1_model_temp_cold,squared=False)
    								rmse_MERRA2_combo_cold_gcell_master.append(rmse_1_model_cold)
    								corr_1_model_cold,_ = pearsonr(blend_1_model_temp_cold,station_temp_cold)
    								corr_MERRA2_combo_cold_gcell_master.append(corr_1_model_cold)    							

    							bias_MERRA2_combo_cold_gcell_mean = mean(bias_MERRA2_combo_cold_gcell_master)
    							bias_MERRA2_combo_cold_master.append(bias_MERRA2_combo_cold_gcell_mean)
    							stdev_MERRA2_combo_cold_gcell_mean = mean(stdev_MERRA2_combo_cold_gcell_master)
    							stdev_MERRA2_combo_cold_master.append(stdev_MERRA2_combo_cold_gcell_mean)
    							rmse_MERRA2_combo_cold_gcell_mean = mean(rmse_MERRA2_combo_cold_gcell_master)
    							rmse_MERRA2_combo_cold_master.append(rmse_MERRA2_combo_cold_gcell_mean)
    							corr_MERRA2_combo_cold_gcell_mean = mean(corr_MERRA2_combo_cold_gcell_master)
    							corr_MERRA2_combo_cold_master.append(corr_MERRA2_combo_cold_gcell_mean)

    						elif (len_i == 2):
    							model_1 = i[0]
    							model_2 = i[1]

    							blend_MERRA2_combo_temp_cold_gcell_master = []
    							bias_MERRA2_combo_cold_gcell_master = []
    							stdev_MERRA2_combo_cold_gcell_master = []
    							rmse_MERRA2_combo_cold_gcell_master = []
    							corr_MERRA2_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								dframe_2_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_2_model[model_2] = model_2_temp_cold
    								dframe_2_model_avg = dframe_2_model.mean(axis=1)
    								blend_2_model_temp_cold = dframe_2_model_avg
    								bias_2_model_cold = bias(blend_2_model_temp_cold,station_temp_cold)
    								bias_MERRA2_combo_cold_gcell_master.append(bias_2_model_cold)
    								stdev_2_model_cold = np.std(blend_2_model_temp_cold)
    								stdev_MERRA2_combo_cold_gcell_master.append(stdev_2_model_cold)
    								rmse_2_model_cold = mean_squared_error(station_temp_cold,blend_2_model_temp_cold,squared=False)
    								rmse_MERRA2_combo_cold_gcell_master.append(rmse_2_model_cold)
    								corr_2_model_cold,_ = pearsonr(blend_2_model_temp_cold,station_temp_cold)
    								corr_MERRA2_combo_cold_gcell_master.append(corr_2_model_cold)    							

    							bias_MERRA2_combo_cold_gcell_mean = mean(bias_MERRA2_combo_cold_gcell_master)
    							bias_MERRA2_combo_cold_master.append(bias_MERRA2_combo_cold_gcell_mean)
    							stdev_MERRA2_combo_cold_gcell_mean = mean(stdev_MERRA2_combo_cold_gcell_master)
    							stdev_MERRA2_combo_cold_master.append(stdev_MERRA2_combo_cold_gcell_mean)
    							rmse_MERRA2_combo_cold_gcell_mean = mean(rmse_MERRA2_combo_cold_gcell_master)
    							rmse_MERRA2_combo_cold_master.append(rmse_MERRA2_combo_cold_gcell_mean)
    							corr_MERRA2_combo_cold_gcell_mean = mean(corr_MERRA2_combo_cold_gcell_master)
    							corr_MERRA2_combo_cold_master.append(corr_MERRA2_combo_cold_gcell_mean)


    						elif (len_i == 3):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]

    							blend_MERRA2_combo_temp_cold_gcell_master = []
    							bias_MERRA2_combo_cold_gcell_master = []
    							stdev_MERRA2_combo_cold_gcell_master = []
    							rmse_MERRA2_combo_cold_gcell_master = []
    							corr_MERRA2_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    								dframe_3_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_3_model[model_2] = model_2_temp_cold
    								dframe_3_model[model_3] = model_3_temp_cold
    								dframe_3_model_avg = dframe_3_model.mean(axis=1)
    								blend_3_model_temp_cold = dframe_3_model_avg
    								bias_3_model_cold = bias(blend_3_model_temp_cold,station_temp_cold)
    								bias_MERRA2_combo_cold_gcell_master.append(bias_3_model_cold)
    								stdev_3_model_cold = np.std(blend_3_model_temp_cold)
    								stdev_MERRA2_combo_cold_gcell_master.append(stdev_3_model_cold)
    								rmse_3_model_cold = mean_squared_error(station_temp_cold,blend_3_model_temp_cold,squared=False)
    								rmse_MERRA2_combo_cold_gcell_master.append(rmse_3_model_cold)
    								corr_3_model_cold,_ = pearsonr(blend_3_model_temp_cold,station_temp_cold)
    								corr_MERRA2_combo_cold_gcell_master.append(corr_3_model_cold)    							

    							bias_MERRA2_combo_cold_gcell_mean = mean(bias_MERRA2_combo_cold_gcell_master)
    							bias_MERRA2_combo_cold_master.append(bias_MERRA2_combo_cold_gcell_mean)
    							stdev_MERRA2_combo_cold_gcell_mean = mean(stdev_MERRA2_combo_cold_gcell_master)
    							stdev_MERRA2_combo_cold_master.append(stdev_MERRA2_combo_cold_gcell_mean)
    							rmse_MERRA2_combo_cold_gcell_mean = mean(rmse_MERRA2_combo_cold_gcell_master)
    							rmse_MERRA2_combo_cold_master.append(rmse_MERRA2_combo_cold_gcell_mean)
    							corr_MERRA2_combo_cold_gcell_mean = mean(corr_MERRA2_combo_cold_gcell_master)
    							corr_MERRA2_combo_cold_master.append(corr_MERRA2_combo_cold_gcell_mean)


    						elif (len_i == 4):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]

    							blend_MERRA2_combo_temp_cold_gcell_master = []
    							bias_MERRA2_combo_cold_gcell_master = []
    							stdev_MERRA2_combo_cold_gcell_master = []
    							rmse_MERRA2_combo_cold_gcell_master = []
    							corr_MERRA2_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    								model_4_temp_cold = dframe_cold_season_gcell[model_4].values
    								dframe_4_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_4_model[model_2] = model_2_temp_cold
    								dframe_4_model[model_3] = model_3_temp_cold
    								dframe_4_model[model_4] = model_4_temp_cold
    								dframe_4_model_avg = dframe_4_model.mean(axis=1)
    								blend_4_model_temp_cold = dframe_4_model_avg
    								bias_4_model_cold = bias(blend_4_model_temp_cold,station_temp_cold)
    								bias_MERRA2_combo_cold_gcell_master.append(bias_4_model_cold)
    								stdev_4_model_cold = np.std(blend_4_model_temp_cold)
    								stdev_MERRA2_combo_cold_gcell_master.append(stdev_4_model_cold)
    								rmse_4_model_cold = mean_squared_error(station_temp_cold,blend_4_model_temp_cold,squared=False)
    								rmse_MERRA2_combo_cold_gcell_master.append(rmse_4_model_cold)
    								corr_4_model_cold,_ = pearsonr(blend_4_model_temp_cold,station_temp_cold)
    								corr_MERRA2_combo_cold_gcell_master.append(corr_4_model_cold)    							

    							bias_MERRA2_combo_cold_gcell_mean = mean(bias_MERRA2_combo_cold_gcell_master)
    							bias_MERRA2_combo_cold_master.append(bias_MERRA2_combo_cold_gcell_mean)
    							stdev_MERRA2_combo_cold_gcell_mean = mean(stdev_MERRA2_combo_cold_gcell_master)
    							stdev_MERRA2_combo_cold_master.append(stdev_MERRA2_combo_cold_gcell_mean)
    							rmse_MERRA2_combo_cold_gcell_mean = mean(rmse_MERRA2_combo_cold_gcell_master)
    							rmse_MERRA2_combo_cold_master.append(rmse_MERRA2_combo_cold_gcell_mean)
    							corr_MERRA2_combo_cold_gcell_mean = mean(corr_MERRA2_combo_cold_gcell_master)
    							corr_MERRA2_combo_cold_master.append(corr_MERRA2_combo_cold_gcell_mean)



    						elif (len_i == 5):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]

    							blend_MERRA2_combo_temp_cold_gcell_master = []
    							bias_MERRA2_combo_cold_gcell_master = []
    							stdev_MERRA2_combo_cold_gcell_master = []
    							rmse_MERRA2_combo_cold_gcell_master = []
    							corr_MERRA2_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    								model_4_temp_cold = dframe_cold_season_gcell[model_4].values
    								model_5_temp_cold = dframe_cold_season_gcell[model_5].values
    								dframe_5_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_5_model[model_2] = model_2_temp_cold
    								dframe_5_model[model_3] = model_3_temp_cold
    								dframe_5_model[model_4] = model_4_temp_cold
    								dframe_5_model[model_5] = model_5_temp_cold
    								dframe_5_model_avg = dframe_5_model.mean(axis=1)
    								blend_5_model_temp_cold = dframe_5_model_avg
    								bias_5_model_cold = bias(blend_5_model_temp_cold,station_temp_cold)
    								bias_MERRA2_combo_cold_gcell_master.append(bias_5_model_cold)
    								stdev_5_model_cold = np.std(blend_5_model_temp_cold)
    								stdev_MERRA2_combo_cold_gcell_master.append(stdev_5_model_cold)
    								rmse_5_model_cold = mean_squared_error(station_temp_cold,blend_5_model_temp_cold,squared=False)
    								rmse_MERRA2_combo_cold_gcell_master.append(rmse_5_model_cold)
    								corr_5_model_cold,_ = pearsonr(blend_5_model_temp_cold,station_temp_cold)
    								corr_MERRA2_combo_cold_gcell_master.append(corr_5_model_cold)    							

    							bias_MERRA2_combo_cold_gcell_mean = mean(bias_MERRA2_combo_cold_gcell_master)
    							bias_MERRA2_combo_cold_master.append(bias_MERRA2_combo_cold_gcell_mean)
    							stdev_MERRA2_combo_cold_gcell_mean = mean(stdev_MERRA2_combo_cold_gcell_master)
    							stdev_MERRA2_combo_cold_master.append(stdev_MERRA2_combo_cold_gcell_mean)
    							rmse_MERRA2_combo_cold_gcell_mean = mean(rmse_MERRA2_combo_cold_gcell_master)
    							rmse_MERRA2_combo_cold_master.append(rmse_MERRA2_combo_cold_gcell_mean)
    							corr_MERRA2_combo_cold_gcell_mean = mean(corr_MERRA2_combo_cold_gcell_master)
    							corr_MERRA2_combo_cold_master.append(corr_MERRA2_combo_cold_gcell_mean)



    						elif (len_i == 6):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]
    							model_6 = i[5]

    							blend_MERRA2_combo_temp_cold_gcell_master = []
    							bias_MERRA2_combo_cold_gcell_master = []
    							stdev_MERRA2_combo_cold_gcell_master = []
    							rmse_MERRA2_combo_cold_gcell_master = []
    							corr_MERRA2_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    								model_4_temp_cold = dframe_cold_season_gcell[model_4].values
    								model_5_temp_cold = dframe_cold_season_gcell[model_5].values
    								model_6_temp_cold = dframe_cold_season_gcell[model_6].values
    								dframe_6_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_6_model[model_2] = model_2_temp_cold
    								dframe_6_model[model_3] = model_3_temp_cold
    								dframe_6_model[model_4] = model_4_temp_cold
    								dframe_6_model[model_5] = model_5_temp_cold
    								dframe_6_model[model_6] = model_6_temp_cold
    								dframe_6_model_avg = dframe_6_model.mean(axis=1)
    								blend_6_model_temp_cold = dframe_6_model_avg
    								bias_6_model_cold = bias(blend_6_model_temp_cold,station_temp_cold)
    								bias_MERRA2_combo_cold_gcell_master.append(bias_6_model_cold)
    								stdev_6_model_cold = np.std(blend_6_model_temp_cold)
    								stdev_MERRA2_combo_cold_gcell_master.append(stdev_6_model_cold)
    								rmse_6_model_cold = mean_squared_error(station_temp_cold,blend_6_model_temp_cold,squared=False)
    								rmse_MERRA2_combo_cold_gcell_master.append(rmse_6_model_cold)
    								corr_6_model_cold,_ = pearsonr(blend_6_model_temp_cold,station_temp_cold)
    								corr_MERRA2_combo_cold_gcell_master.append(corr_6_model_cold)    							

    							bias_MERRA2_combo_cold_gcell_mean = mean(bias_MERRA2_combo_cold_gcell_master)
    							bias_MERRA2_combo_cold_master.append(bias_MERRA2_combo_cold_gcell_mean)
    							stdev_MERRA2_combo_cold_gcell_mean = mean(stdev_MERRA2_combo_cold_gcell_master)
    							stdev_MERRA2_combo_cold_master.append(stdev_MERRA2_combo_cold_gcell_mean)
    							rmse_MERRA2_combo_cold_gcell_mean = mean(rmse_MERRA2_combo_cold_gcell_master)
    							rmse_MERRA2_combo_cold_master.append(rmse_MERRA2_combo_cold_gcell_mean)
    							corr_MERRA2_combo_cold_gcell_mean = mean(corr_MERRA2_combo_cold_gcell_master)
    							corr_MERRA2_combo_cold_master.append(corr_MERRA2_combo_cold_gcell_mean)



    						elif (len_i == 7):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]
    							model_6 = i[5]
    							model_7 = i[6]

    							blend_MERRA2_combo_temp_cold_gcell_master = []
    							bias_MERRA2_combo_cold_gcell_master = []
    							stdev_MERRA2_combo_cold_gcell_master = []
    							rmse_MERRA2_combo_cold_gcell_master = []
    							corr_MERRA2_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    								model_4_temp_cold = dframe_cold_season_gcell[model_4].values
    								model_5_temp_cold = dframe_cold_season_gcell[model_5].values
    								model_6_temp_cold = dframe_cold_season_gcell[model_6].values
    								model_7_temp_cold = dframe_cold_season_gcell[model_7].values
    								dframe_7_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_7_model[model_2] = model_2_temp_cold
    								dframe_7_model[model_3] = model_3_temp_cold
    								dframe_7_model[model_4] = model_4_temp_cold
    								dframe_7_model[model_5] = model_5_temp_cold
    								dframe_7_model[model_6] = model_6_temp_cold
    								dframe_7_model[model_7] = model_7_temp_cold
    								dframe_7_model_avg = dframe_7_model.mean(axis=1)
    								blend_7_model_temp_cold = dframe_7_model_avg
    								bias_7_model_cold = bias(blend_7_model_temp_cold,station_temp_cold)
    								bias_MERRA2_combo_cold_gcell_master.append(bias_7_model_cold)
    								stdev_7_model_cold = np.std(blend_7_model_temp_cold)
    								stdev_MERRA2_combo_cold_gcell_master.append(stdev_7_model_cold)
    								rmse_7_model_cold = mean_squared_error(station_temp_cold,blend_7_model_temp_cold,squared=False)
    								rmse_MERRA2_combo_cold_gcell_master.append(rmse_7_model_cold)
    								corr_7_model_cold,_ = pearsonr(blend_7_model_temp_cold,station_temp_cold)
    								corr_MERRA2_combo_cold_gcell_master.append(corr_7_model_cold)    							

    							bias_MERRA2_combo_cold_gcell_mean = mean(bias_MERRA2_combo_cold_gcell_master)
    							bias_MERRA2_combo_cold_master.append(bias_MERRA2_combo_cold_gcell_mean)
    							stdev_MERRA2_combo_cold_gcell_mean = mean(stdev_MERRA2_combo_cold_gcell_master)
    							stdev_MERRA2_combo_cold_master.append(stdev_MERRA2_combo_cold_gcell_mean)
    							rmse_MERRA2_combo_cold_gcell_mean = mean(rmse_MERRA2_combo_cold_gcell_master)
    							rmse_MERRA2_combo_cold_master.append(rmse_MERRA2_combo_cold_gcell_mean)
    							corr_MERRA2_combo_cold_gcell_mean = mean(corr_MERRA2_combo_cold_gcell_master)
    							corr_MERRA2_combo_cold_master.append(corr_MERRA2_combo_cold_gcell_mean)



    						elif (len_i == 8):
    							bias_8_model_cold = bias_naive_cold_mean
    							bias_MERRA2_combo_cold_master.append(bias_8_model_cold)
    							stdev_8_model_cold = stdev_naive_cold_mean
    							stdev_MERRA2_combo_cold_master.append(stdev_8_model_cold)
    							rmse_8_model_cold = rmse_naive_cold_mean 
    							rmse_MERRA2_combo_cold_master.append(rmse_8_model_cold)
    							corr_8_model_cold = corr_naive_cold_mean
    							corr_MERRA2_combo_cold_master.append(corr_8_model_cold)

    					bias_MERRA2_combo_cold_mean = mean(bias_MERRA2_combo_cold_master)
    					stdev_MERRA2_combo_cold_mean = mean(stdev_MERRA2_combo_cold_master)
    					SDV_MERRA2_combo_cold_mean = stdev_MERRA2_combo_cold_mean/stdev_station_cold
    					rmse_MERRA2_combo_cold_mean = mean(rmse_MERRA2_combo_cold_master)
    					corr_MERRA2_combo_cold_mean = mean(corr_MERRA2_combo_cold_master)



## GLDAS-Noah Model ##

    					bias_GLDAS_combo_cold_master = []
    					rmse_GLDAS_combo_cold_master = []
    					stdev_GLDAS_combo_cold_master = []
    					SDV_GLDAS_combo_cold_master = []
    					corr_GLDAS_combo_cold_master = []

    					for i in GLDAS_array:
    						len_i = len(i)
    						if (len_i == 1):
    							blend_GLDAS_combo_temp_cold_gcell_master = []
    							bias_GLDAS_combo_cold_gcell_master = []
    							stdev_GLDAS_combo_cold_gcell_master = []
    							rmse_GLDAS_combo_cold_gcell_master = []
    							corr_GLDAS_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								blend_1_model_temp_cold = dframe_cold_season_gcell[i[0]].values
    								print(blend_1_model_temp_cold)
    								print(station_temp_cold)
    								bias_1_model_cold = bias(blend_1_model_temp_cold,station_temp_cold)
    								bias_GLDAS_combo_cold_gcell_master.append(bias_1_model_cold)
    								stdev_1_model_cold = np.std(blend_1_model_temp_cold)
    								stdev_GLDAS_combo_cold_gcell_master.append(stdev_1_model_cold)
    								rmse_1_model_cold = mean_squared_error(station_temp_cold,blend_1_model_temp_cold,squared=False)
    								rmse_GLDAS_combo_cold_gcell_master.append(rmse_1_model_cold)
    								corr_1_model_cold,_ = pearsonr(blend_1_model_temp_cold,station_temp_cold)
    								corr_GLDAS_combo_cold_gcell_master.append(corr_1_model_cold)    							

    							bias_GLDAS_combo_cold_gcell_mean = mean(bias_GLDAS_combo_cold_gcell_master)
    							bias_GLDAS_combo_cold_master.append(bias_GLDAS_combo_cold_gcell_mean)
    							stdev_GLDAS_combo_cold_gcell_mean = mean(stdev_GLDAS_combo_cold_gcell_master)
    							stdev_GLDAS_combo_cold_master.append(stdev_GLDAS_combo_cold_gcell_mean)
    							rmse_GLDAS_combo_cold_gcell_mean = mean(rmse_GLDAS_combo_cold_gcell_master)
    							rmse_GLDAS_combo_cold_master.append(rmse_GLDAS_combo_cold_gcell_mean)
    							corr_GLDAS_combo_cold_gcell_mean = mean(corr_GLDAS_combo_cold_gcell_master)
    							corr_GLDAS_combo_cold_master.append(corr_GLDAS_combo_cold_gcell_mean)

    						elif (len_i == 2):
    							model_1 = i[0]
    							model_2 = i[1]

    							blend_GLDAS_combo_temp_cold_gcell_master = []
    							bias_GLDAS_combo_cold_gcell_master = []
    							stdev_GLDAS_combo_cold_gcell_master = []
    							rmse_GLDAS_combo_cold_gcell_master = []
    							corr_GLDAS_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								dframe_2_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_2_model[model_2] = model_2_temp_cold
    								dframe_2_model_avg = dframe_2_model.mean(axis=1)
    								blend_2_model_temp_cold = dframe_2_model_avg
    								bias_2_model_cold = bias(blend_2_model_temp_cold,station_temp_cold)
    								bias_GLDAS_combo_cold_gcell_master.append(bias_2_model_cold)
    								stdev_2_model_cold = np.std(blend_2_model_temp_cold)
    								stdev_GLDAS_combo_cold_gcell_master.append(stdev_2_model_cold)
    								rmse_2_model_cold = mean_squared_error(station_temp_cold,blend_2_model_temp_cold,squared=False)
    								rmse_GLDAS_combo_cold_gcell_master.append(rmse_2_model_cold)
    								corr_2_model_cold,_ = pearsonr(blend_2_model_temp_cold,station_temp_cold)
    								corr_GLDAS_combo_cold_gcell_master.append(corr_2_model_cold)    							

    							bias_GLDAS_combo_cold_gcell_mean = mean(bias_GLDAS_combo_cold_gcell_master)
    							bias_GLDAS_combo_cold_master.append(bias_GLDAS_combo_cold_gcell_mean)
    							stdev_GLDAS_combo_cold_gcell_mean = mean(stdev_GLDAS_combo_cold_gcell_master)
    							stdev_GLDAS_combo_cold_master.append(stdev_GLDAS_combo_cold_gcell_mean)
    							rmse_GLDAS_combo_cold_gcell_mean = mean(rmse_GLDAS_combo_cold_gcell_master)
    							rmse_GLDAS_combo_cold_master.append(rmse_GLDAS_combo_cold_gcell_mean)
    							corr_GLDAS_combo_cold_gcell_mean = mean(corr_GLDAS_combo_cold_gcell_master)
    							corr_GLDAS_combo_cold_master.append(corr_GLDAS_combo_cold_gcell_mean)


    						elif (len_i == 3):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]

    							blend_GLDAS_combo_temp_cold_gcell_master = []
    							bias_GLDAS_combo_cold_gcell_master = []
    							stdev_GLDAS_combo_cold_gcell_master = []
    							rmse_GLDAS_combo_cold_gcell_master = []
    							corr_GLDAS_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    								dframe_3_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_3_model[model_2] = model_2_temp_cold
    								dframe_3_model[model_3] = model_3_temp_cold
    								dframe_3_model_avg = dframe_3_model.mean(axis=1)
    								blend_3_model_temp_cold = dframe_3_model_avg
    								bias_3_model_cold = bias(blend_3_model_temp_cold,station_temp_cold)
    								bias_GLDAS_combo_cold_gcell_master.append(bias_3_model_cold)
    								stdev_3_model_cold = np.std(blend_3_model_temp_cold)
    								stdev_GLDAS_combo_cold_gcell_master.append(stdev_3_model_cold)
    								rmse_3_model_cold = mean_squared_error(station_temp_cold,blend_3_model_temp_cold,squared=False)
    								rmse_GLDAS_combo_cold_gcell_master.append(rmse_3_model_cold)
    								corr_3_model_cold,_ = pearsonr(blend_3_model_temp_cold,station_temp_cold)
    								corr_GLDAS_combo_cold_gcell_master.append(corr_3_model_cold)    							

    							bias_GLDAS_combo_cold_gcell_mean = mean(bias_GLDAS_combo_cold_gcell_master)
    							bias_GLDAS_combo_cold_master.append(bias_GLDAS_combo_cold_gcell_mean)
    							stdev_GLDAS_combo_cold_gcell_mean = mean(stdev_GLDAS_combo_cold_gcell_master)
    							stdev_GLDAS_combo_cold_master.append(stdev_GLDAS_combo_cold_gcell_mean)
    							rmse_GLDAS_combo_cold_gcell_mean = mean(rmse_GLDAS_combo_cold_gcell_master)
    							rmse_GLDAS_combo_cold_master.append(rmse_GLDAS_combo_cold_gcell_mean)
    							corr_GLDAS_combo_cold_gcell_mean = mean(corr_GLDAS_combo_cold_gcell_master)
    							corr_GLDAS_combo_cold_master.append(corr_GLDAS_combo_cold_gcell_mean)


    						elif (len_i == 4):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]

    							blend_GLDAS_combo_temp_cold_gcell_master = []
    							bias_GLDAS_combo_cold_gcell_master = []
    							stdev_GLDAS_combo_cold_gcell_master = []
    							rmse_GLDAS_combo_cold_gcell_master = []
    							corr_GLDAS_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    								model_4_temp_cold = dframe_cold_season_gcell[model_4].values
    								dframe_4_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_4_model[model_2] = model_2_temp_cold
    								dframe_4_model[model_3] = model_3_temp_cold
    								dframe_4_model[model_4] = model_4_temp_cold
    								dframe_4_model_avg = dframe_4_model.mean(axis=1)
    								blend_4_model_temp_cold = dframe_4_model_avg
    								bias_4_model_cold = bias(blend_4_model_temp_cold,station_temp_cold)
    								bias_GLDAS_combo_cold_gcell_master.append(bias_4_model_cold)
    								stdev_4_model_cold = np.std(blend_4_model_temp_cold)
    								stdev_GLDAS_combo_cold_gcell_master.append(stdev_4_model_cold)
    								rmse_4_model_cold = mean_squared_error(station_temp_cold,blend_4_model_temp_cold,squared=False)
    								rmse_GLDAS_combo_cold_gcell_master.append(rmse_4_model_cold)
    								corr_4_model_cold,_ = pearsonr(blend_4_model_temp_cold,station_temp_cold)
    								corr_GLDAS_combo_cold_gcell_master.append(corr_4_model_cold)    							

    							bias_GLDAS_combo_cold_gcell_mean = mean(bias_GLDAS_combo_cold_gcell_master)
    							bias_GLDAS_combo_cold_master.append(bias_GLDAS_combo_cold_gcell_mean)
    							stdev_GLDAS_combo_cold_gcell_mean = mean(stdev_GLDAS_combo_cold_gcell_master)
    							stdev_GLDAS_combo_cold_master.append(stdev_GLDAS_combo_cold_gcell_mean)
    							rmse_GLDAS_combo_cold_gcell_mean = mean(rmse_GLDAS_combo_cold_gcell_master)
    							rmse_GLDAS_combo_cold_master.append(rmse_GLDAS_combo_cold_gcell_mean)
    							corr_GLDAS_combo_cold_gcell_mean = mean(corr_GLDAS_combo_cold_gcell_master)
    							corr_GLDAS_combo_cold_master.append(corr_GLDAS_combo_cold_gcell_mean)



    						elif (len_i == 5):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]

    							blend_GLDAS_combo_temp_cold_gcell_master = []
    							bias_GLDAS_combo_cold_gcell_master = []
    							stdev_GLDAS_combo_cold_gcell_master = []
    							rmse_GLDAS_combo_cold_gcell_master = []
    							corr_GLDAS_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    								model_4_temp_cold = dframe_cold_season_gcell[model_4].values
    								model_5_temp_cold = dframe_cold_season_gcell[model_5].values
    								dframe_5_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_5_model[model_2] = model_2_temp_cold
    								dframe_5_model[model_3] = model_3_temp_cold
    								dframe_5_model[model_4] = model_4_temp_cold
    								dframe_5_model[model_5] = model_5_temp_cold
    								dframe_5_model_avg = dframe_5_model.mean(axis=1)
    								blend_5_model_temp_cold = dframe_5_model_avg
    								bias_5_model_cold = bias(blend_5_model_temp_cold,station_temp_cold)
    								bias_GLDAS_combo_cold_gcell_master.append(bias_5_model_cold)
    								stdev_5_model_cold = np.std(blend_5_model_temp_cold)
    								stdev_GLDAS_combo_cold_gcell_master.append(stdev_5_model_cold)
    								rmse_5_model_cold = mean_squared_error(station_temp_cold,blend_5_model_temp_cold,squared=False)
    								rmse_GLDAS_combo_cold_gcell_master.append(rmse_5_model_cold)
    								corr_5_model_cold,_ = pearsonr(blend_5_model_temp_cold,station_temp_cold)
    								corr_GLDAS_combo_cold_gcell_master.append(corr_5_model_cold)    							

    							bias_GLDAS_combo_cold_gcell_mean = mean(bias_GLDAS_combo_cold_gcell_master)
    							bias_GLDAS_combo_cold_master.append(bias_GLDAS_combo_cold_gcell_mean)
    							stdev_GLDAS_combo_cold_gcell_mean = mean(stdev_GLDAS_combo_cold_gcell_master)
    							stdev_GLDAS_combo_cold_master.append(stdev_GLDAS_combo_cold_gcell_mean)
    							rmse_GLDAS_combo_cold_gcell_mean = mean(rmse_GLDAS_combo_cold_gcell_master)
    							rmse_GLDAS_combo_cold_master.append(rmse_GLDAS_combo_cold_gcell_mean)
    							corr_GLDAS_combo_cold_gcell_mean = mean(corr_GLDAS_combo_cold_gcell_master)
    							corr_GLDAS_combo_cold_master.append(corr_GLDAS_combo_cold_gcell_mean)



    						elif (len_i == 6):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]
    							model_6 = i[5]

    							blend_GLDAS_combo_temp_cold_gcell_master = []
    							bias_GLDAS_combo_cold_gcell_master = []
    							stdev_GLDAS_combo_cold_gcell_master = []
    							rmse_GLDAS_combo_cold_gcell_master = []
    							corr_GLDAS_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    								model_4_temp_cold = dframe_cold_season_gcell[model_4].values
    								model_5_temp_cold = dframe_cold_season_gcell[model_5].values
    								model_6_temp_cold = dframe_cold_season_gcell[model_6].values
    								dframe_6_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_6_model[model_2] = model_2_temp_cold
    								dframe_6_model[model_3] = model_3_temp_cold
    								dframe_6_model[model_4] = model_4_temp_cold
    								dframe_6_model[model_5] = model_5_temp_cold
    								dframe_6_model[model_6] = model_6_temp_cold
    								dframe_6_model_avg = dframe_6_model.mean(axis=1)
    								blend_6_model_temp_cold = dframe_6_model_avg
    								bias_6_model_cold = bias(blend_6_model_temp_cold,station_temp_cold)
    								bias_GLDAS_combo_cold_gcell_master.append(bias_6_model_cold)
    								stdev_6_model_cold = np.std(blend_6_model_temp_cold)
    								stdev_GLDAS_combo_cold_gcell_master.append(stdev_6_model_cold)
    								rmse_6_model_cold = mean_squared_error(station_temp_cold,blend_6_model_temp_cold,squared=False)
    								rmse_GLDAS_combo_cold_gcell_master.append(rmse_6_model_cold)
    								corr_6_model_cold,_ = pearsonr(blend_6_model_temp_cold,station_temp_cold)
    								corr_GLDAS_combo_cold_gcell_master.append(corr_6_model_cold)    							

    							bias_GLDAS_combo_cold_gcell_mean = mean(bias_GLDAS_combo_cold_gcell_master)
    							bias_GLDAS_combo_cold_master.append(bias_GLDAS_combo_cold_gcell_mean)
    							stdev_GLDAS_combo_cold_gcell_mean = mean(stdev_GLDAS_combo_cold_gcell_master)
    							stdev_GLDAS_combo_cold_master.append(stdev_GLDAS_combo_cold_gcell_mean)
    							rmse_GLDAS_combo_cold_gcell_mean = mean(rmse_GLDAS_combo_cold_gcell_master)
    							rmse_GLDAS_combo_cold_master.append(rmse_GLDAS_combo_cold_gcell_mean)
    							corr_GLDAS_combo_cold_gcell_mean = mean(corr_GLDAS_combo_cold_gcell_master)
    							corr_GLDAS_combo_cold_master.append(corr_GLDAS_combo_cold_gcell_mean)



    						elif (len_i == 7):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]
    							model_6 = i[5]
    							model_7 = i[6]

    							blend_GLDAS_combo_temp_cold_gcell_master = []
    							bias_GLDAS_combo_cold_gcell_master = []
    							stdev_GLDAS_combo_cold_gcell_master = []
    							rmse_GLDAS_combo_cold_gcell_master = []
    							corr_GLDAS_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    								model_4_temp_cold = dframe_cold_season_gcell[model_4].values
    								model_5_temp_cold = dframe_cold_season_gcell[model_5].values
    								model_6_temp_cold = dframe_cold_season_gcell[model_6].values
    								model_7_temp_cold = dframe_cold_season_gcell[model_7].values
    								dframe_7_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_7_model[model_2] = model_2_temp_cold
    								dframe_7_model[model_3] = model_3_temp_cold
    								dframe_7_model[model_4] = model_4_temp_cold
    								dframe_7_model[model_5] = model_5_temp_cold
    								dframe_7_model[model_6] = model_6_temp_cold
    								dframe_7_model[model_7] = model_7_temp_cold
    								dframe_7_model_avg = dframe_7_model.mean(axis=1)
    								blend_7_model_temp_cold = dframe_7_model_avg
    								bias_7_model_cold = bias(blend_7_model_temp_cold,station_temp_cold)
    								bias_GLDAS_combo_cold_gcell_master.append(bias_7_model_cold)
    								stdev_7_model_cold = np.std(blend_7_model_temp_cold)
    								stdev_GLDAS_combo_cold_gcell_master.append(stdev_7_model_cold)
    								rmse_7_model_cold = mean_squared_error(station_temp_cold,blend_7_model_temp_cold,squared=False)
    								rmse_GLDAS_combo_cold_gcell_master.append(rmse_7_model_cold)
    								corr_7_model_cold,_ = pearsonr(blend_7_model_temp_cold,station_temp_cold)
    								corr_GLDAS_combo_cold_gcell_master.append(corr_7_model_cold)    							

    							bias_GLDAS_combo_cold_gcell_mean = mean(bias_GLDAS_combo_cold_gcell_master)
    							bias_GLDAS_combo_cold_master.append(bias_GLDAS_combo_cold_gcell_mean)
    							stdev_GLDAS_combo_cold_gcell_mean = mean(stdev_GLDAS_combo_cold_gcell_master)
    							stdev_GLDAS_combo_cold_master.append(stdev_GLDAS_combo_cold_gcell_mean)
    							rmse_GLDAS_combo_cold_gcell_mean = mean(rmse_GLDAS_combo_cold_gcell_master)
    							rmse_GLDAS_combo_cold_master.append(rmse_GLDAS_combo_cold_gcell_mean)
    							corr_GLDAS_combo_cold_gcell_mean = mean(corr_GLDAS_combo_cold_gcell_master)
    							corr_GLDAS_combo_cold_master.append(corr_GLDAS_combo_cold_gcell_mean)



    						elif (len_i == 8):
    							bias_8_model_cold = bias_naive_cold_mean
    							bias_GLDAS_combo_cold_master.append(bias_8_model_cold)
    							stdev_8_model_cold = stdev_naive_cold_mean
    							stdev_GLDAS_combo_cold_master.append(stdev_8_model_cold)
    							rmse_8_model_cold = rmse_naive_cold_mean 
    							rmse_GLDAS_combo_cold_master.append(rmse_8_model_cold)
    							corr_8_model_cold = corr_naive_cold_mean
    							corr_GLDAS_combo_cold_master.append(corr_8_model_cold)

    					bias_GLDAS_combo_cold_mean = mean(bias_GLDAS_combo_cold_master)
    					stdev_GLDAS_combo_cold_mean = mean(stdev_GLDAS_combo_cold_master)
    					SDV_GLDAS_combo_cold_mean = stdev_GLDAS_combo_cold_mean/stdev_station_cold
    					rmse_GLDAS_combo_cold_mean = mean(rmse_GLDAS_combo_cold_master)
    					corr_GLDAS_combo_cold_mean = mean(corr_GLDAS_combo_cold_master)


## GLDAS-CLSM Model ##

    					bias_GLDAS_CLSM_combo_cold_master = []
    					rmse_GLDAS_CLSM_combo_cold_master = []
    					stdev_GLDAS_CLSM_combo_cold_master = []
    					SDV_GLDAS_CLSM_combo_cold_master = []
    					corr_GLDAS_CLSM_combo_cold_master = []

    					for i in GLDAS_CLSM_array:
    						len_i = len(i)
    						if (len_i == 1):
    							blend_GLDAS_CLSM_combo_temp_cold_gcell_master = []
    							bias_GLDAS_CLSM_combo_cold_gcell_master = []
    							stdev_GLDAS_CLSM_combo_cold_gcell_master = []
    							rmse_GLDAS_CLSM_combo_cold_gcell_master = []
    							corr_GLDAS_CLSM_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								blend_1_model_temp_cold = dframe_cold_season_gcell[i[0]].values
    								print(blend_1_model_temp_cold)
    								print(station_temp_cold)
    								bias_1_model_cold = bias(blend_1_model_temp_cold,station_temp_cold)
    								bias_GLDAS_CLSM_combo_cold_gcell_master.append(bias_1_model_cold)
    								stdev_1_model_cold = np.std(blend_1_model_temp_cold)
    								stdev_GLDAS_CLSM_combo_cold_gcell_master.append(stdev_1_model_cold)
    								rmse_1_model_cold = mean_squared_error(station_temp_cold,blend_1_model_temp_cold,squared=False)
    								rmse_GLDAS_CLSM_combo_cold_gcell_master.append(rmse_1_model_cold)
    								corr_1_model_cold,_ = pearsonr(blend_1_model_temp_cold,station_temp_cold)
    								corr_GLDAS_CLSM_combo_cold_gcell_master.append(corr_1_model_cold)    							

    							bias_GLDAS_CLSM_combo_cold_gcell_mean = mean(bias_GLDAS_CLSM_combo_cold_gcell_master)
    							bias_GLDAS_CLSM_combo_cold_master.append(bias_GLDAS_CLSM_combo_cold_gcell_mean)
    							stdev_GLDAS_CLSM_combo_cold_gcell_mean = mean(stdev_GLDAS_CLSM_combo_cold_gcell_master)
    							stdev_GLDAS_CLSM_combo_cold_master.append(stdev_GLDAS_CLSM_combo_cold_gcell_mean)
    							rmse_GLDAS_CLSM_combo_cold_gcell_mean = mean(rmse_GLDAS_CLSM_combo_cold_gcell_master)
    							rmse_GLDAS_CLSM_combo_cold_master.append(rmse_GLDAS_CLSM_combo_cold_gcell_mean)
    							corr_GLDAS_CLSM_combo_cold_gcell_mean = mean(corr_GLDAS_CLSM_combo_cold_gcell_master)
    							corr_GLDAS_CLSM_combo_cold_master.append(corr_GLDAS_CLSM_combo_cold_gcell_mean)

    						elif (len_i == 2):
    							model_1 = i[0]
    							model_2 = i[1]

    							blend_GLDAS_CLSM_combo_temp_cold_gcell_master = []
    							bias_GLDAS_CLSM_combo_cold_gcell_master = []
    							stdev_GLDAS_CLSM_combo_cold_gcell_master = []
    							rmse_GLDAS_CLSM_combo_cold_gcell_master = []
    							corr_GLDAS_CLSM_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								dframe_2_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_2_model[model_2] = model_2_temp_cold
    								dframe_2_model_avg = dframe_2_model.mean(axis=1)
    								blend_2_model_temp_cold = dframe_2_model_avg
    								bias_2_model_cold = bias(blend_2_model_temp_cold,station_temp_cold)
    								bias_GLDAS_CLSM_combo_cold_gcell_master.append(bias_2_model_cold)
    								stdev_2_model_cold = np.std(blend_2_model_temp_cold)
    								stdev_GLDAS_CLSM_combo_cold_gcell_master.append(stdev_2_model_cold)
    								rmse_2_model_cold = mean_squared_error(station_temp_cold,blend_2_model_temp_cold,squared=False)
    								rmse_GLDAS_CLSM_combo_cold_gcell_master.append(rmse_2_model_cold)
    								corr_2_model_cold,_ = pearsonr(blend_2_model_temp_cold,station_temp_cold)
    								corr_GLDAS_CLSM_combo_cold_gcell_master.append(corr_2_model_cold)    							

    							bias_GLDAS_CLSM_combo_cold_gcell_mean = mean(bias_GLDAS_CLSM_combo_cold_gcell_master)
    							bias_GLDAS_CLSM_combo_cold_master.append(bias_GLDAS_CLSM_combo_cold_gcell_mean)
    							stdev_GLDAS_CLSM_combo_cold_gcell_mean = mean(stdev_GLDAS_CLSM_combo_cold_gcell_master)
    							stdev_GLDAS_CLSM_combo_cold_master.append(stdev_GLDAS_CLSM_combo_cold_gcell_mean)
    							rmse_GLDAS_CLSM_combo_cold_gcell_mean = mean(rmse_GLDAS_CLSM_combo_cold_gcell_master)
    							rmse_GLDAS_CLSM_combo_cold_master.append(rmse_GLDAS_CLSM_combo_cold_gcell_mean)
    							corr_GLDAS_CLSM_combo_cold_gcell_mean = mean(corr_GLDAS_CLSM_combo_cold_gcell_master)
    							corr_GLDAS_CLSM_combo_cold_master.append(corr_GLDAS_CLSM_combo_cold_gcell_mean)


    						elif (len_i == 3):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]

    							blend_GLDAS_CLSM_combo_temp_cold_gcell_master = []
    							bias_GLDAS_CLSM_combo_cold_gcell_master = []
    							stdev_GLDAS_CLSM_combo_cold_gcell_master = []
    							rmse_GLDAS_CLSM_combo_cold_gcell_master = []
    							corr_GLDAS_CLSM_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    								dframe_3_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_3_model[model_2] = model_2_temp_cold
    								dframe_3_model[model_3] = model_3_temp_cold
    								dframe_3_model_avg = dframe_3_model.mean(axis=1)
    								blend_3_model_temp_cold = dframe_3_model_avg
    								bias_3_model_cold = bias(blend_3_model_temp_cold,station_temp_cold)
    								bias_GLDAS_CLSM_combo_cold_gcell_master.append(bias_3_model_cold)
    								stdev_3_model_cold = np.std(blend_3_model_temp_cold)
    								stdev_GLDAS_CLSM_combo_cold_gcell_master.append(stdev_3_model_cold)
    								rmse_3_model_cold = mean_squared_error(station_temp_cold,blend_3_model_temp_cold,squared=False)
    								rmse_GLDAS_CLSM_combo_cold_gcell_master.append(rmse_3_model_cold)
    								corr_3_model_cold,_ = pearsonr(blend_3_model_temp_cold,station_temp_cold)
    								corr_GLDAS_CLSM_combo_cold_gcell_master.append(corr_3_model_cold)    							

    							bias_GLDAS_CLSM_combo_cold_gcell_mean = mean(bias_GLDAS_CLSM_combo_cold_gcell_master)
    							bias_GLDAS_CLSM_combo_cold_master.append(bias_GLDAS_CLSM_combo_cold_gcell_mean)
    							stdev_GLDAS_CLSM_combo_cold_gcell_mean = mean(stdev_GLDAS_CLSM_combo_cold_gcell_master)
    							stdev_GLDAS_CLSM_combo_cold_master.append(stdev_GLDAS_CLSM_combo_cold_gcell_mean)
    							rmse_GLDAS_CLSM_combo_cold_gcell_mean = mean(rmse_GLDAS_CLSM_combo_cold_gcell_master)
    							rmse_GLDAS_CLSM_combo_cold_master.append(rmse_GLDAS_CLSM_combo_cold_gcell_mean)
    							corr_GLDAS_CLSM_combo_cold_gcell_mean = mean(corr_GLDAS_CLSM_combo_cold_gcell_master)
    							corr_GLDAS_CLSM_combo_cold_master.append(corr_GLDAS_CLSM_combo_cold_gcell_mean)


    						elif (len_i == 4):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]

    							blend_GLDAS_CLSM_combo_temp_cold_gcell_master = []
    							bias_GLDAS_CLSM_combo_cold_gcell_master = []
    							stdev_GLDAS_CLSM_combo_cold_gcell_master = []
    							rmse_GLDAS_CLSM_combo_cold_gcell_master = []
    							corr_GLDAS_CLSM_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    								model_4_temp_cold = dframe_cold_season_gcell[model_4].values
    								dframe_4_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_4_model[model_2] = model_2_temp_cold
    								dframe_4_model[model_3] = model_3_temp_cold
    								dframe_4_model[model_4] = model_4_temp_cold
    								dframe_4_model_avg = dframe_4_model.mean(axis=1)
    								blend_4_model_temp_cold = dframe_4_model_avg
    								bias_4_model_cold = bias(blend_4_model_temp_cold,station_temp_cold)
    								bias_GLDAS_CLSM_combo_cold_gcell_master.append(bias_4_model_cold)
    								stdev_4_model_cold = np.std(blend_4_model_temp_cold)
    								stdev_GLDAS_CLSM_combo_cold_gcell_master.append(stdev_4_model_cold)
    								rmse_4_model_cold = mean_squared_error(station_temp_cold,blend_4_model_temp_cold,squared=False)
    								rmse_GLDAS_CLSM_combo_cold_gcell_master.append(rmse_4_model_cold)
    								corr_4_model_cold,_ = pearsonr(blend_4_model_temp_cold,station_temp_cold)
    								corr_GLDAS_CLSM_combo_cold_gcell_master.append(corr_4_model_cold)    							

    							bias_GLDAS_CLSM_combo_cold_gcell_mean = mean(bias_GLDAS_CLSM_combo_cold_gcell_master)
    							bias_GLDAS_CLSM_combo_cold_master.append(bias_GLDAS_CLSM_combo_cold_gcell_mean)
    							stdev_GLDAS_CLSM_combo_cold_gcell_mean = mean(stdev_GLDAS_CLSM_combo_cold_gcell_master)
    							stdev_GLDAS_CLSM_combo_cold_master.append(stdev_GLDAS_CLSM_combo_cold_gcell_mean)
    							rmse_GLDAS_CLSM_combo_cold_gcell_mean = mean(rmse_GLDAS_CLSM_combo_cold_gcell_master)
    							rmse_GLDAS_CLSM_combo_cold_master.append(rmse_GLDAS_CLSM_combo_cold_gcell_mean)
    							corr_GLDAS_CLSM_combo_cold_gcell_mean = mean(corr_GLDAS_CLSM_combo_cold_gcell_master)
    							corr_GLDAS_CLSM_combo_cold_master.append(corr_GLDAS_CLSM_combo_cold_gcell_mean)



    						elif (len_i == 5):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]

    							blend_GLDAS_CLSM_combo_temp_cold_gcell_master = []
    							bias_GLDAS_CLSM_combo_cold_gcell_master = []
    							stdev_GLDAS_CLSM_combo_cold_gcell_master = []
    							rmse_GLDAS_CLSM_combo_cold_gcell_master = []
    							corr_GLDAS_CLSM_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    								model_4_temp_cold = dframe_cold_season_gcell[model_4].values
    								model_5_temp_cold = dframe_cold_season_gcell[model_5].values
    								dframe_5_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_5_model[model_2] = model_2_temp_cold
    								dframe_5_model[model_3] = model_3_temp_cold
    								dframe_5_model[model_4] = model_4_temp_cold
    								dframe_5_model[model_5] = model_5_temp_cold
    								dframe_5_model_avg = dframe_5_model.mean(axis=1)
    								blend_5_model_temp_cold = dframe_5_model_avg
    								bias_5_model_cold = bias(blend_5_model_temp_cold,station_temp_cold)
    								bias_GLDAS_CLSM_combo_cold_gcell_master.append(bias_5_model_cold)
    								stdev_5_model_cold = np.std(blend_5_model_temp_cold)
    								stdev_GLDAS_CLSM_combo_cold_gcell_master.append(stdev_5_model_cold)
    								rmse_5_model_cold = mean_squared_error(station_temp_cold,blend_5_model_temp_cold,squared=False)
    								rmse_GLDAS_CLSM_combo_cold_gcell_master.append(rmse_5_model_cold)
    								corr_5_model_cold,_ = pearsonr(blend_5_model_temp_cold,station_temp_cold)
    								corr_GLDAS_CLSM_combo_cold_gcell_master.append(corr_5_model_cold)    							

    							bias_GLDAS_CLSM_combo_cold_gcell_mean = mean(bias_GLDAS_CLSM_combo_cold_gcell_master)
    							bias_GLDAS_CLSM_combo_cold_master.append(bias_GLDAS_CLSM_combo_cold_gcell_mean)
    							stdev_GLDAS_CLSM_combo_cold_gcell_mean = mean(stdev_GLDAS_CLSM_combo_cold_gcell_master)
    							stdev_GLDAS_CLSM_combo_cold_master.append(stdev_GLDAS_CLSM_combo_cold_gcell_mean)
    							rmse_GLDAS_CLSM_combo_cold_gcell_mean = mean(rmse_GLDAS_CLSM_combo_cold_gcell_master)
    							rmse_GLDAS_CLSM_combo_cold_master.append(rmse_GLDAS_CLSM_combo_cold_gcell_mean)
    							corr_GLDAS_CLSM_combo_cold_gcell_mean = mean(corr_GLDAS_CLSM_combo_cold_gcell_master)
    							corr_GLDAS_CLSM_combo_cold_master.append(corr_GLDAS_CLSM_combo_cold_gcell_mean)



    						elif (len_i == 6):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]
    							model_6 = i[5]

    							blend_GLDAS_CLSM_combo_temp_cold_gcell_master = []
    							bias_GLDAS_CLSM_combo_cold_gcell_master = []
    							stdev_GLDAS_CLSM_combo_cold_gcell_master = []
    							rmse_GLDAS_CLSM_combo_cold_gcell_master = []
    							corr_GLDAS_CLSM_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    								model_4_temp_cold = dframe_cold_season_gcell[model_4].values
    								model_5_temp_cold = dframe_cold_season_gcell[model_5].values
    								model_6_temp_cold = dframe_cold_season_gcell[model_6].values
    								dframe_6_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_6_model[model_2] = model_2_temp_cold
    								dframe_6_model[model_3] = model_3_temp_cold
    								dframe_6_model[model_4] = model_4_temp_cold
    								dframe_6_model[model_5] = model_5_temp_cold
    								dframe_6_model[model_6] = model_6_temp_cold
    								dframe_6_model_avg = dframe_6_model.mean(axis=1)
    								blend_6_model_temp_cold = dframe_6_model_avg
    								bias_6_model_cold = bias(blend_6_model_temp_cold,station_temp_cold)
    								bias_GLDAS_CLSM_combo_cold_gcell_master.append(bias_6_model_cold)
    								stdev_6_model_cold = np.std(blend_6_model_temp_cold)
    								stdev_GLDAS_CLSM_combo_cold_gcell_master.append(stdev_6_model_cold)
    								rmse_6_model_cold = mean_squared_error(station_temp_cold,blend_6_model_temp_cold,squared=False)
    								rmse_GLDAS_CLSM_combo_cold_gcell_master.append(rmse_6_model_cold)
    								corr_6_model_cold,_ = pearsonr(blend_6_model_temp_cold,station_temp_cold)
    								corr_GLDAS_CLSM_combo_cold_gcell_master.append(corr_6_model_cold)    							

    							bias_GLDAS_CLSM_combo_cold_gcell_mean = mean(bias_GLDAS_CLSM_combo_cold_gcell_master)
    							bias_GLDAS_CLSM_combo_cold_master.append(bias_GLDAS_CLSM_combo_cold_gcell_mean)
    							stdev_GLDAS_CLSM_combo_cold_gcell_mean = mean(stdev_GLDAS_CLSM_combo_cold_gcell_master)
    							stdev_GLDAS_CLSM_combo_cold_master.append(stdev_GLDAS_CLSM_combo_cold_gcell_mean)
    							rmse_GLDAS_CLSM_combo_cold_gcell_mean = mean(rmse_GLDAS_CLSM_combo_cold_gcell_master)
    							rmse_GLDAS_CLSM_combo_cold_master.append(rmse_GLDAS_CLSM_combo_cold_gcell_mean)
    							corr_GLDAS_CLSM_combo_cold_gcell_mean = mean(corr_GLDAS_CLSM_combo_cold_gcell_master)
    							corr_GLDAS_CLSM_combo_cold_master.append(corr_GLDAS_CLSM_combo_cold_gcell_mean)



    						elif (len_i == 7):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]
    							model_6 = i[5]
    							model_7 = i[6]

    							blend_GLDAS_CLSM_combo_temp_cold_gcell_master = []
    							bias_GLDAS_CLSM_combo_cold_gcell_master = []
    							stdev_GLDAS_CLSM_combo_cold_gcell_master = []
    							rmse_GLDAS_CLSM_combo_cold_gcell_master = []
    							corr_GLDAS_CLSM_combo_cold_gcell_master = []
    							for p in gcell_cold_uq:
    								if (p == 33777):
    									continue
    								dframe_cold_season_gcell = dframe_cold_season[dframe_cold_season['Grid Cell'] == p]
    								if (len(dframe_cold_season_gcell) < 2):
    									continue
    								station_temp_cold = dframe_cold_season_gcell['Station'].values
    								model_1_temp_cold = dframe_cold_season_gcell[model_1].values
    								model_2_temp_cold = dframe_cold_season_gcell[model_2].values
    								model_3_temp_cold = dframe_cold_season_gcell[model_3].values
    								model_4_temp_cold = dframe_cold_season_gcell[model_4].values
    								model_5_temp_cold = dframe_cold_season_gcell[model_5].values
    								model_6_temp_cold = dframe_cold_season_gcell[model_6].values
    								model_7_temp_cold = dframe_cold_season_gcell[model_7].values
    								dframe_7_model = pd.DataFrame(data=model_1_temp_cold, columns=[model_1])
    								dframe_7_model[model_2] = model_2_temp_cold
    								dframe_7_model[model_3] = model_3_temp_cold
    								dframe_7_model[model_4] = model_4_temp_cold
    								dframe_7_model[model_5] = model_5_temp_cold
    								dframe_7_model[model_6] = model_6_temp_cold
    								dframe_7_model[model_7] = model_7_temp_cold
    								dframe_7_model_avg = dframe_7_model.mean(axis=1)
    								blend_7_model_temp_cold = dframe_7_model_avg
    								bias_7_model_cold = bias(blend_7_model_temp_cold,station_temp_cold)
    								bias_GLDAS_CLSM_combo_cold_gcell_master.append(bias_7_model_cold)
    								stdev_7_model_cold = np.std(blend_7_model_temp_cold)
    								stdev_GLDAS_CLSM_combo_cold_gcell_master.append(stdev_7_model_cold)
    								rmse_7_model_cold = mean_squared_error(station_temp_cold,blend_7_model_temp_cold,squared=False)
    								rmse_GLDAS_CLSM_combo_cold_gcell_master.append(rmse_7_model_cold)
    								corr_7_model_cold,_ = pearsonr(blend_7_model_temp_cold,station_temp_cold)
    								corr_GLDAS_CLSM_combo_cold_gcell_master.append(corr_7_model_cold)    							

    							bias_GLDAS_CLSM_combo_cold_gcell_mean = mean(bias_GLDAS_CLSM_combo_cold_gcell_master)
    							bias_GLDAS_CLSM_combo_cold_master.append(bias_GLDAS_CLSM_combo_cold_gcell_mean)
    							stdev_GLDAS_CLSM_combo_cold_gcell_mean = mean(stdev_GLDAS_CLSM_combo_cold_gcell_master)
    							stdev_GLDAS_CLSM_combo_cold_master.append(stdev_GLDAS_CLSM_combo_cold_gcell_mean)
    							rmse_GLDAS_CLSM_combo_cold_gcell_mean = mean(rmse_GLDAS_CLSM_combo_cold_gcell_master)
    							rmse_GLDAS_CLSM_combo_cold_master.append(rmse_GLDAS_CLSM_combo_cold_gcell_mean)
    							corr_GLDAS_CLSM_combo_cold_gcell_mean = mean(corr_GLDAS_CLSM_combo_cold_gcell_master)
    							corr_GLDAS_CLSM_combo_cold_master.append(corr_GLDAS_CLSM_combo_cold_gcell_mean)



    						elif (len_i == 8):
    							bias_8_model_cold = bias_naive_cold_mean
    							bias_GLDAS_CLSM_combo_cold_master.append(bias_8_model_cold)
    							stdev_8_model_cold = stdev_naive_cold_mean
    							stdev_GLDAS_CLSM_combo_cold_master.append(stdev_8_model_cold)
    							rmse_8_model_cold = rmse_naive_cold_mean 
    							rmse_GLDAS_CLSM_combo_cold_master.append(rmse_8_model_cold)
    							corr_8_model_cold = corr_naive_cold_mean
    							corr_GLDAS_CLSM_combo_cold_master.append(corr_8_model_cold)

    					bias_GLDAS_CLSM_combo_cold_mean = mean(bias_GLDAS_CLSM_combo_cold_master)
    					stdev_GLDAS_CLSM_combo_cold_mean = mean(stdev_GLDAS_CLSM_combo_cold_master)
    					SDV_GLDAS_CLSM_combo_cold_mean = stdev_GLDAS_CLSM_combo_cold_mean/stdev_station_cold
    					rmse_GLDAS_CLSM_combo_cold_mean = mean(rmse_GLDAS_CLSM_combo_cold_master)
    					corr_GLDAS_CLSM_combo_cold_mean = mean(corr_GLDAS_CLSM_combo_cold_master)




#### Warm Season ####

## Calculate All Possible Combinations ##

    					gcell_warm = dframe_warm_season['Grid Cell'].values
    					gcell_warm_uq = np.unique(gcell_warm)


    					bias_naive_warm_gcell_master = []
    					stdev_naive_warm_gcell_master = []
    					rmse_naive_warm_gcell_master = []
    					corr_naive_warm_gcell_master = []
    					stdev_stn_warm_gcell_master = []
    					for p in gcell_warm_uq:
    						if (p == 33777):
    							continue
    						dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]

    						if (len(dframe_warm_season_gcell) < 2):
    							continue

    						station_temp_warm = dframe_warm_season_gcell['Station'].values
    						naive_temp_warm = dframe_warm_season_gcell['Naive Blend All'].values
    						bias_naive_warm = bias(naive_temp_warm,station_temp_warm)
    						bias_naive_warm_gcell_master.append(bias_naive_warm)
    						stdev_naive_warm = np.var(naive_temp_warm)
    						stdev_naive_warm_gcell_master.append(stdev_naive_warm)
    						stdev_station_warm = np.var(station_temp_warm)
    						stdev_stn_warm_gcell_master.append(stdev_station_warm)
    						rmse_naive_warm = mean_squared_error(station_temp_warm,naive_temp_warm,squared=False)
    						rmse_naive_warm_gcell_master.append(rmse_naive_warm)
    						corr_naive_warm,_ = pearsonr(naive_temp_warm,station_temp_warm)
    						corr_naive_warm_gcell_master.append(corr_naive_warm)

    					bias_naive_warm_mean = mean(bias_naive_warm_gcell_master)
    					stdev_naive_warm_mean = mean(stdev_naive_warm_gcell_master)
    					stdev_naive_warm_mean = math.sqrt(stdev_naive_warm_mean)
    					stdev_stn_warm_mean = mean(stdev_stn_warm_gcell_master)
    					stdev_station_warm = math.sqrt(stdev_stn_warm_mean)
    					SDV_naive_warm_mean = stdev_stn_warm_mean/stdev_station_warm
    					rmse_naive_warm_mean = mean(rmse_naive_warm_gcell_master)
    					corr_naive_warm_mean = mean(corr_naive_warm_gcell_master) 						


					
# 1 model combos #

    					bias_1_model_warm_master = []
    					rmse_1_model_warm_master = []
    					stdev_1_model_warm_master = []
    					SDV_1_model_warm_master = []
    					corr_1_model_warm_master = []

    					for i in ['CFSR','ERA-Interim','ERA5','ERA5-Land','JRA55','MERRA2','GLDAS-Noah','GLDAS-CLSM']:
    						blend_1_model_temp_warm_gcell_master = []
    						bias_1_model_warm_gcell_master = []
    						stdev_1_model_warm_gcell_master = []
    						rmse_1_model_warm_gcell_master = []
    						corr_1_model_warm_gcell_master = []
    						for p in gcell_warm_uq:
    							if (p == 3377):
    								continue
    							dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]


    							if (len(dframe_warm_season_gcell) < 2):
    								continue
    							station_temp_warm = dframe_warm_season_gcell['Station'].values
    							blend_1_model_temp_warm = dframe_warm_season_gcell[i].values
    							bias_1_model_warm = bias(blend_1_model_temp_warm,station_temp_warm)
    							bias_1_model_warm_gcell_master.append(bias_1_model_warm)
    							stdev_1_model_warm = np.var(blend_1_model_temp_warm)
    							stdev_1_model_warm_gcell_master.append(stdev_1_model_warm)
    							rmse_1_model_warm = mean_squared_error(station_temp_warm,blend_1_model_temp_warm,squared=False)
    							rmse_1_model_warm_gcell_master.append(rmse_1_model_warm)
    							corr_1_model_warm,_ = pearsonr(blend_1_model_temp_warm,station_temp_warm)
    							corr_1_model_warm_gcell_master.append(corr_1_model_warm)
    						bias_1_model_warm_gcell_mean = mean(bias_1_model_warm_gcell_master)
    						bias_1_model_warm_master.append(bias_1_model_warm_gcell_mean)
    						stdev_1_model_warm_gcell_mean = mean(stdev_1_model_warm_gcell_master)
    						stdev_1_model_warm_master.append(stdev_1_model_warm_gcell_mean)
    						rmse_1_model_warm_gcell_mean = mean(rmse_1_model_warm_gcell_master)
    						rmse_1_model_warm_master.append(rmse_1_model_warm_gcell_mean)
    						corr_1_model_warm_gcell_mean = mean(corr_1_model_warm_gcell_master)
    						corr_1_model_warm_master.append(corr_1_model_warm_gcell_mean)						    												
    					bias_1_model_warm_mean = mean(bias_1_model_warm_master)
    					stdev_1_model_warm_mean = mean(stdev_1_model_warm_master)
    					stdev_1_model_warm_mean2 = math.sqrt(stdev_1_model_warm_mean)
    					SDV_1_model_warm_mean = stdev_1_model_warm_mean2/stdev_station_warm
    					rmse_1_model_warm_mean = mean(rmse_1_model_warm_master)
    					corr_1_model_warm_mean = mean(corr_1_model_warm_master)    					

    						
# 2 model combos #

    					bias_2_model_warm_master = []
    					rmse_2_model_warm_master = []
    					stdev_2_model_warm_master = []
    					SDV_2_model_warm_master = []
    					corr_2_model_warm_master = []

    					blend_combos_2 = combinations(['CFSR','ERA-Interim','ERA5','ERA5-Land','JRA55','MERRA2','GLDAS-Noah','GLDAS-CLSM'],2)
    					for i in blend_combos_2:
    						combo = i

    						model_1 = i[0]
    						model_2 = i[1]

    						blend_2_model_temp_warm_gcell_master = []
    						bias_2_model_warm_gcell_master = []
    						stdev_2_model_warm_gcell_master = []
    						rmse_2_model_warm_gcell_master = []
    						corr_2_model_warm_gcell_master = []
    						for p in gcell_warm_uq:
    							dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    							if (len(dframe_warm_season_gcell) < 2):
    								continue
    							station_temp_warm = dframe_warm_season_gcell['Station'].values
    							model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    							model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    							dframe_2_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    							dframe_2_model[model_2] = model_2_temp_warm
    							dframe_2_model_avg = dframe_2_model.mean(axis=1)
    							blend_2_model_temp_warm = dframe_2_model_avg.values
    							bias_2_model_warm = bias(blend_2_model_temp_warm,station_temp_warm)
    							bias_2_model_warm_gcell_master.append(bias_2_model_warm)
    							stdev_2_model_warm = np.std(blend_2_model_temp_warm)
    							stdev_2_model_warm_gcell_master.append(stdev_2_model_warm)
    							rmse_2_model_warm = mean_squared_error(station_temp_warm,blend_2_model_temp_warm,squared=False)
    							rmse_2_model_warm_gcell_master.append(rmse_2_model_warm)
    							corr_2_model_warm,_ = pearsonr(blend_2_model_temp_warm,station_temp_warm)
    							corr_2_model_warm_gcell_master.append(corr_2_model_warm)
    						bias_2_model_warm_gcell_mean = mean(bias_2_model_warm_gcell_master)
    						bias_2_model_warm_master.append(bias_2_model_warm_gcell_mean)
    						stdev_2_model_warm_gcell_mean = mean(stdev_2_model_warm_gcell_master)
    						stdev_2_model_warm_master.append(stdev_2_model_warm_gcell_mean)
    						rmse_2_model_warm_gcell_mean = mean(rmse_2_model_warm_gcell_master)
    						rmse_2_model_warm_master.append(rmse_2_model_warm_gcell_mean)
    						corr_2_model_warm_gcell_mean = mean(corr_2_model_warm_gcell_master)
    						corr_2_model_warm_master.append(corr_2_model_warm_gcell_mean)						    												
    					bias_2_model_warm_mean = mean(bias_2_model_warm_master)
    					stdev_2_model_warm_mean = mean(stdev_2_model_warm_master)
    					stdev_2_model_warm_mean2 = math.sqrt(stdev_2_model_warm_mean)
    					SDV_2_model_warm_mean = stdev_2_model_warm_mean2/stdev_station_warm
    					rmse_2_model_warm_mean = mean(rmse_2_model_warm_master)
    					corr_2_model_warm_mean = mean(corr_2_model_warm_master) 



# 3 model combos #

    					bias_3_model_warm_master = []
    					rmse_3_model_warm_master = []
    					stdev_3_model_warm_master = []
    					SDV_3_model_warm_master = []
    					corr_3_model_warm_master = []

    					blend_combos_3 = combinations(['CFSR','ERA-Interim','ERA5','ERA5-Land','JRA55','MERRA2','GLDAS-Noah','GLDAS-CLSM'],3)
    					for i in blend_combos_3:
    						combo = i

    						model_1 = i[0]
    						model_2 = i[1]
    						model_3 = i[2]

    						blend_3_model_temp_warm_gcell_master = []
    						bias_3_model_warm_gcell_master = []
    						stdev_3_model_warm_gcell_master = []
    						rmse_3_model_warm_gcell_master = []
    						corr_3_model_warm_gcell_master = []
    						for p in gcell_warm_uq:
    							if (p == 3377):
    								continue
    							dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    							if (len(dframe_warm_season_gcell) < 2):
    								continue
    							station_temp_warm = dframe_warm_season_gcell['Station'].values
    							model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    							model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    							model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    							dframe_3_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    							dframe_3_model[model_2] = model_2_temp_warm
    							dframe_3_model[model_3] = model_3_temp_warm						
    							dframe_3_model_avg = dframe_3_model.mean(axis=1)
    							blend_3_model_temp_warm = dframe_3_model_avg.values
    							bias_3_model_warm = bias(blend_3_model_temp_warm,station_temp_warm)
    							bias_3_model_warm_gcell_master.append(bias_3_model_warm)
    							stdev_3_model_warm = np.std(blend_3_model_temp_warm)
    							stdev_3_model_warm_gcell_master.append(stdev_3_model_warm)
    							rmse_3_model_warm = mean_squared_error(station_temp_warm,blend_3_model_temp_warm,squared=False)
    							rmse_3_model_warm_gcell_master.append(rmse_3_model_warm)
    							corr_3_model_warm,_ = pearsonr(blend_3_model_temp_warm,station_temp_warm)
    							corr_3_model_warm_gcell_master.append(corr_3_model_warm)

    						bias_3_model_warm_gcell_mean = mean(bias_3_model_warm_gcell_master)
    						bias_3_model_warm_master.append(bias_3_model_warm_gcell_mean)
    						stdev_3_model_warm_gcell_mean = mean(stdev_3_model_warm_gcell_master)
    						stdev_3_model_warm_master.append(stdev_3_model_warm_gcell_mean)
    						rmse_3_model_warm_gcell_mean = mean(rmse_3_model_warm_gcell_master)
    						rmse_3_model_warm_master.append(rmse_3_model_warm_gcell_mean)
    						corr_3_model_warm_gcell_mean = mean(corr_3_model_warm_gcell_master)
    						corr_3_model_warm_master.append(corr_3_model_warm_gcell_mean)						    												
    					bias_3_model_warm_mean = mean(bias_3_model_warm_master)
    					stdev_3_model_warm_mean = mean(stdev_3_model_warm_master)
    					stdev_3_model_warm_mean2 = math.sqrt(stdev_3_model_warm_mean)
    					SDV_3_model_warm_mean = stdev_3_model_warm_mean2/stdev_station_warm
    					rmse_3_model_warm_mean = mean(rmse_3_model_warm_master)
    					corr_3_model_warm_mean = mean(corr_3_model_warm_master) 



# 4 model combos #

    					bias_4_model_warm_master = []
    					rmse_4_model_warm_master = []
    					stdev_4_model_warm_master = []
    					SDV_4_model_warm_master = []
    					corr_4_model_warm_master = []

    					blend_combos_4 = combinations(['CFSR','ERA-Interim','ERA5','ERA5-Land','JRA55','MERRA2','GLDAS-Noah','GLDAS-CLSM'],4)
    					for i in blend_combos_4:
    						combo = i

    						model_1 = i[0]
    						model_2 = i[1]
    						model_3 = i[2]
    						model_4 = i[3]

    						blend_4_model_temp_warm_gcell_master = []
    						bias_4_model_warm_gcell_master = []
    						stdev_4_model_warm_gcell_master = []
    						rmse_4_model_warm_gcell_master = []
    						corr_4_model_warm_gcell_master = []
    						for p in gcell_warm_uq:
    							if (p == 3377):
    								continue
    							dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    							if (len(dframe_warm_season_gcell) < 2):
    								continue
    							station_temp_warm = dframe_warm_season_gcell['Station'].values
    							model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    							model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    							model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    							model_4_temp_warm = dframe_warm_season_gcell[model_4].values						
    							dframe_4_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    							dframe_4_model[model_2] = model_2_temp_warm
    							dframe_4_model[model_3] = model_3_temp_warm
    							dframe_4_model[model_4] = model_4_temp_warm
    							dframe_4_model_avg = dframe_4_model.mean(axis=1)
    							blend_4_model_temp_warm = dframe_4_model_avg.values
    							bias_4_model_warm = bias(blend_4_model_temp_warm,station_temp_warm)
    							bias_4_model_warm_gcell_master.append(bias_4_model_warm)
    							stdev_4_model_warm = np.std(blend_4_model_temp_warm)
    							stdev_4_model_warm_gcell_master.append(stdev_4_model_warm)
    							rmse_4_model_warm = mean_squared_error(station_temp_warm,blend_4_model_temp_warm,squared=False)
    							rmse_4_model_warm_gcell_master.append(rmse_4_model_warm)
    							corr_4_model_warm,_ = pearsonr(blend_4_model_temp_warm,station_temp_warm)
    							corr_4_model_warm_gcell_master.append(corr_4_model_warm)


    						bias_4_model_warm_gcell_mean = mean(bias_4_model_warm_gcell_master)
    						bias_4_model_warm_master.append(bias_4_model_warm_gcell_mean)
    						stdev_4_model_warm_gcell_mean = mean(stdev_4_model_warm_gcell_master)
    						stdev_4_model_warm_master.append(stdev_4_model_warm_gcell_mean)
    						rmse_4_model_warm_gcell_mean = mean(rmse_4_model_warm_gcell_master)
    						rmse_4_model_warm_master.append(rmse_4_model_warm_gcell_mean)
    						corr_4_model_warm_gcell_mean = mean(corr_4_model_warm_gcell_master)
    						corr_4_model_warm_master.append(corr_4_model_warm_gcell_mean)						    												
    					bias_4_model_warm_mean = mean(bias_4_model_warm_master)
    					stdev_4_model_warm_mean = mean(stdev_4_model_warm_master)
    					stdev_4_model_warm_mean2 = math.sqrt(stdev_4_model_warm_mean)
    					SDV_4_model_warm_mean = stdev_4_model_warm_mean2/stdev_station_warm
    					rmse_4_model_warm_mean = mean(rmse_4_model_warm_master)
    					corr_4_model_warm_mean = mean(corr_4_model_warm_master)

# 5 model combos #

    					bias_5_model_warm_master = []
    					rmse_5_model_warm_master = []
    					stdev_5_model_warm_master = []
    					SDV_5_model_warm_master = []
    					corr_5_model_warm_master = []

    					blend_combos_5 = combinations(['CFSR','ERA-Interim','ERA5','ERA5-Land','JRA55','MERRA2','GLDAS-Noah','GLDAS-CLSM'],5)
    					for i in blend_combos_5:
    						combo = i

    						model_1 = i[0]
    						model_2 = i[1]
    						model_3 = i[2]
    						model_4 = i[3]
    						model_5 = i[4]
    						blend_5_model_temp_warm_gcell_master = []
    						bias_5_model_warm_gcell_master = []
    						stdev_5_model_warm_gcell_master = []
    						rmse_5_model_warm_gcell_master = []
    						corr_5_model_warm_gcell_master = []
    						for p in gcell_warm_uq:
    							if (p == 3377):
    								continue
    							dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    							if (len(dframe_warm_season_gcell) < 2):
    								continue
    							station_temp_warm = dframe_warm_season_gcell['Station'].values
    							model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    							model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    							model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    							model_4_temp_warm = dframe_warm_season_gcell[model_4].values
    							model_5_temp_warm = dframe_warm_season_gcell[model_5].values
    							dframe_5_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    							dframe_5_model[model_2] = model_2_temp_warm
    							dframe_5_model[model_3] = model_3_temp_warm
    							dframe_5_model[model_4] = model_4_temp_warm
    							dframe_5_model[model_5] = model_5_temp_warm
    							dframe_5_model_avg = dframe_5_model.mean(axis=1)
    							blend_5_model_temp_warm = dframe_5_model_avg.values

    							bias_5_model_warm = bias(blend_5_model_temp_warm,station_temp_warm)
    							bias_5_model_warm_gcell_master.append(bias_5_model_warm)
    							stdev_5_model_warm = np.std(blend_5_model_temp_warm)
    							stdev_5_model_warm_gcell_master.append(stdev_5_model_warm)
    							rmse_5_model_warm = mean_squared_error(station_temp_warm,blend_5_model_temp_warm,squared=False)
    							rmse_5_model_warm_gcell_master.append(rmse_5_model_warm)
    							corr_5_model_warm,_ = pearsonr(blend_5_model_temp_warm,station_temp_warm)
    							corr_5_model_warm_gcell_master.append(corr_5_model_warm)

    						bias_5_model_warm_gcell_mean = mean(bias_5_model_warm_gcell_master)
    						bias_5_model_warm_master.append(bias_5_model_warm_gcell_mean)
    						stdev_5_model_warm_gcell_mean = mean(stdev_5_model_warm_gcell_master)
    						stdev_5_model_warm_master.append(stdev_5_model_warm_gcell_mean)
    						rmse_5_model_warm_gcell_mean = mean(rmse_5_model_warm_gcell_master)
    						rmse_5_model_warm_master.append(rmse_5_model_warm_gcell_mean)
    						corr_5_model_warm_gcell_mean = mean(corr_5_model_warm_gcell_master)
    						corr_5_model_warm_master.append(corr_5_model_warm_gcell_mean)						    												
    					bias_5_model_warm_mean = mean(bias_5_model_warm_master)
    					stdev_5_model_warm_mean = mean(stdev_5_model_warm_master)
    					stdev_5_model_warm_mean2 = math.sqrt(stdev_5_model_warm_mean)
    					SDV_5_model_warm_mean = stdev_5_model_warm_mean2/stdev_station_warm
    					rmse_5_model_warm_mean = mean(rmse_5_model_warm_master)
    					corr_5_model_warm_mean = mean(corr_5_model_warm_master) 

# 6 model combos #

    					bias_6_model_warm_master = []
    					rmse_6_model_warm_master = []
    					stdev_6_model_warm_master = []
    					SDV_6_model_warm_master = []
    					corr_6_model_warm_master = []

    					blend_combos_6 = combinations(['CFSR','ERA-Interim','ERA5','ERA5-Land','JRA55','MERRA2','GLDAS-Noah','GLDAS-CLSM'],6)
    					for i in blend_combos_6:
    						combo = i

    						model_1 = i[0]
    						model_2 = i[1]
    						model_3 = i[2]
    						model_4 = i[3]
    						model_5 = i[4]
    						model_6 = i[5]

    						blend_6_model_temp_warm_gcell_master = []
    						bias_6_model_warm_gcell_master = []
    						stdev_6_model_warm_gcell_master = []
    						rmse_6_model_warm_gcell_master = []
    						corr_6_model_warm_gcell_master = []
    						for p in gcell_warm_uq:
    							if (p == 3377):
    								continue
    							dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    							if (len(dframe_warm_season_gcell) < 2):
    								continue
    							station_temp_warm = dframe_warm_season_gcell['Station'].values
    							model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    							model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    							model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    							model_4_temp_warm = dframe_warm_season_gcell[model_4].values
    							model_5_temp_warm = dframe_warm_season_gcell[model_5].values
    							model_6_temp_warm = dframe_warm_season_gcell[model_6].values							
    							dframe_6_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    							dframe_6_model[model_2] = model_2_temp_warm
    							dframe_6_model[model_3] = model_3_temp_warm
    							dframe_6_model[model_4] = model_4_temp_warm
    							dframe_6_model[model_5] = model_5_temp_warm
    							dframe_6_model[model_6] = model_6_temp_warm
    							dframe_6_model_avg = dframe_6_model.mean(axis=1)
    							blend_6_model_temp_warm = dframe_6_model_avg.values
    							bias_6_model_warm = bias(blend_6_model_temp_warm,station_temp_warm)
    							bias_6_model_warm_gcell_master.append(bias_6_model_warm)
    							stdev_6_model_warm = np.std(blend_6_model_temp_warm)
    							stdev_6_model_warm_gcell_master.append(stdev_6_model_warm)
    							rmse_6_model_warm = mean_squared_error(station_temp_warm,blend_6_model_temp_warm,squared=False)
    							rmse_6_model_warm_gcell_master.append(rmse_6_model_warm)
    							corr_6_model_warm,_ = pearsonr(blend_6_model_temp_warm,station_temp_warm)
    							corr_6_model_warm_gcell_master.append(corr_6_model_warm)

    						bias_6_model_warm_gcell_mean = mean(bias_6_model_warm_gcell_master)
    						bias_6_model_warm_master.append(bias_6_model_warm_gcell_mean)
    						stdev_6_model_warm_gcell_mean = mean(stdev_6_model_warm_gcell_master)
    						stdev_6_model_warm_master.append(stdev_6_model_warm_gcell_mean)
    						rmse_6_model_warm_gcell_mean = mean(rmse_6_model_warm_gcell_master)
    						rmse_6_model_warm_master.append(rmse_6_model_warm_gcell_mean)
    						corr_6_model_warm_gcell_mean = mean(corr_6_model_warm_gcell_master)
    						corr_6_model_warm_master.append(corr_6_model_warm_gcell_mean)						    												
    					bias_6_model_warm_mean = mean(bias_6_model_warm_master)
    					stdev_6_model_warm_mean = mean(stdev_6_model_warm_master)
    					stdev_6_model_warm_mean2 = math.sqrt(stdev_6_model_warm_mean)
    					SDV_6_model_warm_mean = stdev_6_model_warm_mean2/stdev_station_warm
    					rmse_6_model_warm_mean = mean(rmse_6_model_warm_master)
    					corr_6_model_warm_mean = mean(corr_6_model_warm_master)




# 7 model combos #

    					bias_7_model_warm_master = []
    					rmse_7_model_warm_master = []
    					stdev_7_model_warm_master = []
    					SDV_7_model_warm_master = []
    					corr_7_model_warm_master = []

    					blend_combos_7 = combinations(['CFSR','ERA-Interim','ERA5','ERA5-Land','JRA55','MERRA2','GLDAS-Noah','GLDAS-CLSM'],7)
    					for i in blend_combos_7:
    						combo = i

    						model_1 = i[0]
    						model_2 = i[1]
    						model_3 = i[2]
    						model_4 = i[3]
    						model_5 = i[4]
    						model_6 = i[5]
    						model_7 = i[6]

    						blend_7_model_temp_warm_gcell_master = []
    						bias_7_model_warm_gcell_master = []
    						stdev_7_model_warm_gcell_master = []
    						rmse_7_model_warm_gcell_master = []
    						corr_7_model_warm_gcell_master = []
    						for p in gcell_warm_uq:
    							if (p == 3377):
    								continue
    							dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    							if (len(dframe_warm_season_gcell) < 2):
    								continue
    							station_temp_warm = dframe_warm_season_gcell['Station'].values
    							model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    							model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    							model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    							model_4_temp_warm = dframe_warm_season_gcell[model_4].values
    							model_5_temp_warm = dframe_warm_season_gcell[model_5].values
    							model_6_temp_warm = dframe_warm_season_gcell[model_6].values
    							model_7_temp_warm = dframe_warm_season_gcell[model_7].values

    							dframe_7_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    							dframe_7_model[model_2] = model_2_temp_warm
    							dframe_7_model[model_3] = model_3_temp_warm
    							dframe_7_model[model_4] = model_4_temp_warm
    							dframe_7_model[model_5] = model_5_temp_warm
    							dframe_7_model[model_6] = model_6_temp_warm
    							dframe_7_model[model_7] = model_7_temp_warm
    							dframe_7_model_avg = dframe_7_model.mean(axis=1)
    							blend_7_model_temp_warm = dframe_7_model_avg.values

    							bias_7_model_warm = bias(blend_7_model_temp_warm,station_temp_warm)
    							bias_7_model_warm_gcell_master.append(bias_7_model_warm)
    							stdev_7_model_warm = np.std(blend_7_model_temp_warm)
    							stdev_7_model_warm_gcell_master.append(stdev_7_model_warm)
    							rmse_7_model_warm = mean_squared_error(station_temp_warm,blend_7_model_temp_warm,squared=False)
    							rmse_7_model_warm_gcell_master.append(rmse_7_model_warm)
    							corr_7_model_warm,_ = pearsonr(blend_7_model_temp_warm,station_temp_warm)
    							corr_7_model_warm_gcell_master.append(corr_7_model_warm)


    						bias_7_model_warm_gcell_mean = mean(bias_7_model_warm_gcell_master)
    						bias_7_model_warm_master.append(bias_7_model_warm_gcell_mean)
    						stdev_7_model_warm_gcell_mean = mean(stdev_7_model_warm_gcell_master)
    						stdev_7_model_warm_master.append(stdev_7_model_warm_gcell_mean)
    						rmse_7_model_warm_gcell_mean = mean(rmse_7_model_warm_gcell_master)
    						rmse_7_model_warm_master.append(rmse_7_model_warm_gcell_mean)
    						corr_7_model_warm_gcell_mean = mean(corr_7_model_warm_gcell_master)
    						corr_7_model_warm_master.append(corr_7_model_warm_gcell_mean)						    												
    					bias_7_model_warm_mean = mean(bias_7_model_warm_master)
    					stdev_7_model_warm_mean = mean(stdev_7_model_warm_master)
    					stdev_7_model_warm_mean2 = math.sqrt(stdev_7_model_warm_mean)
    					SDV_7_model_warm_mean = stdev_7_model_warm_mean2/stdev_station_warm
    					rmse_7_model_warm_mean = mean(rmse_7_model_warm_master)
    					corr_7_model_warm_mean = mean(corr_7_model_warm_master) 

# 8 model combo #


    					bias_8_model_warm_mean = bias_naive_warm_mean
    					stdev_8_model_warm_mean = stdev_naive_warm_mean
    					SDV_8_model_warm_mean = SDV_naive_warm_mean
    					rmse_8_model_warm_mean = rmse_naive_warm_mean
    					corr_8_model_warm_mean = corr_naive_warm_mean


## Calculate Combinations Associated With A Particular Model ##

    					


## CFSR Model ##

    					bias_CFSR_combo_warm_master = []
    					rmse_CFSR_combo_warm_master = []
    					stdev_CFSR_combo_warm_master = []
    					SDV_CFSR_combo_warm_master = []
    					corr_CFSR_combo_warm_master = []

    					for i in CFSR_array:
    						len_i = len(i)
    						if (len_i == 1):
    							blend_CFSR_combo_temp_warm_gcell_master = []
    							bias_CFSR_combo_warm_gcell_master = []
    							stdev_CFSR_combo_warm_gcell_master = []
    							rmse_CFSR_combo_warm_gcell_master = []
    							corr_CFSR_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								blend_1_model_temp_warm = dframe_warm_season_gcell[i[0]].values
    								print(blend_1_model_temp_warm)
    								print(station_temp_warm)
    								bias_1_model_warm = bias(blend_1_model_temp_warm,station_temp_warm)
    								bias_CFSR_combo_warm_gcell_master.append(bias_1_model_warm)
    								stdev_1_model_warm = np.std(blend_1_model_temp_warm)
    								stdev_CFSR_combo_warm_gcell_master.append(stdev_1_model_warm)
    								rmse_1_model_warm = mean_squared_error(station_temp_warm,blend_1_model_temp_warm,squared=False)
    								rmse_CFSR_combo_warm_gcell_master.append(rmse_1_model_warm)
    								corr_1_model_warm,_ = pearsonr(blend_1_model_temp_warm,station_temp_warm)
    								corr_CFSR_combo_warm_gcell_master.append(corr_1_model_warm)    							

    							bias_CFSR_combo_warm_gcell_mean = mean(bias_CFSR_combo_warm_gcell_master)
    							bias_CFSR_combo_warm_master.append(bias_CFSR_combo_warm_gcell_mean)
    							stdev_CFSR_combo_warm_gcell_mean = mean(stdev_CFSR_combo_warm_gcell_master)
    							stdev_CFSR_combo_warm_master.append(stdev_CFSR_combo_warm_gcell_mean)
    							rmse_CFSR_combo_warm_gcell_mean = mean(rmse_CFSR_combo_warm_gcell_master)
    							rmse_CFSR_combo_warm_master.append(rmse_CFSR_combo_warm_gcell_mean)
    							corr_CFSR_combo_warm_gcell_mean = mean(corr_CFSR_combo_warm_gcell_master)
    							corr_CFSR_combo_warm_master.append(corr_CFSR_combo_warm_gcell_mean)

    						elif (len_i == 2):
    							model_1 = i[0]
    							model_2 = i[1]

    							blend_CFSR_combo_temp_warm_gcell_master = []
    							bias_CFSR_combo_warm_gcell_master = []
    							stdev_CFSR_combo_warm_gcell_master = []
    							rmse_CFSR_combo_warm_gcell_master = []
    							corr_CFSR_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								dframe_2_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_2_model[model_2] = model_2_temp_warm
    								dframe_2_model_avg = dframe_2_model.mean(axis=1)
    								blend_2_model_temp_warm = dframe_2_model_avg
    								bias_2_model_warm = bias(blend_2_model_temp_warm,station_temp_warm)
    								bias_CFSR_combo_warm_gcell_master.append(bias_2_model_warm)
    								stdev_2_model_warm = np.std(blend_2_model_temp_warm)
    								stdev_CFSR_combo_warm_gcell_master.append(stdev_2_model_warm)
    								rmse_2_model_warm = mean_squared_error(station_temp_warm,blend_2_model_temp_warm,squared=False)
    								rmse_CFSR_combo_warm_gcell_master.append(rmse_2_model_warm)
    								corr_2_model_warm,_ = pearsonr(blend_2_model_temp_warm,station_temp_warm)
    								corr_CFSR_combo_warm_gcell_master.append(corr_2_model_warm)    							

    							bias_CFSR_combo_warm_gcell_mean = mean(bias_CFSR_combo_warm_gcell_master)
    							bias_CFSR_combo_warm_master.append(bias_CFSR_combo_warm_gcell_mean)
    							stdev_CFSR_combo_warm_gcell_mean = mean(stdev_CFSR_combo_warm_gcell_master)
    							stdev_CFSR_combo_warm_master.append(stdev_CFSR_combo_warm_gcell_mean)
    							rmse_CFSR_combo_warm_gcell_mean = mean(rmse_CFSR_combo_warm_gcell_master)
    							rmse_CFSR_combo_warm_master.append(rmse_CFSR_combo_warm_gcell_mean)
    							corr_CFSR_combo_warm_gcell_mean = mean(corr_CFSR_combo_warm_gcell_master)
    							corr_CFSR_combo_warm_master.append(corr_CFSR_combo_warm_gcell_mean)


    						elif (len_i == 3):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]

    							blend_CFSR_combo_temp_warm_gcell_master = []
    							bias_CFSR_combo_warm_gcell_master = []
    							stdev_CFSR_combo_warm_gcell_master = []
    							rmse_CFSR_combo_warm_gcell_master = []
    							corr_CFSR_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    								dframe_3_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_3_model[model_2] = model_2_temp_warm
    								dframe_3_model[model_3] = model_3_temp_warm
    								dframe_3_model_avg = dframe_3_model.mean(axis=1)
    								blend_3_model_temp_warm = dframe_3_model_avg
    								bias_3_model_warm = bias(blend_3_model_temp_warm,station_temp_warm)
    								bias_CFSR_combo_warm_gcell_master.append(bias_3_model_warm)
    								stdev_3_model_warm = np.std(blend_3_model_temp_warm)
    								stdev_CFSR_combo_warm_gcell_master.append(stdev_3_model_warm)
    								rmse_3_model_warm = mean_squared_error(station_temp_warm,blend_3_model_temp_warm,squared=False)
    								rmse_CFSR_combo_warm_gcell_master.append(rmse_3_model_warm)
    								corr_3_model_warm,_ = pearsonr(blend_3_model_temp_warm,station_temp_warm)
    								corr_CFSR_combo_warm_gcell_master.append(corr_3_model_warm)    							

    							bias_CFSR_combo_warm_gcell_mean = mean(bias_CFSR_combo_warm_gcell_master)
    							bias_CFSR_combo_warm_master.append(bias_CFSR_combo_warm_gcell_mean)
    							stdev_CFSR_combo_warm_gcell_mean = mean(stdev_CFSR_combo_warm_gcell_master)
    							stdev_CFSR_combo_warm_master.append(stdev_CFSR_combo_warm_gcell_mean)
    							rmse_CFSR_combo_warm_gcell_mean = mean(rmse_CFSR_combo_warm_gcell_master)
    							rmse_CFSR_combo_warm_master.append(rmse_CFSR_combo_warm_gcell_mean)
    							corr_CFSR_combo_warm_gcell_mean = mean(corr_CFSR_combo_warm_gcell_master)
    							corr_CFSR_combo_warm_master.append(corr_CFSR_combo_warm_gcell_mean)


    						elif (len_i == 4):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]

    							blend_CFSR_combo_temp_warm_gcell_master = []
    							bias_CFSR_combo_warm_gcell_master = []
    							stdev_CFSR_combo_warm_gcell_master = []
    							rmse_CFSR_combo_warm_gcell_master = []
    							corr_CFSR_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    								model_4_temp_warm = dframe_warm_season_gcell[model_4].values
    								dframe_4_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_4_model[model_2] = model_2_temp_warm
    								dframe_4_model[model_3] = model_3_temp_warm
    								dframe_4_model[model_4] = model_4_temp_warm
    								dframe_4_model_avg = dframe_4_model.mean(axis=1)
    								blend_4_model_temp_warm = dframe_4_model_avg
    								bias_4_model_warm = bias(blend_4_model_temp_warm,station_temp_warm)
    								bias_CFSR_combo_warm_gcell_master.append(bias_4_model_warm)
    								stdev_4_model_warm = np.std(blend_4_model_temp_warm)
    								stdev_CFSR_combo_warm_gcell_master.append(stdev_4_model_warm)
    								rmse_4_model_warm = mean_squared_error(station_temp_warm,blend_4_model_temp_warm,squared=False)
    								rmse_CFSR_combo_warm_gcell_master.append(rmse_4_model_warm)
    								corr_4_model_warm,_ = pearsonr(blend_4_model_temp_warm,station_temp_warm)
    								corr_CFSR_combo_warm_gcell_master.append(corr_4_model_warm)    							

    							bias_CFSR_combo_warm_gcell_mean = mean(bias_CFSR_combo_warm_gcell_master)
    							bias_CFSR_combo_warm_master.append(bias_CFSR_combo_warm_gcell_mean)
    							stdev_CFSR_combo_warm_gcell_mean = mean(stdev_CFSR_combo_warm_gcell_master)
    							stdev_CFSR_combo_warm_master.append(stdev_CFSR_combo_warm_gcell_mean)
    							rmse_CFSR_combo_warm_gcell_mean = mean(rmse_CFSR_combo_warm_gcell_master)
    							rmse_CFSR_combo_warm_master.append(rmse_CFSR_combo_warm_gcell_mean)
    							corr_CFSR_combo_warm_gcell_mean = mean(corr_CFSR_combo_warm_gcell_master)
    							corr_CFSR_combo_warm_master.append(corr_CFSR_combo_warm_gcell_mean)



    						elif (len_i == 5):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]

    							blend_CFSR_combo_temp_warm_gcell_master = []
    							bias_CFSR_combo_warm_gcell_master = []
    							stdev_CFSR_combo_warm_gcell_master = []
    							rmse_CFSR_combo_warm_gcell_master = []
    							corr_CFSR_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    								model_4_temp_warm = dframe_warm_season_gcell[model_4].values
    								model_5_temp_warm = dframe_warm_season_gcell[model_5].values
    								dframe_5_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_5_model[model_2] = model_2_temp_warm
    								dframe_5_model[model_3] = model_3_temp_warm
    								dframe_5_model[model_4] = model_4_temp_warm
    								dframe_5_model[model_5] = model_5_temp_warm
    								dframe_5_model_avg = dframe_5_model.mean(axis=1)
    								blend_5_model_temp_warm = dframe_5_model_avg
    								bias_5_model_warm = bias(blend_5_model_temp_warm,station_temp_warm)
    								bias_CFSR_combo_warm_gcell_master.append(bias_5_model_warm)
    								stdev_5_model_warm = np.std(blend_5_model_temp_warm)
    								stdev_CFSR_combo_warm_gcell_master.append(stdev_5_model_warm)
    								rmse_5_model_warm = mean_squared_error(station_temp_warm,blend_5_model_temp_warm,squared=False)
    								rmse_CFSR_combo_warm_gcell_master.append(rmse_5_model_warm)
    								corr_5_model_warm,_ = pearsonr(blend_5_model_temp_warm,station_temp_warm)
    								corr_CFSR_combo_warm_gcell_master.append(corr_5_model_warm)    							

    							bias_CFSR_combo_warm_gcell_mean = mean(bias_CFSR_combo_warm_gcell_master)
    							bias_CFSR_combo_warm_master.append(bias_CFSR_combo_warm_gcell_mean)
    							stdev_CFSR_combo_warm_gcell_mean = mean(stdev_CFSR_combo_warm_gcell_master)
    							stdev_CFSR_combo_warm_master.append(stdev_CFSR_combo_warm_gcell_mean)
    							rmse_CFSR_combo_warm_gcell_mean = mean(rmse_CFSR_combo_warm_gcell_master)
    							rmse_CFSR_combo_warm_master.append(rmse_CFSR_combo_warm_gcell_mean)
    							corr_CFSR_combo_warm_gcell_mean = mean(corr_CFSR_combo_warm_gcell_master)
    							corr_CFSR_combo_warm_master.append(corr_CFSR_combo_warm_gcell_mean)



    						elif (len_i == 6):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]
    							model_6 = i[5]

    							blend_CFSR_combo_temp_warm_gcell_master = []
    							bias_CFSR_combo_warm_gcell_master = []
    							stdev_CFSR_combo_warm_gcell_master = []
    							rmse_CFSR_combo_warm_gcell_master = []
    							corr_CFSR_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    								model_4_temp_warm = dframe_warm_season_gcell[model_4].values
    								model_5_temp_warm = dframe_warm_season_gcell[model_5].values
    								model_6_temp_warm = dframe_warm_season_gcell[model_6].values
    								dframe_6_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_6_model[model_2] = model_2_temp_warm
    								dframe_6_model[model_3] = model_3_temp_warm
    								dframe_6_model[model_4] = model_4_temp_warm
    								dframe_6_model[model_5] = model_5_temp_warm
    								dframe_6_model[model_6] = model_6_temp_warm
    								dframe_6_model_avg = dframe_6_model.mean(axis=1)
    								blend_6_model_temp_warm = dframe_6_model_avg
    								bias_6_model_warm = bias(blend_6_model_temp_warm,station_temp_warm)
    								bias_CFSR_combo_warm_gcell_master.append(bias_6_model_warm)
    								stdev_6_model_warm = np.std(blend_6_model_temp_warm)
    								stdev_CFSR_combo_warm_gcell_master.append(stdev_6_model_warm)
    								rmse_6_model_warm = mean_squared_error(station_temp_warm,blend_6_model_temp_warm,squared=False)
    								rmse_CFSR_combo_warm_gcell_master.append(rmse_6_model_warm)
    								corr_6_model_warm,_ = pearsonr(blend_6_model_temp_warm,station_temp_warm)
    								corr_CFSR_combo_warm_gcell_master.append(corr_6_model_warm)    							

    							bias_CFSR_combo_warm_gcell_mean = mean(bias_CFSR_combo_warm_gcell_master)
    							bias_CFSR_combo_warm_master.append(bias_CFSR_combo_warm_gcell_mean)
    							stdev_CFSR_combo_warm_gcell_mean = mean(stdev_CFSR_combo_warm_gcell_master)
    							stdev_CFSR_combo_warm_master.append(stdev_CFSR_combo_warm_gcell_mean)
    							rmse_CFSR_combo_warm_gcell_mean = mean(rmse_CFSR_combo_warm_gcell_master)
    							rmse_CFSR_combo_warm_master.append(rmse_CFSR_combo_warm_gcell_mean)
    							corr_CFSR_combo_warm_gcell_mean = mean(corr_CFSR_combo_warm_gcell_master)
    							corr_CFSR_combo_warm_master.append(corr_CFSR_combo_warm_gcell_mean)



    						elif (len_i == 7):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]
    							model_6 = i[5]
    							model_7 = i[6]

    							blend_CFSR_combo_temp_warm_gcell_master = []
    							bias_CFSR_combo_warm_gcell_master = []
    							stdev_CFSR_combo_warm_gcell_master = []
    							rmse_CFSR_combo_warm_gcell_master = []
    							corr_CFSR_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    								model_4_temp_warm = dframe_warm_season_gcell[model_4].values
    								model_5_temp_warm = dframe_warm_season_gcell[model_5].values
    								model_6_temp_warm = dframe_warm_season_gcell[model_6].values
    								model_7_temp_warm = dframe_warm_season_gcell[model_7].values
    								dframe_7_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_7_model[model_2] = model_2_temp_warm
    								dframe_7_model[model_3] = model_3_temp_warm
    								dframe_7_model[model_4] = model_4_temp_warm
    								dframe_7_model[model_5] = model_5_temp_warm
    								dframe_7_model[model_6] = model_6_temp_warm
    								dframe_7_model[model_7] = model_7_temp_warm
    								dframe_7_model_avg = dframe_7_model.mean(axis=1)
    								blend_7_model_temp_warm = dframe_7_model_avg
    								bias_7_model_warm = bias(blend_7_model_temp_warm,station_temp_warm)
    								bias_CFSR_combo_warm_gcell_master.append(bias_7_model_warm)
    								stdev_7_model_warm = np.std(blend_7_model_temp_warm)
    								stdev_CFSR_combo_warm_gcell_master.append(stdev_7_model_warm)
    								rmse_7_model_warm = mean_squared_error(station_temp_warm,blend_7_model_temp_warm,squared=False)
    								rmse_CFSR_combo_warm_gcell_master.append(rmse_7_model_warm)
    								corr_7_model_warm,_ = pearsonr(blend_7_model_temp_warm,station_temp_warm)
    								corr_CFSR_combo_warm_gcell_master.append(corr_7_model_warm)    							

    							bias_CFSR_combo_warm_gcell_mean = mean(bias_CFSR_combo_warm_gcell_master)
    							bias_CFSR_combo_warm_master.append(bias_CFSR_combo_warm_gcell_mean)
    							stdev_CFSR_combo_warm_gcell_mean = mean(stdev_CFSR_combo_warm_gcell_master)
    							stdev_CFSR_combo_warm_master.append(stdev_CFSR_combo_warm_gcell_mean)
    							rmse_CFSR_combo_warm_gcell_mean = mean(rmse_CFSR_combo_warm_gcell_master)
    							rmse_CFSR_combo_warm_master.append(rmse_CFSR_combo_warm_gcell_mean)
    							corr_CFSR_combo_warm_gcell_mean = mean(corr_CFSR_combo_warm_gcell_master)
    							corr_CFSR_combo_warm_master.append(corr_CFSR_combo_warm_gcell_mean)



    						elif (len_i == 8):
    							bias_8_model_warm = bias_naive_warm_mean
    							bias_CFSR_combo_warm_master.append(bias_8_model_warm)
    							stdev_8_model_warm = stdev_naive_warm_mean
    							stdev_CFSR_combo_warm_master.append(stdev_8_model_warm)
    							rmse_8_model_warm = rmse_naive_warm_mean 
    							rmse_CFSR_combo_warm_master.append(rmse_8_model_warm)
    							corr_8_model_warm = corr_naive_warm_mean
    							corr_CFSR_combo_warm_master.append(corr_8_model_warm)

    					bias_CFSR_combo_warm_mean = mean(bias_CFSR_combo_warm_master)
    					stdev_CFSR_combo_warm_mean = mean(stdev_CFSR_combo_warm_master)
    					SDV_CFSR_combo_warm_mean = stdev_CFSR_combo_warm_mean/stdev_station_warm
    					rmse_CFSR_combo_warm_mean = mean(rmse_CFSR_combo_warm_master)
    					corr_CFSR_combo_warm_mean = mean(corr_CFSR_combo_warm_master)



## ERA-Interim Model ##

    					bias_ERAI_combo_warm_master = []
    					rmse_ERAI_combo_warm_master = []
    					stdev_ERAI_combo_warm_master = []
    					SDV_ERAI_combo_warm_master = []
    					corr_ERAI_combo_warm_master = []

    					for i in ERAI_array:
    						len_i = len(i)
    						if (len_i == 1):
    							blend_ERAI_combo_temp_warm_gcell_master = []
    							bias_ERAI_combo_warm_gcell_master = []
    							stdev_ERAI_combo_warm_gcell_master = []
    							rmse_ERAI_combo_warm_gcell_master = []
    							corr_ERAI_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								blend_1_model_temp_warm = dframe_warm_season_gcell[i[0]].values
    								print(blend_1_model_temp_warm)
    								print(station_temp_warm)
    								bias_1_model_warm = bias(blend_1_model_temp_warm,station_temp_warm)
    								bias_ERAI_combo_warm_gcell_master.append(bias_1_model_warm)
    								stdev_1_model_warm = np.std(blend_1_model_temp_warm)
    								stdev_ERAI_combo_warm_gcell_master.append(stdev_1_model_warm)
    								rmse_1_model_warm = mean_squared_error(station_temp_warm,blend_1_model_temp_warm,squared=False)
    								rmse_ERAI_combo_warm_gcell_master.append(rmse_1_model_warm)
    								corr_1_model_warm,_ = pearsonr(blend_1_model_temp_warm,station_temp_warm)
    								corr_ERAI_combo_warm_gcell_master.append(corr_1_model_warm)    							

    							bias_ERAI_combo_warm_gcell_mean = mean(bias_ERAI_combo_warm_gcell_master)
    							bias_ERAI_combo_warm_master.append(bias_ERAI_combo_warm_gcell_mean)
    							stdev_ERAI_combo_warm_gcell_mean = mean(stdev_ERAI_combo_warm_gcell_master)
    							stdev_ERAI_combo_warm_master.append(stdev_ERAI_combo_warm_gcell_mean)
    							rmse_ERAI_combo_warm_gcell_mean = mean(rmse_ERAI_combo_warm_gcell_master)
    							rmse_ERAI_combo_warm_master.append(rmse_ERAI_combo_warm_gcell_mean)
    							corr_ERAI_combo_warm_gcell_mean = mean(corr_ERAI_combo_warm_gcell_master)
    							corr_ERAI_combo_warm_master.append(corr_ERAI_combo_warm_gcell_mean)

    						elif (len_i == 2):
    							model_1 = i[0]
    							model_2 = i[1]

    							blend_ERAI_combo_temp_warm_gcell_master = []
    							bias_ERAI_combo_warm_gcell_master = []
    							stdev_ERAI_combo_warm_gcell_master = []
    							rmse_ERAI_combo_warm_gcell_master = []
    							corr_ERAI_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								dframe_2_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_2_model[model_2] = model_2_temp_warm
    								dframe_2_model_avg = dframe_2_model.mean(axis=1)
    								blend_2_model_temp_warm = dframe_2_model_avg
    								bias_2_model_warm = bias(blend_2_model_temp_warm,station_temp_warm)
    								bias_ERAI_combo_warm_gcell_master.append(bias_2_model_warm)
    								stdev_2_model_warm = np.std(blend_2_model_temp_warm)
    								stdev_ERAI_combo_warm_gcell_master.append(stdev_2_model_warm)
    								rmse_2_model_warm = mean_squared_error(station_temp_warm,blend_2_model_temp_warm,squared=False)
    								rmse_ERAI_combo_warm_gcell_master.append(rmse_2_model_warm)
    								corr_2_model_warm,_ = pearsonr(blend_2_model_temp_warm,station_temp_warm)
    								corr_ERAI_combo_warm_gcell_master.append(corr_2_model_warm)    							

    							bias_ERAI_combo_warm_gcell_mean = mean(bias_ERAI_combo_warm_gcell_master)
    							bias_ERAI_combo_warm_master.append(bias_ERAI_combo_warm_gcell_mean)
    							stdev_ERAI_combo_warm_gcell_mean = mean(stdev_ERAI_combo_warm_gcell_master)
    							stdev_ERAI_combo_warm_master.append(stdev_ERAI_combo_warm_gcell_mean)
    							rmse_ERAI_combo_warm_gcell_mean = mean(rmse_ERAI_combo_warm_gcell_master)
    							rmse_ERAI_combo_warm_master.append(rmse_ERAI_combo_warm_gcell_mean)
    							corr_ERAI_combo_warm_gcell_mean = mean(corr_ERAI_combo_warm_gcell_master)
    							corr_ERAI_combo_warm_master.append(corr_ERAI_combo_warm_gcell_mean)


    						elif (len_i == 3):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]

    							blend_ERAI_combo_temp_warm_gcell_master = []
    							bias_ERAI_combo_warm_gcell_master = []
    							stdev_ERAI_combo_warm_gcell_master = []
    							rmse_ERAI_combo_warm_gcell_master = []
    							corr_ERAI_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    								dframe_3_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_3_model[model_2] = model_2_temp_warm
    								dframe_3_model[model_3] = model_3_temp_warm
    								dframe_3_model_avg = dframe_3_model.mean(axis=1)
    								blend_3_model_temp_warm = dframe_3_model_avg
    								bias_3_model_warm = bias(blend_3_model_temp_warm,station_temp_warm)
    								bias_ERAI_combo_warm_gcell_master.append(bias_3_model_warm)
    								stdev_3_model_warm = np.std(blend_3_model_temp_warm)
    								stdev_ERAI_combo_warm_gcell_master.append(stdev_3_model_warm)
    								rmse_3_model_warm = mean_squared_error(station_temp_warm,blend_3_model_temp_warm,squared=False)
    								rmse_ERAI_combo_warm_gcell_master.append(rmse_3_model_warm)
    								corr_3_model_warm,_ = pearsonr(blend_3_model_temp_warm,station_temp_warm)
    								corr_ERAI_combo_warm_gcell_master.append(corr_3_model_warm)    							

    							bias_ERAI_combo_warm_gcell_mean = mean(bias_ERAI_combo_warm_gcell_master)
    							bias_ERAI_combo_warm_master.append(bias_ERAI_combo_warm_gcell_mean)
    							stdev_ERAI_combo_warm_gcell_mean = mean(stdev_ERAI_combo_warm_gcell_master)
    							stdev_ERAI_combo_warm_master.append(stdev_ERAI_combo_warm_gcell_mean)
    							rmse_ERAI_combo_warm_gcell_mean = mean(rmse_ERAI_combo_warm_gcell_master)
    							rmse_ERAI_combo_warm_master.append(rmse_ERAI_combo_warm_gcell_mean)
    							corr_ERAI_combo_warm_gcell_mean = mean(corr_ERAI_combo_warm_gcell_master)
    							corr_ERAI_combo_warm_master.append(corr_ERAI_combo_warm_gcell_mean)


    						elif (len_i == 4):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]

    							blend_ERAI_combo_temp_warm_gcell_master = []
    							bias_ERAI_combo_warm_gcell_master = []
    							stdev_ERAI_combo_warm_gcell_master = []
    							rmse_ERAI_combo_warm_gcell_master = []
    							corr_ERAI_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    								model_4_temp_warm = dframe_warm_season_gcell[model_4].values
    								dframe_4_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_4_model[model_2] = model_2_temp_warm
    								dframe_4_model[model_3] = model_3_temp_warm
    								dframe_4_model[model_4] = model_4_temp_warm
    								dframe_4_model_avg = dframe_4_model.mean(axis=1)
    								blend_4_model_temp_warm = dframe_4_model_avg
    								bias_4_model_warm = bias(blend_4_model_temp_warm,station_temp_warm)
    								bias_ERAI_combo_warm_gcell_master.append(bias_4_model_warm)
    								stdev_4_model_warm = np.std(blend_4_model_temp_warm)
    								stdev_ERAI_combo_warm_gcell_master.append(stdev_4_model_warm)
    								rmse_4_model_warm = mean_squared_error(station_temp_warm,blend_4_model_temp_warm,squared=False)
    								rmse_ERAI_combo_warm_gcell_master.append(rmse_4_model_warm)
    								corr_4_model_warm,_ = pearsonr(blend_4_model_temp_warm,station_temp_warm)
    								corr_ERAI_combo_warm_gcell_master.append(corr_4_model_warm)    							

    							bias_ERAI_combo_warm_gcell_mean = mean(bias_ERAI_combo_warm_gcell_master)
    							bias_ERAI_combo_warm_master.append(bias_ERAI_combo_warm_gcell_mean)
    							stdev_ERAI_combo_warm_gcell_mean = mean(stdev_ERAI_combo_warm_gcell_master)
    							stdev_ERAI_combo_warm_master.append(stdev_ERAI_combo_warm_gcell_mean)
    							rmse_ERAI_combo_warm_gcell_mean = mean(rmse_ERAI_combo_warm_gcell_master)
    							rmse_ERAI_combo_warm_master.append(rmse_ERAI_combo_warm_gcell_mean)
    							corr_ERAI_combo_warm_gcell_mean = mean(corr_ERAI_combo_warm_gcell_master)
    							corr_ERAI_combo_warm_master.append(corr_ERAI_combo_warm_gcell_mean)



    						elif (len_i == 5):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]

    							blend_ERAI_combo_temp_warm_gcell_master = []
    							bias_ERAI_combo_warm_gcell_master = []
    							stdev_ERAI_combo_warm_gcell_master = []
    							rmse_ERAI_combo_warm_gcell_master = []
    							corr_ERAI_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    								model_4_temp_warm = dframe_warm_season_gcell[model_4].values
    								model_5_temp_warm = dframe_warm_season_gcell[model_5].values
    								dframe_5_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_5_model[model_2] = model_2_temp_warm
    								dframe_5_model[model_3] = model_3_temp_warm
    								dframe_5_model[model_4] = model_4_temp_warm
    								dframe_5_model[model_5] = model_5_temp_warm
    								dframe_5_model_avg = dframe_5_model.mean(axis=1)
    								blend_5_model_temp_warm = dframe_5_model_avg
    								bias_5_model_warm = bias(blend_5_model_temp_warm,station_temp_warm)
    								bias_ERAI_combo_warm_gcell_master.append(bias_5_model_warm)
    								stdev_5_model_warm = np.std(blend_5_model_temp_warm)
    								stdev_ERAI_combo_warm_gcell_master.append(stdev_5_model_warm)
    								rmse_5_model_warm = mean_squared_error(station_temp_warm,blend_5_model_temp_warm,squared=False)
    								rmse_ERAI_combo_warm_gcell_master.append(rmse_5_model_warm)
    								corr_5_model_warm,_ = pearsonr(blend_5_model_temp_warm,station_temp_warm)
    								corr_ERAI_combo_warm_gcell_master.append(corr_5_model_warm)    							

    							bias_ERAI_combo_warm_gcell_mean = mean(bias_ERAI_combo_warm_gcell_master)
    							bias_ERAI_combo_warm_master.append(bias_ERAI_combo_warm_gcell_mean)
    							stdev_ERAI_combo_warm_gcell_mean = mean(stdev_ERAI_combo_warm_gcell_master)
    							stdev_ERAI_combo_warm_master.append(stdev_ERAI_combo_warm_gcell_mean)
    							rmse_ERAI_combo_warm_gcell_mean = mean(rmse_ERAI_combo_warm_gcell_master)
    							rmse_ERAI_combo_warm_master.append(rmse_ERAI_combo_warm_gcell_mean)
    							corr_ERAI_combo_warm_gcell_mean = mean(corr_ERAI_combo_warm_gcell_master)
    							corr_ERAI_combo_warm_master.append(corr_ERAI_combo_warm_gcell_mean)



    						elif (len_i == 6):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]
    							model_6 = i[5]

    							blend_ERAI_combo_temp_warm_gcell_master = []
    							bias_ERAI_combo_warm_gcell_master = []
    							stdev_ERAI_combo_warm_gcell_master = []
    							rmse_ERAI_combo_warm_gcell_master = []
    							corr_ERAI_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    								model_4_temp_warm = dframe_warm_season_gcell[model_4].values
    								model_5_temp_warm = dframe_warm_season_gcell[model_5].values
    								model_6_temp_warm = dframe_warm_season_gcell[model_6].values
    								dframe_6_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_6_model[model_2] = model_2_temp_warm
    								dframe_6_model[model_3] = model_3_temp_warm
    								dframe_6_model[model_4] = model_4_temp_warm
    								dframe_6_model[model_5] = model_5_temp_warm
    								dframe_6_model[model_6] = model_6_temp_warm
    								dframe_6_model_avg = dframe_6_model.mean(axis=1)
    								blend_6_model_temp_warm = dframe_6_model_avg
    								bias_6_model_warm = bias(blend_6_model_temp_warm,station_temp_warm)
    								bias_ERAI_combo_warm_gcell_master.append(bias_6_model_warm)
    								stdev_6_model_warm = np.std(blend_6_model_temp_warm)
    								stdev_ERAI_combo_warm_gcell_master.append(stdev_6_model_warm)
    								rmse_6_model_warm = mean_squared_error(station_temp_warm,blend_6_model_temp_warm,squared=False)
    								rmse_ERAI_combo_warm_gcell_master.append(rmse_6_model_warm)
    								corr_6_model_warm,_ = pearsonr(blend_6_model_temp_warm,station_temp_warm)
    								corr_ERAI_combo_warm_gcell_master.append(corr_6_model_warm)    							

    							bias_ERAI_combo_warm_gcell_mean = mean(bias_ERAI_combo_warm_gcell_master)
    							bias_ERAI_combo_warm_master.append(bias_ERAI_combo_warm_gcell_mean)
    							stdev_ERAI_combo_warm_gcell_mean = mean(stdev_ERAI_combo_warm_gcell_master)
    							stdev_ERAI_combo_warm_master.append(stdev_ERAI_combo_warm_gcell_mean)
    							rmse_ERAI_combo_warm_gcell_mean = mean(rmse_ERAI_combo_warm_gcell_master)
    							rmse_ERAI_combo_warm_master.append(rmse_ERAI_combo_warm_gcell_mean)
    							corr_ERAI_combo_warm_gcell_mean = mean(corr_ERAI_combo_warm_gcell_master)
    							corr_ERAI_combo_warm_master.append(corr_ERAI_combo_warm_gcell_mean)



    						elif (len_i == 7):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]
    							model_6 = i[5]
    							model_7 = i[6]

    							blend_ERAI_combo_temp_warm_gcell_master = []
    							bias_ERAI_combo_warm_gcell_master = []
    							stdev_ERAI_combo_warm_gcell_master = []
    							rmse_ERAI_combo_warm_gcell_master = []
    							corr_ERAI_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    								model_4_temp_warm = dframe_warm_season_gcell[model_4].values
    								model_5_temp_warm = dframe_warm_season_gcell[model_5].values
    								model_6_temp_warm = dframe_warm_season_gcell[model_6].values
    								model_7_temp_warm = dframe_warm_season_gcell[model_7].values
    								dframe_7_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_7_model[model_2] = model_2_temp_warm
    								dframe_7_model[model_3] = model_3_temp_warm
    								dframe_7_model[model_4] = model_4_temp_warm
    								dframe_7_model[model_5] = model_5_temp_warm
    								dframe_7_model[model_6] = model_6_temp_warm
    								dframe_7_model[model_7] = model_7_temp_warm
    								dframe_7_model_avg = dframe_7_model.mean(axis=1)
    								blend_7_model_temp_warm = dframe_7_model_avg
    								bias_7_model_warm = bias(blend_7_model_temp_warm,station_temp_warm)
    								bias_ERAI_combo_warm_gcell_master.append(bias_7_model_warm)
    								stdev_7_model_warm = np.std(blend_7_model_temp_warm)
    								stdev_ERAI_combo_warm_gcell_master.append(stdev_7_model_warm)
    								rmse_7_model_warm = mean_squared_error(station_temp_warm,blend_7_model_temp_warm,squared=False)
    								rmse_ERAI_combo_warm_gcell_master.append(rmse_7_model_warm)
    								corr_7_model_warm,_ = pearsonr(blend_7_model_temp_warm,station_temp_warm)
    								corr_ERAI_combo_warm_gcell_master.append(corr_7_model_warm)    							

    							bias_ERAI_combo_warm_gcell_mean = mean(bias_ERAI_combo_warm_gcell_master)
    							bias_ERAI_combo_warm_master.append(bias_ERAI_combo_warm_gcell_mean)
    							stdev_ERAI_combo_warm_gcell_mean = mean(stdev_ERAI_combo_warm_gcell_master)
    							stdev_ERAI_combo_warm_master.append(stdev_ERAI_combo_warm_gcell_mean)
    							rmse_ERAI_combo_warm_gcell_mean = mean(rmse_ERAI_combo_warm_gcell_master)
    							rmse_ERAI_combo_warm_master.append(rmse_ERAI_combo_warm_gcell_mean)
    							corr_ERAI_combo_warm_gcell_mean = mean(corr_ERAI_combo_warm_gcell_master)
    							corr_ERAI_combo_warm_master.append(corr_ERAI_combo_warm_gcell_mean)



    						elif (len_i == 8):
    							bias_8_model_warm = bias_naive_warm_mean
    							bias_ERAI_combo_warm_master.append(bias_8_model_warm)
    							stdev_8_model_warm = stdev_naive_warm_mean
    							stdev_ERAI_combo_warm_master.append(stdev_8_model_warm)
    							rmse_8_model_warm = rmse_naive_warm_mean 
    							rmse_ERAI_combo_warm_master.append(rmse_8_model_warm)
    							corr_8_model_warm = corr_naive_warm_mean
    							corr_ERAI_combo_warm_master.append(corr_8_model_warm)

    					bias_ERAI_combo_warm_mean = mean(bias_ERAI_combo_warm_master)
    					stdev_ERAI_combo_warm_mean = mean(stdev_ERAI_combo_warm_master)
    					SDV_ERAI_combo_warm_mean = stdev_ERAI_combo_warm_mean/stdev_station_warm
    					rmse_ERAI_combo_warm_mean = mean(rmse_ERAI_combo_warm_master)
    					corr_ERAI_combo_warm_mean = mean(corr_ERAI_combo_warm_master)


## ERA5 Model ##


    					bias_ERA5_combo_warm_master = []
    					rmse_ERA5_combo_warm_master = []
    					stdev_ERA5_combo_warm_master = []
    					SDV_ERA5_combo_warm_master = []
    					corr_ERA5_combo_warm_master = []

    					for i in ERA5_array:
    						len_i = len(i)
    						if (len_i == 1):
    							blend_ERA5_combo_temp_warm_gcell_master = []
    							bias_ERA5_combo_warm_gcell_master = []
    							stdev_ERA5_combo_warm_gcell_master = []
    							rmse_ERA5_combo_warm_gcell_master = []
    							corr_ERA5_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								blend_1_model_temp_warm = dframe_warm_season_gcell[i[0]].values
    								print(blend_1_model_temp_warm)
    								print(station_temp_warm)
    								bias_1_model_warm = bias(blend_1_model_temp_warm,station_temp_warm)
    								bias_ERA5_combo_warm_gcell_master.append(bias_1_model_warm)
    								stdev_1_model_warm = np.std(blend_1_model_temp_warm)
    								stdev_ERA5_combo_warm_gcell_master.append(stdev_1_model_warm)
    								rmse_1_model_warm = mean_squared_error(station_temp_warm,blend_1_model_temp_warm,squared=False)
    								rmse_ERA5_combo_warm_gcell_master.append(rmse_1_model_warm)
    								corr_1_model_warm,_ = pearsonr(blend_1_model_temp_warm,station_temp_warm)
    								corr_ERA5_combo_warm_gcell_master.append(corr_1_model_warm)    							

    							bias_ERA5_combo_warm_gcell_mean = mean(bias_ERA5_combo_warm_gcell_master)
    							bias_ERA5_combo_warm_master.append(bias_ERA5_combo_warm_gcell_mean)
    							stdev_ERA5_combo_warm_gcell_mean = mean(stdev_ERA5_combo_warm_gcell_master)
    							stdev_ERA5_combo_warm_master.append(stdev_ERA5_combo_warm_gcell_mean)
    							rmse_ERA5_combo_warm_gcell_mean = mean(rmse_ERA5_combo_warm_gcell_master)
    							rmse_ERA5_combo_warm_master.append(rmse_ERA5_combo_warm_gcell_mean)
    							corr_ERA5_combo_warm_gcell_mean = mean(corr_ERA5_combo_warm_gcell_master)
    							corr_ERA5_combo_warm_master.append(corr_ERA5_combo_warm_gcell_mean)

    						elif (len_i == 2):
    							model_1 = i[0]
    							model_2 = i[1]

    							blend_ERA5_combo_temp_warm_gcell_master = []
    							bias_ERA5_combo_warm_gcell_master = []
    							stdev_ERA5_combo_warm_gcell_master = []
    							rmse_ERA5_combo_warm_gcell_master = []
    							corr_ERA5_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								dframe_2_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_2_model[model_2] = model_2_temp_warm
    								dframe_2_model_avg = dframe_2_model.mean(axis=1)
    								blend_2_model_temp_warm = dframe_2_model_avg
    								bias_2_model_warm = bias(blend_2_model_temp_warm,station_temp_warm)
    								bias_ERA5_combo_warm_gcell_master.append(bias_2_model_warm)
    								stdev_2_model_warm = np.std(blend_2_model_temp_warm)
    								stdev_ERA5_combo_warm_gcell_master.append(stdev_2_model_warm)
    								rmse_2_model_warm = mean_squared_error(station_temp_warm,blend_2_model_temp_warm,squared=False)
    								rmse_ERA5_combo_warm_gcell_master.append(rmse_2_model_warm)
    								corr_2_model_warm,_ = pearsonr(blend_2_model_temp_warm,station_temp_warm)
    								corr_ERA5_combo_warm_gcell_master.append(corr_2_model_warm)    							

    							bias_ERA5_combo_warm_gcell_mean = mean(bias_ERA5_combo_warm_gcell_master)
    							bias_ERA5_combo_warm_master.append(bias_ERA5_combo_warm_gcell_mean)
    							stdev_ERA5_combo_warm_gcell_mean = mean(stdev_ERA5_combo_warm_gcell_master)
    							stdev_ERA5_combo_warm_master.append(stdev_ERA5_combo_warm_gcell_mean)
    							rmse_ERA5_combo_warm_gcell_mean = mean(rmse_ERA5_combo_warm_gcell_master)
    							rmse_ERA5_combo_warm_master.append(rmse_ERA5_combo_warm_gcell_mean)
    							corr_ERA5_combo_warm_gcell_mean = mean(corr_ERA5_combo_warm_gcell_master)
    							corr_ERA5_combo_warm_master.append(corr_ERA5_combo_warm_gcell_mean)


    						elif (len_i == 3):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]

    							blend_ERA5_combo_temp_warm_gcell_master = []
    							bias_ERA5_combo_warm_gcell_master = []
    							stdev_ERA5_combo_warm_gcell_master = []
    							rmse_ERA5_combo_warm_gcell_master = []
    							corr_ERA5_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    								dframe_3_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_3_model[model_2] = model_2_temp_warm
    								dframe_3_model[model_3] = model_3_temp_warm
    								dframe_3_model_avg = dframe_3_model.mean(axis=1)
    								blend_3_model_temp_warm = dframe_3_model_avg
    								bias_3_model_warm = bias(blend_3_model_temp_warm,station_temp_warm)
    								bias_ERA5_combo_warm_gcell_master.append(bias_3_model_warm)
    								stdev_3_model_warm = np.std(blend_3_model_temp_warm)
    								stdev_ERA5_combo_warm_gcell_master.append(stdev_3_model_warm)
    								rmse_3_model_warm = mean_squared_error(station_temp_warm,blend_3_model_temp_warm,squared=False)
    								rmse_ERA5_combo_warm_gcell_master.append(rmse_3_model_warm)
    								corr_3_model_warm,_ = pearsonr(blend_3_model_temp_warm,station_temp_warm)
    								corr_ERA5_combo_warm_gcell_master.append(corr_3_model_warm)    							

    							bias_ERA5_combo_warm_gcell_mean = mean(bias_ERA5_combo_warm_gcell_master)
    							bias_ERA5_combo_warm_master.append(bias_ERA5_combo_warm_gcell_mean)
    							stdev_ERA5_combo_warm_gcell_mean = mean(stdev_ERA5_combo_warm_gcell_master)
    							stdev_ERA5_combo_warm_master.append(stdev_ERA5_combo_warm_gcell_mean)
    							rmse_ERA5_combo_warm_gcell_mean = mean(rmse_ERA5_combo_warm_gcell_master)
    							rmse_ERA5_combo_warm_master.append(rmse_ERA5_combo_warm_gcell_mean)
    							corr_ERA5_combo_warm_gcell_mean = mean(corr_ERA5_combo_warm_gcell_master)
    							corr_ERA5_combo_warm_master.append(corr_ERA5_combo_warm_gcell_mean)


    						elif (len_i == 4):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]

    							blend_ERA5_combo_temp_warm_gcell_master = []
    							bias_ERA5_combo_warm_gcell_master = []
    							stdev_ERA5_combo_warm_gcell_master = []
    							rmse_ERA5_combo_warm_gcell_master = []
    							corr_ERA5_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    								model_4_temp_warm = dframe_warm_season_gcell[model_4].values
    								dframe_4_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_4_model[model_2] = model_2_temp_warm
    								dframe_4_model[model_3] = model_3_temp_warm
    								dframe_4_model[model_4] = model_4_temp_warm
    								dframe_4_model_avg = dframe_4_model.mean(axis=1)
    								blend_4_model_temp_warm = dframe_4_model_avg
    								bias_4_model_warm = bias(blend_4_model_temp_warm,station_temp_warm)
    								bias_ERA5_combo_warm_gcell_master.append(bias_4_model_warm)
    								stdev_4_model_warm = np.std(blend_4_model_temp_warm)
    								stdev_ERA5_combo_warm_gcell_master.append(stdev_4_model_warm)
    								rmse_4_model_warm = mean_squared_error(station_temp_warm,blend_4_model_temp_warm,squared=False)
    								rmse_ERA5_combo_warm_gcell_master.append(rmse_4_model_warm)
    								corr_4_model_warm,_ = pearsonr(blend_4_model_temp_warm,station_temp_warm)
    								corr_ERA5_combo_warm_gcell_master.append(corr_4_model_warm)    							

    							bias_ERA5_combo_warm_gcell_mean = mean(bias_ERA5_combo_warm_gcell_master)
    							bias_ERA5_combo_warm_master.append(bias_ERA5_combo_warm_gcell_mean)
    							stdev_ERA5_combo_warm_gcell_mean = mean(stdev_ERA5_combo_warm_gcell_master)
    							stdev_ERA5_combo_warm_master.append(stdev_ERA5_combo_warm_gcell_mean)
    							rmse_ERA5_combo_warm_gcell_mean = mean(rmse_ERA5_combo_warm_gcell_master)
    							rmse_ERA5_combo_warm_master.append(rmse_ERA5_combo_warm_gcell_mean)
    							corr_ERA5_combo_warm_gcell_mean = mean(corr_ERA5_combo_warm_gcell_master)
    							corr_ERA5_combo_warm_master.append(corr_ERA5_combo_warm_gcell_mean)



    						elif (len_i == 5):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]

    							blend_ERA5_combo_temp_warm_gcell_master = []
    							bias_ERA5_combo_warm_gcell_master = []
    							stdev_ERA5_combo_warm_gcell_master = []
    							rmse_ERA5_combo_warm_gcell_master = []
    							corr_ERA5_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    								model_4_temp_warm = dframe_warm_season_gcell[model_4].values
    								model_5_temp_warm = dframe_warm_season_gcell[model_5].values
    								dframe_5_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_5_model[model_2] = model_2_temp_warm
    								dframe_5_model[model_3] = model_3_temp_warm
    								dframe_5_model[model_4] = model_4_temp_warm
    								dframe_5_model[model_5] = model_5_temp_warm
    								dframe_5_model_avg = dframe_5_model.mean(axis=1)
    								blend_5_model_temp_warm = dframe_5_model_avg
    								bias_5_model_warm = bias(blend_5_model_temp_warm,station_temp_warm)
    								bias_ERA5_combo_warm_gcell_master.append(bias_5_model_warm)
    								stdev_5_model_warm = np.std(blend_5_model_temp_warm)
    								stdev_ERA5_combo_warm_gcell_master.append(stdev_5_model_warm)
    								rmse_5_model_warm = mean_squared_error(station_temp_warm,blend_5_model_temp_warm,squared=False)
    								rmse_ERA5_combo_warm_gcell_master.append(rmse_5_model_warm)
    								corr_5_model_warm,_ = pearsonr(blend_5_model_temp_warm,station_temp_warm)
    								corr_ERA5_combo_warm_gcell_master.append(corr_5_model_warm)    							

    							bias_ERA5_combo_warm_gcell_mean = mean(bias_ERA5_combo_warm_gcell_master)
    							bias_ERA5_combo_warm_master.append(bias_ERA5_combo_warm_gcell_mean)
    							stdev_ERA5_combo_warm_gcell_mean = mean(stdev_ERA5_combo_warm_gcell_master)
    							stdev_ERA5_combo_warm_master.append(stdev_ERA5_combo_warm_gcell_mean)
    							rmse_ERA5_combo_warm_gcell_mean = mean(rmse_ERA5_combo_warm_gcell_master)
    							rmse_ERA5_combo_warm_master.append(rmse_ERA5_combo_warm_gcell_mean)
    							corr_ERA5_combo_warm_gcell_mean = mean(corr_ERA5_combo_warm_gcell_master)
    							corr_ERA5_combo_warm_master.append(corr_ERA5_combo_warm_gcell_mean)



    						elif (len_i == 6):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]
    							model_6 = i[5]

    							blend_ERA5_combo_temp_warm_gcell_master = []
    							bias_ERA5_combo_warm_gcell_master = []
    							stdev_ERA5_combo_warm_gcell_master = []
    							rmse_ERA5_combo_warm_gcell_master = []
    							corr_ERA5_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    								model_4_temp_warm = dframe_warm_season_gcell[model_4].values
    								model_5_temp_warm = dframe_warm_season_gcell[model_5].values
    								model_6_temp_warm = dframe_warm_season_gcell[model_6].values
    								dframe_6_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_6_model[model_2] = model_2_temp_warm
    								dframe_6_model[model_3] = model_3_temp_warm
    								dframe_6_model[model_4] = model_4_temp_warm
    								dframe_6_model[model_5] = model_5_temp_warm
    								dframe_6_model[model_6] = model_6_temp_warm
    								dframe_6_model_avg = dframe_6_model.mean(axis=1)
    								blend_6_model_temp_warm = dframe_6_model_avg
    								bias_6_model_warm = bias(blend_6_model_temp_warm,station_temp_warm)
    								bias_ERA5_combo_warm_gcell_master.append(bias_6_model_warm)
    								stdev_6_model_warm = np.std(blend_6_model_temp_warm)
    								stdev_ERA5_combo_warm_gcell_master.append(stdev_6_model_warm)
    								rmse_6_model_warm = mean_squared_error(station_temp_warm,blend_6_model_temp_warm,squared=False)
    								rmse_ERA5_combo_warm_gcell_master.append(rmse_6_model_warm)
    								corr_6_model_warm,_ = pearsonr(blend_6_model_temp_warm,station_temp_warm)
    								corr_ERA5_combo_warm_gcell_master.append(corr_6_model_warm)    							

    							bias_ERA5_combo_warm_gcell_mean = mean(bias_ERA5_combo_warm_gcell_master)
    							bias_ERA5_combo_warm_master.append(bias_ERA5_combo_warm_gcell_mean)
    							stdev_ERA5_combo_warm_gcell_mean = mean(stdev_ERA5_combo_warm_gcell_master)
    							stdev_ERA5_combo_warm_master.append(stdev_ERA5_combo_warm_gcell_mean)
    							rmse_ERA5_combo_warm_gcell_mean = mean(rmse_ERA5_combo_warm_gcell_master)
    							rmse_ERA5_combo_warm_master.append(rmse_ERA5_combo_warm_gcell_mean)
    							corr_ERA5_combo_warm_gcell_mean = mean(corr_ERA5_combo_warm_gcell_master)
    							corr_ERA5_combo_warm_master.append(corr_ERA5_combo_warm_gcell_mean)



    						elif (len_i == 7):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]
    							model_6 = i[5]
    							model_7 = i[6]

    							blend_ERA5_combo_temp_warm_gcell_master = []
    							bias_ERA5_combo_warm_gcell_master = []
    							stdev_ERA5_combo_warm_gcell_master = []
    							rmse_ERA5_combo_warm_gcell_master = []
    							corr_ERA5_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    								model_4_temp_warm = dframe_warm_season_gcell[model_4].values
    								model_5_temp_warm = dframe_warm_season_gcell[model_5].values
    								model_6_temp_warm = dframe_warm_season_gcell[model_6].values
    								model_7_temp_warm = dframe_warm_season_gcell[model_7].values
    								dframe_7_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_7_model[model_2] = model_2_temp_warm
    								dframe_7_model[model_3] = model_3_temp_warm
    								dframe_7_model[model_4] = model_4_temp_warm
    								dframe_7_model[model_5] = model_5_temp_warm
    								dframe_7_model[model_6] = model_6_temp_warm
    								dframe_7_model[model_7] = model_7_temp_warm
    								dframe_7_model_avg = dframe_7_model.mean(axis=1)
    								blend_7_model_temp_warm = dframe_7_model_avg
    								bias_7_model_warm = bias(blend_7_model_temp_warm,station_temp_warm)
    								bias_ERA5_combo_warm_gcell_master.append(bias_7_model_warm)
    								stdev_7_model_warm = np.std(blend_7_model_temp_warm)
    								stdev_ERA5_combo_warm_gcell_master.append(stdev_7_model_warm)
    								rmse_7_model_warm = mean_squared_error(station_temp_warm,blend_7_model_temp_warm,squared=False)
    								rmse_ERA5_combo_warm_gcell_master.append(rmse_7_model_warm)
    								corr_7_model_warm,_ = pearsonr(blend_7_model_temp_warm,station_temp_warm)
    								corr_ERA5_combo_warm_gcell_master.append(corr_7_model_warm)    							

    							bias_ERA5_combo_warm_gcell_mean = mean(bias_ERA5_combo_warm_gcell_master)
    							bias_ERA5_combo_warm_master.append(bias_ERA5_combo_warm_gcell_mean)
    							stdev_ERA5_combo_warm_gcell_mean = mean(stdev_ERA5_combo_warm_gcell_master)
    							stdev_ERA5_combo_warm_master.append(stdev_ERA5_combo_warm_gcell_mean)
    							rmse_ERA5_combo_warm_gcell_mean = mean(rmse_ERA5_combo_warm_gcell_master)
    							rmse_ERA5_combo_warm_master.append(rmse_ERA5_combo_warm_gcell_mean)
    							corr_ERA5_combo_warm_gcell_mean = mean(corr_ERA5_combo_warm_gcell_master)
    							corr_ERA5_combo_warm_master.append(corr_ERA5_combo_warm_gcell_mean)



    						elif (len_i == 8):
    							bias_8_model_warm = bias_naive_warm_mean
    							bias_ERA5_combo_warm_master.append(bias_8_model_warm)
    							stdev_8_model_warm = stdev_naive_warm_mean
    							stdev_ERA5_combo_warm_master.append(stdev_8_model_warm)
    							rmse_8_model_warm = rmse_naive_warm_mean 
    							rmse_ERA5_combo_warm_master.append(rmse_8_model_warm)
    							corr_8_model_warm = corr_naive_warm_mean
    							corr_ERA5_combo_warm_master.append(corr_8_model_warm)

    					bias_ERA5_combo_warm_mean = mean(bias_ERA5_combo_warm_master)
    					stdev_ERA5_combo_warm_mean = mean(stdev_ERA5_combo_warm_master)
    					SDV_ERA5_combo_warm_mean = stdev_ERA5_combo_warm_mean/stdev_station_warm
    					rmse_ERA5_combo_warm_mean = mean(rmse_ERA5_combo_warm_master)
    					corr_ERA5_combo_warm_mean = mean(corr_ERA5_combo_warm_master)


## ERA5-Land Model ##

    					bias_ERA5_Land_combo_warm_master = []
    					rmse_ERA5_Land_combo_warm_master = []
    					stdev_ERA5_Land_combo_warm_master = []
    					SDV_ERA5_Land_combo_warm_master = []
    					corr_ERA5_Land_combo_warm_master = []

    					for i in ERA5_Land_array:
    						len_i = len(i)
    						if (len_i == 1):
    							blend_ERA5_Land_combo_temp_warm_gcell_master = []
    							bias_ERA5_Land_combo_warm_gcell_master = []
    							stdev_ERA5_Land_combo_warm_gcell_master = []
    							rmse_ERA5_Land_combo_warm_gcell_master = []
    							corr_ERA5_Land_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								blend_1_model_temp_warm = dframe_warm_season_gcell[i[0]].values
    								print(blend_1_model_temp_warm)
    								print(station_temp_warm)
    								bias_1_model_warm = bias(blend_1_model_temp_warm,station_temp_warm)
    								bias_ERA5_Land_combo_warm_gcell_master.append(bias_1_model_warm)
    								stdev_1_model_warm = np.std(blend_1_model_temp_warm)
    								stdev_ERA5_Land_combo_warm_gcell_master.append(stdev_1_model_warm)
    								rmse_1_model_warm = mean_squared_error(station_temp_warm,blend_1_model_temp_warm,squared=False)
    								rmse_ERA5_Land_combo_warm_gcell_master.append(rmse_1_model_warm)
    								corr_1_model_warm,_ = pearsonr(blend_1_model_temp_warm,station_temp_warm)
    								corr_ERA5_Land_combo_warm_gcell_master.append(corr_1_model_warm)    							

    							bias_ERA5_Land_combo_warm_gcell_mean = mean(bias_ERA5_Land_combo_warm_gcell_master)
    							bias_ERA5_Land_combo_warm_master.append(bias_ERA5_Land_combo_warm_gcell_mean)
    							stdev_ERA5_Land_combo_warm_gcell_mean = mean(stdev_ERA5_Land_combo_warm_gcell_master)
    							stdev_ERA5_Land_combo_warm_master.append(stdev_ERA5_Land_combo_warm_gcell_mean)
    							rmse_ERA5_Land_combo_warm_gcell_mean = mean(rmse_ERA5_Land_combo_warm_gcell_master)
    							rmse_ERA5_Land_combo_warm_master.append(rmse_ERA5_Land_combo_warm_gcell_mean)
    							corr_ERA5_Land_combo_warm_gcell_mean = mean(corr_ERA5_Land_combo_warm_gcell_master)
    							corr_ERA5_Land_combo_warm_master.append(corr_ERA5_Land_combo_warm_gcell_mean)

    						elif (len_i == 2):
    							model_1 = i[0]
    							model_2 = i[1]

    							blend_ERA5_Land_combo_temp_warm_gcell_master = []
    							bias_ERA5_Land_combo_warm_gcell_master = []
    							stdev_ERA5_Land_combo_warm_gcell_master = []
    							rmse_ERA5_Land_combo_warm_gcell_master = []
    							corr_ERA5_Land_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								dframe_2_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_2_model[model_2] = model_2_temp_warm
    								dframe_2_model_avg = dframe_2_model.mean(axis=1)
    								blend_2_model_temp_warm = dframe_2_model_avg
    								bias_2_model_warm = bias(blend_2_model_temp_warm,station_temp_warm)
    								bias_ERA5_Land_combo_warm_gcell_master.append(bias_2_model_warm)
    								stdev_2_model_warm = np.std(blend_2_model_temp_warm)
    								stdev_ERA5_Land_combo_warm_gcell_master.append(stdev_2_model_warm)
    								rmse_2_model_warm = mean_squared_error(station_temp_warm,blend_2_model_temp_warm,squared=False)
    								rmse_ERA5_Land_combo_warm_gcell_master.append(rmse_2_model_warm)
    								corr_2_model_warm,_ = pearsonr(blend_2_model_temp_warm,station_temp_warm)
    								corr_ERA5_Land_combo_warm_gcell_master.append(corr_2_model_warm)    							

    							bias_ERA5_Land_combo_warm_gcell_mean = mean(bias_ERA5_Land_combo_warm_gcell_master)
    							bias_ERA5_Land_combo_warm_master.append(bias_ERA5_Land_combo_warm_gcell_mean)
    							stdev_ERA5_Land_combo_warm_gcell_mean = mean(stdev_ERA5_Land_combo_warm_gcell_master)
    							stdev_ERA5_Land_combo_warm_master.append(stdev_ERA5_Land_combo_warm_gcell_mean)
    							rmse_ERA5_Land_combo_warm_gcell_mean = mean(rmse_ERA5_Land_combo_warm_gcell_master)
    							rmse_ERA5_Land_combo_warm_master.append(rmse_ERA5_Land_combo_warm_gcell_mean)
    							corr_ERA5_Land_combo_warm_gcell_mean = mean(corr_ERA5_Land_combo_warm_gcell_master)
    							corr_ERA5_Land_combo_warm_master.append(corr_ERA5_Land_combo_warm_gcell_mean)


    						elif (len_i == 3):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]

    							blend_ERA5_Land_combo_temp_warm_gcell_master = []
    							bias_ERA5_Land_combo_warm_gcell_master = []
    							stdev_ERA5_Land_combo_warm_gcell_master = []
    							rmse_ERA5_Land_combo_warm_gcell_master = []
    							corr_ERA5_Land_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    								dframe_3_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_3_model[model_2] = model_2_temp_warm
    								dframe_3_model[model_3] = model_3_temp_warm
    								dframe_3_model_avg = dframe_3_model.mean(axis=1)
    								blend_3_model_temp_warm = dframe_3_model_avg
    								bias_3_model_warm = bias(blend_3_model_temp_warm,station_temp_warm)
    								bias_ERA5_Land_combo_warm_gcell_master.append(bias_3_model_warm)
    								stdev_3_model_warm = np.std(blend_3_model_temp_warm)
    								stdev_ERA5_Land_combo_warm_gcell_master.append(stdev_3_model_warm)
    								rmse_3_model_warm = mean_squared_error(station_temp_warm,blend_3_model_temp_warm,squared=False)
    								rmse_ERA5_Land_combo_warm_gcell_master.append(rmse_3_model_warm)
    								corr_3_model_warm,_ = pearsonr(blend_3_model_temp_warm,station_temp_warm)
    								corr_ERA5_Land_combo_warm_gcell_master.append(corr_3_model_warm)    							

    							bias_ERA5_Land_combo_warm_gcell_mean = mean(bias_ERA5_Land_combo_warm_gcell_master)
    							bias_ERA5_Land_combo_warm_master.append(bias_ERA5_Land_combo_warm_gcell_mean)
    							stdev_ERA5_Land_combo_warm_gcell_mean = mean(stdev_ERA5_Land_combo_warm_gcell_master)
    							stdev_ERA5_Land_combo_warm_master.append(stdev_ERA5_Land_combo_warm_gcell_mean)
    							rmse_ERA5_Land_combo_warm_gcell_mean = mean(rmse_ERA5_Land_combo_warm_gcell_master)
    							rmse_ERA5_Land_combo_warm_master.append(rmse_ERA5_Land_combo_warm_gcell_mean)
    							corr_ERA5_Land_combo_warm_gcell_mean = mean(corr_ERA5_Land_combo_warm_gcell_master)
    							corr_ERA5_Land_combo_warm_master.append(corr_ERA5_Land_combo_warm_gcell_mean)


    						elif (len_i == 4):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]

    							blend_ERA5_Land_combo_temp_warm_gcell_master = []
    							bias_ERA5_Land_combo_warm_gcell_master = []
    							stdev_ERA5_Land_combo_warm_gcell_master = []
    							rmse_ERA5_Land_combo_warm_gcell_master = []
    							corr_ERA5_Land_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    								model_4_temp_warm = dframe_warm_season_gcell[model_4].values
    								dframe_4_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_4_model[model_2] = model_2_temp_warm
    								dframe_4_model[model_3] = model_3_temp_warm
    								dframe_4_model[model_4] = model_4_temp_warm
    								dframe_4_model_avg = dframe_4_model.mean(axis=1)
    								blend_4_model_temp_warm = dframe_4_model_avg
    								bias_4_model_warm = bias(blend_4_model_temp_warm,station_temp_warm)
    								bias_ERA5_Land_combo_warm_gcell_master.append(bias_4_model_warm)
    								stdev_4_model_warm = np.std(blend_4_model_temp_warm)
    								stdev_ERA5_Land_combo_warm_gcell_master.append(stdev_4_model_warm)
    								rmse_4_model_warm = mean_squared_error(station_temp_warm,blend_4_model_temp_warm,squared=False)
    								rmse_ERA5_Land_combo_warm_gcell_master.append(rmse_4_model_warm)
    								corr_4_model_warm,_ = pearsonr(blend_4_model_temp_warm,station_temp_warm)
    								corr_ERA5_Land_combo_warm_gcell_master.append(corr_4_model_warm)    							

    							bias_ERA5_Land_combo_warm_gcell_mean = mean(bias_ERA5_Land_combo_warm_gcell_master)
    							bias_ERA5_Land_combo_warm_master.append(bias_ERA5_Land_combo_warm_gcell_mean)
    							stdev_ERA5_Land_combo_warm_gcell_mean = mean(stdev_ERA5_Land_combo_warm_gcell_master)
    							stdev_ERA5_Land_combo_warm_master.append(stdev_ERA5_Land_combo_warm_gcell_mean)
    							rmse_ERA5_Land_combo_warm_gcell_mean = mean(rmse_ERA5_Land_combo_warm_gcell_master)
    							rmse_ERA5_Land_combo_warm_master.append(rmse_ERA5_Land_combo_warm_gcell_mean)
    							corr_ERA5_Land_combo_warm_gcell_mean = mean(corr_ERA5_Land_combo_warm_gcell_master)
    							corr_ERA5_Land_combo_warm_master.append(corr_ERA5_Land_combo_warm_gcell_mean)



    						elif (len_i == 5):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]

    							blend_ERA5_Land_combo_temp_warm_gcell_master = []
    							bias_ERA5_Land_combo_warm_gcell_master = []
    							stdev_ERA5_Land_combo_warm_gcell_master = []
    							rmse_ERA5_Land_combo_warm_gcell_master = []
    							corr_ERA5_Land_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    								model_4_temp_warm = dframe_warm_season_gcell[model_4].values
    								model_5_temp_warm = dframe_warm_season_gcell[model_5].values
    								dframe_5_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_5_model[model_2] = model_2_temp_warm
    								dframe_5_model[model_3] = model_3_temp_warm
    								dframe_5_model[model_4] = model_4_temp_warm
    								dframe_5_model[model_5] = model_5_temp_warm
    								dframe_5_model_avg = dframe_5_model.mean(axis=1)
    								blend_5_model_temp_warm = dframe_5_model_avg
    								bias_5_model_warm = bias(blend_5_model_temp_warm,station_temp_warm)
    								bias_ERA5_Land_combo_warm_gcell_master.append(bias_5_model_warm)
    								stdev_5_model_warm = np.std(blend_5_model_temp_warm)
    								stdev_ERA5_Land_combo_warm_gcell_master.append(stdev_5_model_warm)
    								rmse_5_model_warm = mean_squared_error(station_temp_warm,blend_5_model_temp_warm,squared=False)
    								rmse_ERA5_Land_combo_warm_gcell_master.append(rmse_5_model_warm)
    								corr_5_model_warm,_ = pearsonr(blend_5_model_temp_warm,station_temp_warm)
    								corr_ERA5_Land_combo_warm_gcell_master.append(corr_5_model_warm)    							

    							bias_ERA5_Land_combo_warm_gcell_mean = mean(bias_ERA5_Land_combo_warm_gcell_master)
    							bias_ERA5_Land_combo_warm_master.append(bias_ERA5_Land_combo_warm_gcell_mean)
    							stdev_ERA5_Land_combo_warm_gcell_mean = mean(stdev_ERA5_Land_combo_warm_gcell_master)
    							stdev_ERA5_Land_combo_warm_master.append(stdev_ERA5_Land_combo_warm_gcell_mean)
    							rmse_ERA5_Land_combo_warm_gcell_mean = mean(rmse_ERA5_Land_combo_warm_gcell_master)
    							rmse_ERA5_Land_combo_warm_master.append(rmse_ERA5_Land_combo_warm_gcell_mean)
    							corr_ERA5_Land_combo_warm_gcell_mean = mean(corr_ERA5_Land_combo_warm_gcell_master)
    							corr_ERA5_Land_combo_warm_master.append(corr_ERA5_Land_combo_warm_gcell_mean)



    						elif (len_i == 6):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]
    							model_6 = i[5]

    							blend_ERA5_Land_combo_temp_warm_gcell_master = []
    							bias_ERA5_Land_combo_warm_gcell_master = []
    							stdev_ERA5_Land_combo_warm_gcell_master = []
    							rmse_ERA5_Land_combo_warm_gcell_master = []
    							corr_ERA5_Land_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    								model_4_temp_warm = dframe_warm_season_gcell[model_4].values
    								model_5_temp_warm = dframe_warm_season_gcell[model_5].values
    								model_6_temp_warm = dframe_warm_season_gcell[model_6].values
    								dframe_6_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_6_model[model_2] = model_2_temp_warm
    								dframe_6_model[model_3] = model_3_temp_warm
    								dframe_6_model[model_4] = model_4_temp_warm
    								dframe_6_model[model_5] = model_5_temp_warm
    								dframe_6_model[model_6] = model_6_temp_warm
    								dframe_6_model_avg = dframe_6_model.mean(axis=1)
    								blend_6_model_temp_warm = dframe_6_model_avg
    								bias_6_model_warm = bias(blend_6_model_temp_warm,station_temp_warm)
    								bias_ERA5_Land_combo_warm_gcell_master.append(bias_6_model_warm)
    								stdev_6_model_warm = np.std(blend_6_model_temp_warm)
    								stdev_ERA5_Land_combo_warm_gcell_master.append(stdev_6_model_warm)
    								rmse_6_model_warm = mean_squared_error(station_temp_warm,blend_6_model_temp_warm,squared=False)
    								rmse_ERA5_Land_combo_warm_gcell_master.append(rmse_6_model_warm)
    								corr_6_model_warm,_ = pearsonr(blend_6_model_temp_warm,station_temp_warm)
    								corr_ERA5_Land_combo_warm_gcell_master.append(corr_6_model_warm)    							

    							bias_ERA5_Land_combo_warm_gcell_mean = mean(bias_ERA5_Land_combo_warm_gcell_master)
    							bias_ERA5_Land_combo_warm_master.append(bias_ERA5_Land_combo_warm_gcell_mean)
    							stdev_ERA5_Land_combo_warm_gcell_mean = mean(stdev_ERA5_Land_combo_warm_gcell_master)
    							stdev_ERA5_Land_combo_warm_master.append(stdev_ERA5_Land_combo_warm_gcell_mean)
    							rmse_ERA5_Land_combo_warm_gcell_mean = mean(rmse_ERA5_Land_combo_warm_gcell_master)
    							rmse_ERA5_Land_combo_warm_master.append(rmse_ERA5_Land_combo_warm_gcell_mean)
    							corr_ERA5_Land_combo_warm_gcell_mean = mean(corr_ERA5_Land_combo_warm_gcell_master)
    							corr_ERA5_Land_combo_warm_master.append(corr_ERA5_Land_combo_warm_gcell_mean)



    						elif (len_i == 7):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]
    							model_6 = i[5]
    							model_7 = i[6]

    							blend_ERA5_Land_combo_temp_warm_gcell_master = []
    							bias_ERA5_Land_combo_warm_gcell_master = []
    							stdev_ERA5_Land_combo_warm_gcell_master = []
    							rmse_ERA5_Land_combo_warm_gcell_master = []
    							corr_ERA5_Land_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    								model_4_temp_warm = dframe_warm_season_gcell[model_4].values
    								model_5_temp_warm = dframe_warm_season_gcell[model_5].values
    								model_6_temp_warm = dframe_warm_season_gcell[model_6].values
    								model_7_temp_warm = dframe_warm_season_gcell[model_7].values
    								dframe_7_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_7_model[model_2] = model_2_temp_warm
    								dframe_7_model[model_3] = model_3_temp_warm
    								dframe_7_model[model_4] = model_4_temp_warm
    								dframe_7_model[model_5] = model_5_temp_warm
    								dframe_7_model[model_6] = model_6_temp_warm
    								dframe_7_model[model_7] = model_7_temp_warm
    								dframe_7_model_avg = dframe_7_model.mean(axis=1)
    								blend_7_model_temp_warm = dframe_7_model_avg
    								bias_7_model_warm = bias(blend_7_model_temp_warm,station_temp_warm)
    								bias_ERA5_Land_combo_warm_gcell_master.append(bias_7_model_warm)
    								stdev_7_model_warm = np.std(blend_7_model_temp_warm)
    								stdev_ERA5_Land_combo_warm_gcell_master.append(stdev_7_model_warm)
    								rmse_7_model_warm = mean_squared_error(station_temp_warm,blend_7_model_temp_warm,squared=False)
    								rmse_ERA5_Land_combo_warm_gcell_master.append(rmse_7_model_warm)
    								corr_7_model_warm,_ = pearsonr(blend_7_model_temp_warm,station_temp_warm)
    								corr_ERA5_Land_combo_warm_gcell_master.append(corr_7_model_warm)    							

    							bias_ERA5_Land_combo_warm_gcell_mean = mean(bias_ERA5_Land_combo_warm_gcell_master)
    							bias_ERA5_Land_combo_warm_master.append(bias_ERA5_Land_combo_warm_gcell_mean)
    							stdev_ERA5_Land_combo_warm_gcell_mean = mean(stdev_ERA5_Land_combo_warm_gcell_master)
    							stdev_ERA5_Land_combo_warm_master.append(stdev_ERA5_Land_combo_warm_gcell_mean)
    							rmse_ERA5_Land_combo_warm_gcell_mean = mean(rmse_ERA5_Land_combo_warm_gcell_master)
    							rmse_ERA5_Land_combo_warm_master.append(rmse_ERA5_Land_combo_warm_gcell_mean)
    							corr_ERA5_Land_combo_warm_gcell_mean = mean(corr_ERA5_Land_combo_warm_gcell_master)
    							corr_ERA5_Land_combo_warm_master.append(corr_ERA5_Land_combo_warm_gcell_mean)



    						elif (len_i == 8):
    							bias_8_model_warm = bias_naive_warm_mean
    							bias_ERA5_Land_combo_warm_master.append(bias_8_model_warm)
    							stdev_8_model_warm = stdev_naive_warm_mean
    							stdev_ERA5_Land_combo_warm_master.append(stdev_8_model_warm)
    							rmse_8_model_warm = rmse_naive_warm_mean 
    							rmse_ERA5_Land_combo_warm_master.append(rmse_8_model_warm)
    							corr_8_model_warm = corr_naive_warm_mean
    							corr_ERA5_Land_combo_warm_master.append(corr_8_model_warm)

    					bias_ERA5_Land_combo_warm_mean = mean(bias_ERA5_Land_combo_warm_master)
    					stdev_ERA5_Land_combo_warm_mean = mean(stdev_ERA5_Land_combo_warm_master)
    					SDV_ERA5_Land_combo_warm_mean = stdev_ERA5_Land_combo_warm_mean/stdev_station_warm
    					rmse_ERA5_Land_combo_warm_mean = mean(rmse_ERA5_Land_combo_warm_master)
    					corr_ERA5_Land_combo_warm_mean = mean(corr_ERA5_Land_combo_warm_master)


## JRA-55 Model ##

    					bias_JRA_combo_warm_master = []
    					rmse_JRA_combo_warm_master = []
    					stdev_JRA_combo_warm_master = []
    					SDV_JRA_combo_warm_master = []
    					corr_JRA_combo_warm_master = []

    					for i in JRA_array:
    						len_i = len(i)
    						if (len_i == 1):
    							blend_JRA_combo_temp_warm_gcell_master = []
    							bias_JRA_combo_warm_gcell_master = []
    							stdev_JRA_combo_warm_gcell_master = []
    							rmse_JRA_combo_warm_gcell_master = []
    							corr_JRA_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								blend_1_model_temp_warm = dframe_warm_season_gcell[i[0]].values
    								print(blend_1_model_temp_warm)
    								print(station_temp_warm)
    								bias_1_model_warm = bias(blend_1_model_temp_warm,station_temp_warm)
    								bias_JRA_combo_warm_gcell_master.append(bias_1_model_warm)
    								stdev_1_model_warm = np.std(blend_1_model_temp_warm)
    								stdev_JRA_combo_warm_gcell_master.append(stdev_1_model_warm)
    								rmse_1_model_warm = mean_squared_error(station_temp_warm,blend_1_model_temp_warm,squared=False)
    								rmse_JRA_combo_warm_gcell_master.append(rmse_1_model_warm)
    								corr_1_model_warm,_ = pearsonr(blend_1_model_temp_warm,station_temp_warm)
    								corr_JRA_combo_warm_gcell_master.append(corr_1_model_warm)    							

    							bias_JRA_combo_warm_gcell_mean = mean(bias_JRA_combo_warm_gcell_master)
    							bias_JRA_combo_warm_master.append(bias_JRA_combo_warm_gcell_mean)
    							stdev_JRA_combo_warm_gcell_mean = mean(stdev_JRA_combo_warm_gcell_master)
    							stdev_JRA_combo_warm_master.append(stdev_JRA_combo_warm_gcell_mean)
    							rmse_JRA_combo_warm_gcell_mean = mean(rmse_JRA_combo_warm_gcell_master)
    							rmse_JRA_combo_warm_master.append(rmse_JRA_combo_warm_gcell_mean)
    							corr_JRA_combo_warm_gcell_mean = mean(corr_JRA_combo_warm_gcell_master)
    							corr_JRA_combo_warm_master.append(corr_JRA_combo_warm_gcell_mean)

    						elif (len_i == 2):
    							model_1 = i[0]
    							model_2 = i[1]

    							blend_JRA_combo_temp_warm_gcell_master = []
    							bias_JRA_combo_warm_gcell_master = []
    							stdev_JRA_combo_warm_gcell_master = []
    							rmse_JRA_combo_warm_gcell_master = []
    							corr_JRA_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								dframe_2_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_2_model[model_2] = model_2_temp_warm
    								dframe_2_model_avg = dframe_2_model.mean(axis=1)
    								blend_2_model_temp_warm = dframe_2_model_avg
    								bias_2_model_warm = bias(blend_2_model_temp_warm,station_temp_warm)
    								bias_JRA_combo_warm_gcell_master.append(bias_2_model_warm)
    								stdev_2_model_warm = np.std(blend_2_model_temp_warm)
    								stdev_JRA_combo_warm_gcell_master.append(stdev_2_model_warm)
    								rmse_2_model_warm = mean_squared_error(station_temp_warm,blend_2_model_temp_warm,squared=False)
    								rmse_JRA_combo_warm_gcell_master.append(rmse_2_model_warm)
    								corr_2_model_warm,_ = pearsonr(blend_2_model_temp_warm,station_temp_warm)
    								corr_JRA_combo_warm_gcell_master.append(corr_2_model_warm)    							

    							bias_JRA_combo_warm_gcell_mean = mean(bias_JRA_combo_warm_gcell_master)
    							bias_JRA_combo_warm_master.append(bias_JRA_combo_warm_gcell_mean)
    							stdev_JRA_combo_warm_gcell_mean = mean(stdev_JRA_combo_warm_gcell_master)
    							stdev_JRA_combo_warm_master.append(stdev_JRA_combo_warm_gcell_mean)
    							rmse_JRA_combo_warm_gcell_mean = mean(rmse_JRA_combo_warm_gcell_master)
    							rmse_JRA_combo_warm_master.append(rmse_JRA_combo_warm_gcell_mean)
    							corr_JRA_combo_warm_gcell_mean = mean(corr_JRA_combo_warm_gcell_master)
    							corr_JRA_combo_warm_master.append(corr_JRA_combo_warm_gcell_mean)


    						elif (len_i == 3):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]

    							blend_JRA_combo_temp_warm_gcell_master = []
    							bias_JRA_combo_warm_gcell_master = []
    							stdev_JRA_combo_warm_gcell_master = []
    							rmse_JRA_combo_warm_gcell_master = []
    							corr_JRA_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    								dframe_3_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_3_model[model_2] = model_2_temp_warm
    								dframe_3_model[model_3] = model_3_temp_warm
    								dframe_3_model_avg = dframe_3_model.mean(axis=1)
    								blend_3_model_temp_warm = dframe_3_model_avg
    								bias_3_model_warm = bias(blend_3_model_temp_warm,station_temp_warm)
    								bias_JRA_combo_warm_gcell_master.append(bias_3_model_warm)
    								stdev_3_model_warm = np.std(blend_3_model_temp_warm)
    								stdev_JRA_combo_warm_gcell_master.append(stdev_3_model_warm)
    								rmse_3_model_warm = mean_squared_error(station_temp_warm,blend_3_model_temp_warm,squared=False)
    								rmse_JRA_combo_warm_gcell_master.append(rmse_3_model_warm)
    								corr_3_model_warm,_ = pearsonr(blend_3_model_temp_warm,station_temp_warm)
    								corr_JRA_combo_warm_gcell_master.append(corr_3_model_warm)    							

    							bias_JRA_combo_warm_gcell_mean = mean(bias_JRA_combo_warm_gcell_master)
    							bias_JRA_combo_warm_master.append(bias_JRA_combo_warm_gcell_mean)
    							stdev_JRA_combo_warm_gcell_mean = mean(stdev_JRA_combo_warm_gcell_master)
    							stdev_JRA_combo_warm_master.append(stdev_JRA_combo_warm_gcell_mean)
    							rmse_JRA_combo_warm_gcell_mean = mean(rmse_JRA_combo_warm_gcell_master)
    							rmse_JRA_combo_warm_master.append(rmse_JRA_combo_warm_gcell_mean)
    							corr_JRA_combo_warm_gcell_mean = mean(corr_JRA_combo_warm_gcell_master)
    							corr_JRA_combo_warm_master.append(corr_JRA_combo_warm_gcell_mean)


    						elif (len_i == 4):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]

    							blend_JRA_combo_temp_warm_gcell_master = []
    							bias_JRA_combo_warm_gcell_master = []
    							stdev_JRA_combo_warm_gcell_master = []
    							rmse_JRA_combo_warm_gcell_master = []
    							corr_JRA_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    								model_4_temp_warm = dframe_warm_season_gcell[model_4].values
    								dframe_4_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_4_model[model_2] = model_2_temp_warm
    								dframe_4_model[model_3] = model_3_temp_warm
    								dframe_4_model[model_4] = model_4_temp_warm
    								dframe_4_model_avg = dframe_4_model.mean(axis=1)
    								blend_4_model_temp_warm = dframe_4_model_avg
    								bias_4_model_warm = bias(blend_4_model_temp_warm,station_temp_warm)
    								bias_JRA_combo_warm_gcell_master.append(bias_4_model_warm)
    								stdev_4_model_warm = np.std(blend_4_model_temp_warm)
    								stdev_JRA_combo_warm_gcell_master.append(stdev_4_model_warm)
    								rmse_4_model_warm = mean_squared_error(station_temp_warm,blend_4_model_temp_warm,squared=False)
    								rmse_JRA_combo_warm_gcell_master.append(rmse_4_model_warm)
    								corr_4_model_warm,_ = pearsonr(blend_4_model_temp_warm,station_temp_warm)
    								corr_JRA_combo_warm_gcell_master.append(corr_4_model_warm)    							

    							bias_JRA_combo_warm_gcell_mean = mean(bias_JRA_combo_warm_gcell_master)
    							bias_JRA_combo_warm_master.append(bias_JRA_combo_warm_gcell_mean)
    							stdev_JRA_combo_warm_gcell_mean = mean(stdev_JRA_combo_warm_gcell_master)
    							stdev_JRA_combo_warm_master.append(stdev_JRA_combo_warm_gcell_mean)
    							rmse_JRA_combo_warm_gcell_mean = mean(rmse_JRA_combo_warm_gcell_master)
    							rmse_JRA_combo_warm_master.append(rmse_JRA_combo_warm_gcell_mean)
    							corr_JRA_combo_warm_gcell_mean = mean(corr_JRA_combo_warm_gcell_master)
    							corr_JRA_combo_warm_master.append(corr_JRA_combo_warm_gcell_mean)



    						elif (len_i == 5):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]

    							blend_JRA_combo_temp_warm_gcell_master = []
    							bias_JRA_combo_warm_gcell_master = []
    							stdev_JRA_combo_warm_gcell_master = []
    							rmse_JRA_combo_warm_gcell_master = []
    							corr_JRA_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    								model_4_temp_warm = dframe_warm_season_gcell[model_4].values
    								model_5_temp_warm = dframe_warm_season_gcell[model_5].values
    								dframe_5_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_5_model[model_2] = model_2_temp_warm
    								dframe_5_model[model_3] = model_3_temp_warm
    								dframe_5_model[model_4] = model_4_temp_warm
    								dframe_5_model[model_5] = model_5_temp_warm
    								dframe_5_model_avg = dframe_5_model.mean(axis=1)
    								blend_5_model_temp_warm = dframe_5_model_avg
    								bias_5_model_warm = bias(blend_5_model_temp_warm,station_temp_warm)
    								bias_JRA_combo_warm_gcell_master.append(bias_5_model_warm)
    								stdev_5_model_warm = np.std(blend_5_model_temp_warm)
    								stdev_JRA_combo_warm_gcell_master.append(stdev_5_model_warm)
    								rmse_5_model_warm = mean_squared_error(station_temp_warm,blend_5_model_temp_warm,squared=False)
    								rmse_JRA_combo_warm_gcell_master.append(rmse_5_model_warm)
    								corr_5_model_warm,_ = pearsonr(blend_5_model_temp_warm,station_temp_warm)
    								corr_JRA_combo_warm_gcell_master.append(corr_5_model_warm)    							

    							bias_JRA_combo_warm_gcell_mean = mean(bias_JRA_combo_warm_gcell_master)
    							bias_JRA_combo_warm_master.append(bias_JRA_combo_warm_gcell_mean)
    							stdev_JRA_combo_warm_gcell_mean = mean(stdev_JRA_combo_warm_gcell_master)
    							stdev_JRA_combo_warm_master.append(stdev_JRA_combo_warm_gcell_mean)
    							rmse_JRA_combo_warm_gcell_mean = mean(rmse_JRA_combo_warm_gcell_master)
    							rmse_JRA_combo_warm_master.append(rmse_JRA_combo_warm_gcell_mean)
    							corr_JRA_combo_warm_gcell_mean = mean(corr_JRA_combo_warm_gcell_master)
    							corr_JRA_combo_warm_master.append(corr_JRA_combo_warm_gcell_mean)



    						elif (len_i == 6):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]
    							model_6 = i[5]

    							blend_JRA_combo_temp_warm_gcell_master = []
    							bias_JRA_combo_warm_gcell_master = []
    							stdev_JRA_combo_warm_gcell_master = []
    							rmse_JRA_combo_warm_gcell_master = []
    							corr_JRA_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    								model_4_temp_warm = dframe_warm_season_gcell[model_4].values
    								model_5_temp_warm = dframe_warm_season_gcell[model_5].values
    								model_6_temp_warm = dframe_warm_season_gcell[model_6].values
    								dframe_6_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_6_model[model_2] = model_2_temp_warm
    								dframe_6_model[model_3] = model_3_temp_warm
    								dframe_6_model[model_4] = model_4_temp_warm
    								dframe_6_model[model_5] = model_5_temp_warm
    								dframe_6_model[model_6] = model_6_temp_warm
    								dframe_6_model_avg = dframe_6_model.mean(axis=1)
    								blend_6_model_temp_warm = dframe_6_model_avg
    								bias_6_model_warm = bias(blend_6_model_temp_warm,station_temp_warm)
    								bias_JRA_combo_warm_gcell_master.append(bias_6_model_warm)
    								stdev_6_model_warm = np.std(blend_6_model_temp_warm)
    								stdev_JRA_combo_warm_gcell_master.append(stdev_6_model_warm)
    								rmse_6_model_warm = mean_squared_error(station_temp_warm,blend_6_model_temp_warm,squared=False)
    								rmse_JRA_combo_warm_gcell_master.append(rmse_6_model_warm)
    								corr_6_model_warm,_ = pearsonr(blend_6_model_temp_warm,station_temp_warm)
    								corr_JRA_combo_warm_gcell_master.append(corr_6_model_warm)    							

    							bias_JRA_combo_warm_gcell_mean = mean(bias_JRA_combo_warm_gcell_master)
    							bias_JRA_combo_warm_master.append(bias_JRA_combo_warm_gcell_mean)
    							stdev_JRA_combo_warm_gcell_mean = mean(stdev_JRA_combo_warm_gcell_master)
    							stdev_JRA_combo_warm_master.append(stdev_JRA_combo_warm_gcell_mean)
    							rmse_JRA_combo_warm_gcell_mean = mean(rmse_JRA_combo_warm_gcell_master)
    							rmse_JRA_combo_warm_master.append(rmse_JRA_combo_warm_gcell_mean)
    							corr_JRA_combo_warm_gcell_mean = mean(corr_JRA_combo_warm_gcell_master)
    							corr_JRA_combo_warm_master.append(corr_JRA_combo_warm_gcell_mean)



    						elif (len_i == 7):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]
    							model_6 = i[5]
    							model_7 = i[6]

    							blend_JRA_combo_temp_warm_gcell_master = []
    							bias_JRA_combo_warm_gcell_master = []
    							stdev_JRA_combo_warm_gcell_master = []
    							rmse_JRA_combo_warm_gcell_master = []
    							corr_JRA_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    								model_4_temp_warm = dframe_warm_season_gcell[model_4].values
    								model_5_temp_warm = dframe_warm_season_gcell[model_5].values
    								model_6_temp_warm = dframe_warm_season_gcell[model_6].values
    								model_7_temp_warm = dframe_warm_season_gcell[model_7].values
    								dframe_7_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_7_model[model_2] = model_2_temp_warm
    								dframe_7_model[model_3] = model_3_temp_warm
    								dframe_7_model[model_4] = model_4_temp_warm
    								dframe_7_model[model_5] = model_5_temp_warm
    								dframe_7_model[model_6] = model_6_temp_warm
    								dframe_7_model[model_7] = model_7_temp_warm
    								dframe_7_model_avg = dframe_7_model.mean(axis=1)
    								blend_7_model_temp_warm = dframe_7_model_avg
    								bias_7_model_warm = bias(blend_7_model_temp_warm,station_temp_warm)
    								bias_JRA_combo_warm_gcell_master.append(bias_7_model_warm)
    								stdev_7_model_warm = np.std(blend_7_model_temp_warm)
    								stdev_JRA_combo_warm_gcell_master.append(stdev_7_model_warm)
    								rmse_7_model_warm = mean_squared_error(station_temp_warm,blend_7_model_temp_warm,squared=False)
    								rmse_JRA_combo_warm_gcell_master.append(rmse_7_model_warm)
    								corr_7_model_warm,_ = pearsonr(blend_7_model_temp_warm,station_temp_warm)
    								corr_JRA_combo_warm_gcell_master.append(corr_7_model_warm)    							

    							bias_JRA_combo_warm_gcell_mean = mean(bias_JRA_combo_warm_gcell_master)
    							bias_JRA_combo_warm_master.append(bias_JRA_combo_warm_gcell_mean)
    							stdev_JRA_combo_warm_gcell_mean = mean(stdev_JRA_combo_warm_gcell_master)
    							stdev_JRA_combo_warm_master.append(stdev_JRA_combo_warm_gcell_mean)
    							rmse_JRA_combo_warm_gcell_mean = mean(rmse_JRA_combo_warm_gcell_master)
    							rmse_JRA_combo_warm_master.append(rmse_JRA_combo_warm_gcell_mean)
    							corr_JRA_combo_warm_gcell_mean = mean(corr_JRA_combo_warm_gcell_master)
    							corr_JRA_combo_warm_master.append(corr_JRA_combo_warm_gcell_mean)



    						elif (len_i == 8):
    							bias_8_model_warm = bias_naive_warm_mean
    							bias_JRA_combo_warm_master.append(bias_8_model_warm)
    							stdev_8_model_warm = stdev_naive_warm_mean
    							stdev_JRA_combo_warm_master.append(stdev_8_model_warm)
    							rmse_8_model_warm = rmse_naive_warm_mean 
    							rmse_JRA_combo_warm_master.append(rmse_8_model_warm)
    							corr_8_model_warm = corr_naive_warm_mean
    							corr_JRA_combo_warm_master.append(corr_8_model_warm)

    					bias_JRA_combo_warm_mean = mean(bias_JRA_combo_warm_master)
    					stdev_JRA_combo_warm_mean = mean(stdev_JRA_combo_warm_master)
    					SDV_JRA_combo_warm_mean = stdev_JRA_combo_warm_mean/stdev_station_warm
    					rmse_JRA_combo_warm_mean = mean(rmse_JRA_combo_warm_master)
    					corr_JRA_combo_warm_mean = mean(corr_JRA_combo_warm_master)



## MERRA2 Model ##

    					bias_MERRA2_combo_warm_master = []
    					rmse_MERRA2_combo_warm_master = []
    					stdev_MERRA2_combo_warm_master = []
    					SDV_MERRA2_combo_warm_master = []
    					corr_MERRA2_combo_warm_master = []

    					for i in MERRA2_array:
    						len_i = len(i)
    						if (len_i == 1):
    							blend_MERRA2_combo_temp_warm_gcell_master = []
    							bias_MERRA2_combo_warm_gcell_master = []
    							stdev_MERRA2_combo_warm_gcell_master = []
    							rmse_MERRA2_combo_warm_gcell_master = []
    							corr_MERRA2_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								blend_1_model_temp_warm = dframe_warm_season_gcell[i[0]].values
    								print(blend_1_model_temp_warm)
    								print(station_temp_warm)
    								bias_1_model_warm = bias(blend_1_model_temp_warm,station_temp_warm)
    								bias_MERRA2_combo_warm_gcell_master.append(bias_1_model_warm)
    								stdev_1_model_warm = np.std(blend_1_model_temp_warm)
    								stdev_MERRA2_combo_warm_gcell_master.append(stdev_1_model_warm)
    								rmse_1_model_warm = mean_squared_error(station_temp_warm,blend_1_model_temp_warm,squared=False)
    								rmse_MERRA2_combo_warm_gcell_master.append(rmse_1_model_warm)
    								corr_1_model_warm,_ = pearsonr(blend_1_model_temp_warm,station_temp_warm)
    								corr_MERRA2_combo_warm_gcell_master.append(corr_1_model_warm)    							

    							bias_MERRA2_combo_warm_gcell_mean = mean(bias_MERRA2_combo_warm_gcell_master)
    							bias_MERRA2_combo_warm_master.append(bias_MERRA2_combo_warm_gcell_mean)
    							stdev_MERRA2_combo_warm_gcell_mean = mean(stdev_MERRA2_combo_warm_gcell_master)
    							stdev_MERRA2_combo_warm_master.append(stdev_MERRA2_combo_warm_gcell_mean)
    							rmse_MERRA2_combo_warm_gcell_mean = mean(rmse_MERRA2_combo_warm_gcell_master)
    							rmse_MERRA2_combo_warm_master.append(rmse_MERRA2_combo_warm_gcell_mean)
    							corr_MERRA2_combo_warm_gcell_mean = mean(corr_MERRA2_combo_warm_gcell_master)
    							corr_MERRA2_combo_warm_master.append(corr_MERRA2_combo_warm_gcell_mean)

    						elif (len_i == 2):
    							model_1 = i[0]
    							model_2 = i[1]

    							blend_MERRA2_combo_temp_warm_gcell_master = []
    							bias_MERRA2_combo_warm_gcell_master = []
    							stdev_MERRA2_combo_warm_gcell_master = []
    							rmse_MERRA2_combo_warm_gcell_master = []
    							corr_MERRA2_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								dframe_2_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_2_model[model_2] = model_2_temp_warm
    								dframe_2_model_avg = dframe_2_model.mean(axis=1)
    								blend_2_model_temp_warm = dframe_2_model_avg
    								bias_2_model_warm = bias(blend_2_model_temp_warm,station_temp_warm)
    								bias_MERRA2_combo_warm_gcell_master.append(bias_2_model_warm)
    								stdev_2_model_warm = np.std(blend_2_model_temp_warm)
    								stdev_MERRA2_combo_warm_gcell_master.append(stdev_2_model_warm)
    								rmse_2_model_warm = mean_squared_error(station_temp_warm,blend_2_model_temp_warm,squared=False)
    								rmse_MERRA2_combo_warm_gcell_master.append(rmse_2_model_warm)
    								corr_2_model_warm,_ = pearsonr(blend_2_model_temp_warm,station_temp_warm)
    								corr_MERRA2_combo_warm_gcell_master.append(corr_2_model_warm)    							

    							bias_MERRA2_combo_warm_gcell_mean = mean(bias_MERRA2_combo_warm_gcell_master)
    							bias_MERRA2_combo_warm_master.append(bias_MERRA2_combo_warm_gcell_mean)
    							stdev_MERRA2_combo_warm_gcell_mean = mean(stdev_MERRA2_combo_warm_gcell_master)
    							stdev_MERRA2_combo_warm_master.append(stdev_MERRA2_combo_warm_gcell_mean)
    							rmse_MERRA2_combo_warm_gcell_mean = mean(rmse_MERRA2_combo_warm_gcell_master)
    							rmse_MERRA2_combo_warm_master.append(rmse_MERRA2_combo_warm_gcell_mean)
    							corr_MERRA2_combo_warm_gcell_mean = mean(corr_MERRA2_combo_warm_gcell_master)
    							corr_MERRA2_combo_warm_master.append(corr_MERRA2_combo_warm_gcell_mean)


    						elif (len_i == 3):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]

    							blend_MERRA2_combo_temp_warm_gcell_master = []
    							bias_MERRA2_combo_warm_gcell_master = []
    							stdev_MERRA2_combo_warm_gcell_master = []
    							rmse_MERRA2_combo_warm_gcell_master = []
    							corr_MERRA2_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    								dframe_3_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_3_model[model_2] = model_2_temp_warm
    								dframe_3_model[model_3] = model_3_temp_warm
    								dframe_3_model_avg = dframe_3_model.mean(axis=1)
    								blend_3_model_temp_warm = dframe_3_model_avg
    								bias_3_model_warm = bias(blend_3_model_temp_warm,station_temp_warm)
    								bias_MERRA2_combo_warm_gcell_master.append(bias_3_model_warm)
    								stdev_3_model_warm = np.std(blend_3_model_temp_warm)
    								stdev_MERRA2_combo_warm_gcell_master.append(stdev_3_model_warm)
    								rmse_3_model_warm = mean_squared_error(station_temp_warm,blend_3_model_temp_warm,squared=False)
    								rmse_MERRA2_combo_warm_gcell_master.append(rmse_3_model_warm)
    								corr_3_model_warm,_ = pearsonr(blend_3_model_temp_warm,station_temp_warm)
    								corr_MERRA2_combo_warm_gcell_master.append(corr_3_model_warm)    							

    							bias_MERRA2_combo_warm_gcell_mean = mean(bias_MERRA2_combo_warm_gcell_master)
    							bias_MERRA2_combo_warm_master.append(bias_MERRA2_combo_warm_gcell_mean)
    							stdev_MERRA2_combo_warm_gcell_mean = mean(stdev_MERRA2_combo_warm_gcell_master)
    							stdev_MERRA2_combo_warm_master.append(stdev_MERRA2_combo_warm_gcell_mean)
    							rmse_MERRA2_combo_warm_gcell_mean = mean(rmse_MERRA2_combo_warm_gcell_master)
    							rmse_MERRA2_combo_warm_master.append(rmse_MERRA2_combo_warm_gcell_mean)
    							corr_MERRA2_combo_warm_gcell_mean = mean(corr_MERRA2_combo_warm_gcell_master)
    							corr_MERRA2_combo_warm_master.append(corr_MERRA2_combo_warm_gcell_mean)


    						elif (len_i == 4):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]

    							blend_MERRA2_combo_temp_warm_gcell_master = []
    							bias_MERRA2_combo_warm_gcell_master = []
    							stdev_MERRA2_combo_warm_gcell_master = []
    							rmse_MERRA2_combo_warm_gcell_master = []
    							corr_MERRA2_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    								model_4_temp_warm = dframe_warm_season_gcell[model_4].values
    								dframe_4_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_4_model[model_2] = model_2_temp_warm
    								dframe_4_model[model_3] = model_3_temp_warm
    								dframe_4_model[model_4] = model_4_temp_warm
    								dframe_4_model_avg = dframe_4_model.mean(axis=1)
    								blend_4_model_temp_warm = dframe_4_model_avg
    								bias_4_model_warm = bias(blend_4_model_temp_warm,station_temp_warm)
    								bias_MERRA2_combo_warm_gcell_master.append(bias_4_model_warm)
    								stdev_4_model_warm = np.std(blend_4_model_temp_warm)
    								stdev_MERRA2_combo_warm_gcell_master.append(stdev_4_model_warm)
    								rmse_4_model_warm = mean_squared_error(station_temp_warm,blend_4_model_temp_warm,squared=False)
    								rmse_MERRA2_combo_warm_gcell_master.append(rmse_4_model_warm)
    								corr_4_model_warm,_ = pearsonr(blend_4_model_temp_warm,station_temp_warm)
    								corr_MERRA2_combo_warm_gcell_master.append(corr_4_model_warm)    							

    							bias_MERRA2_combo_warm_gcell_mean = mean(bias_MERRA2_combo_warm_gcell_master)
    							bias_MERRA2_combo_warm_master.append(bias_MERRA2_combo_warm_gcell_mean)
    							stdev_MERRA2_combo_warm_gcell_mean = mean(stdev_MERRA2_combo_warm_gcell_master)
    							stdev_MERRA2_combo_warm_master.append(stdev_MERRA2_combo_warm_gcell_mean)
    							rmse_MERRA2_combo_warm_gcell_mean = mean(rmse_MERRA2_combo_warm_gcell_master)
    							rmse_MERRA2_combo_warm_master.append(rmse_MERRA2_combo_warm_gcell_mean)
    							corr_MERRA2_combo_warm_gcell_mean = mean(corr_MERRA2_combo_warm_gcell_master)
    							corr_MERRA2_combo_warm_master.append(corr_MERRA2_combo_warm_gcell_mean)



    						elif (len_i == 5):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]

    							blend_MERRA2_combo_temp_warm_gcell_master = []
    							bias_MERRA2_combo_warm_gcell_master = []
    							stdev_MERRA2_combo_warm_gcell_master = []
    							rmse_MERRA2_combo_warm_gcell_master = []
    							corr_MERRA2_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    								model_4_temp_warm = dframe_warm_season_gcell[model_4].values
    								model_5_temp_warm = dframe_warm_season_gcell[model_5].values
    								dframe_5_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_5_model[model_2] = model_2_temp_warm
    								dframe_5_model[model_3] = model_3_temp_warm
    								dframe_5_model[model_4] = model_4_temp_warm
    								dframe_5_model[model_5] = model_5_temp_warm
    								dframe_5_model_avg = dframe_5_model.mean(axis=1)
    								blend_5_model_temp_warm = dframe_5_model_avg
    								bias_5_model_warm = bias(blend_5_model_temp_warm,station_temp_warm)
    								bias_MERRA2_combo_warm_gcell_master.append(bias_5_model_warm)
    								stdev_5_model_warm = np.std(blend_5_model_temp_warm)
    								stdev_MERRA2_combo_warm_gcell_master.append(stdev_5_model_warm)
    								rmse_5_model_warm = mean_squared_error(station_temp_warm,blend_5_model_temp_warm,squared=False)
    								rmse_MERRA2_combo_warm_gcell_master.append(rmse_5_model_warm)
    								corr_5_model_warm,_ = pearsonr(blend_5_model_temp_warm,station_temp_warm)
    								corr_MERRA2_combo_warm_gcell_master.append(corr_5_model_warm)    							

    							bias_MERRA2_combo_warm_gcell_mean = mean(bias_MERRA2_combo_warm_gcell_master)
    							bias_MERRA2_combo_warm_master.append(bias_MERRA2_combo_warm_gcell_mean)
    							stdev_MERRA2_combo_warm_gcell_mean = mean(stdev_MERRA2_combo_warm_gcell_master)
    							stdev_MERRA2_combo_warm_master.append(stdev_MERRA2_combo_warm_gcell_mean)
    							rmse_MERRA2_combo_warm_gcell_mean = mean(rmse_MERRA2_combo_warm_gcell_master)
    							rmse_MERRA2_combo_warm_master.append(rmse_MERRA2_combo_warm_gcell_mean)
    							corr_MERRA2_combo_warm_gcell_mean = mean(corr_MERRA2_combo_warm_gcell_master)
    							corr_MERRA2_combo_warm_master.append(corr_MERRA2_combo_warm_gcell_mean)



    						elif (len_i == 6):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]
    							model_6 = i[5]

    							blend_MERRA2_combo_temp_warm_gcell_master = []
    							bias_MERRA2_combo_warm_gcell_master = []
    							stdev_MERRA2_combo_warm_gcell_master = []
    							rmse_MERRA2_combo_warm_gcell_master = []
    							corr_MERRA2_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    								model_4_temp_warm = dframe_warm_season_gcell[model_4].values
    								model_5_temp_warm = dframe_warm_season_gcell[model_5].values
    								model_6_temp_warm = dframe_warm_season_gcell[model_6].values
    								dframe_6_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_6_model[model_2] = model_2_temp_warm
    								dframe_6_model[model_3] = model_3_temp_warm
    								dframe_6_model[model_4] = model_4_temp_warm
    								dframe_6_model[model_5] = model_5_temp_warm
    								dframe_6_model[model_6] = model_6_temp_warm
    								dframe_6_model_avg = dframe_6_model.mean(axis=1)
    								blend_6_model_temp_warm = dframe_6_model_avg
    								bias_6_model_warm = bias(blend_6_model_temp_warm,station_temp_warm)
    								bias_MERRA2_combo_warm_gcell_master.append(bias_6_model_warm)
    								stdev_6_model_warm = np.std(blend_6_model_temp_warm)
    								stdev_MERRA2_combo_warm_gcell_master.append(stdev_6_model_warm)
    								rmse_6_model_warm = mean_squared_error(station_temp_warm,blend_6_model_temp_warm,squared=False)
    								rmse_MERRA2_combo_warm_gcell_master.append(rmse_6_model_warm)
    								corr_6_model_warm,_ = pearsonr(blend_6_model_temp_warm,station_temp_warm)
    								corr_MERRA2_combo_warm_gcell_master.append(corr_6_model_warm)    							

    							bias_MERRA2_combo_warm_gcell_mean = mean(bias_MERRA2_combo_warm_gcell_master)
    							bias_MERRA2_combo_warm_master.append(bias_MERRA2_combo_warm_gcell_mean)
    							stdev_MERRA2_combo_warm_gcell_mean = mean(stdev_MERRA2_combo_warm_gcell_master)
    							stdev_MERRA2_combo_warm_master.append(stdev_MERRA2_combo_warm_gcell_mean)
    							rmse_MERRA2_combo_warm_gcell_mean = mean(rmse_MERRA2_combo_warm_gcell_master)
    							rmse_MERRA2_combo_warm_master.append(rmse_MERRA2_combo_warm_gcell_mean)
    							corr_MERRA2_combo_warm_gcell_mean = mean(corr_MERRA2_combo_warm_gcell_master)
    							corr_MERRA2_combo_warm_master.append(corr_MERRA2_combo_warm_gcell_mean)



    						elif (len_i == 7):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]
    							model_6 = i[5]
    							model_7 = i[6]

    							blend_MERRA2_combo_temp_warm_gcell_master = []
    							bias_MERRA2_combo_warm_gcell_master = []
    							stdev_MERRA2_combo_warm_gcell_master = []
    							rmse_MERRA2_combo_warm_gcell_master = []
    							corr_MERRA2_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    								model_4_temp_warm = dframe_warm_season_gcell[model_4].values
    								model_5_temp_warm = dframe_warm_season_gcell[model_5].values
    								model_6_temp_warm = dframe_warm_season_gcell[model_6].values
    								model_7_temp_warm = dframe_warm_season_gcell[model_7].values
    								dframe_7_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_7_model[model_2] = model_2_temp_warm
    								dframe_7_model[model_3] = model_3_temp_warm
    								dframe_7_model[model_4] = model_4_temp_warm
    								dframe_7_model[model_5] = model_5_temp_warm
    								dframe_7_model[model_6] = model_6_temp_warm
    								dframe_7_model[model_7] = model_7_temp_warm
    								dframe_7_model_avg = dframe_7_model.mean(axis=1)
    								blend_7_model_temp_warm = dframe_7_model_avg
    								bias_7_model_warm = bias(blend_7_model_temp_warm,station_temp_warm)
    								bias_MERRA2_combo_warm_gcell_master.append(bias_7_model_warm)
    								stdev_7_model_warm = np.std(blend_7_model_temp_warm)
    								stdev_MERRA2_combo_warm_gcell_master.append(stdev_7_model_warm)
    								rmse_7_model_warm = mean_squared_error(station_temp_warm,blend_7_model_temp_warm,squared=False)
    								rmse_MERRA2_combo_warm_gcell_master.append(rmse_7_model_warm)
    								corr_7_model_warm,_ = pearsonr(blend_7_model_temp_warm,station_temp_warm)
    								corr_MERRA2_combo_warm_gcell_master.append(corr_7_model_warm)    							

    							bias_MERRA2_combo_warm_gcell_mean = mean(bias_MERRA2_combo_warm_gcell_master)
    							bias_MERRA2_combo_warm_master.append(bias_MERRA2_combo_warm_gcell_mean)
    							stdev_MERRA2_combo_warm_gcell_mean = mean(stdev_MERRA2_combo_warm_gcell_master)
    							stdev_MERRA2_combo_warm_master.append(stdev_MERRA2_combo_warm_gcell_mean)
    							rmse_MERRA2_combo_warm_gcell_mean = mean(rmse_MERRA2_combo_warm_gcell_master)
    							rmse_MERRA2_combo_warm_master.append(rmse_MERRA2_combo_warm_gcell_mean)
    							corr_MERRA2_combo_warm_gcell_mean = mean(corr_MERRA2_combo_warm_gcell_master)
    							corr_MERRA2_combo_warm_master.append(corr_MERRA2_combo_warm_gcell_mean)



    						elif (len_i == 8):
    							bias_8_model_warm = bias_naive_warm_mean
    							bias_MERRA2_combo_warm_master.append(bias_8_model_warm)
    							stdev_8_model_warm = stdev_naive_warm_mean
    							stdev_MERRA2_combo_warm_master.append(stdev_8_model_warm)
    							rmse_8_model_warm = rmse_naive_warm_mean 
    							rmse_MERRA2_combo_warm_master.append(rmse_8_model_warm)
    							corr_8_model_warm = corr_naive_warm_mean
    							corr_MERRA2_combo_warm_master.append(corr_8_model_warm)

    					bias_MERRA2_combo_warm_mean = mean(bias_MERRA2_combo_warm_master)
    					stdev_MERRA2_combo_warm_mean = mean(stdev_MERRA2_combo_warm_master)
    					SDV_MERRA2_combo_warm_mean = stdev_MERRA2_combo_warm_mean/stdev_station_warm
    					rmse_MERRA2_combo_warm_mean = mean(rmse_MERRA2_combo_warm_master)
    					corr_MERRA2_combo_warm_mean = mean(corr_MERRA2_combo_warm_master)



## GLDAS-Noah Model ##

    					bias_GLDAS_combo_warm_master = []
    					rmse_GLDAS_combo_warm_master = []
    					stdev_GLDAS_combo_warm_master = []
    					SDV_GLDAS_combo_warm_master = []
    					corr_GLDAS_combo_warm_master = []

    					for i in GLDAS_array:
    						len_i = len(i)
    						if (len_i == 1):
    							blend_GLDAS_combo_temp_warm_gcell_master = []
    							bias_GLDAS_combo_warm_gcell_master = []
    							stdev_GLDAS_combo_warm_gcell_master = []
    							rmse_GLDAS_combo_warm_gcell_master = []
    							corr_GLDAS_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								blend_1_model_temp_warm = dframe_warm_season_gcell[i[0]].values
    								print(blend_1_model_temp_warm)
    								print(station_temp_warm)
    								bias_1_model_warm = bias(blend_1_model_temp_warm,station_temp_warm)
    								bias_GLDAS_combo_warm_gcell_master.append(bias_1_model_warm)
    								stdev_1_model_warm = np.std(blend_1_model_temp_warm)
    								stdev_GLDAS_combo_warm_gcell_master.append(stdev_1_model_warm)
    								rmse_1_model_warm = mean_squared_error(station_temp_warm,blend_1_model_temp_warm,squared=False)
    								rmse_GLDAS_combo_warm_gcell_master.append(rmse_1_model_warm)
    								corr_1_model_warm,_ = pearsonr(blend_1_model_temp_warm,station_temp_warm)
    								corr_GLDAS_combo_warm_gcell_master.append(corr_1_model_warm)    							

    							bias_GLDAS_combo_warm_gcell_mean = mean(bias_GLDAS_combo_warm_gcell_master)
    							bias_GLDAS_combo_warm_master.append(bias_GLDAS_combo_warm_gcell_mean)
    							stdev_GLDAS_combo_warm_gcell_mean = mean(stdev_GLDAS_combo_warm_gcell_master)
    							stdev_GLDAS_combo_warm_master.append(stdev_GLDAS_combo_warm_gcell_mean)
    							rmse_GLDAS_combo_warm_gcell_mean = mean(rmse_GLDAS_combo_warm_gcell_master)
    							rmse_GLDAS_combo_warm_master.append(rmse_GLDAS_combo_warm_gcell_mean)
    							corr_GLDAS_combo_warm_gcell_mean = mean(corr_GLDAS_combo_warm_gcell_master)
    							corr_GLDAS_combo_warm_master.append(corr_GLDAS_combo_warm_gcell_mean)

    						elif (len_i == 2):
    							model_1 = i[0]
    							model_2 = i[1]

    							blend_GLDAS_combo_temp_warm_gcell_master = []
    							bias_GLDAS_combo_warm_gcell_master = []
    							stdev_GLDAS_combo_warm_gcell_master = []
    							rmse_GLDAS_combo_warm_gcell_master = []
    							corr_GLDAS_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								dframe_2_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_2_model[model_2] = model_2_temp_warm
    								dframe_2_model_avg = dframe_2_model.mean(axis=1)
    								blend_2_model_temp_warm = dframe_2_model_avg
    								bias_2_model_warm = bias(blend_2_model_temp_warm,station_temp_warm)
    								bias_GLDAS_combo_warm_gcell_master.append(bias_2_model_warm)
    								stdev_2_model_warm = np.std(blend_2_model_temp_warm)
    								stdev_GLDAS_combo_warm_gcell_master.append(stdev_2_model_warm)
    								rmse_2_model_warm = mean_squared_error(station_temp_warm,blend_2_model_temp_warm,squared=False)
    								rmse_GLDAS_combo_warm_gcell_master.append(rmse_2_model_warm)
    								corr_2_model_warm,_ = pearsonr(blend_2_model_temp_warm,station_temp_warm)
    								corr_GLDAS_combo_warm_gcell_master.append(corr_2_model_warm)    							

    							bias_GLDAS_combo_warm_gcell_mean = mean(bias_GLDAS_combo_warm_gcell_master)
    							bias_GLDAS_combo_warm_master.append(bias_GLDAS_combo_warm_gcell_mean)
    							stdev_GLDAS_combo_warm_gcell_mean = mean(stdev_GLDAS_combo_warm_gcell_master)
    							stdev_GLDAS_combo_warm_master.append(stdev_GLDAS_combo_warm_gcell_mean)
    							rmse_GLDAS_combo_warm_gcell_mean = mean(rmse_GLDAS_combo_warm_gcell_master)
    							rmse_GLDAS_combo_warm_master.append(rmse_GLDAS_combo_warm_gcell_mean)
    							corr_GLDAS_combo_warm_gcell_mean = mean(corr_GLDAS_combo_warm_gcell_master)
    							corr_GLDAS_combo_warm_master.append(corr_GLDAS_combo_warm_gcell_mean)


    						elif (len_i == 3):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]

    							blend_GLDAS_combo_temp_warm_gcell_master = []
    							bias_GLDAS_combo_warm_gcell_master = []
    							stdev_GLDAS_combo_warm_gcell_master = []
    							rmse_GLDAS_combo_warm_gcell_master = []
    							corr_GLDAS_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    								dframe_3_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_3_model[model_2] = model_2_temp_warm
    								dframe_3_model[model_3] = model_3_temp_warm
    								dframe_3_model_avg = dframe_3_model.mean(axis=1)
    								blend_3_model_temp_warm = dframe_3_model_avg
    								bias_3_model_warm = bias(blend_3_model_temp_warm,station_temp_warm)
    								bias_GLDAS_combo_warm_gcell_master.append(bias_3_model_warm)
    								stdev_3_model_warm = np.std(blend_3_model_temp_warm)
    								stdev_GLDAS_combo_warm_gcell_master.append(stdev_3_model_warm)
    								rmse_3_model_warm = mean_squared_error(station_temp_warm,blend_3_model_temp_warm,squared=False)
    								rmse_GLDAS_combo_warm_gcell_master.append(rmse_3_model_warm)
    								corr_3_model_warm,_ = pearsonr(blend_3_model_temp_warm,station_temp_warm)
    								corr_GLDAS_combo_warm_gcell_master.append(corr_3_model_warm)    							

    							bias_GLDAS_combo_warm_gcell_mean = mean(bias_GLDAS_combo_warm_gcell_master)
    							bias_GLDAS_combo_warm_master.append(bias_GLDAS_combo_warm_gcell_mean)
    							stdev_GLDAS_combo_warm_gcell_mean = mean(stdev_GLDAS_combo_warm_gcell_master)
    							stdev_GLDAS_combo_warm_master.append(stdev_GLDAS_combo_warm_gcell_mean)
    							rmse_GLDAS_combo_warm_gcell_mean = mean(rmse_GLDAS_combo_warm_gcell_master)
    							rmse_GLDAS_combo_warm_master.append(rmse_GLDAS_combo_warm_gcell_mean)
    							corr_GLDAS_combo_warm_gcell_mean = mean(corr_GLDAS_combo_warm_gcell_master)
    							corr_GLDAS_combo_warm_master.append(corr_GLDAS_combo_warm_gcell_mean)


    						elif (len_i == 4):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]

    							blend_GLDAS_combo_temp_warm_gcell_master = []
    							bias_GLDAS_combo_warm_gcell_master = []
    							stdev_GLDAS_combo_warm_gcell_master = []
    							rmse_GLDAS_combo_warm_gcell_master = []
    							corr_GLDAS_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    								model_4_temp_warm = dframe_warm_season_gcell[model_4].values
    								dframe_4_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_4_model[model_2] = model_2_temp_warm
    								dframe_4_model[model_3] = model_3_temp_warm
    								dframe_4_model[model_4] = model_4_temp_warm
    								dframe_4_model_avg = dframe_4_model.mean(axis=1)
    								blend_4_model_temp_warm = dframe_4_model_avg
    								bias_4_model_warm = bias(blend_4_model_temp_warm,station_temp_warm)
    								bias_GLDAS_combo_warm_gcell_master.append(bias_4_model_warm)
    								stdev_4_model_warm = np.std(blend_4_model_temp_warm)
    								stdev_GLDAS_combo_warm_gcell_master.append(stdev_4_model_warm)
    								rmse_4_model_warm = mean_squared_error(station_temp_warm,blend_4_model_temp_warm,squared=False)
    								rmse_GLDAS_combo_warm_gcell_master.append(rmse_4_model_warm)
    								corr_4_model_warm,_ = pearsonr(blend_4_model_temp_warm,station_temp_warm)
    								corr_GLDAS_combo_warm_gcell_master.append(corr_4_model_warm)    							

    							bias_GLDAS_combo_warm_gcell_mean = mean(bias_GLDAS_combo_warm_gcell_master)
    							bias_GLDAS_combo_warm_master.append(bias_GLDAS_combo_warm_gcell_mean)
    							stdev_GLDAS_combo_warm_gcell_mean = mean(stdev_GLDAS_combo_warm_gcell_master)
    							stdev_GLDAS_combo_warm_master.append(stdev_GLDAS_combo_warm_gcell_mean)
    							rmse_GLDAS_combo_warm_gcell_mean = mean(rmse_GLDAS_combo_warm_gcell_master)
    							rmse_GLDAS_combo_warm_master.append(rmse_GLDAS_combo_warm_gcell_mean)
    							corr_GLDAS_combo_warm_gcell_mean = mean(corr_GLDAS_combo_warm_gcell_master)
    							corr_GLDAS_combo_warm_master.append(corr_GLDAS_combo_warm_gcell_mean)



    						elif (len_i == 5):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]

    							blend_GLDAS_combo_temp_warm_gcell_master = []
    							bias_GLDAS_combo_warm_gcell_master = []
    							stdev_GLDAS_combo_warm_gcell_master = []
    							rmse_GLDAS_combo_warm_gcell_master = []
    							corr_GLDAS_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    								model_4_temp_warm = dframe_warm_season_gcell[model_4].values
    								model_5_temp_warm = dframe_warm_season_gcell[model_5].values
    								dframe_5_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_5_model[model_2] = model_2_temp_warm
    								dframe_5_model[model_3] = model_3_temp_warm
    								dframe_5_model[model_4] = model_4_temp_warm
    								dframe_5_model[model_5] = model_5_temp_warm
    								dframe_5_model_avg = dframe_5_model.mean(axis=1)
    								blend_5_model_temp_warm = dframe_5_model_avg
    								bias_5_model_warm = bias(blend_5_model_temp_warm,station_temp_warm)
    								bias_GLDAS_combo_warm_gcell_master.append(bias_5_model_warm)
    								stdev_5_model_warm = np.std(blend_5_model_temp_warm)
    								stdev_GLDAS_combo_warm_gcell_master.append(stdev_5_model_warm)
    								rmse_5_model_warm = mean_squared_error(station_temp_warm,blend_5_model_temp_warm,squared=False)
    								rmse_GLDAS_combo_warm_gcell_master.append(rmse_5_model_warm)
    								corr_5_model_warm,_ = pearsonr(blend_5_model_temp_warm,station_temp_warm)
    								corr_GLDAS_combo_warm_gcell_master.append(corr_5_model_warm)    							

    							bias_GLDAS_combo_warm_gcell_mean = mean(bias_GLDAS_combo_warm_gcell_master)
    							bias_GLDAS_combo_warm_master.append(bias_GLDAS_combo_warm_gcell_mean)
    							stdev_GLDAS_combo_warm_gcell_mean = mean(stdev_GLDAS_combo_warm_gcell_master)
    							stdev_GLDAS_combo_warm_master.append(stdev_GLDAS_combo_warm_gcell_mean)
    							rmse_GLDAS_combo_warm_gcell_mean = mean(rmse_GLDAS_combo_warm_gcell_master)
    							rmse_GLDAS_combo_warm_master.append(rmse_GLDAS_combo_warm_gcell_mean)
    							corr_GLDAS_combo_warm_gcell_mean = mean(corr_GLDAS_combo_warm_gcell_master)
    							corr_GLDAS_combo_warm_master.append(corr_GLDAS_combo_warm_gcell_mean)



    						elif (len_i == 6):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]
    							model_6 = i[5]

    							blend_GLDAS_combo_temp_warm_gcell_master = []
    							bias_GLDAS_combo_warm_gcell_master = []
    							stdev_GLDAS_combo_warm_gcell_master = []
    							rmse_GLDAS_combo_warm_gcell_master = []
    							corr_GLDAS_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    								model_4_temp_warm = dframe_warm_season_gcell[model_4].values
    								model_5_temp_warm = dframe_warm_season_gcell[model_5].values
    								model_6_temp_warm = dframe_warm_season_gcell[model_6].values
    								dframe_6_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_6_model[model_2] = model_2_temp_warm
    								dframe_6_model[model_3] = model_3_temp_warm
    								dframe_6_model[model_4] = model_4_temp_warm
    								dframe_6_model[model_5] = model_5_temp_warm
    								dframe_6_model[model_6] = model_6_temp_warm
    								dframe_6_model_avg = dframe_6_model.mean(axis=1)
    								blend_6_model_temp_warm = dframe_6_model_avg
    								bias_6_model_warm = bias(blend_6_model_temp_warm,station_temp_warm)
    								bias_GLDAS_combo_warm_gcell_master.append(bias_6_model_warm)
    								stdev_6_model_warm = np.std(blend_6_model_temp_warm)
    								stdev_GLDAS_combo_warm_gcell_master.append(stdev_6_model_warm)
    								rmse_6_model_warm = mean_squared_error(station_temp_warm,blend_6_model_temp_warm,squared=False)
    								rmse_GLDAS_combo_warm_gcell_master.append(rmse_6_model_warm)
    								corr_6_model_warm,_ = pearsonr(blend_6_model_temp_warm,station_temp_warm)
    								corr_GLDAS_combo_warm_gcell_master.append(corr_6_model_warm)    							

    							bias_GLDAS_combo_warm_gcell_mean = mean(bias_GLDAS_combo_warm_gcell_master)
    							bias_GLDAS_combo_warm_master.append(bias_GLDAS_combo_warm_gcell_mean)
    							stdev_GLDAS_combo_warm_gcell_mean = mean(stdev_GLDAS_combo_warm_gcell_master)
    							stdev_GLDAS_combo_warm_master.append(stdev_GLDAS_combo_warm_gcell_mean)
    							rmse_GLDAS_combo_warm_gcell_mean = mean(rmse_GLDAS_combo_warm_gcell_master)
    							rmse_GLDAS_combo_warm_master.append(rmse_GLDAS_combo_warm_gcell_mean)
    							corr_GLDAS_combo_warm_gcell_mean = mean(corr_GLDAS_combo_warm_gcell_master)
    							corr_GLDAS_combo_warm_master.append(corr_GLDAS_combo_warm_gcell_mean)



    						elif (len_i == 7):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]
    							model_6 = i[5]
    							model_7 = i[6]

    							blend_GLDAS_combo_temp_warm_gcell_master = []
    							bias_GLDAS_combo_warm_gcell_master = []
    							stdev_GLDAS_combo_warm_gcell_master = []
    							rmse_GLDAS_combo_warm_gcell_master = []
    							corr_GLDAS_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    								model_4_temp_warm = dframe_warm_season_gcell[model_4].values
    								model_5_temp_warm = dframe_warm_season_gcell[model_5].values
    								model_6_temp_warm = dframe_warm_season_gcell[model_6].values
    								model_7_temp_warm = dframe_warm_season_gcell[model_7].values
    								dframe_7_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_7_model[model_2] = model_2_temp_warm
    								dframe_7_model[model_3] = model_3_temp_warm
    								dframe_7_model[model_4] = model_4_temp_warm
    								dframe_7_model[model_5] = model_5_temp_warm
    								dframe_7_model[model_6] = model_6_temp_warm
    								dframe_7_model[model_7] = model_7_temp_warm
    								dframe_7_model_avg = dframe_7_model.mean(axis=1)
    								blend_7_model_temp_warm = dframe_7_model_avg
    								bias_7_model_warm = bias(blend_7_model_temp_warm,station_temp_warm)
    								bias_GLDAS_combo_warm_gcell_master.append(bias_7_model_warm)
    								stdev_7_model_warm = np.std(blend_7_model_temp_warm)
    								stdev_GLDAS_combo_warm_gcell_master.append(stdev_7_model_warm)
    								rmse_7_model_warm = mean_squared_error(station_temp_warm,blend_7_model_temp_warm,squared=False)
    								rmse_GLDAS_combo_warm_gcell_master.append(rmse_7_model_warm)
    								corr_7_model_warm,_ = pearsonr(blend_7_model_temp_warm,station_temp_warm)
    								corr_GLDAS_combo_warm_gcell_master.append(corr_7_model_warm)    							

    							bias_GLDAS_combo_warm_gcell_mean = mean(bias_GLDAS_combo_warm_gcell_master)
    							bias_GLDAS_combo_warm_master.append(bias_GLDAS_combo_warm_gcell_mean)
    							stdev_GLDAS_combo_warm_gcell_mean = mean(stdev_GLDAS_combo_warm_gcell_master)
    							stdev_GLDAS_combo_warm_master.append(stdev_GLDAS_combo_warm_gcell_mean)
    							rmse_GLDAS_combo_warm_gcell_mean = mean(rmse_GLDAS_combo_warm_gcell_master)
    							rmse_GLDAS_combo_warm_master.append(rmse_GLDAS_combo_warm_gcell_mean)
    							corr_GLDAS_combo_warm_gcell_mean = mean(corr_GLDAS_combo_warm_gcell_master)
    							corr_GLDAS_combo_warm_master.append(corr_GLDAS_combo_warm_gcell_mean)



    						elif (len_i == 8):
    							bias_8_model_warm = bias_naive_warm_mean
    							bias_GLDAS_combo_warm_master.append(bias_8_model_warm)
    							stdev_8_model_warm = stdev_naive_warm_mean
    							stdev_GLDAS_combo_warm_master.append(stdev_8_model_warm)
    							rmse_8_model_warm = rmse_naive_warm_mean 
    							rmse_GLDAS_combo_warm_master.append(rmse_8_model_warm)
    							corr_8_model_warm = corr_naive_warm_mean
    							corr_GLDAS_combo_warm_master.append(corr_8_model_warm)

    					bias_GLDAS_combo_warm_mean = mean(bias_GLDAS_combo_warm_master)
    					stdev_GLDAS_combo_warm_mean = mean(stdev_GLDAS_combo_warm_master)
    					SDV_GLDAS_combo_warm_mean = stdev_GLDAS_combo_warm_mean/stdev_station_warm
    					rmse_GLDAS_combo_warm_mean = mean(rmse_GLDAS_combo_warm_master)
    					corr_GLDAS_combo_warm_mean = mean(corr_GLDAS_combo_warm_master)


## GLDAS-CLSM Model ##

    					bias_GLDAS_CLSM_combo_warm_master = []
    					rmse_GLDAS_CLSM_combo_warm_master = []
    					stdev_GLDAS_CLSM_combo_warm_master = []
    					SDV_GLDAS_CLSM_combo_warm_master = []
    					corr_GLDAS_CLSM_combo_warm_master = []

    					for i in GLDAS_CLSM_array:
    						len_i = len(i)
    						if (len_i == 1):
    							blend_GLDAS_CLSM_combo_temp_warm_gcell_master = []
    							bias_GLDAS_CLSM_combo_warm_gcell_master = []
    							stdev_GLDAS_CLSM_combo_warm_gcell_master = []
    							rmse_GLDAS_CLSM_combo_warm_gcell_master = []
    							corr_GLDAS_CLSM_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								blend_1_model_temp_warm = dframe_warm_season_gcell[i[0]].values
    								print(blend_1_model_temp_warm)
    								print(station_temp_warm)
    								bias_1_model_warm = bias(blend_1_model_temp_warm,station_temp_warm)
    								bias_GLDAS_CLSM_combo_warm_gcell_master.append(bias_1_model_warm)
    								stdev_1_model_warm = np.std(blend_1_model_temp_warm)
    								stdev_GLDAS_CLSM_combo_warm_gcell_master.append(stdev_1_model_warm)
    								rmse_1_model_warm = mean_squared_error(station_temp_warm,blend_1_model_temp_warm,squared=False)
    								rmse_GLDAS_CLSM_combo_warm_gcell_master.append(rmse_1_model_warm)
    								corr_1_model_warm,_ = pearsonr(blend_1_model_temp_warm,station_temp_warm)
    								corr_GLDAS_CLSM_combo_warm_gcell_master.append(corr_1_model_warm)    							

    							bias_GLDAS_CLSM_combo_warm_gcell_mean = mean(bias_GLDAS_CLSM_combo_warm_gcell_master)
    							bias_GLDAS_CLSM_combo_warm_master.append(bias_GLDAS_CLSM_combo_warm_gcell_mean)
    							stdev_GLDAS_CLSM_combo_warm_gcell_mean = mean(stdev_GLDAS_CLSM_combo_warm_gcell_master)
    							stdev_GLDAS_CLSM_combo_warm_master.append(stdev_GLDAS_CLSM_combo_warm_gcell_mean)
    							rmse_GLDAS_CLSM_combo_warm_gcell_mean = mean(rmse_GLDAS_CLSM_combo_warm_gcell_master)
    							rmse_GLDAS_CLSM_combo_warm_master.append(rmse_GLDAS_CLSM_combo_warm_gcell_mean)
    							corr_GLDAS_CLSM_combo_warm_gcell_mean = mean(corr_GLDAS_CLSM_combo_warm_gcell_master)
    							corr_GLDAS_CLSM_combo_warm_master.append(corr_GLDAS_CLSM_combo_warm_gcell_mean)

    						elif (len_i == 2):
    							model_1 = i[0]
    							model_2 = i[1]

    							blend_GLDAS_CLSM_combo_temp_warm_gcell_master = []
    							bias_GLDAS_CLSM_combo_warm_gcell_master = []
    							stdev_GLDAS_CLSM_combo_warm_gcell_master = []
    							rmse_GLDAS_CLSM_combo_warm_gcell_master = []
    							corr_GLDAS_CLSM_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								dframe_2_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_2_model[model_2] = model_2_temp_warm
    								dframe_2_model_avg = dframe_2_model.mean(axis=1)
    								blend_2_model_temp_warm = dframe_2_model_avg
    								bias_2_model_warm = bias(blend_2_model_temp_warm,station_temp_warm)
    								bias_GLDAS_CLSM_combo_warm_gcell_master.append(bias_2_model_warm)
    								stdev_2_model_warm = np.std(blend_2_model_temp_warm)
    								stdev_GLDAS_CLSM_combo_warm_gcell_master.append(stdev_2_model_warm)
    								rmse_2_model_warm = mean_squared_error(station_temp_warm,blend_2_model_temp_warm,squared=False)
    								rmse_GLDAS_CLSM_combo_warm_gcell_master.append(rmse_2_model_warm)
    								corr_2_model_warm,_ = pearsonr(blend_2_model_temp_warm,station_temp_warm)
    								corr_GLDAS_CLSM_combo_warm_gcell_master.append(corr_2_model_warm)    							

    							bias_GLDAS_CLSM_combo_warm_gcell_mean = mean(bias_GLDAS_CLSM_combo_warm_gcell_master)
    							bias_GLDAS_CLSM_combo_warm_master.append(bias_GLDAS_CLSM_combo_warm_gcell_mean)
    							stdev_GLDAS_CLSM_combo_warm_gcell_mean = mean(stdev_GLDAS_CLSM_combo_warm_gcell_master)
    							stdev_GLDAS_CLSM_combo_warm_master.append(stdev_GLDAS_CLSM_combo_warm_gcell_mean)
    							rmse_GLDAS_CLSM_combo_warm_gcell_mean = mean(rmse_GLDAS_CLSM_combo_warm_gcell_master)
    							rmse_GLDAS_CLSM_combo_warm_master.append(rmse_GLDAS_CLSM_combo_warm_gcell_mean)
    							corr_GLDAS_CLSM_combo_warm_gcell_mean = mean(corr_GLDAS_CLSM_combo_warm_gcell_master)
    							corr_GLDAS_CLSM_combo_warm_master.append(corr_GLDAS_CLSM_combo_warm_gcell_mean)


    						elif (len_i == 3):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]

    							blend_GLDAS_CLSM_combo_temp_warm_gcell_master = []
    							bias_GLDAS_CLSM_combo_warm_gcell_master = []
    							stdev_GLDAS_CLSM_combo_warm_gcell_master = []
    							rmse_GLDAS_CLSM_combo_warm_gcell_master = []
    							corr_GLDAS_CLSM_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    								dframe_3_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_3_model[model_2] = model_2_temp_warm
    								dframe_3_model[model_3] = model_3_temp_warm
    								dframe_3_model_avg = dframe_3_model.mean(axis=1)
    								blend_3_model_temp_warm = dframe_3_model_avg
    								bias_3_model_warm = bias(blend_3_model_temp_warm,station_temp_warm)
    								bias_GLDAS_CLSM_combo_warm_gcell_master.append(bias_3_model_warm)
    								stdev_3_model_warm = np.std(blend_3_model_temp_warm)
    								stdev_GLDAS_CLSM_combo_warm_gcell_master.append(stdev_3_model_warm)
    								rmse_3_model_warm = mean_squared_error(station_temp_warm,blend_3_model_temp_warm,squared=False)
    								rmse_GLDAS_CLSM_combo_warm_gcell_master.append(rmse_3_model_warm)
    								corr_3_model_warm,_ = pearsonr(blend_3_model_temp_warm,station_temp_warm)
    								corr_GLDAS_CLSM_combo_warm_gcell_master.append(corr_3_model_warm)    							

    							bias_GLDAS_CLSM_combo_warm_gcell_mean = mean(bias_GLDAS_CLSM_combo_warm_gcell_master)
    							bias_GLDAS_CLSM_combo_warm_master.append(bias_GLDAS_CLSM_combo_warm_gcell_mean)
    							stdev_GLDAS_CLSM_combo_warm_gcell_mean = mean(stdev_GLDAS_CLSM_combo_warm_gcell_master)
    							stdev_GLDAS_CLSM_combo_warm_master.append(stdev_GLDAS_CLSM_combo_warm_gcell_mean)
    							rmse_GLDAS_CLSM_combo_warm_gcell_mean = mean(rmse_GLDAS_CLSM_combo_warm_gcell_master)
    							rmse_GLDAS_CLSM_combo_warm_master.append(rmse_GLDAS_CLSM_combo_warm_gcell_mean)
    							corr_GLDAS_CLSM_combo_warm_gcell_mean = mean(corr_GLDAS_CLSM_combo_warm_gcell_master)
    							corr_GLDAS_CLSM_combo_warm_master.append(corr_GLDAS_CLSM_combo_warm_gcell_mean)


    						elif (len_i == 4):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]

    							blend_GLDAS_CLSM_combo_temp_warm_gcell_master = []
    							bias_GLDAS_CLSM_combo_warm_gcell_master = []
    							stdev_GLDAS_CLSM_combo_warm_gcell_master = []
    							rmse_GLDAS_CLSM_combo_warm_gcell_master = []
    							corr_GLDAS_CLSM_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    								model_4_temp_warm = dframe_warm_season_gcell[model_4].values
    								dframe_4_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_4_model[model_2] = model_2_temp_warm
    								dframe_4_model[model_3] = model_3_temp_warm
    								dframe_4_model[model_4] = model_4_temp_warm
    								dframe_4_model_avg = dframe_4_model.mean(axis=1)
    								blend_4_model_temp_warm = dframe_4_model_avg
    								bias_4_model_warm = bias(blend_4_model_temp_warm,station_temp_warm)
    								bias_GLDAS_CLSM_combo_warm_gcell_master.append(bias_4_model_warm)
    								stdev_4_model_warm = np.std(blend_4_model_temp_warm)
    								stdev_GLDAS_CLSM_combo_warm_gcell_master.append(stdev_4_model_warm)
    								rmse_4_model_warm = mean_squared_error(station_temp_warm,blend_4_model_temp_warm,squared=False)
    								rmse_GLDAS_CLSM_combo_warm_gcell_master.append(rmse_4_model_warm)
    								corr_4_model_warm,_ = pearsonr(blend_4_model_temp_warm,station_temp_warm)
    								corr_GLDAS_CLSM_combo_warm_gcell_master.append(corr_4_model_warm)    							

    							bias_GLDAS_CLSM_combo_warm_gcell_mean = mean(bias_GLDAS_CLSM_combo_warm_gcell_master)
    							bias_GLDAS_CLSM_combo_warm_master.append(bias_GLDAS_CLSM_combo_warm_gcell_mean)
    							stdev_GLDAS_CLSM_combo_warm_gcell_mean = mean(stdev_GLDAS_CLSM_combo_warm_gcell_master)
    							stdev_GLDAS_CLSM_combo_warm_master.append(stdev_GLDAS_CLSM_combo_warm_gcell_mean)
    							rmse_GLDAS_CLSM_combo_warm_gcell_mean = mean(rmse_GLDAS_CLSM_combo_warm_gcell_master)
    							rmse_GLDAS_CLSM_combo_warm_master.append(rmse_GLDAS_CLSM_combo_warm_gcell_mean)
    							corr_GLDAS_CLSM_combo_warm_gcell_mean = mean(corr_GLDAS_CLSM_combo_warm_gcell_master)
    							corr_GLDAS_CLSM_combo_warm_master.append(corr_GLDAS_CLSM_combo_warm_gcell_mean)



    						elif (len_i == 5):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]

    							blend_GLDAS_CLSM_combo_temp_warm_gcell_master = []
    							bias_GLDAS_CLSM_combo_warm_gcell_master = []
    							stdev_GLDAS_CLSM_combo_warm_gcell_master = []
    							rmse_GLDAS_CLSM_combo_warm_gcell_master = []
    							corr_GLDAS_CLSM_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    								model_4_temp_warm = dframe_warm_season_gcell[model_4].values
    								model_5_temp_warm = dframe_warm_season_gcell[model_5].values
    								dframe_5_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_5_model[model_2] = model_2_temp_warm
    								dframe_5_model[model_3] = model_3_temp_warm
    								dframe_5_model[model_4] = model_4_temp_warm
    								dframe_5_model[model_5] = model_5_temp_warm
    								dframe_5_model_avg = dframe_5_model.mean(axis=1)
    								blend_5_model_temp_warm = dframe_5_model_avg
    								bias_5_model_warm = bias(blend_5_model_temp_warm,station_temp_warm)
    								bias_GLDAS_CLSM_combo_warm_gcell_master.append(bias_5_model_warm)
    								stdev_5_model_warm = np.std(blend_5_model_temp_warm)
    								stdev_GLDAS_CLSM_combo_warm_gcell_master.append(stdev_5_model_warm)
    								rmse_5_model_warm = mean_squared_error(station_temp_warm,blend_5_model_temp_warm,squared=False)
    								rmse_GLDAS_CLSM_combo_warm_gcell_master.append(rmse_5_model_warm)
    								corr_5_model_warm,_ = pearsonr(blend_5_model_temp_warm,station_temp_warm)
    								corr_GLDAS_CLSM_combo_warm_gcell_master.append(corr_5_model_warm)    							

    							bias_GLDAS_CLSM_combo_warm_gcell_mean = mean(bias_GLDAS_CLSM_combo_warm_gcell_master)
    							bias_GLDAS_CLSM_combo_warm_master.append(bias_GLDAS_CLSM_combo_warm_gcell_mean)
    							stdev_GLDAS_CLSM_combo_warm_gcell_mean = mean(stdev_GLDAS_CLSM_combo_warm_gcell_master)
    							stdev_GLDAS_CLSM_combo_warm_master.append(stdev_GLDAS_CLSM_combo_warm_gcell_mean)
    							rmse_GLDAS_CLSM_combo_warm_gcell_mean = mean(rmse_GLDAS_CLSM_combo_warm_gcell_master)
    							rmse_GLDAS_CLSM_combo_warm_master.append(rmse_GLDAS_CLSM_combo_warm_gcell_mean)
    							corr_GLDAS_CLSM_combo_warm_gcell_mean = mean(corr_GLDAS_CLSM_combo_warm_gcell_master)
    							corr_GLDAS_CLSM_combo_warm_master.append(corr_GLDAS_CLSM_combo_warm_gcell_mean)



    						elif (len_i == 6):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]
    							model_6 = i[5]

    							blend_GLDAS_CLSM_combo_temp_warm_gcell_master = []
    							bias_GLDAS_CLSM_combo_warm_gcell_master = []
    							stdev_GLDAS_CLSM_combo_warm_gcell_master = []
    							rmse_GLDAS_CLSM_combo_warm_gcell_master = []
    							corr_GLDAS_CLSM_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    								model_4_temp_warm = dframe_warm_season_gcell[model_4].values
    								model_5_temp_warm = dframe_warm_season_gcell[model_5].values
    								model_6_temp_warm = dframe_warm_season_gcell[model_6].values
    								dframe_6_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_6_model[model_2] = model_2_temp_warm
    								dframe_6_model[model_3] = model_3_temp_warm
    								dframe_6_model[model_4] = model_4_temp_warm
    								dframe_6_model[model_5] = model_5_temp_warm
    								dframe_6_model[model_6] = model_6_temp_warm
    								dframe_6_model_avg = dframe_6_model.mean(axis=1)
    								blend_6_model_temp_warm = dframe_6_model_avg
    								bias_6_model_warm = bias(blend_6_model_temp_warm,station_temp_warm)
    								bias_GLDAS_CLSM_combo_warm_gcell_master.append(bias_6_model_warm)
    								stdev_6_model_warm = np.std(blend_6_model_temp_warm)
    								stdev_GLDAS_CLSM_combo_warm_gcell_master.append(stdev_6_model_warm)
    								rmse_6_model_warm = mean_squared_error(station_temp_warm,blend_6_model_temp_warm,squared=False)
    								rmse_GLDAS_CLSM_combo_warm_gcell_master.append(rmse_6_model_warm)
    								corr_6_model_warm,_ = pearsonr(blend_6_model_temp_warm,station_temp_warm)
    								corr_GLDAS_CLSM_combo_warm_gcell_master.append(corr_6_model_warm)    							

    							bias_GLDAS_CLSM_combo_warm_gcell_mean = mean(bias_GLDAS_CLSM_combo_warm_gcell_master)
    							bias_GLDAS_CLSM_combo_warm_master.append(bias_GLDAS_CLSM_combo_warm_gcell_mean)
    							stdev_GLDAS_CLSM_combo_warm_gcell_mean = mean(stdev_GLDAS_CLSM_combo_warm_gcell_master)
    							stdev_GLDAS_CLSM_combo_warm_master.append(stdev_GLDAS_CLSM_combo_warm_gcell_mean)
    							rmse_GLDAS_CLSM_combo_warm_gcell_mean = mean(rmse_GLDAS_CLSM_combo_warm_gcell_master)
    							rmse_GLDAS_CLSM_combo_warm_master.append(rmse_GLDAS_CLSM_combo_warm_gcell_mean)
    							corr_GLDAS_CLSM_combo_warm_gcell_mean = mean(corr_GLDAS_CLSM_combo_warm_gcell_master)
    							corr_GLDAS_CLSM_combo_warm_master.append(corr_GLDAS_CLSM_combo_warm_gcell_mean)



    						elif (len_i == 7):
    							model_1 = i[0]
    							model_2 = i[1]
    							model_3 = i[2]
    							model_4 = i[3]
    							model_5 = i[4]
    							model_6 = i[5]
    							model_7 = i[6]

    							blend_GLDAS_CLSM_combo_temp_warm_gcell_master = []
    							bias_GLDAS_CLSM_combo_warm_gcell_master = []
    							stdev_GLDAS_CLSM_combo_warm_gcell_master = []
    							rmse_GLDAS_CLSM_combo_warm_gcell_master = []
    							corr_GLDAS_CLSM_combo_warm_gcell_master = []
    							for p in gcell_warm_uq:
    								if (p == 33777):
    									continue
    								dframe_warm_season_gcell = dframe_warm_season[dframe_warm_season['Grid Cell'] == p]
    								if (len(dframe_warm_season_gcell) < 2):
    									continue
    								station_temp_warm = dframe_warm_season_gcell['Station'].values
    								model_1_temp_warm = dframe_warm_season_gcell[model_1].values
    								model_2_temp_warm = dframe_warm_season_gcell[model_2].values
    								model_3_temp_warm = dframe_warm_season_gcell[model_3].values
    								model_4_temp_warm = dframe_warm_season_gcell[model_4].values
    								model_5_temp_warm = dframe_warm_season_gcell[model_5].values
    								model_6_temp_warm = dframe_warm_season_gcell[model_6].values
    								model_7_temp_warm = dframe_warm_season_gcell[model_7].values
    								dframe_7_model = pd.DataFrame(data=model_1_temp_warm, columns=[model_1])
    								dframe_7_model[model_2] = model_2_temp_warm
    								dframe_7_model[model_3] = model_3_temp_warm
    								dframe_7_model[model_4] = model_4_temp_warm
    								dframe_7_model[model_5] = model_5_temp_warm
    								dframe_7_model[model_6] = model_6_temp_warm
    								dframe_7_model[model_7] = model_7_temp_warm
    								dframe_7_model_avg = dframe_7_model.mean(axis=1)
    								blend_7_model_temp_warm = dframe_7_model_avg
    								bias_7_model_warm = bias(blend_7_model_temp_warm,station_temp_warm)
    								bias_GLDAS_CLSM_combo_warm_gcell_master.append(bias_7_model_warm)
    								stdev_7_model_warm = np.std(blend_7_model_temp_warm)
    								stdev_GLDAS_CLSM_combo_warm_gcell_master.append(stdev_7_model_warm)
    								rmse_7_model_warm = mean_squared_error(station_temp_warm,blend_7_model_temp_warm,squared=False)
    								rmse_GLDAS_CLSM_combo_warm_gcell_master.append(rmse_7_model_warm)
    								corr_7_model_warm,_ = pearsonr(blend_7_model_temp_warm,station_temp_warm)
    								corr_GLDAS_CLSM_combo_warm_gcell_master.append(corr_7_model_warm)    							

    							bias_GLDAS_CLSM_combo_warm_gcell_mean = mean(bias_GLDAS_CLSM_combo_warm_gcell_master)
    							bias_GLDAS_CLSM_combo_warm_master.append(bias_GLDAS_CLSM_combo_warm_gcell_mean)
    							stdev_GLDAS_CLSM_combo_warm_gcell_mean = mean(stdev_GLDAS_CLSM_combo_warm_gcell_master)
    							stdev_GLDAS_CLSM_combo_warm_master.append(stdev_GLDAS_CLSM_combo_warm_gcell_mean)
    							rmse_GLDAS_CLSM_combo_warm_gcell_mean = mean(rmse_GLDAS_CLSM_combo_warm_gcell_master)
    							rmse_GLDAS_CLSM_combo_warm_master.append(rmse_GLDAS_CLSM_combo_warm_gcell_mean)
    							corr_GLDAS_CLSM_combo_warm_gcell_mean = mean(corr_GLDAS_CLSM_combo_warm_gcell_master)
    							corr_GLDAS_CLSM_combo_warm_master.append(corr_GLDAS_CLSM_combo_warm_gcell_mean)



    						elif (len_i == 8):
    							bias_8_model_warm = bias_naive_warm_mean
    							bias_GLDAS_CLSM_combo_warm_master.append(bias_8_model_warm)
    							stdev_8_model_warm = stdev_naive_warm_mean
    							stdev_GLDAS_CLSM_combo_warm_master.append(stdev_8_model_warm)
    							rmse_8_model_warm = rmse_naive_warm_mean 
    							rmse_GLDAS_CLSM_combo_warm_master.append(rmse_8_model_warm)
    							corr_8_model_warm = corr_naive_warm_mean
    							corr_GLDAS_CLSM_combo_warm_master.append(corr_8_model_warm)

    					bias_GLDAS_CLSM_combo_warm_mean = mean(bias_GLDAS_CLSM_combo_warm_master)
    					stdev_GLDAS_CLSM_combo_warm_mean = mean(stdev_GLDAS_CLSM_combo_warm_master)
    					SDV_GLDAS_CLSM_combo_warm_mean = stdev_GLDAS_CLSM_combo_warm_mean/stdev_station_warm
    					rmse_GLDAS_CLSM_combo_warm_mean = mean(rmse_GLDAS_CLSM_combo_warm_master)
    					corr_GLDAS_CLSM_combo_warm_mean = mean(corr_GLDAS_CLSM_combo_warm_master)




##### Create Final Dataframe ####

    					dict_model_combo = {"Bias Cold Season": pd.Series([bias_CFSR_combo_cold_mean, bias_ERAI_combo_cold_mean,bias_ERA5_combo_cold_mean,bias_ERA5_Land_combo_cold_mean,bias_JRA_combo_cold_mean,bias_MERRA2_combo_cold_mean,bias_GLDAS_combo_cold_mean,bias_GLDAS_CLSM_combo_cold_mean],
					index=["CFSR","ERA-Interim","ERA5","ERA5-Land","JRA55","MERRA2","GLDAS-Noah","GLDAS-CLSM"]),
					"Bias Warm Season": pd.Series([bias_CFSR_combo_warm_mean, bias_ERAI_combo_warm_mean,bias_ERA5_combo_warm_mean,bias_ERA5_Land_combo_warm_mean,bias_JRA_combo_warm_mean,bias_MERRA2_combo_warm_mean,bias_GLDAS_combo_warm_mean,bias_GLDAS_CLSM_combo_warm_mean],
					index=["CFSR","ERA-Interim","ERA5","ERA5-Land","JRA55","MERRA2","GLDAS-Noah","GLDAS-CLSM"]), 
					"SDEV Cold Season": pd.Series([stdev_CFSR_combo_cold_mean, stdev_ERAI_combo_cold_mean,stdev_ERA5_combo_cold_mean,stdev_ERA5_Land_combo_cold_mean,stdev_JRA_combo_cold_mean,stdev_MERRA2_combo_cold_mean,stdev_GLDAS_combo_cold_mean,stdev_GLDAS_CLSM_combo_cold_mean],
					index=["CFSR","ERA-Interim","ERA5","ERA5-Land","JRA55","MERRA2","GLDAS-Noah","GLDAS-CLSM"]), 
					"SDEV Warm Season": pd.Series([stdev_CFSR_combo_warm_mean, stdev_ERAI_combo_warm_mean,stdev_ERA5_combo_warm_mean,stdev_ERA5_Land_combo_warm_mean,stdev_JRA_combo_warm_mean,stdev_MERRA2_combo_warm_mean,stdev_GLDAS_combo_warm_mean,stdev_GLDAS_CLSM_combo_warm_mean],
					index=["CFSR","ERA-Interim","ERA5","ERA5-Land","JRA55","MERRA2","GLDAS-Noah","GLDAS-CLSM"]), 
					"Norm SDV Cold Season": pd.Series([SDV_CFSR_combo_cold_mean, SDV_ERAI_combo_cold_mean,SDV_ERA5_combo_cold_mean,SDV_ERA5_Land_combo_cold_mean,SDV_JRA_combo_cold_mean,SDV_MERRA2_combo_cold_mean,SDV_GLDAS_combo_cold_mean,SDV_GLDAS_CLSM_combo_cold_mean],
					index=["CFSR","ERA-Interim","ERA5","ERA5-Land","JRA55","MERRA2","GLDAS-Noah","GLDAS-CLSM"]), 
					"Norm SDV Warm Season": pd.Series([SDV_CFSR_combo_warm_mean, SDV_ERAI_combo_warm_mean,SDV_ERA5_combo_warm_mean,SDV_ERA5_Land_combo_warm_mean,SDV_JRA_combo_warm_mean,SDV_MERRA2_combo_warm_mean,SDV_GLDAS_combo_warm_mean,SDV_GLDAS_CLSM_combo_warm_mean],
					index=["CFSR","ERA-Interim","ERA5","ERA5-Land","JRA55","MERRA2","GLDAS-Noah","GLDAS-CLSM"]), 
					"RMSE Cold Season": pd.Series([rmse_CFSR_combo_cold_mean, rmse_ERAI_combo_cold_mean,rmse_ERA5_combo_cold_mean,rmse_ERA5_Land_combo_cold_mean,rmse_JRA_combo_cold_mean,rmse_MERRA2_combo_cold_mean,rmse_GLDAS_combo_cold_mean,rmse_GLDAS_CLSM_combo_cold_mean],
					index=["CFSR","ERA-Interim","ERA5","ERA5-Land","JRA55","MERRA2","GLDAS-Noah","GLDAS-CLSM"]), 
					"RMSE Warm Season": pd.Series([rmse_CFSR_combo_warm_mean, rmse_ERAI_combo_warm_mean,rmse_ERA5_combo_warm_mean,rmse_ERA5_Land_combo_warm_mean,rmse_JRA_combo_warm_mean,rmse_MERRA2_combo_warm_mean,rmse_GLDAS_combo_warm_mean,rmse_GLDAS_CLSM_combo_warm_mean],
					index=["CFSR","ERA-Interim","ERA5","ERA5-Land","JRA55","MERRA2","GLDAS-Noah","GLDAS-CLSM"]), 
					"Pearson Correlation Cold Season": pd.Series([corr_CFSR_combo_cold_mean, corr_ERAI_combo_cold_mean,corr_ERA5_combo_cold_mean,corr_ERA5_Land_combo_cold_mean,corr_JRA_combo_cold_mean,corr_MERRA2_combo_cold_mean,corr_GLDAS_combo_cold_mean,corr_GLDAS_CLSM_combo_cold_mean],
					index=["CFSR","ERA-Interim","ERA5","ERA5-Land","JRA55","MERRA2","GLDAS-Noah","GLDAS-CLSM"]), 
					"Pearson Correlation Warm Season": pd.Series([corr_CFSR_combo_warm_mean, corr_ERAI_combo_warm_mean,corr_ERA5_combo_warm_mean,corr_ERA5_Land_combo_warm_mean,corr_JRA_combo_warm_mean,corr_MERRA2_combo_warm_mean,corr_GLDAS_combo_warm_mean,corr_GLDAS_CLSM_combo_warm_mean],
					index=["CFSR","ERA-Interim","ERA5","ERA5-Land","JRA55","MERRA2","GLDAS-Noah","GLDAS-CLSM"])}
    					df_model_combo = pd.DataFrame(dict_model_combo)
    					metrics_model_combo = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CMOS_poster/CLSM_res/'+str(remap_type)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_naive_metrics_new_data_CMOS_CLSM_Model_Combos_gcell_BEST_Sep2021_sites_1yr.csv'])
    					df_model_combo.to_csv(metrics_model_combo)

    					dict_model_number = {"Bias Cold Season": pd.Series([bias_1_model_cold_mean, bias_2_model_cold_mean,bias_3_model_cold_mean,bias_4_model_cold_mean,bias_5_model_cold_mean,bias_6_model_cold_mean,bias_7_model_cold_mean,bias_8_model_cold_mean],
					index=["1 Model","2 Model","3 Model","4 Model","5 Model","6 Model","7 Model","8 Model"]),
					"Bias Warm Season": pd.Series([bias_1_model_warm_mean, bias_2_model_warm_mean,bias_3_model_warm_mean,bias_4_model_warm_mean,bias_5_model_warm_mean,bias_6_model_warm_mean,bias_7_model_warm_mean,bias_8_model_warm_mean],
					index=["1 Model","2 Model","3 Model","4 Model","5 Model","6 Model","7 Model","8 Model"]), 
					"SDEV Cold Season": pd.Series([stdev_1_model_cold_mean, stdev_2_model_cold_mean,stdev_3_model_cold_mean,stdev_4_model_cold_mean,stdev_5_model_cold_mean,stdev_6_model_cold_mean,stdev_7_model_cold_mean,stdev_8_model_cold_mean],
					index=["1 Model","2 Model","3 Model","4 Model","5 Model","6 Model","7 Model","8 Model"]),
					"SDEV Warm Season": pd.Series([stdev_1_model_warm_mean, stdev_2_model_warm_mean,stdev_3_model_warm_mean,stdev_4_model_warm_mean,stdev_5_model_warm_mean,stdev_6_model_warm_mean,stdev_7_model_warm_mean,stdev_8_model_warm_mean],
					index=["1 Model","2 Model","3 Model","4 Model","5 Model","6 Model","7 Model","8 Model"]), 
					"Norm SDV Cold Season": pd.Series([SDV_1_model_cold_mean, SDV_2_model_cold_mean,SDV_3_model_cold_mean,SDV_4_model_cold_mean,SDV_5_model_cold_mean,SDV_6_model_cold_mean,SDV_7_model_cold_mean,SDV_8_model_cold_mean],
					index=["1 Model","2 Model","3 Model","4 Model","5 Model","6 Model","7 Model","8 Model"]), 
					"Norm SDV Warm Season": pd.Series([SDV_1_model_warm_mean, SDV_2_model_warm_mean,SDV_3_model_warm_mean,SDV_4_model_warm_mean,SDV_5_model_warm_mean,SDV_6_model_warm_mean,SDV_7_model_warm_mean,SDV_8_model_warm_mean],
					index=["1 Model","2 Model","3 Model","4 Model","5 Model","6 Model","7 Model","8 Model"]), 
					"RMSE Cold Season": pd.Series([rmse_1_model_cold_mean, rmse_2_model_cold_mean,rmse_3_model_cold_mean,rmse_4_model_cold_mean,rmse_5_model_cold_mean,rmse_6_model_cold_mean,rmse_7_model_cold_mean,rmse_8_model_cold_mean],
					index=["1 Model","2 Model","3 Model","4 Model","5 Model","6 Model","7 Model","8 Model"]), 
					"RMSE Warm Season": pd.Series([rmse_1_model_warm_mean, rmse_2_model_warm_mean,rmse_3_model_warm_mean,rmse_4_model_warm_mean,rmse_5_model_warm_mean,rmse_6_model_warm_mean,rmse_7_model_warm_mean,rmse_8_model_warm_mean],
					index=["1 Model","2 Model","3 Model","4 Model","5 Model","6 Model","7 Model","8 Model"]), 
					"Pearson Correlation Cold Season": pd.Series([corr_1_model_cold_mean, corr_2_model_cold_mean,corr_3_model_cold_mean,corr_4_model_cold_mean,corr_5_model_cold_mean,corr_6_model_cold_mean,corr_7_model_cold_mean,corr_8_model_cold_mean],
					index=["1 Model","2 Model","3 Model","4 Model","5 Model","6 Model","7 Model","8 Model"]), 
					"Pearson Correlation Warm Season": pd.Series([corr_1_model_warm_mean, corr_2_model_warm_mean,corr_3_model_warm_mean,corr_4_model_warm_mean,corr_5_model_warm_mean,corr_6_model_warm_mean,corr_7_model_warm_mean,corr_8_model_warm_mean],
					index=["1 Model","2 Model","3 Model","4 Model","5 Model","6 Model","7 Model","8 Model"])}
    					df_model_number =  pd.DataFrame(dict_model_number)
    					metrics_model_number = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/cold_warm_season/new_data/CMOS_poster/CLSM_res/'+str(remap_type)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_naive_metrics_new_data_CMOS_CLSM_Model_Number_gcell_BEST_Sep2021_sites_1yr.csv'])					
    					df_model_number.to_csv(metrics_model_number)
