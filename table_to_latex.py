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
from dateutil.relativedelta import *

########## Define global functions #########
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

############ Set Directories ############

base_dir = "/mnt/data/users/herringtont/soil_temp/naive_blended_metrics/grid_cell_stn/"
temp_type = ['raw_temp']
blend_type = ['simple_average']


############ loop through files ############

for i in temp_type:
    temp_type_i = i
    
    for j in blend_type:
    	blend_type_j = j
    	blend_dir = ''.join([base_dir+str(temp_type_i)+'/'+str(blend_type_j)+'/'])

    	pathlist = os.listdir(blend_dir)
    	pathlist_sorted = natural_sort(pathlist)

    	for path in pathlist_sorted:
    		metrics_fil = ''.join([blend_dir,path])
    		print(metrics_fil)
    		fil_basename = os.path.basename(metrics_fil)
    		fil_noext = os.path.splitext(fil_basename)
    		tex_fil = ''.join([str(blend_dir)+'tex/'+str(fil_noext[0])+'.tex'])
    		print(tex_fil)		
############ create LaTex files ############
    		df_metrics = pd.read_csv(metrics_fil)
    		gcell =  df_metrics['Grid Cell']
    		lat = df_metrics['Central Lat']
    		lat_rnd = [round(x,2) for x in lat]
    		lon = df_metrics['Central Lon']
    		lon_180 = [(((x + 180) % 360) - 180) for x in lon]
    		lon_rnd = [round(x,2) for x in lon_180]
		
    		naive_bias = df_metrics['Naive Blend Bias']
    		naive_bias_rnd = [round(x,3) for x in naive_bias]
    		CFSR_bias = df_metrics['CFSR Bias']
    		CFSR_bias_rnd = [round(x,3) for x in CFSR_bias]
    		ERAI_bias = df_metrics['ERA-Interim Bias']
    		ERAI_bias_rnd = [round(x,3) for x in ERAI_bias]
    		ERA5_bias = df_metrics['ERA5 Bias']
    		ERA5_bias_rnd = [round(x,3) for x in ERA5_bias]
    		JRA_bias = df_metrics['JRA-55 Bias']
    		JRA_bias_rnd = [round(x,3) for x in JRA_bias]
    		MERRA2_bias = df_metrics['MERRA2 Bias']
    		MERRA2_bias_rnd = [round(x,3) for x in MERRA2_bias]
    		GLDAS_bias = df_metrics['GLDAS Bias']
    		GLDAS_bias_rnd = [round(x,3) for x in GLDAS_bias]

    		naive_rmse = df_metrics['Naive Blend RMSE']
    		naive_rmse_rnd = [round(x,3) for x in naive_rmse]
    		CFSR_rmse = df_metrics['CFSR RMSE']
    		CFSR_rmse_rnd = [round(x,3) for x in CFSR_rmse]
    		ERAI_rmse = df_metrics['ERA-Interim RMSE']
    		ERAI_rmse_rnd = [round(x,3) for x in ERAI_rmse]
    		ERA5_rmse = df_metrics['ERA5 RMSE']
    		ERA5_rmse_rnd = [round(x,3) for x in ERA5_rmse]
    		JRA_rmse = df_metrics['JRA-55 RMSE']
    		JRA_rmse_rnd = [round(x,3) for x in JRA_rmse]
    		MERRA2_rmse = df_metrics['MERRA2 RMSE']
    		MERRA2_rmse_rnd = [round(x,3) for x in MERRA2_rmse]
    		GLDAS_rmse = df_metrics['GLDAS RMSE']
    		GLDAS_rmse_rnd = [round(x,3) for x in GLDAS_rmse]

    		naive_corr = df_metrics['Naive Blend corr']
    		naive_corr_rnd = [round(x,3) for x in naive_corr]
    		CFSR_corr = df_metrics['CFSR corr']
    		CFSR_corr_rnd = [round(x,3) for x in CFSR_corr]
    		ERAI_corr = df_metrics['ERA-Interim corr']
    		ERAI_corr_rnd = [round(x,3) for x in ERAI_corr]
    		ERA5_corr = df_metrics['ERA5 corr']
    		ERA5_corr_rnd = [round(x,3) for x in ERA5_corr]
    		JRA_corr = df_metrics['JRA-55 corr']
    		JRA_corr_rnd = [round(x,3) for x in JRA_corr]
    		MERRA2_corr = df_metrics['MERRA2 corr']
    		MERRA2_corr_rnd = [round(x,3) for x in MERRA2_corr]
    		GLDAS_corr = df_metrics['GLDAS corr']
    		GLDAS_corr_rnd = [round(x,3) for x in GLDAS_corr]

    		df_tex = pd.DataFrame(data=gcell, columns=['Grid Cell']) 
    		df_tex['Central Lat'] = lat_rnd
    		df_tex['Central Lon'] = lon_rnd
    		df_tex['Naive Blend Bias'] = naive_bias_rnd
    		df_tex['CFSR Bias'] = CFSR_bias_rnd
    		df_tex['ERA-Interim Bias'] = ERAI_bias_rnd
    		df_tex['ERA5 Bias'] = ERA5_bias_rnd
    		df_tex['JRA-55 Bias'] = JRA_bias_rnd
    		df_tex['MERRA2 Bias'] = MERRA2_bias_rnd
    		df_tex['GLDAS Bias'] = GLDAS_bias_rnd
    		df_tex['Naive Blend RMSE'] = naive_rmse_rnd
    		df_tex['CFSR RMSE'] = CFSR_rmse_rnd
    		df_tex['ERA-Interim RMSE'] = ERAI_rmse_rnd
    		df_tex['ERA5 RMSE'] = ERA5_rmse_rnd
    		df_tex['JRA-55 RMSE'] = JRA_rmse_rnd
    		df_tex['MERRA2 RMSE'] = MERRA2_rmse_rnd
    		df_tex['GLDAS RMSE'] = GLDAS_rmse_rnd
    		df_tex['Naive Blend corr'] = naive_corr_rnd
    		df_tex['CFSR corr'] = CFSR_corr_rnd
    		df_tex['ERA-Interim corr'] = ERAI_corr_rnd
    		df_tex['ERA5 corr'] = ERA5_corr_rnd
    		df_tex['JRA-55 corr'] = JRA_corr_rnd
    		df_tex['MERRA2 corr'] = MERRA2_corr_rnd
    		df_tex['GLDAS corr'] = GLDAS_corr_rnd
    		df_tex.to_latex(tex_fil)
