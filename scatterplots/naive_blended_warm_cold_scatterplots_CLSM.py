import os
import glob
import netCDF4
import csv
import datetime
import matplotlib as mpl
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


########## Define Functions ##########

def scatter_subset(x, y, hue, mask, **kws):
    sns.scatterplot(x=x[mask], y=y[mask], hue=hue[mask], **kws)
    
    
############# Set Directories ############

naive_type = 'simple_average'
olr = 'zscore'
lyr_top = 'top_30cm'
lyr_btm = '30cm_300cm'
thr = '100'
remap_type = 'remapcon'
temp_thr = '-2C'

#sns.set_context(rc={"axes.labelsize":20}, font_scale=1.0)
############# Grab Data ###############

cold_fil_top = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/CLSM_res/subset/remapcon_simple_average_zscore_top_30cm_thr_100_dframe_cold_season_temp_master_BEST_-2C_CMOS_CLSM_subset_permafrost.csv'])
cold_fil_btm = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/CLSM_res/subset/remapcon_simple_average_zscore_30cm_300cm_thr_100_dframe_cold_season_temp_master_BEST_-2C_CMOS_CLSM_subset_permafrost.csv'])

warm_fil_top = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/CLSM_res/subset/remapcon_simple_average_zscore_top_30cm_thr_100_dframe_warm_season_temp_master_BEST_-2C_CMOS_CLSM_subset_permafrost.csv'])
warm_fil_btm = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/CLSM_res/subset/remapcon_simple_average_zscore_30cm_300cm_thr_100_dframe_warm_season_temp_master_BEST_-2C_CMOS_CLSM_subset_permafrost.csv'])

scatter_fil_top = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/CLSM_res/subset/remapcon_top_30cm_thr_100_dframe_scatterplot_CMOS_CLSM_subset_permafrost_cold_warm_BEST.csv'])
scatter_fil_btm = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/CLSM_res/subset/remapcon_30cm_300cm_thr_100_dframe_scatterplot_CMOS_CLSM_subset_permafrost_cold_warm_BEST.csv'])


dframe_cold_top = pd.read_csv(cold_fil_top)
dframe_cold_top['Layer'] = '0cm - 30cm'
station_cold_top = dframe_cold_top['Station'].values
naive_cold_top = dframe_cold_top['Naive Blend'].values
naive_noJRA_cold_top = dframe_cold_top['Naive Blend no JRA55'].values
naive_noJRAold_cold_top = dframe_cold_top['Naive Blend no JRA55 Old'].values
naive_all_cold_top = dframe_cold_top['Naive Blend All'].values
CFSR_cold_top = dframe_cold_top['CFSR'].values
ERAI_cold_top = dframe_cold_top['ERA-Interim'].values
ERA5_cold_top = dframe_cold_top['ERA5'].values
ERA5_Land_cold_top = dframe_cold_top['ERA5-Land'].values
JRA_cold_top = dframe_cold_top['JRA55'].values
MERRA2_cold_top = dframe_cold_top['MERRA2'].values
GLDAS_cold_top = dframe_cold_top['GLDAS-Noah'].values
GLDAS_CLSM_cold_top = dframe_cold_top['GLDAS-CLSM'].values

dframe_cold_btm = pd.read_csv(cold_fil_btm)
dframe_cold_btm['Layer'] = '30cm - 300cm'
station_cold_btm = dframe_cold_btm['Station'].values
naive_cold_btm = dframe_cold_btm['Naive Blend'].values
naive_noJRA_cold_btm = dframe_cold_btm['Naive Blend no JRA55'].values
naive_noJRAold_cold_btm = dframe_cold_btm['Naive Blend no JRA55 Old'].values
naive_all_cold_btm = dframe_cold_btm['Naive Blend All'].values
CFSR_cold_btm = dframe_cold_btm['CFSR'].values
ERAI_cold_btm = dframe_cold_btm['ERA-Interim'].values
ERA5_cold_btm = dframe_cold_btm['ERA5'].values
ERA5_Land_cold_btm = dframe_cold_btm['ERA5-Land'].values
JRA_cold_btm = dframe_cold_btm['JRA55'].values
MERRA2_cold_btm = dframe_cold_btm['MERRA2'].values
GLDAS_cold_btm = dframe_cold_btm['GLDAS-Noah'].values
GLDAS_CLSM_cold_btm = dframe_cold_btm['GLDAS-CLSM'].values

dframe_warm_top = pd.read_csv(warm_fil_top)
dframe_warm_top['Layer'] = '0cm - 30cm'
station_warm_top = dframe_warm_top['Station'].values
naive_warm_top = dframe_warm_top['Naive Blend'].values
naive_noJRA_warm_top = dframe_warm_top['Naive Blend no JRA55'].values
naive_noJRAold_warm_top = dframe_warm_top['Naive Blend no JRA55 Old'].values
naive_all_warm_top = dframe_warm_top['Naive Blend All'].values
CFSR_warm_top = dframe_warm_top['CFSR'].values
ERAI_warm_top = dframe_warm_top['ERA-Interim'].values
ERA5_warm_top = dframe_warm_top['ERA5'].values
ERA5_Land_warm_top = dframe_warm_top['ERA5-Land'].values
JRA_warm_top = dframe_warm_top['JRA55'].values
MERRA2_warm_top = dframe_warm_top['MERRA2'].values
GLDAS_warm_top = dframe_warm_top['GLDAS-Noah'].values
GLDAS_CLSM_warm_top = dframe_warm_top['GLDAS-CLSM'].values

dframe_warm_btm = pd.read_csv(warm_fil_top)
dframe_warm_btm['Layer'] = '30cm - 300cm'
station_warm_btm = dframe_warm_btm['Station'].values
naive_warm_btm = dframe_warm_btm['Naive Blend'].values
naive_noJRA_warm_btm = dframe_warm_btm['Naive Blend no JRA55'].values
naive_noJRAold_warm_btm = dframe_warm_btm['Naive Blend no JRA55 Old'].values
naive_all_warm_btm = dframe_warm_btm['Naive Blend All'].values
CFSR_warm_btm = dframe_warm_btm['CFSR'].values
ERAI_warm_btm = dframe_warm_btm['ERA-Interim'].values
ERA5_warm_btm = dframe_warm_btm['ERA5'].values
ERA5_Land_warm_btm = dframe_warm_btm['ERA5-Land'].values
JRA_warm_btm = dframe_warm_btm['JRA55'].values
MERRA2_warm_btm = dframe_warm_btm['MERRA2'].values
GLDAS_warm_btm = dframe_warm_btm['GLDAS-Noah'].values
GLDAS_CLSM_warm_btm = dframe_warm_btm['GLDAS-CLSM'].values

dframe_scatter_top = pd.read_csv(scatter_fil_top)
dframe_scatter_top['Layer'] = '0cm - 30cm'
station_scatter_top = dframe_scatter_top['Station'].values
naive_scatter_top = dframe_scatter_top['Naive Blend'].values
naive_noJRA_scatter_top = dframe_scatter_top['Naive Blend no JRA55'].values
naive_noJRAold_scatter_top = dframe_scatter_top['Naive Blend no JRA55 Old'].values
naive_all_scatter_top = dframe_scatter_top['Naive Blend All'].values
CFSR_scatter_top = dframe_scatter_top['CFSR'].values
ERAI_scatter_top = dframe_scatter_top['ERA-Interim'].values
ERA5_scatter_top = dframe_scatter_top['ERA5'].values
ERA5_Land_scatter_top = dframe_scatter_top['ERA5-Land'].values
JRA_scatter_top = dframe_scatter_top['JRA55'].values
MERRA2_scatter_top = dframe_scatter_top['MERRA2'].values
GLDAS_scatter_top = dframe_scatter_top['GLDAS-Noah'].values
GLDAS_CLSM_scatter_top = dframe_scatter_top['GLDAS-CLSM'].values
season_scatter_top = dframe_scatter_top['Season'].values
layer_scatter_top =  dframe_scatter_top['Layer'].values


season_depth_top = []
for i in season_scatter_top:
    seas_i = i
    if (seas_i == 'Cold'):
    	seas_dep = 'Cold_Top'

    elif (seas_i == 'Warm'):
    	seas_dep = 'Warm_Top'

    season_depth_top.append(seas_dep)
    
dframe_scatter_btm = pd.read_csv(scatter_fil_btm)
dframe_scatter_btm['Layer'] = '30cm - 300cm'
station_scatter_btm = dframe_scatter_btm['Station'].values
naive_scatter_btm = dframe_scatter_btm['Naive Blend'].values
naive_noJRA_scatter_btm = dframe_scatter_btm['Naive Blend no JRA55'].values
naive_noJRAold_scatter_btm = dframe_scatter_btm['Naive Blend no JRA55 Old'].values
naive_all_scatter_btm = dframe_scatter_btm['Naive Blend All'].values
CFSR_scatter_btm = dframe_scatter_btm['CFSR'].values
ERAI_scatter_btm = dframe_scatter_btm['ERA-Interim'].values
ERA5_scatter_btm = dframe_scatter_btm['ERA5'].values
ERA5_Land_scatter_btm = dframe_scatter_btm['ERA5-Land'].values
JRA_scatter_btm = dframe_scatter_btm['JRA55'].values
MERRA2_scatter_btm = dframe_scatter_btm['MERRA2'].values
GLDAS_scatter_btm = dframe_scatter_btm['GLDAS-Noah'].values
GLDAS_CLSM_scatter_btm = dframe_scatter_btm['GLDAS-CLSM'].values
season_scatter_btm = dframe_scatter_btm['Season'].values
layer_scatter_btm = dframe_scatter_btm['Layer'].values


season_depth_btm = []
for i in season_scatter_btm:
    seas_i = i
    if (seas_i == 'Cold'):
    	seas_dep = 'Cold_Btm'

    elif (seas_i == 'Warm'):
    	seas_dep = 'Warm_Btm'

    season_depth_btm.append(seas_dep)

station_master = []
naive_all_master = []
CFSR_master = []
ERAI_master = []
ERA5_master = []
ERA5_Land_master = []
JRA_master = []
MERRA2_master = []
GLDAS_master = []
GLDAS_CLSM_master = []
season_master = []
layer_master = []
size_master = []
season_depth_master = []

station_master.append(station_scatter_top)
station_master.append(station_scatter_btm)
naive_all_master.append(naive_all_scatter_top)
naive_all_master.append(naive_all_scatter_btm)
CFSR_master.append(CFSR_scatter_top)
CFSR_master.append(CFSR_scatter_btm)
ERAI_master.append(ERAI_scatter_top)
ERAI_master.append(ERAI_scatter_btm)
ERA5_master.append(ERA5_scatter_top)
ERA5_master.append(ERA5_scatter_btm)
ERA5_Land_master.append(ERA5_Land_scatter_top)
ERA5_Land_master.append(ERA5_Land_scatter_btm)
JRA_master.append(JRA_scatter_top)
JRA_master.append(JRA_scatter_btm)
MERRA2_master.append(MERRA2_scatter_top)
MERRA2_master.append(MERRA2_scatter_btm)
GLDAS_master.append(GLDAS_scatter_top)
GLDAS_master.append(GLDAS_scatter_btm)
GLDAS_CLSM_master.append(GLDAS_CLSM_scatter_top)
GLDAS_CLSM_master.append(GLDAS_CLSM_scatter_btm)
season_master.append(season_scatter_top)
season_master.append(season_scatter_btm)
season_depth_master.append(season_depth_top)
season_depth_master.append(season_depth_btm)
layer_master.append(layer_scatter_top)
layer_master.append(layer_scatter_btm)

station_master = [i for sub in station_master for i in sub]
naive_all_master = [i for sub in naive_all_master for i in sub]
CFSR_master = [i for sub in CFSR_master for i in sub]
ERAI_master = [i for sub in ERAI_master for i in sub]
ERA5_master = [i for sub in ERA5_master for i in sub]
ERA5_Land_master = [i for sub in ERA5_Land_master for i in sub]
JRA_master = [i for sub in JRA_master for i in sub]
MERRA2_master = [i for sub in MERRA2_master for i in sub]
GLDAS_master = [i for sub in GLDAS_master for i in sub]
GLDAS_CLSM_master = [i for sub in GLDAS_CLSM_master for i in sub]
season_master = [i for sub in season_master for i in sub]
season_depth_master = [i for sub in season_depth_master for i in sub]
layer_master = [i for sub in layer_master for i in sub]

dframe_master = pd.DataFrame({'Station':station_master,'Season':season_master,'Layer':layer_master,'Ensemble Mean':naive_all_master,'CFSR':CFSR_master,'ERA-Interim':ERAI_master,'ERA5':ERA5_master,'ERA5-Land':ERA5_Land_master,'JRA55':JRA_master,'MERRA2':MERRA2_master,'GLDAS-Noah':GLDAS_master,'GLDAS-CLSM':GLDAS_CLSM_master})
total_fil = '/mnt/data/users/herringtont/soil_temp/scripts/dframe_master.csv'
dframe_master.to_csv(total_fil,index=False,na_rep=np.nan)

print(dframe_master)

dframe_scatter_top2 = pd.DataFrame({'Station':station_scatter_top,'Season':season_scatter_top,'Ensemble Mean':naive_all_scatter_top,'CFSR':CFSR_scatter_top,'ERA-Interim':ERAI_scatter_top,'ERA5':ERA5_scatter_top,'ERA5-Land':ERA5_Land_scatter_top,'JRA55':JRA_scatter_top,'MERRA2':MERRA2_scatter_top,'GLDAS-Noah':GLDAS_scatter_top,'GLDAS-CLSM':GLDAS_CLSM_scatter_top})
dframe_scatter_btm2 = pd.DataFrame({'Station':station_scatter_btm,'Season':season_scatter_btm,'Ensemble Mean':naive_all_scatter_btm,'CFSR':CFSR_scatter_btm,'ERA-Interim':ERAI_scatter_btm,'ERA5':ERA5_scatter_btm,'ERA5-Land':ERA5_Land_scatter_btm,'JRA55':JRA_scatter_btm,'MERRA2':MERRA2_scatter_btm,'GLDAS-Noah':GLDAS_scatter_btm,'GLDAS-CLSM':GLDAS_CLSM_scatter_btm})

top_fil = '/mnt/data/users/herringtont/soil_temp/scripts/dframe_top.csv'
btm_fil = '/mnt/data/users/herringtont/soil_temp/scripts/dframe_btm.csv'
dframe_scatter_top2.to_csv(top_fil,index=False,na_rep=np.nan)
dframe_scatter_btm2.to_csv(btm_fil,index=False,na_rep=np.nan)

#################### Create Scatterplot Matrices ####################					

mpl.rcParams['font.size'] = 45

scatter1 = sn.PairGrid(dframe_master,hue='Season')


sns.set_palette(sns.color_palette('Set2'))
#scatter1.map_upper(scatter_subset, mask=dframe_master['Layer'] == '0cm - 30cm', cmap=plt.get_cmap('Pastel1'))
scatter1.map_upper(scatter_subset, mask=dframe_master['Layer'] == '0cm - 30cm')
sns.set_palette(sns.color_palette('Dark2'))
#scatter1.map_diag(sns.histplot, legend = False, color=plt.get_cmap('Pastel1'))
scatter1.map_diag(sns.histplot, legend = False)
#scatter1.map_lower(scatter_subset, mask=dframe_master['Layer'] == '30cm - 300cm',cmap=plt.get_cmap('Set1'))
sns.set_palette(sns.color_palette('Dark2'))
scatter1.map_lower(scatter_subset, mask=dframe_master['Layer'] == '30cm - 300cm')


scatter1.set(xlim=(-40,40),ylim=(-40,40))
plt.savefig('/mnt/data/users/herringtont/soil_temp/plots/naive_blend_scatterplots/new_data/CMOS_poster/CLSM_res/'+str(remap_type)+'_thr_100_scatterplot_all_temp_new_data_CMOS_CLSM_all_layers.png')
plt.close()








































