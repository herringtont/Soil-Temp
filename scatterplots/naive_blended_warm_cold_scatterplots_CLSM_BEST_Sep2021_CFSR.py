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

scatter_fil_top = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/CFSR_res/remapcon_top_30cm_thr_100_dframe_scatterplot_CMOS_CLSM_subset_permafrost_cold_warm_BEST_Sep2021_airtemp_CFSR.csv'])
scatter_fil_btm = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/CFSR_res/remapcon_30cm_300cm_thr_100_dframe_scatterplot_CMOS_CLSM_subset_permafrost_cold_warm_BEST_Sep2021_airtemp_CFSR.csv'])


dframe_cold_top = pd.read_csv(scatter_fil_top)
dframe_cold_top = dframe_cold_top[dframe_cold_top['Season']  == 'Cold']
dframe_cold_top['Layer'] = '0cm - 30cm'
station_cold_top = dframe_cold_top['Station'].values
naive_all_cold_top = dframe_cold_top['Ensemble Mean'].values
CFSR_cold_top = dframe_cold_top['CFSR'].values
ERA5_cold_top = dframe_cold_top['ERA5'].values
ERA5_Land_cold_top = dframe_cold_top['ERA5-Land'].values
GLDAS_cold_top = dframe_cold_top['GLDAS-Noah'].values

dframe_cold_btm = pd.read_csv(scatter_fil_btm)
dframe_cold_btm = dframe_cold_btm[dframe_cold_btm['Season']  == 'Cold']
dframe_cold_btm['Layer'] = '30cm - 300cm'
station_cold_btm = dframe_cold_btm['Station'].values
naive_all_cold_btm = dframe_cold_btm['Ensemble Mean'].values
CFSR_cold_btm = dframe_cold_btm['CFSR'].values
ERA5_cold_btm = dframe_cold_btm['ERA5'].values
ERA5_Land_cold_btm = dframe_cold_btm['ERA5-Land'].values
GLDAS_cold_btm = dframe_cold_btm['GLDAS-Noah'].values

dframe_warm_top = pd.read_csv(scatter_fil_btm)
dframe_warm_top = dframe_warm_top[dframe_warm_top['Season']  == 'Warm']
dframe_warm_top['Layer'] = '0cm - 30cm'
station_warm_top = dframe_warm_top['Station'].values
naive_all_warm_top = dframe_warm_top['Ensemble Mean'].values
CFSR_warm_top = dframe_warm_top['CFSR'].values
ERA5_warm_top = dframe_warm_top['ERA5'].values
ERA5_Land_warm_top = dframe_warm_top['ERA5-Land'].values
GLDAS_warm_top = dframe_warm_top['GLDAS-Noah'].values

dframe_warm_btm = pd.read_csv(scatter_fil_btm)
dframe_warm_btm = dframe_warm_btm[dframe_warm_btm['Season']  == 'Warm']
dframe_warm_btm['Layer'] = '30cm - 300cm'
station_warm_btm = dframe_warm_btm['Station'].values
naive_all_warm_btm = dframe_warm_btm['Ensemble Mean'].values
CFSR_warm_btm = dframe_warm_btm['CFSR'].values
ERA5_warm_btm = dframe_warm_btm['ERA5'].values
ERA5_Land_warm_btm = dframe_warm_btm['ERA5-Land'].values
GLDAS_warm_btm = dframe_warm_btm['GLDAS-Noah'].values

dframe_scatter_top = pd.read_csv(scatter_fil_top)
dframe_scatter_top['Layer'] = '0cm - 30cm'
station_scatter_top = dframe_scatter_top['Station'].values
naive_all_scatter_top = dframe_scatter_top['Ensemble Mean'].values
CFSR_scatter_top = dframe_scatter_top['CFSR'].values
ERA5_scatter_top = dframe_scatter_top['ERA5'].values
ERA5_Land_scatter_top = dframe_scatter_top['ERA5-Land'].values
GLDAS_scatter_top = dframe_scatter_top['GLDAS-Noah'].values
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
naive_all_scatter_btm = dframe_scatter_btm['Ensemble Mean'].values
CFSR_scatter_btm = dframe_scatter_btm['CFSR'].values
ERA5_scatter_btm = dframe_scatter_btm['ERA5'].values
ERA5_Land_scatter_btm = dframe_scatter_btm['ERA5-Land'].values
GLDAS_scatter_btm = dframe_scatter_btm['GLDAS-Noah'].values
season_scatter_btm = dframe_scatter_btm['Season'].values
layer_scatter_btm =  dframe_scatter_btm['Layer'].values


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
ERA5_master.append(ERA5_scatter_top)
ERA5_master.append(ERA5_scatter_btm)
ERA5_Land_master.append(ERA5_Land_scatter_top)
ERA5_Land_master.append(ERA5_Land_scatter_btm)
GLDAS_master.append(GLDAS_scatter_top)
GLDAS_master.append(GLDAS_scatter_btm)
season_master.append(season_scatter_top)
season_master.append(season_scatter_btm)
season_depth_master.append(season_depth_top)
season_depth_master.append(season_depth_btm)
layer_master.append(layer_scatter_top)
layer_master.append(layer_scatter_btm)

station_master = [i for sub in station_master for i in sub]
naive_all_master = [i for sub in naive_all_master for i in sub]
CFSR_master = [i for sub in CFSR_master for i in sub]
ERA5_master = [i for sub in ERA5_master for i in sub]
ERA5_Land_master = [i for sub in ERA5_Land_master for i in sub]
GLDAS_master = [i for sub in GLDAS_master for i in sub]
season_master = [i for sub in season_master for i in sub]
season_depth_master = [i for sub in season_depth_master for i in sub]
layer_master = [i for sub in layer_master for i in sub]

dframe_master = pd.DataFrame({'Station':station_master,'Season':season_master,'Layer':layer_master,'Ensemble Mean':naive_all_master,'CFSR':CFSR_master,'ERA5':ERA5_master,'ERA5-Land':ERA5_Land_master,'GLDAS-Noah':GLDAS_master})
total_fil = '/mnt/data/users/herringtont/soil_temp/scripts/dframe_master.csv'
dframe_master.to_csv(total_fil,index=False,na_rep=np.nan)

print(dframe_master)

dframe_scatter_top2 = pd.DataFrame({'Station':station_scatter_top,'Season':season_scatter_top,'Ensemble Mean':naive_all_scatter_top,'CFSR':CFSR_scatter_top,'ERA5':ERA5_scatter_top,'ERA5-Land':ERA5_Land_scatter_top,'GLDAS-Noah':GLDAS_scatter_top})
dframe_scatter_btm2 = pd.DataFrame({'Station':station_scatter_btm,'Season':season_scatter_btm,'Ensemble Mean':naive_all_scatter_btm,'CFSR':CFSR_scatter_btm,'ERA5':ERA5_scatter_btm,'ERA5-Land':ERA5_Land_scatter_btm,'GLDAS-Noah':GLDAS_scatter_btm})

top_fil = '/mnt/data/users/herringtont/soil_temp/scripts/dframe_top_Sep2021_CFSR.csv'
btm_fil = '/mnt/data/users/herringtont/soil_temp/scripts/dframe_btm_Sep2021_CFSR.csv'
dframe_scatter_top2.to_csv(top_fil,index=False,na_rep=np.nan)
dframe_scatter_btm2.to_csv(btm_fil,index=False,na_rep=np.nan)

#################### Create Scatterplot Matrices ####################					

mpl.rcParams['font.size'] = 25

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
plt.savefig('/mnt/data/users/herringtont/soil_temp/plots/naive_blend_scatterplots/CFSR_res/'+str(remap_type)+'_thr_100_scatterplot_GHCN_Sep2021_CFSR.tiff',format='tiff',dpi=1000)
plt.close()








































